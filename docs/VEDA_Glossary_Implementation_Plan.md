# VEDA — Domain-Aware (Vertical) Glossary — Implementation Plan
## Including the admin-panel vertical selector

## Confirmed problem (verified in code)

Glossary generation is hardcoded to ONE vertical (BFSI/AML/compliance) at every
layer, regardless of what database is actually connected:

| Layer | File | Hardcoded to |
|---|---|---|
| LLM prompt (Stage 2 + Stage 4) | `config.py::GLOSSARY_DOMAIN_DESCRIPTION` | "Compliance and risk management, fraud detection, AML/KYC, incident investigation" — one global string for every source |
| Layer C — static glossary | `ingestion/domain_glossary.py::STATIC_AML_GLOSSARY` | `sar`, `ctr`, `pep`, `kyc`, `aml`, `cdd`, `edd`, `ubo`, `ofac`, `fatf`... |
| Layer B — HF dataset | `ingestion/domain_glossary.py` | comment: "BFSI/AML domain vocabulary" |

`apps/sources/models.py::Source` has **no industry/vertical field at all**. Point
a Real Estate DB at VEDA today and it still gets AML/KYC framing and AML terms
injected into its synonym index.

## Decision (confirmed)
- **Explicit selection** at source registration — no auto-detection.
- **5 verticals:** BFSI, Real Estate, Healthcare, Retail, Generic (default/fallback).
- Selectable **in the Django admin panel**, where `Source` rows are already
  managed (`apps/sources/admin.py::SourceAdmin`).

---

## Design

### 1. New field on `Source` + admin panel dropdown

```python
# apps/sources/models.py
class IndustryVertical(models.TextChoices):
    BFSI        = "bfsi",        "BFSI / Banking & Financial Services"
    REAL_ESTATE = "real_estate", "Real Estate"
    HEALTHCARE  = "healthcare",  "Healthcare"
    RETAIL      = "retail",      "Retail"
    GENERIC     = "generic",     "Generic / Other"

class Source(models.Model):
    ...
    industry_vertical = models.CharField(
        max_length=32, choices=IndustryVertical.choices,
        default=IndustryVertical.GENERIC,
        help_text="Drives which domain glossary and LLM domain-framing is used "
                  "during L3 enrichment. Set once at registration.")
```

Because this is a `TextChoices`-backed `CharField`, **Django admin automatically
renders it as a `<select>` dropdown** in the add/change form — no custom widget
needed. Two small `SourceAdmin` additions make it visible in the list view too:

```python
# apps/sources/admin.py
@admin.register(Source)
class SourceAdmin(admin.ModelAdmin):
    list_display = ("name", "dialect", "industry_vertical", "status", "ready", "last_ingested_at")
    list_filter  = ("dialect", "industry_vertical", "status", "ready")
    actions = ["ingest", "test_connection"]
    ...
```
`list_filter` adds a right-hand sidebar filter so an admin can see/filter all
sources by vertical at a glance — useful once there are several sources
registered across different verticals.

**Admin workflow (end-to-end):**
1. Admin opens `Source` → "Add Source" (or edits an existing one) in Django admin.
2. Fills connection fields as today (`host`, `dbname`, `dialect`, etc.).
3. Picks **Industry vertical** from the new dropdown (defaults to "Generic / Other"
   if left unset).
4. Saves. Runs the existing "Ingest source" admin action.
5. The selected vertical rides through to ingestion automatically — no other
   manual step.

### 2. Thread it through to L3 (the only consumer) — reuses the existing env-injection rail

```
Source.industry_vertical
   └─ Source.as_engine_env() → VEDA_INDUSTRY_VERTICAL=<value>   (same rail as DB connection injection)
        └─ SourceContext.industry_vertical (new field, read in from_env())
             └─ l3_enrich.run(ctx, ...) → run_full_semantic_layer(..., industry_vertical=ctx.industry_vertical)
                  ├─ glossary_builder.load_or_generate_glossary(industry_vertical=...)   [Stage 2]
                  └─ stage4_column_understanding(..., industry_vertical=...)              [Stage 4 domain framing]
```
This is the SAME mechanism already used for per-source DB connection injection
(`as_engine_env()` → subprocess env → `SourceContext.from_env()`) — no new
plumbing pattern, one more field riding the same rail.

### 3. Vertical-keyed domain description

```python
# config.py — replaces the single GLOSSARY_DOMAIN_DESCRIPTION default
VERTICAL_DOMAIN_DESCRIPTIONS = {
    "bfsi":        "Compliance and risk management, fraud detection, AML/KYC, incident investigation",
    "real_estate": "Real estate transactions, property listings, leasing, sales pipeline, brokerage",
    "healthcare":  "Patient care, clinical records, appointments, billing, compliance (HIPAA)",
    "retail":      "Inventory, orders, customers, point-of-sale, merchandising, fulfillment",
    "generic":     "General business operations",
}
```

### 4. Glossary registry (replaces the single AML-only Layer B/C)

```python
STATIC_GLOSSARY_REGISTRY = {
    "bfsi":        STATIC_AML_GLOSSARY,           # existing, unchanged
    "real_estate": STATIC_REAL_ESTATE_GLOSSARY,   # NEW
    "healthcare":  STATIC_HEALTHCARE_GLOSSARY,    # NEW
    "retail":      STATIC_RETAIL_GLOSSARY,        # NEW
    "generic":     {},                            # no static injection
}
```
Layer A (SLM-generated synonyms from the connected schema's own columns) is
**unchanged** — already schema-derived, works for any vertical as-is.

### 5. Real HF datasets selected (verified on HF Hub, not invented)

| Vertical | Dataset | Extraction |
|---|---|---|
| BFSI | `PolyAI/banking77`, `financial_phrasebank`, `gbharti/finance-alpaca` | unchanged (existing) |
| Real Estate | `divarofficial/real-estate-ads` (~1M listings; fields: `property_type`, `city_slug`, `user_type`, `description`, `title`) | field VALUES → glossary terms |
| Retail | `bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset` (intents: `ORDER: cancel_order, track_order`; `DELIVERY: track_delivery`; `PRODUCT: exchange_product`) | near drop-in reuse of the banking77 label-split extractor |
| Healthcare | `gretelai/symptom_to_diagnosis` (1,065 rows, 22 diagnosis labels — same shape as banking77); optional supplement `gamino/wiki_medical_terms` | same banking77-style label extraction |
| Generic | none (by design) | Layer A only |

Curated Layer-C additions (~15-25 terms each):
- **Real Estate:** `RERA`, `carpet_area`, `built_up_area`, `possession_date`,
  `broker_commission`, `lease_term`, `security_deposit`, `inventory`,
  `listing_status`, `sale_type`.
- **Retail:** `SKU`, `inventory_count`, `order_status`, `refund_status`,
  `fulfillment`, `POS`, `discount_code`.
- **Healthcare:** `ICD_code`, `admission_date`, `discharge_date`,
  `diagnosis_code`, `EMR`, `HIPAA`, `patient_id`, `provider_id`.

### 6. Per-vertical/per-source cache scoping

`domain_glossary.py`'s cache files (`slm_glossary.json`, `hf_glossary.json`,
`static_glossary.json`, `domain_glossary.json`) must not collide across
sources. Route them through the existing `artifact_path()` / artifact-scope
mechanism (already used for `SEMANTIC_MODEL_FILE`) so each source's glossary
cache lands under its own scoped path.

---

## Subtasks

### G0 — Source model + admin panel + plumbing
- **G0.1** Add `Source.industry_vertical` field + migration.
- **G0.2** `SourceAdmin`: add `industry_vertical` to `list_display` + `list_filter`
  (dropdown in the form is automatic from `TextChoices`, no extra work needed there).
- **G0.3** `Source.as_engine_env()` → include `VEDA_INDUSTRY_VERTICAL`.
- **G0.4** `ingestion/contracts.py::SourceContext` → add
  `industry_vertical: str = "generic"`; `from_env()` reads `VEDA_INDUSTRY_VERTICAL`.
- Acceptance: admin can select a vertical from a dropdown when adding/editing a
  Source; the list view shows/filters by vertical; registering a Source with
  vertical=`real_estate` results in `ctx.industry_vertical == "real_estate"`
  inside the ingestion subprocess.

### G1 — Domain description registry
- **G1.1** `config.py`: add `VERTICAL_DOMAIN_DESCRIPTIONS` dict; keep
  `GLOSSARY_DOMAIN_DESCRIPTION` as the `generic` fallback value.
- **G1.2** `glossary_builder.generate_glossary()` — accept `industry_vertical`,
  look up from the registry.
- **G1.3** `semantic_layer_v2.py::stage4_column_understanding` — same lookup for
  the `_DOMAIN_DESC` used in the column-understanding prompt.
- Acceptance: a BFSI source and a Real-Estate source produce visibly different
  LLM prompts (domain line differs).

### G2 — Static glossary registry (Layer C)
- **G2.1** Keep `STATIC_AML_GLOSSARY` as the `bfsi` entry, unchanged.
- **G2.2** Author `STATIC_REAL_ESTATE_GLOSSARY`, `STATIC_HEALTHCARE_GLOSSARY`,
  `STATIC_RETAIL_GLOSSARY` (§5 term lists above, same dict shape as the AML one).
- **G2.3** `STATIC_GLOSSARY_REGISTRY` dict selecting by vertical; `generic` → `{}`.
- Acceptance: a Real Estate source's synonym index contains real-estate terms,
  zero AML terms; BFSI behavior is unchanged (no regression).

### G3 — HF dataset registry (Layer B)
- **G3.1** BFSI: unchanged (existing `banking77`/`financial_phrasebank`/`finance-alpaca`).
- **G3.2** Real Estate: `_build_hf_glossary_real_estate()` from
  `divarofficial/real-estate-ads` field values.
- **G3.3** Retail: `_build_hf_glossary_retail()` from
  `bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset` intent labels
  (near-verbatim reuse of the banking77 extractor pattern).
- **G3.4** Healthcare: `_build_hf_glossary_healthcare()` from
  `gretelai/symptom_to_diagnosis` diagnosis labels (same banking77-style
  extraction); optional supplement from `gamino/wiki_medical_terms`.
- **G3.5** Wire all into `HF_DATASET_REGISTRY` keyed by vertical; `generic` → `[]`.
- Acceptance: each vertical's HF layer produces vertical-relevant terms with
  zero overlap with other verticals (spot-check: no `pep`/`ofac` in Real-Estate,
  no `RERA`/`carpet_area` in BFSI); failures are non-fatal (per the existing L3
  contract — a vertical without a working dataset still gets Layer A + Layer C).

### G4 — Wiring into L3_ENRICH
- **G4.1** `ingestion/layers/l3_enrich.py::run()` — pass `ctx.industry_vertical`
  into `run_full_semantic_layer(...)`.
- **G4.2** `run_full_semantic_layer()` signature gains
  `industry_vertical: str = "generic"`, threads to `glossary_builder` and
  `stage4_column_understanding`.
- Acceptance: end-to-end — selecting "Real Estate" in the admin dropdown and
  ingesting produces a real-estate-flavored semantic model + glossary with no
  code change needed per new source (just the dropdown selection).

### G5 — Per-vertical/per-source cache scoping
- **G5.1** Route `domain_glossary.py`'s cache paths through the existing
  `artifact_path()` scoping mechanism.
- Acceptance: ingesting a BFSI source then a Real-Estate source produces two
  distinct glossary files, neither overwriting the other.

---

## Dependency order

```
G0 (Source field + admin dropdown + plumbing) ─► G1 (domain description registry) ─┐
                                               ├► G2 (static glossary registry)    ├─► G4 (wire into L3)
                                               └► G3 (HF registry)                 ┘
G4 ─► G5 (cache scoping)
```

**Do first:** G0 (admin panel + plumbing — unblocks everything, and is the
user-facing piece). Then G1 + G2 in parallel (independent registries) → G3 →
G4 (wiring) → G5 (scoping).

---

## Why this design
- Explicit dropdown beats auto-detect: one-time admin action, no ongoing
  inference/misclassification risk, and it's already how `Source` connection
  fields are set (same admin form, same workflow).
- Reuses two mechanisms VEDA already has — the env-injection rail
  (`as_engine_env` → `SourceContext.from_env`) and the artifact-scope mechanism
  — instead of inventing new plumbing.
- Layer A needs no change (already vertical-agnostic); only the two hardcoded
  layers (B, C) and the LLM domain-framing string become vertical-aware.
- `generic` default means unsupported/uncommon verticals still work today with a
  clean, neutral glossary — no source is ever blocked.
