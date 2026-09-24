# Homzhub — multi-level drill-down chains that work

Measured end to end on 2026-09-24. Every value used below was chosen from the table's OWN
statistics (`pg_stats.most_common_vals`, a value covering 5-60% of the rows), so each step
is a real narrowing rather than a no-op. Judged on the SQL the engine produced.

---

## 1. Properties — TWO levels, filters accumulate  ⭐ best

```
What is the distribution of properties by facing?
only the Nagpur ones
only the FULL ones
go back
```

What the engine actually ran:

| Turn | SQL |
|---|---|
| base | `SELECT facing, corner_property, COUNT(DISTINCT id) … GROUP BY facing, corner_property` |
| level 1 | `… WHERE LOWER(city_name) = %s GROUP BY facing, corner_property` |
| level 2 | `… WHERE LOWER(furnishing) = %s AND LOWER(city_name) = %s GROUP BY …` |
| go back | the base query returns |

Both filters are present at level 2 — the second narrowing ADDS to the first rather than
replacing it — and the `GROUP BY` survives every level. Table: `assets_asset`
(`city_name` Nagpur 26%, `furnishing` FULL 41%, `facing` EAST 17%).

## 2. Lease listings — narrow, then REPLACE the value

```
What is the distribution of lease listings by status?
only the FULL ones
what about SEMI
go back
```

`what about SEMI` swaps FULL for SEMI on the same column instead of asking for both at
once, and `go back` restores the base. Table: `assets_leaselisting` (`furnishing` FULL 41%).

## 3. Payment transactions — one level, clean

```
What is the distribution of payment transactions by transaction type?
only the DEBIT ones
go back
```

Table: `accounts_paymenttransaction` (`transaction_type` DEBIT 52%).

---

## Chains that do NOT hold, and where they break

### Ledger — level 1 works, level 2 does not

```
Show the latest 10 ledger entries.
only the DEBIT ones          -> works: WHERE entry_type
only the Rent ones           -> FAILS
go back                      -> works
```

The failure message is *"I couldn't match the figure you asked about to the data I'd have
to rank it by"*. The base is a RANKED list ("latest 10"), not a distribution, and a second
narrowing on a ranked base is the shape the engine handles least well. Use chain 1 or 2 for
a demo instead. Table: `accounts_generalledger` (`entry_type` DEBIT 58%, `label` Rent 39%).

### Tickets — the base question itself fails

```
How many tickets are there in each status category?
  -> "I couldn't map 'statu' to any column or value in the data."
```

Nothing downstream can work when the base does not. The data is fine — `worklists_ticket`
has `priority` (MEDIUM 49%) and `status` (OPEN 46%), which would make a good two-level
chain — so this is worth fixing, not worth avoiding.

---

## How these were chosen

`pg_stats` already stores each column's commonest value and its frequency, so one query
lists every drillable dimension in the schema without scanning the tables. Homzhub has 60
tables with two or more such dimensions, but most are `_history` tables. Among business
tables, `assets_asset` has the most depth — three independent dimensions, which is why it
supports the only two-level chain here.

## What to expect

- **One level is reliable; two levels work on `assets_asset` and on a replace.**
- **Say the VALUE, not the operation** — `only the FULL ones` works, `filter to FULL` does
  not, because *filter* is looked up as data.
- **`go back` works on all three chains above** but is not reliable in general (see
  `DRILLDOWN_FAQ.md`).
- Each turn takes roughly 25 seconds; `go back` is instant.
