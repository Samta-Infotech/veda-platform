# =============================================================================
# query/target_selection.py
# VEDA — Stage 1 of the join pipeline: EVIDENCE-BASED TARGET SELECTION.
#
# Identifies the REQUESTED entities of a query (not the required join tree — that's
# the JoinPlanner's job; junctions/hubs are introduced there, never here). Each
# candidate carries its evidence so downstream decisions (accept / refuse-on-ambiguity)
# are explainable and tunable from data rather than hidden in a boolean heuristic.
#
# Strict interface: this stage consumes pre-computed lexical/retrieval signals and
# returns entities only. It does NOT plan joins, resolve keys, or aggregate.
# =============================================================================

from dataclasses import dataclass, field


@dataclass
class Target:
    table: str
    matched_tokens: list = field(default_factory=list)
    lexical_score: float = 0.0     # 0..1  fraction of the table's distinctive name tokens the query names
    retrieval_score: float = 0.0   # 0..1  normalized max column retrieval score for the table
    confidence: float = 0.0        # W_LEX*lexical + W_RET*retrieval


@dataclass
class TargetResult:
    anchor: str
    requested: list = field(default_factory=list)   # accepted Targets (confidence ≥ ACCEPT)
    ambiguous: list = field(default_factory=list)    # (Target, competitor) token-ties within DELTA
    uncertain: list = field(default_factory=list)    # named but REJECT ≤ conf < ACCEPT
    rejected: list = field(default_factory=list)     # conf < REJECT or dominated


def select_targets(anchor, candidates, *, qtoks, anchor_toks, anchor_col_toks,
                   retrieval, junctions, name_toks, cfg):
    """Score candidate tables and bucket them by evidence.

    candidates       : iterable of candidate table names (anchor already removed upstream is fine)
    qtoks            : singularized query tokens (set)
    anchor_toks      : singularized tokens of the anchor's NAME (set)
    anchor_col_toks  : singularized tokens of the anchor's COLUMN names (set)
    retrieval        : {table: normalized 0..1 retrieval score}
    junctions        : set of bridge tables (excluded — they're required, never requested)
    name_toks        : callable(table_name) -> set of singularized entity tokens
    cfg              : config.TARGET_SELECTION dict (weights + thresholds)
    """
    w_lex, w_ret = cfg["W_LEX"], cfg["W_RET"]
    accept, reject, delta = cfg["ACCEPT"], cfg["REJECT"], cfg["DELTA"]

    evals = []
    for t in dict.fromkeys(candidates):
        if t == anchor or t in junctions:
            continue                                   # junctions are required, not requested
        distinctive = {d for d in name_toks(t) if d not in anchor_toks}
        if not distinctive:
            continue                                   # nothing that isn't already the anchor
        # A concept the anchor already provides as a COLUMN is not a join target
        # ("incident status and workflow state" → incident.workflow_state).
        if distinctive <= anchor_col_toks:
            continue
        matched = distinctive & qtoks
        lexical = len(matched) / len(distinctive)
        retr = retrieval.get(t, 0.0)
        conf = w_lex * lexical + w_ret * retr
        evals.append(Target(t, sorted(matched), round(lexical, 3), round(retr, 3), round(conf, 3)))

    # Domination — two complementary rules, both "a more specific table explains this
    # candidate away":
    #  (1) strict-subset of MATCHED tokens: "documents" ⊂ "document_category_master".
    #  (2) SAME matched tokens but the other names its entity more COMPLETELY: the query
    #      token "user" fully accounts for `user` (coverage 1.0) but only partially for
    #      `user_profile` (the unnamed "profile" means the user didn't ask for it). So
    #      `user` dominates user_profile / user_prompt / user_user_permissions / external_users.
    #      Without this, every table merely CONTAINING the token floods `ambiguous`.
    def _dominated(tg):
        ms = set(tg.matched_tokens)
        for o in evals:
            if o.table == tg.table:
                continue
            os = set(o.matched_tokens)
            if ms < os:
                return True
            if ms == os and o.lexical_score > tg.lexical_score:
                return True
        return False

    result = TargetResult(anchor=anchor)
    for tg in evals:
        if _dominated(tg):
            result.rejected.append(tg)
        elif tg.confidence >= accept:
            result.requested.append(tg)
        elif tg.confidence >= reject and tg.matched_tokens:
            result.uncertain.append(tg)              # named but not confident → caller refuses
        else:
            result.rejected.append(tg)

    # Ambiguity: two surviving candidates that COMPETE for the same matched token(s) and
    # are within DELTA of each other (e.g. role 0.78 vs another 0.76) — a coin-flip, not
    # a selection. Caller refuses rather than win by a hair.
    live = result.requested + result.uncertain
    for i, a in enumerate(live):
        for b in live[i + 1:]:
            if set(a.matched_tokens) & set(b.matched_tokens) and abs(a.confidence - b.confidence) < delta:
                result.ambiguous.append((a, b))

    result.requested.sort(key=lambda tg: tg.confidence, reverse=True)
    return result
