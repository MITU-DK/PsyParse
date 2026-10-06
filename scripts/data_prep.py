#!/usr/bin/env python3
"""
Phase 0 - Data Foundation (Owner: Manish)

Turns the raw Psy-Insight English file into:
    data/therapy_database.json   (case_profile + therapy_tag per DB session)
    data/db_sessions.json        (raw DB sessions)
    data/eval_scenarios.json     (N=20 held-out scenarios - DO NOT TOUCH until Phase 7)
    data/dev_scenarios.json      (3-5 scenarios for debugging, Phases 2-6)
    data/app_table.json          (App(T) lift-over-prior table, plan Step 1.5)
    data/data_stats.txt          (counts / distributions)
    .env, .gitignore

Usage (run from the Project/ root):
    python scripts/data_prep.py --clone      # Step 0.1  (git clone Psy-Insight into data/raw/)
    python scripts/data_prep.py --inspect    # Step 0.2  (print schema of a few sessions)
    python scripts/data_prep.py              # Steps 0.3 - 0.9 (full run)

SCHEMA ASSUMPTIONS (the plan names these fields but I could not open the real file):
    chain key, client-name key, speaker labels. They are set in the CONFIG block below.
    Run --inspect first and fix the CONFIG block if the real names differ.
"""
import argparse
import json
import math
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REPO_URL = "https://github.com/ckqqqq/Psy-Insight.git"
EN_FILENAME = "en_data_version7.json"
EXPECTED_SESSIONS = 520

SEED = 42
N_EVAL = 20
N_DEV = 4                    # plan: 3-5
MIN_TURNS = 6                # eligible sessions need >= 6 turns
SMALL_GROUP_MAX = 3          # "small group" = 3 or fewer sessions
MIN_NON_SFBT_EVAL = 6        # non-SFBT quota inside the 20 eval scenarios
RARE_THERAPY_MAX = 5         # types with <= 5 sessions stay strictly in the DB
GIANT_COMPONENT_WARN = 0.30  # warn if one group swallows >30% of sessions

# --- field names that are ASSUMED (verify with --inspect) ---
CHAIN_KEYS = ("chain", "chain_id", "dialog_chain")
PARTICIPANT_KEYS = ("participant", "participants", "client", "client_name")
CLIENT_HINTS = ("client", "patient", "visitor", "seeker", "user")
THERAPIST_HINTS = ("therap", "counsel", "doctor", "assistant", "supporter")
GENERIC_NAMES = {
    "client", "patient", "therapist", "counselor", "counsellor", "user", "visitor",
    "seeker", "helper", "assistant", "human", "participant", "speaker", "unknown",
    "none", "n/a", "anonymous", "resident", "student", "mother", "father", "wife", "husband",
    "daughter", "son", "parent", "parents", "teacher", "boy", "girl", "both", "young person",
    "man", "woman", "brother", "sister", "friend", "employee", "client to therapist",
}

JUNK_VALUES = {"", "unknown", "others", "other", "neutral"}

# ----------------------------------------------------------------------------
# Therapy-name normalization (Step 0.4)
# ----------------------------------------------------------------------------
# (regex on lower-cased raw label, short name). First match wins.
THERAPY_RULES = [
    (r"cognitive.?behaviou?r", "CBT"),
    (r"solution.?focused", "SFBT"),
    (r"psychoanaly", "Psychoanalytic"),
    (r"psychodynamic", "Psychodynamic"),
    (r"acceptance", "ACT"),
    (r"dialectical", "DBT"),
    (r"rational.?emotive", "REBT"),
    (r"motivational", "MI"),
    (r"person.?cent(?:ered|red)|client.?cent(?:ered|red)|rogerian", "PCT"),
    (r"humanistic", "Humanistic"),
    (r"narrative", "Narrative"),
    (r"gestalt", "Gestalt"),
    (r"interpersonal", "IPT"),
    (r"marriage|family.?systems", "Family Systems"),
    (r"family", "Family"),
    (r"reality", "Reality"),
    (r"postmodern", "Postmodern"),
    (r"adlerian", "Adlerian"),
    (r"emotion.?focused", "EFT"),
    (r"mindfulness", "Mindfulness"),
    (r"existential", "Existential"),
    (r"behaviou?r", "Behavioral"),
]
KNOWN_ABBREVIATIONS = {"CBT", "SFBT", "ACT", "DBT", "REBT", "MI", "PCT", "IPT", "EFT"}

# Rules whose names are distinctive enough to strip anywhere in text.
_DISTINCTIVE = [
    r"cognitive.?behaviou?r\w*", r"solution.?focused", r"psychoanaly\w*", r"psychodynamic",
    r"marriage and family systems", r"family systems", r"adlerian", r"gestalt", r"postmodern",
    r"acceptance(?: and)? commitment", r"dialectical", r"rational.?emotive", r"motivational interview\w*",
    r"person.?cent(?:ered|red)", r"client.?cent(?:ered|red)", r"rogerian", r"emotion.?focused", r"interpersonal (?:psycho)?therapy",
]
# Rules that are also ordinary words: strip only when followed by therapy/approach/treatment.
_GENERIC = [r"humanistic", r"narrative", r"family", r"mindfulness", r"existential", r"reality", r"behaviou?r\w*"]
_TAIL = r"(?:\s+(?:brief|behaviou?r\w*|therapy|therapies|treatment|approach|counsel+ing))*"
_STRIP_RES = (
    [re.compile(rf"\b{p}{_TAIL}", re.I) for p in _DISTINCTIVE]
    + [re.compile(rf"\b{p}\s+(?:therapy|therapies|treatment|approach|counsel+ing)", re.I) for p in _GENERIC]
    + [re.compile(r"\b(?:%s)\b" % "|".join(sorted(KNOWN_ABBREVIATIONS)))]  # case-sensitive on purpose ("act" is a verb)
)


def normalize_therapy(raw):
    """'Cognitive Behaviour Therapy' -> 'CBT'.  Returns '' for Unknown/empty."""
    if raw is None:
        return ""
    s = str(raw).strip()
    if s.lower() in JUNK_VALUES:
        return ""
    if s.upper() in KNOWN_ABBREVIATIONS:
        return s.upper()
    low = s.lower()
    for pat, short in THERAPY_RULES:
        if re.search(pat, low):
            return short
    # fallback: tidy the raw label (reported in the mapping table so you can review it)
    return re.sub(r"\s*therapy\s*$", "", s, flags=re.I).strip().title()


_RAW_RES = []


def set_raw_names(raw_names):
    _RAW_RES.clear()
    for raw in raw_names:
        if raw and not is_junk(raw):
            words = [re.escape(w) for w in re.split(r"[\s\-]+", raw.strip()) if w]
            _RAW_RES.append(re.compile(r"\b" + r"[\s\-]+".join(words) + r"\b", re.I))


def strip_therapy_names(text):
    for rx in _RAW_RES + _STRIP_RES:
        text = rx.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip(" -:;,.")


def contains_therapy_name(text, raw_names=()):
    if any(rx.search(text) for rx in _STRIP_RES):
        return True
    low = text.lower()
    return any(len(r) >= 6 and r.lower() in low for r in raw_names)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
_LOG = []


def log(msg=""):
    print(msg)
    _LOG.append(str(msg))


def as_list(v):
    """Flatten str / list / dict values into a list of non-empty stripped strings."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [x for item in v for x in as_list(item)]
    if isinstance(v, dict):
        return [x for item in v.values() for x in as_list(item)]
    s = str(v).strip()
    return [s] if s else []


def as_text(v):
    return " ".join(as_list(v))


def unique(items):
    seen, out = set(), []
    for x in items:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def is_junk(s):
    return s.strip().lower() in JUNK_VALUES


def is_client_turn(turn):
    sp = str(turn.get("speaker", "")).strip().lower()
    if any(h in sp for h in THERAPIST_HINTS):
        return False
    return any(h in sp for h in CLIENT_HINTS)


def get_turns(s):
    d = s.get("dialog") or []
    return [t for t in d if isinstance(t, dict)]


def get_chain(s):
    for k in CHAIN_KEYS:
        if s.get(k) not in (None, ""):
            return str(s[k]).strip()
    return None


def get_client_names(s):
    """Non-generic client name(s) from the participant field."""
    fromTurns = []
    for t in get_turns(s):
        if is_client_turn(t):
            n = str(t.get("participant") or "").strip()
            if len(n) > 1 and n.lower() not in GENERIC_NAMES:
                fromTurns.append(n)
    if fromTurns:
        return unique(fromTurns)
    p = None
    for k in PARTICIPANT_KEYS:
        if s.get(k):
            p = s[k]
            break
    if p is None:
        return []
    out = []

    def add(x):
        x = re.sub(r"^(client|patient)\s*[:\-]\s*", "", str(x).strip(), flags=re.I).strip()
        if len(x) > 1 and x.lower() not in GENERIC_NAMES:
            out.append(x)

    def handle_str(txt):
        for part in re.split(r"[;|\n]", txt):
            part = part.strip()
            if re.match(r"^(therap|counsel)", part, re.I):
                continue
            add(part)

    if isinstance(p, dict):
        for k, v in p.items():
            if any(h in str(k).lower() for h in CLIENT_HINTS):
                for x in as_list(v):
                    add(x)
    elif isinstance(p, (list, tuple)):
        for x in p:
            if isinstance(x, dict):
                role = " ".join(str(v) for k, v in x.items() if k in ("role", "speaker", "type")).lower()
                if any(h in role for h in THERAPIST_HINTS):
                    continue
                for k, v in x.items():
                    if k in ("name", "id", "participant"):
                        add(v)
            else:
                handle_str(str(x))
    else:
        handle_str(str(p))
    return unique(out)


def name_key(n):
    return re.sub(r"\s+", " ", n.strip().lower())


# ----------------------------------------------------------------------------
# Steps 0.1 / 0.2 / 0.3
# ----------------------------------------------------------------------------
def step_clone():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "Psy-Insight"
    if target.exists():
        print(f"{target} already exists - skipping clone")
        return
    subprocess.run(["git", "clone", REPO_URL, str(target)], check=True)


def find_raw_file():
    hits = sorted(RAW_DIR.rglob(EN_FILENAME))
    if not hits:
        sys.exit(f"Could not find {EN_FILENAME} under {RAW_DIR}. Run:  python scripts/data_prep.py --clone")
    return hits[0]


def load_sessions(path):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        items = []
        for k, v in raw.items():
            if isinstance(v, dict):
                v = dict(v)
                v.setdefault("dialog_id", str(k))
                items.append(v)
        raw = items
    sessions = []
    for i, s in enumerate(raw):
        s = dict(s)
        s["dialog_id"] = str(s.get("dialog_id") or s.get("id") or f"en_{i + 1:03d}")
        sessions.append(s)
    assert len({s["dialog_id"] for s in sessions}) == len(sessions), "dialog_id values are not unique"
    sessions.sort(key=lambda s: s["dialog_id"])
    # the real file has no chain field: a session with is_same_session == 1 continues the previous one
    if not any(get_chain(s) for s in sessions) and all("is_same_session" in s for s in sessions):
        chain = 0
        for s in sessions:
            if str(s["is_same_session"]).strip().lower() not in ("1", "true"):
                chain += 1
            s["chain_id"] = chain
    return sessions


def step_inspect(sessions):
    log(f"Top-level: {len(sessions)} sessions")
    key_counts = Counter(k for s in sessions for k in s)
    log("Session-level keys (count):")
    for k, c in key_counts.most_common():
        log(f"  {k}: {c}")
    turn_keys = Counter(k for s in sessions for t in get_turns(s) for k in t)
    log("Turn-level keys (count):")
    for k, c in turn_keys.most_common():
        log(f"  {k}: {c}")
    log("Speaker values: " + str(Counter(str(t.get('speaker')) for s in sessions for t in get_turns(s)).most_common(8)))
    log("Raw psychotherapy values: " + str(Counter(str(s.get('psychotherapy')) for s in sessions).most_common()))
    for s in sessions[:4]:
        log("-" * 60)
        for k, v in s.items():
            if k == "dialog":
                continue
            log(f"  {k}: {str(v)[:120]!r}")
        for t in get_turns(s)[:2]:
            log("  turn: " + json.dumps(t, ensure_ascii=False)[:300])
        log(f"  >> chain={get_chain(s)!r}  client_names={get_client_names(s)}  "
            f"client_turns={sum(is_client_turn(t) for t in get_turns(s))}/{len(get_turns(s))}")
    log("\nIf chain / client-name / speaker detection above looks wrong, edit the CONFIG block at the top of this file.")


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def step_english(sessions):
    assert len(sessions) == EXPECTED_SESSIONS, f"expected {EXPECTED_SESSIONS} sessions, got {len(sessions)}"
    bad = [s["dialog_id"] for s in sessions if _CJK.search(json.dumps(s, ensure_ascii=False))]
    assert not bad, f"CJK characters found in {len(bad)} sessions, e.g. {bad[:5]}"
    log(f"[0.3] {len(sessions)} English sessions loaded, no CJK characters")


# ----------------------------------------------------------------------------
# Step 0.4
# ----------------------------------------------------------------------------
def step_filter_therapy(sessions):
    raw_counts = Counter(str(s.get("psychotherapy")) for s in sessions)
    kept = []
    for s in sessions:
        t = normalize_therapy(s.get("psychotherapy"))
        if t:
            s["_therapy"] = t
            kept.append(s)
    log(f"[0.4] removed {len(sessions) - len(kept)} sessions with Unknown/empty psychotherapy; {len(kept)} remain")
    log("      raw label -> normalized label:")
    for raw, c in raw_counts.most_common():
        log(f"        {raw!r:50} -> {normalize_therapy(raw)!r}  ({c})")
    types = Counter(s["_therapy"] for s in kept)
    assert len(types) >= 3, f"only {len(types)} distinct therapy types"
    log(f"      {len(types)} distinct therapy types: {dict(types.most_common())}")
    return kept


# ----------------------------------------------------------------------------
# Step 0.5 - leakage-safe split
# ----------------------------------------------------------------------------
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_components(sessions):
    """Connected components of (chain U client name). Returns list[int] component id per session."""
    dsu = DSU(len(sessions))
    first_seen = {}
    for i, s in enumerate(sessions):
        tokens = []
        ch = get_chain(s)
        if ch:
            tokens.append("chain:" + ch.lower())
        tokens += ["client:" + name_key(n) for n in get_client_names(s)]
        for tok in tokens:
            if tok in first_seen:
                dsu.union(first_seen[tok], i)
            else:
                first_seen[tok] = i
    return [dsu.find(i) for i in range(len(sessions))]


def round_robin(pool, k, rng):
    """Pick k items cycling through therapy types (loose stratification)."""
    by = defaultdict(list)
    for item in pool:
        by[item["_therapy"]].append(item)
    for v in by.values():
        rng.shuffle(v)
    order = sorted(by)
    rng.shuffle(order)
    picked = []
    while len(picked) < k and any(by.values()):
        for t in order:
            if by[t] and len(picked) < k:
                picked.append(by[t].pop())
    return picked


def step_split(sessions):
    rng = random.Random(SEED)
    sessions = sorted(sessions, key=lambda s: s["dialog_id"])
    comp = build_components(sessions)
    comp_members = defaultdict(list)
    for i, c in enumerate(comp):
        comp_members[c].append(i)
    sizes = sorted((len(v) for v in comp_members.values()), reverse=True)
    log(f"[0.5] {len(comp_members)} connected components (chain U client name); largest sizes: {sizes[:8]}")
    if sizes[0] > GIANT_COMPONENT_WARN * len(sessions):
        log(f"      WARNING: largest group has {sizes[0]} sessions (> {int(GIANT_COMPONENT_WARN * 100)}%). "
            "The chain/client-name keys are probably matching something generic - check with --inspect.")

    type_counts = Counter(s["_therapy"] for s in sessions)
    rare = {t for t, c in type_counts.items() if c <= RARE_THERAPY_MAX}
    log(f"      rare therapy types kept strictly in DB (<= {RARE_THERAPY_MAX} sessions): {sorted(rare) or 'none'}")

    # eligible = one random eligible session per small, non-rare component
    pool = []
    for c, members in sorted(comp_members.items()):
        if len(members) > SMALL_GROUP_MAX:
            continue
        if any(sessions[i]["_therapy"] in rare for i in members):
            continue
        ok = [i for i in members
              if str(sessions[i].get("topic") or "").strip() and len(get_turns(sessions[i])) >= MIN_TURNS]
        if ok:
            pool.append((c, sessions[rng.choice(ok)]))
    items = []
    for c, s in pool:
        item = dict(s)
        item["_comp"] = c
        items.append(item)
    log(f"      {len(items)} eligible small groups (<= {SMALL_GROUP_MAX} sessions, topic present, >= {MIN_TURNS} turns)")

    non_sfbt = [x for x in items if x["_therapy"] != "SFBT"]
    assert len(non_sfbt) >= MIN_NON_SFBT_EVAL, f"only {len(non_sfbt)} non-SFBT eligible groups; need {MIN_NON_SFBT_EVAL}"
    eval_q = round_robin(non_sfbt, MIN_NON_SFBT_EVAL, rng)
    chosen_ids = {x["dialog_id"] for x in eval_q}
    rest = [x for x in items if x["dialog_id"] not in chosen_ids]
    assert len(rest) >= (N_EVAL - MIN_NON_SFBT_EVAL) + N_DEV, "not enough eligible groups for eval + dev"
    # remaining picks are random, so small therapy types dont get drained out of the db
    eval_fill = rng.sample(rest, N_EVAL - MIN_NON_SFBT_EVAL)
    eval_items = eval_q + eval_fill
    chosen_ids |= {x["dialog_id"] for x in eval_fill}
    rest = [x for x in items if x["dialog_id"] not in chosen_ids]
    dev_items = round_robin(rest, N_DEV, rng)

    held_comps = {x["_comp"] for x in eval_items + dev_items}
    by_id = {s["dialog_id"]: s for s in sessions}
    eval_sessions = sorted((by_id[x["dialog_id"]] for x in eval_items), key=lambda s: s["dialog_id"])
    dev_sessions = sorted((by_id[x["dialog_id"]] for x in dev_items), key=lambda s: s["dialog_id"])
    scenario_ids = {s["dialog_id"] for s in eval_sessions + dev_sessions}
    db_sessions, leak_guard = [], []
    for i, s in enumerate(sessions):
        if comp[i] in held_comps:
            if s["dialog_id"] not in scenario_ids:
                leak_guard.append(s["dialog_id"])
        else:
            db_sessions.append(s)
    log(f"      eval={len(eval_sessions)}  dev={len(dev_sessions)}  db={len(db_sessions)}  "
        f"(+{len(leak_guard)} sessions of held-out groups dropped from DB as a leakage guard)")
    n_empty_topic = sum(1 for s in db_sessions if not str(s.get("topic") or "").strip())
    log(f"      {n_empty_topic} DB sessions have an empty topic -> bucketed as 'Unknown' in app_table")
    return db_sessions, eval_sessions, dev_sessions, leak_guard


# ----------------------------------------------------------------------------
# Steps 0.6 / 0.7 / 0.8 - zero-LLM extraction
# ----------------------------------------------------------------------------
def strip_proper_names(text, names):
    for n in sorted(names, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(n)}\b", " ", text, flags=re.I)
    out, at_start = [], True
    for tok in re.split(r"(\s+)", text):
        if not tok or tok.isspace():
            out.append(tok)
            continue
        core = re.sub(r"^\W+|\W+$", "", tok)
        if not at_start and re.fullmatch(r"[A-Z][a-z]+", core):
            tok = tok.replace(core, "")
        at_start = tok.rstrip().endswith((".", "!", "?"))
        out.append(tok)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def name_tokens(s):
    toks = []
    for n in get_client_names(s):
        toks.append(n)
        toks += [t for t in re.split(r"\W+", n) if len(t) >= 3 and t.lower() not in GENERIC_NAMES]
    return unique(toks)


def extract_case_profile(s):
    turns = get_turns(s)
    labels = [x for t in turns for x in as_list(t.get("emotional label", t.get("emotional_label"))) if not is_junk(x)]
    obs = [x for t in turns if is_client_turn(t) for x in as_list(t.get("observation")) if not is_junk(x)]
    names = name_tokens(s)
    symptoms = strip_proper_names(" ".join(unique(obs)), names)
    return {
        "core_problems": as_text(s.get("background")),
        "emotional_states": " ".join(unique(labels)),
        "symptoms": symptoms,
    }


def strip_theme_prefix(theme, raw_therapy):
    """'Cognitive Behaviour Therapy: Managing worry' -> 'Managing worry'."""
    theme = theme.strip()
    if ":" in theme:
        head, tail = theme.split(":", 1)
        if contains_therapy_name(head) or normalize_therapy(head) == normalize_therapy(raw_therapy) \
                or head.strip().lower() == str(raw_therapy).strip().lower():
            theme = tail
    return theme.strip()


_CLINICAL_TERMS = [
    (r"depress", "depression"), (r"anxi", "anxiety"), (r"panic", "panic"), (r"stress", "stress"),
    (r"trauma|ptsd", "trauma"), (r"grief|bereave|mourning", "grief"), (r"\bangr|irritab|rage", "anger"),
    (r"insomnia|sleep", "sleep problems"), (r"lonel", "loneliness"),
    (r"self.?esteem|worthless|inferior", "low self-esteem"), (r"marital|couple|relationship", "relationship issues"),
    (r"burn.?out|exhaust", "burnout"), (r"addict|substance|alcohol", "substance use"),
    (r"obsess|compuls", "obsessive-compulsive"), (r"phobi", "phobia"), (r"suicid", "suicidal ideation"),
    (r"eating|binge|anorex|bulim", "eating issues"), (r"procrastinat", "procrastination"),
    (r"perfection", "perfectionism"), (r"guilt|shame", "guilt and shame"), (r"fear", "fear"),
]


def clinical_terms(text):
    low = text.lower()
    return [term for pat, term in _CLINICAL_TERMS if re.search(pat, low)]


def extract_therapy_tag(s, raw_names):
    turns = get_turns(s)
    raw_t = s.get("psychotherapy")
    techniques = []
    techniques += as_list(strip_theme_prefix(as_text(s.get("theme")), raw_t))
    techniques += as_list(s.get("reasoning"))
    techniques += as_list(s.get("summary"))
    for t in turns:                                   # SOP Step P6: parse the `strategy` field
        for item in as_list(t.get("strategy")):
            techniques += [p.strip() for p in re.split(r"[;,/|]", item)]
    cleaned = []
    for x in techniques:
        x = strip_therapy_names(x)
        if len(x) >= 2 and not is_junk(x):
            cleaned.append(x)
    techniques = unique(cleaned)
    assert not any(contains_therapy_name(x, raw_names) for x in techniques), \
        f"therapy-name token left in techniques of {s['dialog_id']}"

    topic = str(s.get("topic") or "").strip()
    text = " ".join([as_text(s.get("background")), topic]
                    + [x for t in turns if is_client_turn(t) for x in as_list(t.get("observation"))])
    conditions = unique(([topic] if topic else []) + clinical_terms(text))
    return {"therapy_type": s["_therapy"], "techniques": techniques, "applicable_conditions": conditions}


def build_database(db_sessions):
    raw_names = sorted({str(s.get("psychotherapy")) for s in db_sessions})
    set_raw_names(raw_names)
    db = []
    for s in db_sessions:
        db.append({
            "dialog_id": s["dialog_id"],
            "case_profile": extract_case_profile(s),
            "therapy_tag": extract_therapy_tag(s, raw_names),
        })
    return db


# ----------------------------------------------------------------------------
# App(T) table (plan Step 1.5 definition, stored as a Phase 0 output)
# ----------------------------------------------------------------------------
def build_app_table(db_sessions):
    topics = {s["dialog_id"]: (str(s.get("topic") or "").strip() or "Unknown") for s in db_sessions}
    n = len(db_sessions)
    types = sorted({s["_therapy"] for s in db_sessions})
    topic_list = sorted(set(topics.values()))
    type_count = Counter(s["_therapy"] for s in db_sessions)
    topic_total = Counter(topics.values())
    pair_count = Counter((topics[s["dialog_id"]], s["_therapy"]) for s in db_sessions)
    log_lift = {}
    for t in topic_list:
        for th in types:
            prior = (type_count[th] + 1) / (n + 1)                       # Laplace as in the plan
            cond = (pair_count[(t, th)] + 1) / (topic_total[t] + 1)
            log_lift[(t, th)] = math.log(cond / prior)
    lo, hi = min(log_lift.values()), max(log_lift.values())
    norm = {k: (0.5 if hi == lo else (v - lo) / (hi - lo)) for k, v in log_lift.items()}
    return {
        "meta": {"definition": "App = minmax(log((count(T,topic)+1)/(total(topic)+1) / ((count(T)+1)/(N+1))))",
                 "n_db_sessions": n, "min_log_lift": lo, "max_log_lift": hi},
        "case_topic": topics,
        "app": {t: {th: norm[(t, th)] for th in types} for t in topic_list},
        "log_lift": {t: {th: log_lift[(t, th)] for th in types} for t in topic_list},
    }


# ----------------------------------------------------------------------------
# Step 0.9 - validation + reporting
# ----------------------------------------------------------------------------
def validate(db, db_sessions, eval_sessions, dev_sessions, leak_guard):
    assert db, "empty database"
    empty = [c["dialog_id"] for c in db
             if not (c["case_profile"]["core_problems"].strip() and c["case_profile"]["symptoms"].strip())]
    assert not empty, f"{len(empty)} cases with empty core_problems or symptoms, e.g. {empty[:5]}"
    noEmo = sum(1 for c in db if not c["case_profile"]["emotional_states"].strip())
    log(f"      {noEmo} cases have empty emotional_states (all labels were Unknown/Others/Neutral) - kept, the embedder skips empty fields")
    assert all(c["therapy_tag"]["therapy_type"] and c["therapy_tag"]["techniques"] for c in db), \
        "a case has an empty therapy_tag (type or techniques)"
    types = Counter(c["therapy_tag"]["therapy_type"] for c in db)
    assert len(types) >= 3, f"only {len(types)} therapy types in DB"
    assert len(db) >= 100, f"only {len(db)} cases (< 100)"
    if len(db) < 300:
        log(f"      NOTE: only {len(db)} DB cases - the plan says to investigate if < 300")
    assert len(eval_sessions) == N_EVAL, f"eval has {len(eval_sessions)} scenarios"
    assert 3 <= len(dev_sessions) <= 5, f"dev has {len(dev_sessions)} scenarios"

    def keys(sessions):
        ids = {s["dialog_id"] for s in sessions}
        chains = {get_chain(s).lower() for s in sessions if get_chain(s)}
        clients = {name_key(n) for s in sessions for n in get_client_names(s)}
        return ids, chains, clients

    sets = {"db": keys(db_sessions), "eval": keys(eval_sessions), "dev": keys(dev_sessions)}
    for a, b in (("db", "eval"), ("db", "dev"), ("eval", "dev")):
        for label, ia, ib in zip(("dialog_id", "chain", "client name"), sets[a], sets[b]):
            assert not (ia & ib), f"{a} and {b} overlap on {label}: {sorted(ia & ib)[:5]}"
    name_leaks = 0
    by_id = {s["dialog_id"]: s for s in db_sessions}
    for c in db:
        for n in get_client_names(by_id[c["dialog_id"]]):
            if re.search(rf"\b{re.escape(n)}\b", c["case_profile"]["symptoms"], re.I):
                name_leaks += 1
    assert name_leaks == 0, f"{name_leaks} client names still present in symptoms"
    log("[0.9] all assertions passed (no empty profiles, >=3 therapy types, >=100 cases, "
        "zero overlap on dialog_id / chain / client name, names stripped, no therapy names in techniques)")

    log("\nDB distribution (cases per therapy type):")
    for t, c in types.most_common():
        log(f"  {t:16} {c}")
    for name, sess in (("EVAL", eval_sessions), ("DEV", dev_sessions)):
        log(f"\n{name} distribution:")
        for t, c in Counter(s["_therapy"] for s in sess).most_common():
            log(f"  {t:16} {c}")
    log(f"\nCases with no applicable_conditions: {sum(1 for c in db if not c['therapy_tag']['applicable_conditions'])}")
    log(f"Held-out groups' extra sessions dropped from DB (leak guard): {len(leak_guard)}")


def clean_for_save(sessions):
    return [{k: v for k, v in s.items() if not k.startswith("_")} for s in sessions]


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_support_files():
    env = ROOT / ".env"
    if not env.exists():
        env.write_text("DEEPSEEK_API_KEY=\n", encoding="utf-8")
    gi = ROOT / ".gitignore"
    wanted = [".env", "*.json", "data/raw/", "*.index", "*.pkl", "*.npy", "logs/", "__pycache__/"]
    existing = gi.read_text(encoding="utf-8").splitlines() if gi.exists() else []
    gi.write_text("\n".join(existing + [w for w in wanted if w not in existing]) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", action="store_true", help="Step 0.1: git clone Psy-Insight into data/raw/")
    ap.add_argument("--inspect", action="store_true", help="Step 0.2: print schema, then stop")
    args = ap.parse_args()

    if args.clone:
        step_clone()
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = find_raw_file()
    log(f"[0.1] using {path}")
    sessions = load_sessions(path)
    if args.inspect:
        step_inspect(sessions)
        return

    step_english(sessions)                                     # 0.3
    sessions = step_filter_therapy(sessions)                   # 0.4
    db_sessions, eval_sessions, dev_sessions, leak = step_split(sessions)   # 0.5
    db = build_database(db_sessions)                           # 0.6 - 0.8
    validate(db, db_sessions, eval_sessions, dev_sessions, leak)            # 0.9

    write_json(DATA_DIR / "therapy_database.json", db)
    write_json(DATA_DIR / "db_sessions.json", clean_for_save(db_sessions))
    write_json(DATA_DIR / "eval_scenarios.json", clean_for_save(eval_sessions))
    write_json(DATA_DIR / "dev_scenarios.json", clean_for_save(dev_sessions))
    write_json(DATA_DIR / "app_table.json", build_app_table(db_sessions))
    (DATA_DIR / "data_stats.txt").write_text("\n".join(_LOG) + "\n", encoding="utf-8")
    write_support_files()
    print(f"\nWrote outputs to {DATA_DIR}")


if __name__ == "__main__":
    main()
