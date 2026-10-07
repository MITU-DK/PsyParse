import json
import re

from ..agents.evaluation_agent import _parse_json
from ..agents.therapist_agent import TherapistAgent
from ..retrieval.hybrid_search import getRetriever, hybridRetrieve, scoreTherapies, selectTopK2


def _tiebreak_key(c):
    # tie-breakers: higher m_s first, then higher app, then alphabetical by type
    return (-c.get("m_s", 0.0), -c.get("app", 0.0), c.get("therapy_type", ""))


def run_stage2(embed_profile, keywords, full_profile):
    ret = getRetriever()

    # hybrid retrieval -> top-10 candidates
    cands = hybridRetrieve(embed_profile, keywords, retriever=ret)

    # score candidates (M_s + App) entirely in python, no LLM
    scored = scoreTherapies(embed_profile, cands, retriever=ret)

    # apply tie-breakers before selecting top-k2
    scored.sort(key=_tiebreak_key)

    # escape hatch: if entire top-10 pool has only one therapy type, skip diversity check
    distinct_types = {c["therapy_type"] for c in scored}
    single_type_pool = len(distinct_types) == 1

    if single_type_pool:
        # bypass diversity, take best 3 (or fewer if pool < 3)
        top_k2 = scored[:3]
        print(f"[warn] escape hatch triggered - all top-10 are '{list(distinct_types)[0]}', skipping diversity check")
    else:
        # drop bottom 30% (lowest 3 of 10), then pick top-k2=3 distinct types
        top_k2 = selectTopK2(scored, k2=3, dropFrac=0.3)

        # diversity rescue: if fewer than 2 distinct types remain, pull from discarded pile
        kept_types = {c["therapy_type"] for c in top_k2}
        if len(kept_types) < 2:
            # find the best case from a different type in the full scored list
            for c in scored:
                if c["therapy_type"] not in kept_types:
                    top_k2.append(c)
                    kept_types.add(c["therapy_type"])
                    print(f"[info] rescued '{c['therapy_type']}' from discarded pile to meet diversity requirement")
                    break

    # therapist synthesizes guidance framework
    therapist = TherapistAgent()
    raw_framework = therapist.synthesize_guidance(top_k2)

    # parse the framework JSON
    try:
        framework = _parse_json(raw_framework)
    except ValueError:
        print("[error] could not parse guidance framework JSON - flagged for regeneration")
        framework = {"frameworks": []}

    # diversity validation on synthesis output
    frameworks_list = framework.get("frameworks", [])
    synth_types = {f.get("therapy", "") for f in frameworks_list}
    if len(frameworks_list) < 2 and not single_type_pool:
        print(f"[warn] synthesis produced only {len(frameworks_list)} framework(s) - flagged for regeneration")

    return {
        "top_k2": top_k2,           # scored candidate cases
        "framework": framework,      # full guidance framework dict
        "frameworks_list": frameworks_list,  # list of per-therapy slices for stage 3a
    }
