# TODO: not run on the real data yet (sandbox had no faiss / sentence-transformers), run `python psyparse/search.py build` then `test` first
import json
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

root = Path(__file__).resolve().parent.parent
dataDir = root / "data"
fields = ["core_problems", "emotional_states", "symptoms"]
defAlpha = 0.5
defK1 = 10
poolSize = 50
cache = {}


def tokenize(text):
    return [w for w in re.findall(r"[a-z]{2,}", str(text).lower()) if w not in ENGLISH_STOP_WORDS]


def isClient(turn):
    sp = str(turn.get("speaker", "")).lower()
    if any(h in sp for h in ("therap", "counsel", "doctor", "assistant", "supporter")):
        return False
    return any(h in sp for h in ("client", "patient", "visitor", "seeker", "user"))


def getModel():
    if "model" not in cache:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        m.max_seq_length = 256
        cache["model"] = m
    return cache["model"]


def embedFields(items):
    # embed every field on its own then mean pool, so nothing gets cut at 256 tokens
    model = getModel()
    flat = []
    owner = []
    for i, texts in enumerate(items):
        for t in texts:
            if t and t.strip():
                flat.append(t)
                owner.append(i)
    vecs = model.encode(flat, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    out = np.zeros((len(items), vecs.shape[1]), dtype="float32")
    cnt = np.zeros(len(items))
    for v, o in zip(vecs, owner):
        out[o] += v
        cnt[o] += 1
    out = out / np.maximum(cnt, 1)[:, None]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return (out / np.maximum(norms, 1e-12)).astype("float32")


def profileFields(p):
    if isinstance(p, dict):
        res = []
        for f in fields:
            v = p.get(f) or ""
            res.append(" ".join(v) if isinstance(v, list) else str(v))
        return res
    return [str(p)]


def loadJson(name):
    with open(dataDir / name, encoding="utf-8") as f:
        return json.load(f)


def build():
    import faiss
    from rank_bm25 import BM25Okapi

    db = loadJson("therapy_database.json")
    sessions = loadJson("db_sessions.json")
    ids = [c["dialog_id"] for c in db]

    # step 1.1
    tok = getattr(getModel(), "tokenizer", None)
    if tok is not None:
        longOnes = sum(1 for c in db if len(tok(" ".join(c["case_profile"][f] for f in fields))["input_ids"]) > 256)
        print("profiles over 256 tokens when concatenated:", longOnes, "(fields are embedded separately so ok)")
    emb = embedFields([[c["case_profile"][f] for f in fields] for c in db])
    np.save(dataDir / "case_embeddings.npy", emb)

    types = sorted({c["therapy_tag"]["therapy_type"] for c in db})
    condText = []
    for t in types:
        conds = []
        for c in db:
            if c["therapy_tag"]["therapy_type"] == t:
                for x in c["therapy_tag"]["applicable_conditions"]:
                    if x.lower() not in [y.lower() for y in conds]:
                        conds.append(x)
        condText.append(" ".join(conds) if conds else t)
    typeEmb = embedFields([[x] for x in condText])
    np.save(dataDir / "therapy_type_embeddings.npy", typeEmb)
    with open(dataDir / "therapy_types.json", "w", encoding="utf-8") as f:
        json.dump(types, f)

    # step 1.2
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(dataDir / "therapy_faiss.index"))
    D, I = index.search(emb[:5], 1)
    print("quick self check, top1 ids:", I[:, 0].tolist())

    # step 1.3
    docs = []
    for c in db:
        tag = c["therapy_tag"]
        docs.append(tokenize(" ".join([tag["therapy_type"]] + tag["techniques"] + tag["applicable_conditions"])))
    bm25 = BM25Okapi(docs)
    with open(dataDir / "therapy_bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)

    byId = {s["dialog_id"]: s for s in sessions}
    texts = []
    for i in ids:
        texts.append(" ".join(str(t.get("content", "")) for t in byId[i].get("dialog", []) if isClient(t)))
    tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
    tfidf.fit(texts)
    with open(dataDir / "tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    print("built", len(db), "cases,", len(types), "therapy types, tfidf vocab", len(tfidf.vocabulary_))


class Retriever:
    def __init__(self):
        import faiss

        self.db = loadJson("therapy_database.json")
        self.appTable = loadJson("app_table.json")
        self.index = faiss.read_index(str(dataDir / "therapy_faiss.index"))
        self.caseEmb = np.load(dataDir / "case_embeddings.npy")
        self.typeEmb = np.load(dataDir / "therapy_type_embeddings.npy")
        self.types = loadJson("therapy_types.json")
        with open(dataDir / "therapy_bm25.pkl", "rb") as f:
            saved = pickle.load(f)
        self.bm25 = saved["bm25"]
        if saved["ids"] != [c["dialog_id"] for c in self.db]:
            raise ValueError("bm25 ids dont match therapy_database.json, rebuild the indices")
        if self.index.ntotal != len(self.db):
            raise ValueError("faiss index size dont match therapy_database.json, rebuild the indices")

    def embedProfile(self, p):
        return embedFields([profileFields(p)])[0]


def getRetriever():
    if "ret" not in cache:
        cache["ret"] = Retriever()
    return cache["ret"]


def hybridRetrieve(p, kw, alpha=defAlpha, k1=defK1, retriever=None):
    ret = retriever or getRetriever()
    q = ret.embedProfile(p)
    n = len(ret.db)
    D, I = ret.index.search(q[None, :].astype("float32"), min(poolSize, n))
    cosScores = D[0]
    idx = I[0]

    kwText = " ".join(kw) if isinstance(kw, (list, tuple)) else str(kw)
    raw = np.asarray(ret.bm25.get_scores(tokenize(kwText)), dtype=float)
    # minmax over the whole db, all zero scores would divide by 0 so fallback to 0
    if raw.max() > raw.min():
        bmNorm = (raw - raw.min()) / (raw.max() - raw.min())
    else:
        bmNorm = np.zeros_like(raw)

    res = []
    for c, i in zip(cosScores, idx):
        case = ret.db[i]
        score = alpha * float(c) + (1 - alpha) * float(bmNorm[i])
        res.append({
            "case_id": case["dialog_id"],
            "score": score,
            "cos": float(c),
            "bm25": float(bmNorm[i]),
            "therapy_type": case["therapy_tag"]["therapy_type"],
            "techniques": case["therapy_tag"]["techniques"],
            "applicable_conditions": case["therapy_tag"]["applicable_conditions"],
            "topic": ret.appTable["case_topic"].get(case["dialog_id"], "Unknown"),
        })
    res.sort(key=lambda r: r["score"], reverse=True)
    return res[:k1]


def scoreTherapies(p, cands, w1=0.5, w2=0.5, retriever=None):
    ret = retriever or getRetriever()
    q = ret.embedProfile(p)
    # live patient gets the majority topic of the retrieved cases (ties go to the better ranked one)
    topic = Counter(c["topic"] for c in cands).most_common(1)[0][0]
    appRow = ret.appTable["app"].get(topic) or ret.appTable["app"].get("Unknown") or {}
    res = []
    for c in cands:
        t = c["therapy_type"]
        mS = 0.0
        if t in ret.types:
            mS = float(np.clip(np.dot(q, ret.typeEmb[ret.types.index(t)]), 0.0, 1.0))
        app = appRow.get(t, 0.5)
        new = dict(c)
        new["m_s"] = mS
        new["app"] = app
        new["s_th"] = w1 * mS + w2 * app
        res.append(new)
    res.sort(key=lambda r: r["s_th"], reverse=True)
    return res


def selectTopK2(scored, k2=3, dropFrac=0.3):
    ranked = sorted(scored, key=lambda c: c["s_th"], reverse=True)
    nDrop = int(len(ranked) * dropFrac + 1e-9)
    kept = ranked[:len(ranked) - nDrop]
    dropped = ranked[len(ranked) - nDrop:]

    # one case per type, best first. grouping by type means sfbt can only take 1 of the k2 slots
    best = {}
    for c in kept:
        if c["therapy_type"] not in best:
            best[c["therapy_type"]] = c
    if len(best) < k2:
        # cut left too few types, pull from the dropped ones, non sfbt first
        for c in sorted(dropped, key=lambda c: c["therapy_type"] == "SFBT"):
            if len(best) >= k2:
                break
            if c["therapy_type"] not in best:
                best[c["therapy_type"]] = c
    out = sorted(best.values(), key=lambda c: c["s_th"], reverse=True)
    return out[:k2]


hybrid_retrieve = hybridRetrieve
score_therapies = scoreTherapies
select_top_k2 = selectTopK2


def runTests():
    results = []

    def check(name, ok, info=""):
        results.append(ok)
        print(("PASS" if ok else "FAIL"), "-", name, info)

    ret = Retriever()
    db = ret.db

    # faiss self retrieval
    D, I = ret.index.search(ret.caseEmb, 1)
    selfScores = np.sum(ret.caseEmb * ret.caseEmb, axis=1)
    good = sum(1 for i in range(len(db)) if I[i, 0] == i or abs(D[i, 0] - selfScores[i]) < 1e-5)
    check("faiss self retrieval", good == len(db), f"({good}/{len(db)})")

    # bm25 keyword test
    scores = np.asarray(ret.bm25.get_scores(tokenize("CBT depression")))
    top = np.argsort(-scores)[:10]
    nCbt = sum(1 for i in top if db[i]["therapy_tag"]["therapy_type"] == "CBT")
    check("bm25 'CBT depression' returns CBT cases", nCbt >= 5, f"({nCbt}/10 are CBT)")

    queries = [
        ("I feel anxious all the time, cant sleep and keep worrying about work", "anxiety worry sleep work stress"),
        ("feeling hopeless and low, no motivation, lost interest in everything", "depression hopeless motivation sadness"),
        ("fighting with my family and feel alone and misunderstood", "family conflict lonely relationship"),
    ]
    mixed = True
    kOk = True
    for text, kw in queries:
        p = {"core_problems": text, "emotional_states": "", "symptoms": text}
        cands = hybridRetrieve(p, kw, retriever=ret)
        cnt = Counter(c["therapy_type"] for c in cands)
        if len(cnt) < 2:
            mixed = False
        top3 = selectTopK2(scoreTherapies(p, cands, retriever=ret))
        nTypes = len({c["therapy_type"] for c in top3})
        if nTypes < min(2, len(cnt)) or len(top3) > 3:
            kOk = False
        print("   query:", text[:40], "| top10 types:", dict(cnt), "| top-k2:", [c["therapy_type"] for c in top3])
    check("hybrid retrieval returns mixed therapy types", mixed)
    check("top-k2 has >=2 distinct therapy types", kOk)

    # reload from disk gives the same thing
    cache.clear()
    ret2 = Retriever()
    p = {"core_problems": queries[0][0], "emotional_states": "", "symptoms": queries[0][0]}
    a = [c["case_id"] for c in hybridRetrieve(p, queries[0][1], retriever=ret)]
    b = [c["case_id"] for c in hybridRetrieve(p, queries[0][1], retriever=ret2)]
    with open(dataDir / "tfidf.pkl", "rb") as f:
        tf = pickle.load(f)
    check("indices reload from disk", a == b and ret2.index.ntotal == len(db) and len(tf.vocabulary_) > 0)

    print("ALL PASSED" if all(results) else "SOME FAILED")
    return all(results)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build":
        build()
    elif cmd == "test":
        sys.exit(0 if runTests() else 1)
    else:
        print("usage: python psyparse/search.py build|test")
