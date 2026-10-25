import re
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

import config

FIELDS = ["core_problems", "emotional_states", "symptoms"]

# everyday chat words. the db is written in clinical language, so these look "rare" to its idf and pushed out
# the words that actually describe the problem (old keywords were mostly feel/like/want/think)
CHAT_WORDS = set("""
feel feeling feels felt like know want wanted think thinking thought really just lot things thing way going make makes
made hard okay yeah good able try trying start doing does constantly enjoy people time times maybe kind sort bit little
stuff actually right sure need needs got getting say said saying tell told mean guess pretty big end head worse wrong
waiting letting talk talking instead away hand used best place places forward stay live looking inside figure notice
moments life making help come came goes better different long day days thank thanks especially probably
doesn didn isn wasn couldn wouldn shouldn aren haven won
""".split())


# same tokenizer the bm25 index in data/ was built with
def tokenize(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z]{2,}", str(text).lower()) if w not in ENGLISH_STOP_WORDS]


def as_text(v: Any) -> str:
    if isinstance(v, list):
        return " ".join(str(x) for x in v)
    return str(v or "")


class HybridRetriever:
    """
    Implements multi-therapy RAG:
      - Equation 1: Hybrid Retrieval S(P, C_i) = alpha * Cos(E(P), E(C_i)) + (1 - alpha) * B_norm(K, T_i)
      - Equation 2: Therapy Suitability S_th(P, T_i) = w1 * M_s(P, T_i) + w2 * App(T_i)
      - Percentile pruning (bottom 30%) and top-k2 grouping by distinct therapy type.
    """
    def __init__(self, data_dir: Optional[Path] = None, model_name: str = "all-MiniLM-L6-v2"):
        d = Path(data_dir or config.DATA_DIR)
        self.encoder = SentenceTransformer(model_name)
        self.encoder.max_seq_length = 256

        with open(d / "therapy_database.json", encoding="utf-8") as f:
            self.database: List[Dict[str, Any]] = json.load(f)
        with open(d / "app_table.json", encoding="utf-8") as f:
            self.app_table = json.load(f)
        with open(d / "therapy_types.json", encoding="utf-8") as f:
            self.types: List[str] = json.load(f)
        self.type_emb = np.load(d / "therapy_type_embeddings.npy")
        self.faiss_index = faiss.read_index(str(d / "therapy_faiss.index"))
        with open(d / "therapy_bm25.pkl", "rb") as f:
            saved = pickle.load(f)
        self.bm25 = saved["bm25"] if isinstance(saved, dict) else saved

        ids = [c["dialog_id"] for c in self.database]
        if isinstance(saved, dict) and saved.get("ids") != ids:
            raise ValueError("bm25 ids dont match therapy_database.json, rebuild the indices")
        if self.faiss_index.ntotal != len(self.database):
            raise ValueError("faiss index size dont match therapy_database.json, rebuild the indices")

        # idf from the db so stage 1 keywords are real tf-idf (not just word counts of one text)
        docs = []
        for c in self.database:
            tag = c.get("therapy_tag", {})
            docs.append(" ".join([as_text(c["case_profile"].get(k)) for k in FIELDS] + tag.get("applicable_conditions", [])))
        self.tfidf = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None)
        self.tfidf.fit(docs)
        # doc freq back from the smooth idf, idf = ln((1+n)/(1+df)) + 1
        self.df = np.round((len(docs) + 1) / np.exp(self.tfidf.idf_ - 1) - 1)

    def keywords(self, texts: List[str], top_n: int = 10) -> List[str]:
        vec = self.tfidf.transform([" ".join(texts)]).toarray()[0]
        vocab = self.tfidf.get_feature_names_out()
        # keep words that also show up in the db (df >= 2), else they can't match anything in bm25 anyway
        res = []
        for i in np.argsort(vec)[::-1]:
            if vec[i] <= 0 or len(res) >= top_n:
                break
            w = vocab[i]
            if w not in CHAT_WORDS and len(w) >= 4 and self.df[i] >= 2:
                res.append(w)
        return res

    def embed_profile(self, profile: Dict[str, Any]) -> np.ndarray:
        # embed each field on its own and mean pool, same as the case embeddings in the index
        texts = [as_text(profile.get(k)) for k in FIELDS]
        texts = [t for t in texts if t.strip()] or [json.dumps(profile)]
        vecs = self.encoder.encode(texts, convert_to_numpy=True)
        q = vecs.mean(axis=0)
        q = q / max(np.linalg.norm(q), 1e-12)
        return q.astype("float32")

    def retrieve(
        self,
        patient_profile: Dict[str, Any],
        keywords: List[str],
        alpha: float = config.ALPHA,
        w1: float = config.W1,
        w2: float = config.W2,
        k1: int = config.RAG_TOP_K1,
        k2: int = config.RAG_TOP_K2,
        drop_frac: float = config.RAG_DROP_FRAC,
    ) -> List[Dict[str, Any]]:
        n = len(self.database)
        q = self.embed_profile(patient_profile)

        # 1. semantic search over all cases
        sims, idxs = self.faiss_index.search(q[None, :], n)
        cos = np.zeros(n)
        for s, i in zip(sims[0], idxs[0]):
            cos[i] = float(s)

        # 2. bm25 on keywords vs therapy tags, minmax over the whole db
        raw = np.asarray(self.bm25.get_scores(tokenize(" ".join(keywords))), dtype=float)
        bm = (raw - raw.min()) / (raw.max() - raw.min()) if raw.max() > raw.min() else np.zeros(n)

        # 3. hybrid score (Eq. 1), keep top k1 cases
        hyb = alpha * cos + (1.0 - alpha) * bm
        order = np.argsort(hyb)[::-1]
        top = order[:k1]
        # next cases, only used to refill when the top k1 has fewer than k2 therapy types (db is ~69% SFBT)
        extra = order[k1:3 * k1]

        # 4. therapy suitability (Eq. 2)
        # live patient has no topic label, so use the majority topic of the retrieved cases for App
        case_topic = self.app_table.get("case_topic", {})
        topic = Counter(case_topic.get(self.database[i]["dialog_id"], "Unknown") for i in top).most_common(1)[0][0]
        app_row = self.app_table.get("app", {}).get(topic) or self.app_table.get("app", {}).get("Unknown") or {}

        cands, more = [], []
        for pos, i in enumerate(list(top) + list(extra)):
            case = self.database[i]
            tag = case.get("therapy_tag", {})
            th = tag.get("therapy_type", "CBT")
            m_s = 0.0
            if th in self.types:
                m_s = float(np.clip(np.dot(q, self.type_emb[self.types.index(th)]), 0.0, 1.0))
            app = float(app_row.get(th, 0.5))
            (cands if pos < len(top) else more).append({
                "case_id": case["dialog_id"],
                "score": float(hyb[i]),
                "cos": float(cos[i]),
                "bm25": float(bm[i]),
                "topic": topic,
                "therapy_type": th,
                "techniques": tag.get("techniques", []),
                "applicable_conditions": tag.get("applicable_conditions", []),
                "m_s": m_s,
                "app": app,
                "s_th": w1 * m_s + w2 * app,
            })

        # 5. drop the bottom 30% by s_th
        cands.sort(key=lambda c: c["s_th"], reverse=True)
        n_drop = int(len(cands) * drop_frac + 1e-9)
        kept, dropped = cands[:len(cands) - n_drop], cands[len(cands) - n_drop:]

        # 6. best case per therapy type, refill from dropped (non-SFBT first, db is ~69% SFBT)
        best: Dict[str, Dict[str, Any]] = {}
        for c in kept:
            best.setdefault(c["therapy_type"], c)
        for pool in [dropped, more]:
            for c in sorted(pool, key=lambda c: (c["therapy_type"] == "SFBT", -c["s_th"])):
                if len(best) >= k2:
                    break
                best.setdefault(c["therapy_type"], c)

        out = sorted(best.values(), key=lambda c: c["s_th"], reverse=True)
        return out[:k2]
