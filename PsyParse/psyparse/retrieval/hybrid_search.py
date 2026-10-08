import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple

class HybridRetriever:
    """
    Implements multi-therapy RAG:
      - Equation 1: Hybrid Retrieval S(P, C_i) = alpha * Cos(E(P), E(C_i)) + (1 - alpha) * B_norm(K, T_i)
      - Equation 2: Therapy Suitability S_th(P, T_i) = w1 * M_s(P, T_i) + w2 * App(T_i)
      - Percentile pruning (bottom 30%) and top-k2 diversity grouping with fallback escape hatches.
    """
    def __init__(
        self,
        db_path: str = "data/therapy_database.json",
        faiss_path: str = "data/therapy_faiss.index",
        bm25_path: str = "data/therapy_bm25.pkl",
        app_table_path: str = "data/app_table.json",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.encoder = SentenceTransformer(model_name)
        with open(db_path, "r", encoding="utf-8") as f:
            self.database: List[Dict[str, Any]] = json.load(f)

        self.faiss_index = faiss.read_index(faiss_path)
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
            if isinstance(bm25_data, dict) and "bm25" in bm25_data:
                self.bm25: BM25Okapi = bm25_data["bm25"]
            else:
                self.bm25: BM25Okapi = bm25_data

        self.app_table: Dict[str, Dict[str, float]] = {}
        if os.path.exists(app_table_path):
            with open(app_table_path, "r", encoding="utf-8") as f:
                self.app_table = json.load(f)

    def retrieve(
        self,
        patient_profile: Dict[str, Any],
        keywords: List[str],
        alpha: float = 0.5,
        w1: float = 0.5,
        w2: float = 0.5,
        k1: int = 10,
        k2: int = 3,
    ) -> List[Dict[str, Any]]:
        # Schema subset matching the indexed Case Profiles
        shared_text = " ".join([
            str(patient_profile.get("core_problems", "")),
            str(patient_profile.get("emotional_states", "")),
            str(patient_profile.get("symptoms", "")),
        ])[:1000]

        # 1. Semantic Search (FAISS IndexFlatIP)
        p_emb = self.encoder.encode([shared_text], convert_to_numpy=True)
        faiss.normalize_L2(p_emb)
        sims, indices = self.faiss_index.search(p_emb, len(self.database))
        cos_scores = np.zeros(len(self.database))
        for score, idx in zip(sims[0], indices[0]):
            cos_scores[idx] = float(score)

        # 2. Lexical Search (BM25 with Min-Max Normalization)
        tokenized_k = [k.lower() for k in keywords]
        bm25_raw = np.array(self.bm25.get_scores(tokenized_k))
        b_min, b_max = bm25_raw.min(), bm25_raw.max()
        bm25_norm = np.zeros_like(bm25_raw) if b_max == b_min else (bm25_raw - b_min) / (b_max - b_min)

        # 3. Hybrid Score (Eq. 1)
        hybrid_scores = alpha * cos_scores + (1.0 - alpha) * bm25_norm
        ranked_pool_indices = np.argsort(hybrid_scores)[::-1][:k1]

        # 4. Therapy Suitability Scoring (Eq. 2)
        candidate_cases = []
        sym_emb = self.encoder.encode([str(patient_profile.get("symptoms", ""))], convert_to_numpy=True)
        faiss.normalize_L2(sym_emb)

        for idx in ranked_pool_indices:
            case = self.database[idx]
            tag = case.get("therapy_tag", {})
            th_type = tag.get("therapy_type", "CBT")
            app_conditions = " ".join(tag.get("applicable_conditions", []))

            # Symptom match M_s (cosine similarity)
            app_emb = self.encoder.encode([app_conditions], convert_to_numpy=True)
            faiss.normalize_L2(app_emb)
            m_s = max(0.0, float(np.dot(sym_emb, app_emb.T)[0][0]))

            # Applicability App
            topic = str(patient_profile.get("core_problems", "General"))
            app_val = self.app_table.get(topic, {}).get(th_type, 0.5)

            s_th = (w1 * m_s) + (w2 * app_val)
            candidate_cases.append({
                "case": case,
                "s_th": s_th,
                "m_s": m_s,
                "app": app_val,
                "therapy_type": th_type,
                "techniques": tag.get("techniques", []),
                "applicable_conditions": tag.get("applicable_conditions", []),
            })

        # 5. Percentile Filtering (discard bottom 30%)
        candidate_cases.sort(key=lambda x: x["s_th"], reverse=True)
        keep_count = max(1, int(len(candidate_cases) * 0.7))
        filtered = candidate_cases[:keep_count]

        # 6. Distinct Grouping by Therapy Type with diversity override
        type_to_best: Dict[str, Dict[str, Any]] = {}
        for c in filtered:
            tt = c["therapy_type"]
            if tt not in type_to_best:
                type_to_best[tt] = c

        selected = list(type_to_best.values())

        # If filtering reduced distinct types below 2, rescue from the discarded pool
        if len(selected) < 2 and len(candidate_cases) > len(selected):
            existing_types = {s["therapy_type"] for s in selected}
            for c in candidate_cases:
                if c["therapy_type"] not in existing_types:
                    selected.append(c)
                    existing_types.add(c["therapy_type"])
                    if len(selected) >= 2:
                        break

        selected.sort(key=lambda x: x["s_th"], reverse=True)
        return selected[:k2]