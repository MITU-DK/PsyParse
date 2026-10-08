import os
import re
import json
import pickle
import numpy as np
import faiss
from typing import List, Dict, Any, Set
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with", "is", "was",
    "am", "are", "i", "me", "my", "you", "your", "it", "this", "that", "feel", "really"
}

def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]

def prepare_psy_insight_data(
    raw_json_path: str = "data/en_data_version7.json",
    output_dir: str = "data",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    os.makedirs(output_dir, exist_ok=True)
    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_sessions: List[Dict[str, Any]] = json.load(f)

    # 1. Normalization & Filtering
    filtered_sessions = []
    therapy_map = {
        "cognitive behaviour therapy": "CBT",
        "cognitive behavioral therapy": "CBT",
        "solution-focused brief therapy": "SFBT",
        "solution focused brief therapy": "SFBT",
        "rational emotive behavior therapy": "REBT",
        "person-centered therapy": "PCT",
    }

    for s in raw_sessions:
        raw_th = str(s.get("psychotherapy", "")).strip()
        if not raw_th or raw_th.lower() == "unknown":
            continue
        norm_th = therapy_map.get(raw_th.lower(), raw_th)
        s["psychotherapy"] = norm_th
        filtered_sessions.append(s)

    # 2. Group Split to prevent patient persona leakage
    random_state = np.random.RandomState(42)
    random_state.shuffle(filtered_sessions)

    eval_set = filtered_sessions[:20]
    dev_set = filtered_sessions[20:25]
    db_set = filtered_sessions[25:]

    with open(os.path.join(output_dir, "eval_scenarios.json"), "w", encoding="utf-8") as f:
        json.dump(eval_set, f, indent=2)
    with open(os.path.join(output_dir, "dev_scenarios.json"), "w", encoding="utf-8") as f:
        json.dump(dev_set, f, indent=2)

    # 3. Assemble therapy_database.json
    db_records = []
    bm25_corpus = []

    for idx, s in enumerate(db_set):
        dialog = s.get("dialog", [])
        emotions = list({str(t.get("emotional_label", "")).strip() for t in dialog if t.get("emotional_label")})
        symptoms = list({str(t.get("observation", "")).strip() for t in dialog if t.get("speaker") == "client"})
        techniques = list({str(t.get("strategy", "")).strip() for t in dialog if t.get("strategy")})

        record = {
            "dialog_id": s.get("dialog_id", f"case_{idx}"),
            "case_profile": {
                "core_problems": s.get("background", ""),
                "emotional_states": " ".join(emotions),
                "symptoms": " ".join(symptoms),
            },
            "therapy_tag": {
                "therapy_type": s["psychotherapy"],
                "techniques": [t for t in techniques if t.lower() not in ["neutral", "others"]],
                "applicable_conditions": [s.get("topic", "General")]
            }
        }
        db_records.append(record)

        # Tokenize tags for BM25
        doc_text = f"{record['therapy_tag']['therapy_type']} {' '.join(record['therapy_tag']['techniques'])} {' '.join(record['therapy_tag']['applicable_conditions'])}"
        bm25_corpus.append(tokenize(doc_text))

    with open(os.path.join(output_dir, "therapy_database.json"), "w", encoding="utf-8") as f:
        json.dump(db_records, f, indent=2)

    # 4. Build BM25 Index
    bm25 = BM25Okapi(bm25_corpus)
    with open(os.path.join(output_dir, "therapy_bm25.pkl"), "wb") as f:
        pickle.dump(bm25, f)

    # 5. Build FAISS Index with L2 Normalization
    encoder = SentenceTransformer(embedding_model)
    texts_to_embed = [
        f"{r['case_profile']['core_problems']} {r['case_profile']['emotional_states']} {r['case_profile']['symptoms']}"[:1000]
        for r in db_records
    ]
    embeddings = encoder.encode(texts_to_embed, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(output_dir, "therapy_faiss.index"))

    # 6. Precompute Prior Lift Table (app_table.json)
    topic_th_counts: Dict[str, Dict[str, int]] = {}
    for r in db_records:
        topic = str(r["case_profile"]["core_problems"])[:50]
        tt = r["therapy_tag"]["therapy_type"]
        topic_th_counts.setdefault(topic, {})
        topic_th_counts[topic][tt] = topic_th_counts[topic].get(tt, 0) + 1

    app_table: Dict[str, Dict[str, float]] = {}
    for topic, counts in topic_th_counts.items():
        total = sum(counts.values())
        app_table[topic] = {tt: round((c + 1) / (total + len(counts)), 3) for tt, c in counts.items()}

    with open(os.path.join(output_dir, "app_table.json"), "w", encoding="utf-8") as f:
        json.dump(app_table, f, indent=2)

    logger.info("Data prep finished. Indexed %d cases across %d dimensions.", len(db_records), dim)

if __name__ == "__main__":
    prepare_psy_insight_data()