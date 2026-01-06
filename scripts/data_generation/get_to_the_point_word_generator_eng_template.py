import json
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist


GLOVE_PATH = "glove.6B.100d.txt"  # from: https://nlp.stanford.edu/data/glove.6B.zip
WORDLIST_PATH = "filtered_wordlist.json"  
MAX_WORDS = 10000  
OUTPUT_FILE = "word_pairs_by_similarity.json"


LOW_MAX = 0.40
HIGH_MIN = 0.60
HIGH_MAX = 0.65


def load_filtered_wordlist(path):
    with open(path, 'r') as f:
        return set(json.load(f))


def load_glove_embeddings(glove_path, allowed_words=None, max_words=None):
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading GloVe"):
            if max_words and len(embeddings) >= max_words:
                break
            parts = line.strip().split()
            word = parts[0]
            if allowed_words is None or word in allowed_words:
                vec = np.array(parts[1:], dtype=float)
                embeddings[word] = vec
    return embeddings


def generate_similarity_pairs(word_vecs):
    words = list(word_vecs.keys())
    vectors = np.vstack([word_vecs[w] for w in words])
    similarities = 1 - cdist(vectors, vectors, metric='cosine')

    low_sim_pairs = []
    high_sim_pairs = []

    for i in tqdm(range(len(words)), desc="Finding similarity-based pairs"):
        sim_row = similarities[i]
        sim_row[i] = -1  # skip self similarity
        sorted_indices = np.argsort(sim_row)[::-1]  # most similar first

        found_low = found_high = False

        for j in sorted_indices:
            sim = sim_row[j]
            if not found_high and HIGH_MIN < sim <= HIGH_MAX:
                high_sim_pairs.append({
                    "start": words[i],
                    "target": words[j],
                    "similarity": round(float(sim),3),
                    "similarity_level": "high"
                })
                found_high = True
            elif not found_low and sim <= LOW_MAX:
                low_sim_pairs.append({
                    "start": words[i],
                    "target": words[j],
                    "similarity": round(float(sim), 3),
                    "similarity_level": "low"
                })
                found_low = True

            # Break if both categories for the current word are found
            if found_low and found_high:
                break

    return {
        "low": low_sim_pairs,
        "high": high_sim_pairs
    }


def save_pairs_by_level(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved pairs by similarity level to {path}")


def main():
    print("Loading filtered word list...")
    allowed_words = load_filtered_wordlist(WORDLIST_PATH)

    print("Loading GloVe embeddings...")
    glove = load_glove_embeddings(GLOVE_PATH, allowed_words, max_words=MAX_WORDS)

    print(f"Loaded {len(glove)} valid GloVe words.")

    print("Generating similarity-based word pairs (Low and High levels only)...") # Updated print message
    similarity_groups = generate_similarity_pairs(glove)

    save_pairs_by_level(similarity_groups, OUTPUT_FILE)


if __name__ == "__main__":
    main()