import json
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import os
import fasttext 

URDU_EMBEDDINGS_PATH = "cc.ur.300.bin" # Changed to .bin
WORDLIST_PATH = "filtered_wordlist_urdu.json"
MAX_WORDS = 6000 # approx 6k words in wordlist
OUTPUT_FILE = "word_pairs_by_similarity_urdu.json"

LOW_MIN = 0.30
LOW_MAX = 0.50

HIGH_MIN = 0.70
HIGH_MAX = 0.90


fasttext_model = None

def load_fasttext_model(path):
    """Loads a FastText model from a .bin file."""
    global fasttext_model
    if not os.path.exists(path):
        raise FileNotFoundError(f"FastText model file not found: {path}. Please download the .bin model.")
    print(f"Loading FastText model from {path} (this might take a moment)...")
    fasttext_model = fasttext.load_model(path)
    print("FastText model loaded.")
    print(f"FastText model vector dimension: {fasttext_model.get_dimension()}")
    return fasttext_model


def load_filtered_wordlist(path):
    """Loads the set of filtered Urdu nouns from a JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Wordlist file not found: {path}. Please ensure it exists.")
    with open(path, 'r', encoding='utf-8') as f:
        return set(json.load(f))


def get_word_embeddings_from_fasttext(fasttext_model_obj, allowed_words=None, max_words=None):
    """
    Retrieves word embeddings from a loaded FastText model.
    Filters by allowed_words and limits by max_words.
    """
    embeddings = {}
    words_processed = 0
    
    if allowed_words is None:
        print("Warning: 'allowed_words' is None. Cannot load specific embeddings without a word list.")
        return embeddings

    for word in tqdm(sorted(list(allowed_words)), desc="Getting embeddings for allowed words"): # Sorting for consistent progress bar
        if max_words and words_processed >= max_words:
            break
        
        vec = fasttext_model_obj.get_word_vector(word)
        embeddings[word] = vec
        words_processed += 1

    return embeddings


def generate_similarity_pairs(word_vecs):
    """
    Generates pairs of words with 'low' and 'high' cosine similarity.
    """
    words = list(word_vecs.keys())
    if not words:
        print("No words with embeddings to process. Returning empty pairs.")
        return {"low": [], "high": []}

    vectors = np.vstack([word_vecs[w] for w in words])
    similarities = 1 - cdist(vectors, vectors, metric='cosine')

    low_sim_pairs = []
    high_sim_pairs = []

    for i in tqdm(range(len(words)), desc="Finding similarity-based pairs"):
        start_word = words[i]
        sim_row = similarities[i]
        sim_row[i] = -1.0

        sorted_indices = np.argsort(sim_row)[::-1]

        found_low = False
        found_high = False

        for j in sorted_indices:
            target_word = words[j]
            sim = sim_row[j]

            if not found_high and HIGH_MIN <= sim < HIGH_MAX:
                high_sim_pairs.append({
                    "start": start_word,
                    "target": target_word,
                    "similarity": round(float(sim), 3),
                    "similarity_level": "high"
                })
                found_high = True
            elif not found_low and LOW_MIN <= sim < LOW_MAX:
                low_sim_pairs.append({
                    "start": start_word,
                    "target": target_word,
                    "similarity": round(float(sim), 3),
                    "similarity_level": "low"
                })
                found_low = True

            if found_low and found_high:
                break

    return {
        "low": low_sim_pairs,
        "high": high_sim_pairs
    }


def save_pairs_by_level(data, path):
    """Saves the generated word pairs to a JSON file."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data['low'])} low similarity pairs and {len(data['high'])} high similarity pairs to {path}")


    


if __name__ == "__main__":
    # Step 1: Load the FastText model (global variable for easy access)
    #global fasttext_model
    fasttext_model = load_fasttext_model(URDU_EMBEDDINGS_PATH)
    if fasttext_model is None:
        print("Failed to load FastText model. Exiting.")

    print("Loading filtered Urdu noun word list...")
    allowed_words = load_filtered_wordlist(WORDLIST_PATH)
    print(f"Loaded {len(allowed_words)} unique Urdu nouns from wordlist.")

    print("Getting embeddings for allowed words from FastText model...")
    # Use the new function to get embeddings
    urdu_embeddings = get_word_embeddings_from_fasttext(fasttext_model, allowed_words, max_words=MAX_WORDS)

    print(f"Found embeddings for {len(urdu_embeddings)} words out of {len(allowed_words)} allowed words.")

    if not urdu_embeddings:
        print("No Urdu embeddings retrieved for the word list. Cannot generate similarity pairs. Exiting.")

    print("Generating similarity-based Urdu word pairs (Low and High levels only)...")
    similarity_groups = generate_similarity_pairs(urdu_embeddings)

    save_pairs_by_level(similarity_groups, OUTPUT_FILE)