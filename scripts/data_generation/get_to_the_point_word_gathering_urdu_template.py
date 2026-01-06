import stanza
import json
import re
from tqdm import tqdm
from rapidfuzz.distance import Levenshtein

stanza.download('ur')

nlp = stanza.Pipeline(lang='ur', processors='tokenize,pos,lemma', use_gpu=False)

LIST_PATH = "urd_news_2020_30K-words.txt" # from https://wortschatz.uni-leipzig.de/en/download/Urdu
WORDLIST_OUTPUT = "filtered_wordlist_urdu.json"
MIN_WORD_LENGTH = 3
SIMILARITY_THRESHOLD = 0.8

def get_urdu_stopwords():
    stopwords_list = {
        "اور", "کا", "کی", "کو", "میں", "ہے", "ہیں", "تھا", "تھی", "تھے",
        "سے", "پر", "نے", "کے", "ایک", "اس", "یہ", "وہ", "بھی", "کہ",
        "!", '"', '%', "'", "(", ")", "*", "+", ",", "-", ".", "/", ":", "?", "§", "¿", "“", "”", "‘", "’", "…", "؟",
        "و", "ا",
    }
    return set(stopwords_list)

def filter_stopwords_from_list(words_list):
    """
    Removes common Urdu stopwords from a list of words.
    """
    stop_words = get_urdu_stopwords() 
    filtered = [w for w in words_list if w not in stop_words]
    print(f"Removed {len(words_list) - len(filtered)} stopwords.")
    return filtered

def get_urdu_nouns(file_path, min_length=MIN_WORD_LENGTH):
    nouns = set()

    # regex specifically for urd_news_2020_30K-words.txt
    urdu_char_pattern = re.compile(r'^[ \u0600-\u06FF]+$')

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        words_to_process = []
        for line in lines:
            parts = line.strip().split(maxsplit=2)
            if len(parts) >= 2:
                word_text = parts[1]
                if len(word_text) >= min_length and urdu_char_pattern.match(word_text):
                    words_to_process.append(word_text)


        batch_size = 1000
        for i in tqdm(range(0, len(words_to_process), batch_size), desc="Stanza Processing Batches"):
            batch = words_to_process[i:i + batch_size]
            
            text_batch = " ".join(batch)
            
            try:
                doc = nlp(text_batch)
                for sentence in doc.sentences:
                    for word in sentence.words:
                        # Stanza's UPOS tags: NOUN, PROPN
                        if word.upos in ['NOUN', 'PROPN']:
                            
                            lemma = word.lemma.lower()
                            if len(lemma) >= min_length and urdu_char_pattern.match(lemma):
                                nouns.add(lemma)
            except Exception as stanza_e:
                print(f"Error processing batch: {stanza_e}")

    except FileNotFoundError:
        print(f"Error: List file '{file_path}' was not found.")
        return set()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return set()

    return sorted(list(nouns))


def is_similar(w1, w2, threshold=SIMILARITY_THRESHOLD):
    """Return True if words w1 and w2 are similar beyond threshold."""
    similarity = Levenshtein.normalized_similarity(w1, w2)
    return similarity > threshold

def remove_similar_words(words_list, similarity_threshold=SIMILARITY_THRESHOLD):
    """Remove words very similar to earlier words to avoid duplicates."""
    filtered_unique_words = []
    for word in tqdm(words_list, desc="Removing near-duplicate words"):
        if not any(is_similar(word, existing_word, similarity_threshold) for existing_word in filtered_unique_words):
            filtered_unique_words.append(word)
    return filtered_unique_words

def save_wordlist(words_list, filename):
    """Saves the list of words to a JSON file."""
    print(f"Saving wordlist to {filename}...")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(words_list, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(words_list)} words to {filename}")
    except Exception as e:
        print(f"Error saving wordlist to {filename}: {e}")


if __name__ == "__main__":

    print("1. Extracting Urdu noun lemmas from Leipzig frequency list using Stanza...")
    urdu_nouns_raw = get_urdu_nouns(LIST_PATH, min_length=MIN_WORD_LENGTH)
    print(f"Initial count of unique Urdu noun lemmas: {len(urdu_nouns_raw)}")

    print("2. Filtering out common Urdu stopwords...")
    urdu_nouns_filtered_stopwords = filter_stopwords_from_list(urdu_nouns_raw)
    print(f"Count after stopword removal: {len(urdu_nouns_filtered_stopwords)}")

    print(f"3. Removing near-duplicate words using Levenshtein similarity (threshold={SIMILARITY_THRESHOLD})...")
    final_clean_urdu_nouns = remove_similar_words(urdu_nouns_raw, similarity_threshold=SIMILARITY_THRESHOLD)
    print(f"Final count of clean, unique Urdu nouns: {len(final_clean_urdu_nouns)}")

    print("4. Saving the final wordlist...")
    save_wordlist(final_clean_urdu_nouns, WORDLIST_OUTPUT)

    print("--- Urdu Noun Wordlist Generation Complete ---")