import nltk
import json
from nltk.corpus import wordnet as wn, stopwords
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm
import os


def download_nltk_data():
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download('wordnet')
    try:
        stopwords.words('english')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')


def get_wordnet_lemmas(min_length=3):
    lemmas = set()
    for synset in wn.all_synsets(pos=wn.NOUN): 
        for lemma in synset.lemmas():
            word = lemma.name().lower()
            if '_' not in word and word.isalpha() and len(word) >= min_length:
                lemmas.add(word)
    return sorted(lemmas)


def filter_stopwords(words):
    stop_words = set(stopwords.words('english'))
    return [w for w in words if w not in stop_words]


def is_similar(w1, w2, threshold=0.8):
    similarity = Levenshtein.normalized_similarity(w1, w2)
    return similarity > threshold


def remove_similar_words(words, similarity_threshold=0.8):
    filtered = []
    for w in tqdm(words, desc="Removing similar words", unit="word"):
        if not any(is_similar(w, fw, similarity_threshold) for fw in filtered):
            filtered.append(w)
    return filtered


def save_wordlist(words, filename):
    with open(filename, 'w') as f:
        json.dump(words, f, indent=2)
    print(f"Saved {len(words)} words to {filename}")


def main():
    print("Downloading required NLTK data...")
    download_nltk_data()

    print("Extracting WordNet noun lemmas...")
    lemmas = get_wordnet_lemmas() 
    print(f"Initial noun lemmas count: {len(lemmas)}")

    print("Filtering out stopwords...")
    filtered = filter_stopwords(lemmas)
    print(f"After stopwords removal: {len(filtered)}")

    print("Removing near-duplicate similar words (fuzzy match)...")
    clean_list = remove_similar_words(filtered, similarity_threshold=0.8)
    print(f"Final filtered noun word count: {len(clean_list)}")

    output_file = "filtered_wordlist.json" 
    save_wordlist(clean_list, output_file)


if __name__ == "__main__":
    main()