from gensim.models import Word2Vec
import gensim.downloader as api
import numpy as np

def detect_bias(model):
    
    # Detects bias by checking cosine similarity between neutral professions 
    # and gendered terms 'man' and 'woman'.
    
    neutral_words = ['programmer', 'doctor', 'nurse', 'engineer', 'teacher', 'homemaker']
    gender_terms = ['man', 'woman']
    
    results = {}
    for nw in neutral_words:
        if nw in model.wv:
            sims = {}
            for gt in gender_terms:
                if gt in model.wv:
                    sims[gt] = model.wv.similarity(nw, gt)
            results[nw] = sims
    return results

def main():
    # Load trained model
    try:
        model = Word2Vec.load("word2vec_gensim.model")
        print("Successfully loaded trained Gensim model.")
    except FileNotFoundError:
        print("Model file not found. Please run train_gensim.py first.")
        return

    # Compare with pre-trained vectors from Gensim downloader
    print("\n--- Comparison with Pre-trained Vectors (glove-wiki-gigaword-100) ---")
    pretrained = api.load("glove-wiki-gigaword-100")
    
    pairs = [('king', 'queen'), ('man', 'woman'), ('apple', 'fruit'), ('car', 'engine')]
    print(f"{'Pair':<20} | {'My Model Sim':<15} | {'Gensim Pretrained Sim':<15}")
    print("-" * 60)
    for w1, w2 in pairs:
        my_sim = model.wv.similarity(w1, w2) if w1 in model.wv and w2 in model.wv else "N/A"
        pre_sim = pretrained.similarity(w1, w2) if w1 in pretrained and w2 in pretrained else "N/A"
        
        my_sim_str = f"{my_sim:.4f}" if isinstance(my_sim, (float, np.float32)) else my_sim
        pre_sim_str = f"{pre_sim:.4f}" if isinstance(pre_sim, (float, np.float32)) else pre_sim
        print(f"{f'{w1}-{w2}':<20} | {my_sim_str:<15} | {pre_sim_str:<15}")

    # Word Analogy Task
    print("\n--- Word Analogy Task: king - man + woman ---")
    try:
        # Analogy: king - man + woman = ?
        result = model.wv.most_similar(positive=['king', 'woman'], negative=['man'], topn=5)
        for word, score in result:
            print(f"{word}: {score:.4f}")
    except KeyError as e:
        print(f"Analogy failed: {e}")

    # For Bias Detection
    print("\n--- Bias Detection (Proximity to Gendered Words) ---")
    bias_data = detect_bias(model)
    for word, sims in bias_data.items():
        if 'man' in sims and 'woman' in sims:
            diff = sims['man'] - sims['woman']
            bias_label = "Man-biased" if diff > 0.05 else ("Woman-biased" if diff < -0.05 else "Neutral")
            print(f"{word:<12}: Man={sims['man']:.3f}, Woman={sims['woman']:.3f} (Diff={diff:.3f}) -> {bias_label}")

if __name__ == "__main__":
    main()