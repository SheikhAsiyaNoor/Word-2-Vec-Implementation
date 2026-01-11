from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
import os
import urllib.request
import zipfile
import time

def main():
    # Ensure data is available
    url = "http://mattmahoney.net/dc/text8.zip"
    zip_path = "text8.zip"
    text_path = "text8"

    if not os.path.exists(text_path):
        if not os.path.exists(zip_path):
            print("Downloading text8 dataset...")
            urllib.request.urlretrieve(url, zip_path)
        
        print("Extracting text8...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall()

    # Setup Corpus (Streaming)
    print("Loading corpus...")
    sentences = Text8Corpus(text_path)

    # Train Model using Gensim (optimized C implementation)
    # sg=1 means Skip-gram
    # negative=5 specifies negative sampling
    # vector_size=100 is the embedding dimension
    # window=5 is the context window
    print("Starting training with Gensim (Skip-gram with Negative Sampling)...")
    start_time = time.time()
    
    model = Word2Vec(
        sentences=sentences, 
        vector_size=100, 
        window=5, 
        min_count=5, 
        workers=4, 
        sg=1, 
        negative=5, 
        epochs=5
    )

    end_time = time.time()
    print(f"Training finished in {(end_time - start_time)/60:.2f} minutes.")

    # Save the model
    model.save("word2vec_gensim.model")
    print("Model saved as word2vec_gensim.model")

if __name__ == "__main__":
    main()