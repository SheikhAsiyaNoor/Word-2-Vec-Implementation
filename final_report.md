# Word2Vec Implementation Report

## 1. Cosine Similarity Comparison
| Pair | My Model Sim | Gensim Pretrained Sim |
| :--- | :--- | :--- |
| king-queen | 0.7120 | 0.7508 |
| man-woman | 0.7144 | 0.8323 |
| apple-fruit | 0.4265 | 0.5359 |
| car-engine | 0.6088 | 0.6330 |

## 2. Word Analogy Task: king - man + woman
- **daughter**: 0.6555
- **queen**: 0.6381
- **consort**: 0.6315
- **princess**: 0.6269
- **throne**: 0.6251

## 3. Bias Detection
| Profession | Man Sim | Woman Sim | Bias |
| :--- | :--- | :--- | :--- |
| programmer | 0.298 | 0.259 | Neutral |
| doctor | 0.484 | 0.420 | Man-biased |
| nurse | 0.301 | 0.474 | Woman-biased |
| engineer | 0.340 | 0.210 | Man-biased |
| teacher | 0.445 | 0.460 | Neutral |
