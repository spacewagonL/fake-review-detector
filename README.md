# Fake Review Detection using Machine Learning

## Overview
This project aims to detect fake reviews using various machine learning techniques. We analyze review text data and compare the performance of traditional supervised learning models (SVM, Logistic Regression, Random Forest, XGBoost) against deep learning models (LSTM). The goal is to determine which approach is most effective in distinguishing fake vs. real reviews.

## Technologies Used
- Python (NumPy, Pandas, Matplotlib, Seaborn)
- Natural Language Processing (NLP) (NLTK, TF-IDF)
- Machine Learning Models (SVM, Logistic Regression, Random Forest, XGBoost)
- Deep Learning (LSTM) (TensorFlow, Keras)
- Google Colab (for training and visualization)

## Methodology
1. **Data Preprocessing**
   - Cleaned text by lowercasing, removing stopwords, and applying lemmatization
   - Converted text into numerical features using TF-IDF
   - Split dataset into training (80%) and testing (20%)

2. **Supervised Learning Models**
   - Trained and evaluated SVM, Logistic Regression, Random Forest, and XGBoost
   - SVM and Logistic Regression performed the best with around 88% accuracy

3. **Deep Learning with LSTM**
   - Built an LSTM-based model for text classification
   - The model struggled to converge, achieving around 50% accuracy
   - Further tuning is needed for LSTM to be effective

## Results & Findings
- Supervised learning models (SVM & Logistic Regression) outperformed deep learning models (LSTM)
- LSTM did not generalize well, likely due to insufficient data or lack of pre-trained embeddings
- TF-IDF proved to be an effective feature extraction method for detecting fake reviews

## Future Work
- Fine-tune deep learning models (LSTM, BiLSTM, GRU) with pre-trained word embeddings (GloVe, Word2Vec)
- Implement BERT (Bidirectional Encoder Representations from Transformers) for better context understanding
- Explore unsupervised learning techniques to detect fake reviews without labeled data
- Use ensemble methods combining supervised and deep learning models for higher accuracy

## How to Run
- Clone the repository:
   ```bash
   git clone https://github.com/spacewagonL/fake-review-detector/fake-review-detection.git
   cd fake-review-detection
