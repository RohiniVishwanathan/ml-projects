Natural Language Processing Portfolio – Rohini Vishwanathan
This folder contains my natural language processing coursework and applied projects completed as part of a graduate-level NLP class at Claremont Graduate University. The work spans the full NLP pipeline—preprocessing, feature engineering, vectorization, topic modeling, BERT embedding, and text classification—applied on real-world datasets such as Yelp reviews and large textual corpora.
Each lab includes both:
A Jupyter Notebook containing full code
A PDF report summarizing methodology, results, and findings
----Contents-----
Lab 1: Text Preprocessing
Cleaning → tokenization → lemmatization → stopword removal
Built a preprocessing pipeline using NLTK and spaCy
Implemented rules to remove digits, punctuation, and noise
Output: clean token sets ready for feature extraction
Files:
Lab1_Rohini_Text_Preprocessing.ipynb
Lab1_Rohini_Text_Preprocessing.pdf
Lab 2: Basic NLP Functions
Utility NLP methods for downstream feature engineering
Created functions for text normalization, tokenization, stemming, POS tagging
Implemented sentence-level utilities and word-level transformations
Files:
Lab2_Rohini_NLP_Functions.ipynb
Lab2_Rohini_NLP_Functions.pdf
Lab 3: Feature Vectorization & Sentiment Analysis
BoW → TF-IDF → Word2Vec → Sentiment Models
Generated word embeddings using BoW, TF-IDF, and Word2Vec
Applied sentiment scorers (TextBlob & VADER)
Compared vectorization techniques for downstream classification
Files:
Lab3_Rohini_Vectorization_Sentiment.ipynb
Lab3_Rohini_Vectorization_Sentiment.pdf
Lab 4: Pre-trained BERT Embeddings
Transformer-based contextual vectorization
Used BERT and DistilBERT models from HuggingFace
Created sentence embeddings for Yelp reviews
Visualized embedding distribution & compared with classical embeddings
Files:
Lab4_Rohini_BERT_Embeddings.ipynb
Lab4_Rohini_BERT_Embeddings.pdf
Lab 5: Text Classification
Traditional ML models + dimensionality reduction with SVD
Built classification models including Logistic Regression, SVM, and GBM
Applied SVD for dimensionality reduction on high-dimensional text data
Evaluated performance using accuracy & macro-F1
Files:
Lab5_Rohini_Text_Classification.ipynb
Lab5_Rohini_Text_Classification.pdf
Lab 6: Topic Modeling (LSA, TF-IDF LSA, and LDA)
Unsupervised topic extraction on 63k+ Yelp reviews
Built LSA models using BoW and TF-IDF
Tuned number of topics using coherence scores
Built LDA topic model with Gensim and visualized results using pyLDAvis
Selected and labeled optimal topics for business interpretation
Files:
Lab6_Rohini_Topic_Modeling.ipynb
Lab6_Rohini_Topic_Modeling.pdf
Take-Home NLP Exam
Hybrid modeling + feature engineering + multi-step NLP pipeline
Combined TF-IDF, LIWC features, and metadata features
Modeled personality traits using traditional ML and hybrid feature sets
Delivered interpretation-focused evaluation and model selection
Files:
TakeHome_Rohini_NLP_Exam.ipynb
TakeHome_Rohini_NLP_Exam.pdf
-----Skills Demonstrated------
Text preprocessing (tokenization, lemmatization, POS tagging)
Feature engineering (TF-IDF, BoW, Word2Vec, FastText, BERT)
Topic modeling (LSA, LDA, coherence)
Text classification (Logistic, SVM, Gradient Boosting)
Transformer models & contextual embeddings
pyLDAvis interactive visualization
Sentiment analysis (TextBlob, VADER)
Pipeline building & reproducible ML workflows
-----Goal of This Portfolio--------
To demonstrate practical, end-to-end NLP capability using both classical and modern transformer-based methods, with a focus on real-world interpretability and clean analytical workflow.
