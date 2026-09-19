# Fake News Detection Using Python and Machine Learning
A machine learning project for classifying news articles as FAKE or REAL using text-based features and a Passive Aggressive Classifier.
# Project Overview
Fake news can spread rapidly through online platforms and can make it difficult for readers to distinguish false information from genuine news. This project explores a machine learning approach for detecting fake news based on the textual content of news articles.
The project uses a curated dataset containing news articles labeled as `FAKE` or `REAL`. The article text is converted into numerical features using TF-IDF vectorization, and a PassiveAggressiveClassifier is trained to perform the classification.
The project was developed using Python and Google Colab.
# Objectives
- Collect and load a labeled fake-news dataset.
- Preprocess news text for machine learning.
- Convert text into TF-IDF feature vectors.
- Train a Passive Aggressive classification model.
- Evaluate the model using accuracy and a confusion matrix.
- Classify news articles as `FAKE` or `REAL`.
# Technologies Used
- Python
- Google Colab
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Passive Aggressive Classifier
# Project Workflow
```text
News Dataset
     ↓
Load Dataset
     ↓
DataFrame Preparation
     ↓
Train/Test Split
     ↓
TF-IDF Vectorization
     ↓
Passive Aggressive Classifier
     ↓
Prediction
     ↓
Accuracy & Confusion Matrix
```
The report's proposed framework also describes data pre-processing through tokenization, stop-word removal and stemming, followed by term-frequency/document-term processing and model evaluation.
# Dataset
The project uses a CSV dataset named:
```text
Fake_News.csv
```
The report describes the dataset as containing news articles from multiple domains with labels indicating whether each article is fake or real. The dataset was obtained from Kaggle.
Expected relevant columns include:
```text
text
label
```
Example labels:
```text
FAKE
REAL
```
> Note: The dataset file is not included in this repository unless you add it yourself. Make sure the filename and column names match the code.
# Installation
Install the required Python packages:
```bash
pip install numpy pandas scikit-learn
```
If you are using Google Colab, these libraries are generally available by default.
# How to Run
# 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```
# 2. Add the dataset
Place `Fake_News.csv` in the project directory.
# 3. Run the Python notebook/script
The original project was implemented in Google Colab. The main steps are:
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
```
Load the dataset:
```python
df = pd.read_csv("Fake_News.csv")
```

Get the labels:
```python
labels = df.label
```
Split the data:
```python
x_train, x_test, y_train, y_test = train_test_split(
    df["text"],
    labels,
    test_size=0.2,
    random_state=7
)
```
Create TF-IDF features:
```python
tfidf_vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7
)
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)
```
Train the classifier:
```python
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
```
Make predictions and calculate accuracy:
```python
y_pred = pac.predict(tfidf_test)

score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score * 100, 2)}%")
```
Generate the confusion matrix:
```python
confusion_matrix(
    y_test,
    y_pred,
    labels=["FAKE", "REAL"]
)
```
# Results
The reported experiment achieved an accuracy of:
92.19%
The confusion matrix reported in the project is:
```text
[[585, 53],
 [46, 583]]
```
The conclusion section of the report additionally states:
- True positives: 589
- True negatives: 587
- False positives: 42
- False negatives: 49
These figures appear inconsistent with the displayed confusion matrix and should therefore be treated as report-reported results rather than independently reconciled metrics.
# Key Concepts
# TF-IDF
TF-IDF (Term Frequency–Inverse Document Frequency) converts raw text into numerical features based on the importance of words within documents and across the corpus.
In this project, `TfidfVectorizer` is configured with English stop-word removal and `max_df=0.7`.
# Passive Aggressive Classifier
Passive Aggressive algorithms are online learning algorithms. The classifier remains passive when predictions are correct and updates its model when a classification is incorrect.
# Suggested Repository Structure
```text
fake-news-detection/
│
├── Fake_News.csv
├── fake_news_detection.ipynb
├── README.md
└── requirements.txt
```
A suitable `requirements.txt` could contain:
```text
numpy
pandas
scikit-learn
```
# Future Scope
The project report identifies several possible directions for future development:
- Identifying key sources involved in the spread of fake news.
- Applying graph theory and machine learning to analyze news propagation.
- Extending fake-news identification to video content.
- Combining fake-news classification with fact detection.
- Adding stance detection as another component of a larger detection system.
# References
The project report references research on:
- Hybrid deep models for fake news detection.
- Fake news stance detection using ensemble classifiers.
- Naive Bayes approaches to fake news detection.
- Event adversarial neural networks for multimodal fake news detection.
- Benchmark datasets for fake news detection.
- Satire detection using machine learning.
# Disclaimer
This project is an academic machine learning implementation for classifying news text. A model prediction should not be treated as definitive proof that a news article is true or false. Real-world verification should involve reliable sources and fact-checking.
Fake News Detection Using Python | 2023–24


