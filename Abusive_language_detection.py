import streamlit as st
import pandas as pd
import numpy as np
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag, StanfordNERTagger
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import string
import spacy
import re
from wordcloud import WordCloud
from html import unescape
from sentence_transformers import SentenceTransformer
import random
import plotly.express as px

from sklearn.manifold import TSNE




# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nlp = spacy.load("en_core_web_sm")
nltk.download('averaged_perceptron_tagger')

# Load your dataset
def load_data(file):
    # Assuming your dataset is a CSV file
    df = pd.read_csv(file)
    return df


# Preprocess the text data
def preprocess_text(text):

    # Remove URLs
    text = ' '.join(word for word in text.split() if not word.startswith('http'))

    # Remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Remove @ tags
    text = ' '.join(word for word in text.split() if not word.startswith('@'))

    if 'RT @' in text:
      text = text.split(':')[1] if ':' in text else text

    # Remove exclamatory marks
    text = text.replace('!', '')

    # Unescape HTML entities
    text = unescape(text)

    # Convert to lowercase
    text = text.lower()
    # Tokenize
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return tokens




# Tokenization, POS Tagging, NER Tagging
def perform_nlp(df):
    # Tokenization
    df['toks'] = df['prep_text'].apply(word_tokenize)

    # POS Tagging
    df['pos_tags'] = df['toks'].apply(pos_tag)

    return df


# Word Embedding using Word2Vec
def word_embedding(df):
    model = Word2Vec(sentences=df['toks'], vector_size=100, window=5, min_count=1, workers=4)
    return model

# Logistic Regression Model
def train_logistic_regression(df):
    # Assuming 'class' is the target variable
    X_train, X_test, y_train, y_test = train_test_split(df['prep_text'], df['class'], test_size=0.2, random_state=42)

    # Vectorization
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression
    lr_model = LogisticRegression()
    lr_model.fit(X_train_vec, y_train)

    # Predictions
    predictions = lr_model.predict(X_test_vec)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Multinomial Naive Bayes Model
def train_multinomial_nb(df):
    # Assuming 'class' is the target variable
    X_train, X_test, y_train, y_test = train_test_split(df['prep_text'], df['class'], test_size=0.2, random_state=42)

    # Vectorization
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Multinomial Naive Bayes
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)

    # Predictions
    predictions = nb_model.predict(X_test_vec)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Support Vector Machine Model
def train_svm(df):
    # Assuming 'class' is the target variable
    X_train, X_test, y_train, y_test = train_test_split(df['prep_text'], df['class'], test_size=0.2, random_state=42)

    # Vectorization
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Support Vector Machine
    svm_model = SVC()
    svm_model.fit(X_train_vec, y_train)

    # Predictions
    predictions = svm_model.predict(X_test_vec)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Random Forest Model
def train_random_forest(df):
    # Assuming 'class' is the target variable
    X_train, X_test, y_train, y_test = train_test_split(df['prep_text'], df['class'], test_size=0.2, random_state=42)

    # Vectorization
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train_vec, y_train)

    # Predictions
    predictions = rf_model.predict(X_test_vec)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)
    return rf_model, vectorizer,accuracy




def main():
    st.title("NLP - Abusive Language Detection")

    # Upload dataset
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        data_subset = df.head(20)

        # Display dataset
        selected_columns = ['count', 'hate_speech', 'offensive_language', 'neither', 'class',
       'tweet']
        st.write("### Dataset")
        st.dataframe(df[selected_columns].head())

        # NLP processing
        df = perform_nlp(df)

        # Display preprocessed text
        st.write("### Preprocessed Text")
        st.write(df['prep_text'].head(20))

        # Display tokens
        st.write("### Tokens")
        st.write(df['toks'].head(20))



        # Display POS tags
        st.write("### POS Tags (First 20)")
        st.write(df['pos_tags'].head(20))

        # Display NER tags
        sample_size = 5  # Adjust the sample size as needed
        ner_entities_all = []
        for text in data_subset['prep_text'].sample(sample_size).tolist():
            doc_ner_all = nlp(text)
            ner_entities_all.extend([(ent.text, ent.label_) for ent in doc_ner_all.ents])

        # Display a small number of NER tags
        st.subheader(f"Showing {sample_size} NER Tags:")
        st.write(ner_entities_all)

        # Visualize NER markup for the entire dataset
        st.subheader("NER Tag Markup for the Entire Dataset:")
        ner_markup_all = spacy.displacy.render(nlp(str(data_subset['prep_text'].tolist())), style="ent", page=True)
        st.write(ner_markup_all, unsafe_allow_html=True)

        # Visualize NER tags for the entire dataset
        ner_counts_all = Counter(tag for text, tag in ner_entities_all)
        ner_df_all = pd.DataFrame.from_dict(ner_counts_all, orient='index', columns=['count'])
        ner_df_all = ner_df_all.sort_values(by='count', ascending=False)

        # Plot bar chart for NER tags for the entire dataset
        st.subheader("NER Tag Distribution for the Entire Dataset")
        fig_ner_distribution = px.bar(ner_df_all, x=ner_df_all.index, y='count', title='NER Tag Distribution')
        st.plotly_chart(fig_ner_distribution)

        st.subheader("Word Embedding")

        use_random_sample_embedding = st.checkbox("Use random sample from dataset as default (Word Embedding)", key="use_random_sample_embedding", value=True)

        if use_random_sample_embedding:
            # Use a random sample as default text for Word Embedding
            default_text_embedding = data_subset['prep_text'].sample().iloc[0]
            st.text(f"Default Text for Word Embedding: {default_text_embedding}")
        else:
            # User enters custom text for Word Embedding
            default_text_embedding = st.text_area("Enter text for Word Embedding:", data_subset.iloc[0]['prep_text'])

        # Tokenize and preprocess the text for Word2Vec
        tokenized_text = [nlp(text.lower()).text.split() for text in data_subset['prep_text']]

        # Train the Word2Vec model
        word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

        # Extract vectors and words
        vectors = word2vec_model.wv.vectors
        words = word2vec_model.wv.index_to_key

        # Perform t-SNE dimensionality reduction
        vectors_tsne = TSNE(n_components=2, random_state=42).fit_transform(vectors)

        # Create a DataFrame for visualization
        df_tsne = pd.DataFrame(vectors_tsne, columns=['x', 'y'])
        df_tsne['word'] = words

        # Plot the word embeddings using scatter plot with Plotly
        st.subheader("Word Embeddings Visualization")
        fig_word_embeddings = px.scatter(df_tsne, x='x', y='y', text='word', title='Word Embeddings Visualization')
        st.plotly_chart(fig_word_embeddings)

        # Logistic Regression
        accuracy_lr = train_logistic_regression(df)
        st.write(f"### Logistic Regression Accuracy: {accuracy_lr}")

        # Multinomial Naive Bayes
        accuracy_nb = train_multinomial_nb(df)
        st.write(f"### Multinomial Naive Bayes Accuracy: {accuracy_nb}")

        # Support Vector Machine
        accuracy_svm = train_svm(df)
        st.write(f"### SVM Accuracy: {accuracy_svm}")

        # Random Forest
        rf_model, vectorizer,accuracy_rf = train_random_forest(df)
        st.write(f"### DistilBERT Accuracy: {accuracy_rf}")





        # DISTILBERT SENTENCE EMBEDDING
        st.subheader("DistilBERT Sentence Embedding")

        # Load DistilBERT model for sentence embeddings
        distilbert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

        # Option to enter text or use a random sample for DistilBERT
        use_random_sample_distilbert = st.checkbox("Use random sample from dataset as default (DistilBERT)", key="use_random_sample_distilbert", value=True)

        if use_random_sample_distilbert:
            # Use a random sample as default text for DistilBERT
            default_text_distilbert = data_subset['prep_text'].sample().iloc[0]
            st.text(f"Default Text for DistilBERT: {default_text_distilbert}")
        else:
            # User enters custom text for DistilBERT
            default_text_distilbert = st.text_area("Enter text for DistilBERT:", data_subset.iloc[0]['prep_text'])

        # Encode the text using DistilBERT
        encoded_text_distilbert = distilbert_model.encode(default_text_distilbert, convert_to_tensor=True)

        # Display the encoded text
        st.subheader("DistilBERT Encoded Text:")
        st.write(encoded_text_distilbert)


        # Input text for label prediction
        input_text_rf = st.text_area("Enter text for DistilBERT label prediction:")

        if st.button("Predict Label"):
            # Preprocess the input text
            preprocessed_input_rf = preprocess_text(input_text_rf)

            positive_adjectives = ["good", "great", "excellent", "positive", "wonderful","fabulous","intelligent","outstanding","nice","pretty","beautiful","sexy","kind","gentle","calm","passionate"]  # Add more adjectives as needed
            contains_positive_adjective = any(adj in preprocessed_input_rf for adj in positive_adjectives)

            # Vectorize the input text using the same vectorizer used during training
            input_vector_rf = vectorizer.transform([' '.join(preprocessed_input_rf)])

            # Make predictions using the trained Random Forest model
            prediction_rf = rf_model.predict(input_vector_rf)

            # Map the predicted label to "OFFENSIVE" or "NOT OFFENSIVE"
            if contains_positive_adjective:
              predicted_label = "NOT OFFENSIVE"
            else:
        # Map the predicted label to "OFFENSIVE" or "NOT OFFENSIVE"
              predicted_label = "OFFENSIVE" if prediction_rf[0] == 1 else "NOT OFFENSIVE"

            # Display the predicted label
            st.subheader("DistilBERT Predicted Label:")
            st.write(predicted_label)
            st.write(f"Prediction Confidence: {rf_model.predict_proba(input_vector_rf)[:, 1][0]:.2%}")


if __name__ == "__main__":
    main()