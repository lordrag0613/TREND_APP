import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import jaccard_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
from sklearn.metrics import multilabel_confusion_matrix , confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import ConfusionMatrixDisplay
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.multiclass import is_multilabel
from sklearn.dummy import DummyClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
import re
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import pickle

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Utilisez BeautifulSoup pour extraire le texte
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
   
    # Utilisez re.sub pour remplacer les caractères non alphabétiques par un espace
    clean_text = re.sub("[^a-zA-Z]", " ", text)
   
    # Tokenization
    words = word_tokenize(clean_text)
   
    # Lemmatization
    lemmatized_words = [lemmatizer.lemmatize(word.lower(), pos='v') for word in words]  # Lemmatize verbs
    lemmatized_words = [lemmatizer.lemmatize(word.lower(), pos='n') for word in lemmatized_words]  # Lemmatize nouns
   
    # Joining words back into sentence
    lemmatized_text = ' '.join(lemmatized_words)
   
    return lemmatized_text






labels = pd.read_csv('data/my_labels.csv')
labels = labels.values.tolist()
labels = [element[0] for element in labels]


with open('models/logistic_regression.sav', 'rb') as file:
   model = pickle.load(file)

with open('models/multi_tfidf_vectorizer.sav', 'rb') as file:
    vectorizer = pickle.load(file)

def predict_tags(title, body, vectorizer, model , label):
    
    # Prétraiter le titre et le corps
    title_clean = preprocess_text(title)
    body_clean = preprocess_text(body)
    
    # Combiner le titre et le corps
    combined_text = title_clean + " " + body_clean
    
    # Transformer le texte combiné en vecteur TF-IDF
    text_vector = vectorizer.transform([combined_text])
    
    # Faire la prédiction avec le modèle
    predicted_tags_vector = model.predict(text_vector)
    
    predicted_tags = [label[i] for i in range(len(predicted_tags_vector[0])) if predicted_tags_vector[0][i] == 1]
    return predicted_tags


# Exemple d'utilisation
# Assumons que vectorizer, rscv_random_forest, et mlb sont déjà définis


# Interface utilisateur Streamlit
st.title("Predict Tags from Title and Body")

# Entrées utilisateur pour le titre et le corps du texte
title = st.text_input("Enter Title:", "Using jQuery to disable a button in Materialize CSS")
body = st.text_area("Enter Body:", "According to documentation, the way to disable a button is simply by adding a class named 'disabled'. \
However, I think that by simply adding the class 'disabled',  \
the button will automatically be disabled, but it is not working. \
Here is my code:\n\nHTML:\n<button id='submit-btn' class='btn waves-effect waves-light submit red' \
type='button' name='action'>\njQuery:\n$('#submit-btn').off('click').on('click', function() \
{\n  $('#submit-btn').addClass('disabled');\n});\n\nHow can I disable a button using jQuery in Materialize CSS? ")

# Bouton "Go" pour déclencher la prédiction
if st.button("Go"):
    if title != "" and body != "":
        # Prédire les étiquettes
        predicted_tags_vector = predict_tags(title, body, vectorizer, model , labels)
        
        # Afficher les étiquettes prédites
        st.write("Predicted Tags:")
        st.write(predicted_tags_vector)
    else:
        st.warning("Please enter both title and body text.")


