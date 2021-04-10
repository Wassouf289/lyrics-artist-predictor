
#lyricsgenius provides a simple interface to the song, artist, and lyrics data stored on Genius.com.
#Using this library you can convienently access the content on Genius.com And much more using the public API.
#pip install lyricsgenius

import lyricsgenius as genius
import os
import nltk
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import spacy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import re
import config

nlp= spacy.load('en_core_web_md')  # load medium spacy model


#Before we start, we will need Genius API credentials, got it through registeration here
#https://docs.genius.com/#/getting-started-h1
#you need your -client-access-token
#In terms of filtering, we’ll ignore lyrics that aren’t from official songs.It’s also a good idea to set 
#remove_section_headers to True assuming we want our dataset to focus solely on spoken song lyrics and to 
#exclude song metadata.


#define list of artists
artists = ['Frank Sinatra','Ed Sheeran','Taylor Swift']
max_songs=150



#search for songs for each artist in the list and save the songs to the artist folder
def collect_songs(artists,max_Songs):
    
    api=genius.Genius(config.token,
                      excluded_terms = ["(Remix)", "(Live)"] ,
                      skip_non_songs=True,
                      remove_section_headers=True)

    for artist in artists:
        songs = (api.search_artist(artist, max_songs=max_Songs, sort='popularity')).songs
        c=0
        for song in songs:
            fileName = os.path.join('lyrics/'+artist,"songnumber"+str(c)+".txt")
            file = open(fileName, "w")
            file.write(song.lyrics)
            file.close()
            c+=1


# ## Data cleaning
#   1- Natural-Language tool-kit(NLTK)
#    
#     Load the raw text.
#     Split into tokens.
#     Stem Words: Stemming refers to the process of reducing each word to its root or base
#     Convert to lowercase.
#     Remove punctuation from each token.
#     Filter out remaining tokens that are not alphabetic.
#     Filter out tokens that are stop words.
# 


#nltk.download('punkt')
#nltk.download('stopwords')

def clean_songs_text_NLTK(artists,max_songs):
    for artist in artists:
        for i in range(max_songs):
            fileName = os.path.join('lyrics/'+artist,"songnumber"+str(i)+".txt")
            file = open(fileName, 'r')
            text = file.read()
            file.close()
            # split into words
            tokens = word_tokenize(text)
            # stemming of words
            porter = PorterStemmer()
            stemmed = [porter.stem(word) for word in tokens]
            # convert to lower case
            l_tokens = [w.lower() for w in stemmed]
            # remove punctuation from each word
            table = str.maketrans('', '', string.punctuation)
            stripped = [w.translate(table) for w in l_tokens]
            # remove remaining tokens that are not alphabetic
            words = [word for word in stripped if word.isalpha()]
            # filter out stop words
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if not w in stop_words]
            file_Name = os.path.join('lyrics_cleaned/'+artist,"song_cleaned_number"+str(i)+".txt")
            f = open(file_Name, 'w')
            for word in words:
                f.write("%s " % word)
            f.close()



# 2- clean text data with Spacy

def spacy_cleaner(artists,max_songs):
    for artist in artists:
        for i in range(max_songs):
            fileName = os.path.join('lyrics/'+artist,"songnumber"+str(i)+".txt")
            file = open(fileName, 'r')
            text = file.read()
            file.close()
            # Apply spacy to the text
            doc=nlp(text)
            # Lemmatization,remove noise (stopwords, digit, puntuaction and single characters)
            tokens=[token.lemma_.strip() for token in doc if 
                not token.is_stop and not nlp.vocab[token.lemma_].is_stop # remove StopWords
                and not token.is_punct # Remove puntuaction
                and not token.is_digit # Remove digit
                and not token.is_space
                and not token.is_quote
                and not token.is_bracket
                and not token.like_num
                and not token.is_currency
               ]
            # Remove empty tokens and one letter tokens
            tokens = [token for token in tokens if token != "" and len(token)>1]
            # Recreation of the text
            new_text=" ".join(tokens)
            # Remove non alphabetic characters
            new_text = re.sub(r"[^a-zA-Z]", " ", new_text)
            # remove non-Unicode characters
            new_text = re.sub(r"[^\x00-\x7F]+", "", new_text)

            new_text=new_text.lower()

            file_Name = os.path.join('lyrics_cleaned/'+artist,"song_cleaned_number"+str(i)+".txt")
            f =open(file_Name, 'w')
            f.write(new_text)
            f.close()

# ### Construct a Text Corpus

def create_corpus(artists_list):
    CORPUS=[]
    for artist in artists_list:
        for fn in os.listdir('lyrics_cleaned/'+artist):
            text = open( 'lyrics_cleaned/'+ artist+'/'+ fn).read()
            CORPUS.append(text)
    return CORPUS


def create_labels(artists_list,max_songs):
    LABELS=[]
    for artist in artists_list:
        for _ in range(max_songs):
            LABELS.append(artist)
    return LABELS



"""train model with Random Forest classifier """
"""we want to use 1500 most occurring words as features for training our classifier.
    So we only include those words that occur in at least 5 documents
    Here 0.7 means that we should include only those words that occur in a maximum of (70%) of all the documents"""
def train_model_RF(X_train,y_train):
    # Vectorize the text input and use Random Forest classifier
    pipeline = make_pipeline(
    TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7,ngram_range=(1,2),stop_words='english'),
    RandomForestClassifier(n_estimators=1000,max_depth=7)
    
    )
    pipeline.fit(X_train,y_train)
    return pipeline

"""train model with Random Forest classifier + GridSearch"""
def train_model_RF_GridSearch(X_train,y_train):
    pipeline = Pipeline([
        ('Tfidf',TfidfVectorizer()),
        ('RF',RandomForestClassifier())
    ])

    params = {
    'Tfidf__max_features':[1000,2000],
    'Tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'Tfidf__min_df':[5,7,10],
    'Tfidf__max_df':[0.5,0.6,0.7],
    'RF__n_estimators':[1000,2000],
    'RF__max_depth':[3,5,7],
    }

    tfidf_gs = GridSearchCV(pipeline, param_grid=params, cv = 5, verbose = 1,scoring='accuracy',n_jobs=-1)
    tfidf_gs.fit(X_train,y_train)
    print('Best parameters:',tfidf_gs.best_params_)
    best_model=tfidf_gs.best_estimator_
    return best_model

""" train model with Naive bias using """
def train_model_NB(X_train,y_train):
    
    pipeline = make_pipeline(
    
    TfidfVectorizer(max_features=2000, min_df=5, max_df=0.5,ngram_range=(1,2),stop_words='english'),
    MultinomialNB(alpha=0.1)
    )
    pipeline.fit(X_train,y_train)
    return pipeline


""" train model with Naive bias using GridSearch"""
def train_model_NB_GridSearch(X_train,y_train):

    pipeline = Pipeline([
    
    ('Tfidf',TfidfVectorizer(stop_words='english')),
    ('NB',MultinomialNB())
    ])
    params = {
    'Tfidf__max_features':[1000,2000,4000],
    'Tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    'Tfidf__min_df':[5,7,10],
    'Tfidf__max_df':[0.5,0.6,0.7],
    'NB__alpha':[1,0.5,0.1, 0.01, 0.001, 0.0001],
    }

    tfidf_gs = GridSearchCV(pipeline, param_grid=params, cv = 5, verbose = 1,scoring='accuracy',n_jobs=-1)
    print('start training\n')
    tfidf_gs.fit(X_train,y_train)
    print('Best parameters:',tfidf_gs.best_params_)
    best_model=tfidf_gs.best_estimator_
    return best_model


def predict(model,text):
    """
    Takes the pre-trained pipeline model and predicts the artist.
    """
    prediction = model.predict(text)
    probs = model.predict_proba(text)
    return prediction[0], probs.max()



#collect_songs(artists,max_songs)
#clean_songs_text_NLTK(artists,max_songs)
#spacy_cleaner(artists,max_songs)
CORPUS=create_corpus(artists)
LABELS=create_labels(artists,max_songs)
# Split Data
X_train,X_test,y_train,y_test =train_test_split(CORPUS,LABELS,test_size=0.2,random_state=42)
#pipeline=train_model_RF(X_train,y_train)
#pipeline=train_model_RF_GridSearch(X_train,y_train)
pipeline=train_model_NB(X_train,y_train)
#pipeline=train_model_NB_GridSearch(X_train,y_train)


if __name__ == '__main__':
    #evaluate on test data
    y_pred=pipeline.predict(X_test)
    print(" \n Model accuracy:",round(accuracy_score(y_test, y_pred),2))
    # Evaluate the model
    print('\n\n confusion matrix')
    print(confusion_matrix(y_test,y_pred))
    print('\n\n classification report')
    print(classification_report(y_test,y_pred))
