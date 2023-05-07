# import libraries
import pandas as pd
import numpy as np
import re
import sys
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    engine = create_engine('sqlite:///DisasterResponse.db')
    df = pd.read_sql_table('DisasterResponsePipeline', engine)

    X = df.message.values
    Y = df.loc[:, 'related':'direct_report'].values
    return X, Y


def tokenize(text):
    '''function for NLP'''
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words('english')]
    stemmed = [PorterStemmer().stem(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    tokens = lemmed
    return tokens


def build_model():
    '''building the pipeline'''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    #parameters have been kept in low dimensions due to performance issues
    parameters = {
        # due to performance issues just two parameters
        'vect__ngram_range': ((1, 1),(1,2)),
        'clf__estimator__n_estimators': [5]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=6, verbose=3)
    return cv

#optional function to evaluate the model
def evaluate_model(cv, y_test, y_pred):
    labels = np.unique(y_pred)
    # input for confusions matrix must be a list of predictions, not one hot encodings --> call argmax!
    confusion_mat = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1), labels=labels)
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Confusion Matrix:\n", confusion_mat)
    print("Accuracy:", accuracy)
    print("\nBest Parameters:", cv.best_params_)


def save_model(model, model_filepath):
    '''Saving model's best_estimator with pickle
        '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        print('Evaluating model...')
        #if needed one can evaluate the model with following line of code (excluded due to performance) ->
        #evaluate_model(model, Y_test, Y_pred)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
