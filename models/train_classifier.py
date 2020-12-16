# import libraries
import sys
import pandas as pd
import numpy as np
import pickle

import re

import nltk
nltk.download('punkt')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


from sqlalchemy import create_engine





def load_data(database_filepath):
    '''This function loads database files
        
    Args:
        database_filepath (str)
    
    Returns:
        X (pandas.DataFrame)
        Y (pandas.DataFrame)
        category_names (Index)
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('messages', engine)    
    
    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis='columns')
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    ''' cleanes tokens from text data by performing the below operations
        - Removes punctuation
        - Lemmatizes tokens
        - Converts to lowercase and removes extra whitespace
    Args: 
        text (str)
    Returns:
        clean_tokens (list): list of tokens
    '''
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    # tokenize
    tokens = word_tokenize(text)
    
    # initalize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for token in tokens:
        clean_tokens.append(lemmatizer.lemmatize(token).lower().strip())
        
    return clean_tokens



def build_model():
    
    '''this builds a parameter optimized model
    
    Returns:
        cv (GridSearchCV object as model)
    '''
  
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
      ])
    
    parameters = {
        'vect__stop_words':['english'],
        'vect__ngram_range':[(1,2)],
        'vect__max_features':[40000],
        'clf__estimator__learning_rate':[0.75, 1.0],
        'clf__estimator__n_estimators':[50, 75]
    }
    
    cv = GridSearchCV(pipeline, parameters, cv=3)    
    return cv



def score_model(scorer, y_true, y_pred, average):
    '''Runs the scorer on all the paired columns
    
    Args:
        scorer: The scoring function to use
        y_true (pandas.DataFrame)
        y_pred (pandas.DataFrame)
        average (str): the average parameter passed to scorer
    
    Returns:
        scores :  scores  depending on
            which average option was used
    '''
    score = scorer(y_true.values.flatten(), y_pred.values.flatten(), average=average)
    return score






def evaluate_model(model, X_test, Y_test, category_names):
    '''Evaluates a multi-label classifier and prints the results
    
    Args:
        model: classifier model
        X_test: test data
        Y_test (pandas.DataFrame): test labels
        category_names: names for test label columns
    '''
    Y_pred = pd.DataFrame(model.predict(X_test))
    
    for i in range(Y_test.shape[1]):
        print('Column {}, {}'.format(i, category_names[i]))
        print(classification_report(Y_test.iloc[:,i], Y_pred.iloc[:,i]))
    
    average = 'micro' # reports only pos_label
    scorer = f1_score
    scores = score_model(scorer, Y_test, Y_pred, average)
    
    print(scorer.__name__, average, np.mean(scores))

    return None


def save_model(model, model_filepath):
    '''This saves the model as a pickle file'''
    try:
        f = open(model_filepath, 'wb')
        pickle.dump(model, f)
        return True
    except IOError:
        return False

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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