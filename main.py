import os, json, time, pickle 
import pandas as pd
import numpy as np
from keras import optimizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score, accuracy_score
from iterstrat.ml_stratifiers import RepeatedMultilabelStratifiedKFold
from build_model import build_model
from utils import ROCCallback, train_model_cv

'''
This script runs a single experiment with the current model from build_model.py file by 
using cross validation. The number of CV folds can be set in N_FOLDS variable.
Results of the experiment are saved in experiments/<experiment_id> directory, where <experiment_id> 
is unique id created during experiment. Some metric reults are also saved to tcc_val_results.txt file in csv format. 
'''

N_FOLDS = 5
RANDOM_STATE = 42
USE_TENSORBOARD = False # please see readme for further info about using tensorboard
MODELS_DATA_PATH = 'models/'
TRAIN_DATA_PATH = 'train_data/train.csv'
TEST_DATA_PATH = 'test_data/test.csv'

LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def load_train_data(file_path, train_column, label_columns):
    print('Loading train data...')
    data_frame = pd.read_csv(file_path)
    return data_frame[train_column].tolist(), data_frame[label_columns].values

def load_test_data(file_path, train_column):
    print('Loading test data...')
    data_frame = pd.read_csv(file_path)
    return data_frame[train_column].tolist(), data_frame['id']

# baseline model -> all zeros
def evaluate_baselines(y_true, average='micro'):
    y_pred = np.zeros_like(y_true)
    roc_auc = roc_auc_score(y_true, y_pred, average=average)
    print('All-zeros', average, 'ROC AUC baseline score:', roc_auc)
    print('Macro accuracy baseline score:', accuracy_score(y_true, y_pred))

# can be used for repeated cross validation, n_repeats=1 by default
def split_dataset_to_train_val_folds(X_data, y_data, n_folds=N_FOLDS, n_repeats=1, random_state=RANDOM_STATE):
    rmskf = RepeatedMultilabelStratifiedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)
    train_val_indices = []
    for train_indices, val_indices in rmskf.split(X_data, y_data):
        train_val_indices.append((train_indices, val_indices))
    return train_val_indices

# creates Tokenizer instance and fits it
def fit_tokenizer(X_texts, hparams):
    print('Fitting tokenizer...')
    tokenizer = Tokenizer(num_words=hparams['max_words'], filters=hparams['tokenizer_filters'], 
                            lower=hparams['tokenizer_lower'], split=hparams['tokenizer_split'], 
                            char_level=hparams['tokenizer_char_level'], oov_token=hparams['tokenizer_oov_token'])
    tokenizer.fit_on_texts(X_texts)
    return tokenizer

def create_padded_sequences(X_texts, tokenizer, hparams):
    print('Converting texts to sequences...')
    X_sequences = tokenizer.texts_to_sequences(X_texts)
    print('Padding sequences...')
    X_padded = pad_sequences(X_sequences, maxlen=hparams['max_length'], padding=hparams['padding'], truncating=hparams['truncating'])
    return X_padded

# main
if __name__ == '__main__':
    X_train_texts, y_train = load_train_data(file_path=TRAIN_DATA_PATH, 
                                                train_column='comment_text', 
                                                label_columns=LABEL_COLUMNS)

    X_test_texts, _ = load_test_data(TEST_DATA_PATH, train_column='comment_text') # test data will be used only for tokenizer fitting

    # main hyperparameters can be set here
    hparams = {
        'max_words': 50000, # for Tokenizer
        'max_length': 180,
        'batch_size': 512,
        'epochs': 100, # leave 100 or more if using early stopping - true by default
        'optimizer': optimizers.RMSprop(),
        'tokenizer_filters': '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        'tokenizer_lower': True,
        'tokenizer_split': " ",
        'tokenizer_char_level': False,
        'padding': 'post',
        'truncating': 'post',
        'tokenizer_oov_token': '<UNK>', # not a real hyperparameter
        'n_classes': y_train.shape[1] # not a real hyperparameter
    }

    print('Evaluating all-zeros baseline...')
    evaluate_baselines(y_train)

    tokenizer = fit_tokenizer(X_train_texts + X_test_texts, hparams)
    VOCAB_SIZE = len(tokenizer.word_index)
    if hparams['max_words'] is None:
        hparams['max_words'] = VOCAB_SIZE + 1
    else:
        hparams['max_words'] += 1
    print('Found', VOCAB_SIZE, 'unique train tokens.')
    print('MAX WORDS:', hparams['max_words'])

    X_train_padded = create_padded_sequences(X_train_texts, tokenizer, hparams)

    print('Splitting train set to', N_FOLDS, 'cross validation folds...')
    train_val_indices = split_dataset_to_train_val_folds(X_train_padded, y_train)

    model = build_model(hparams)
    data = {'X': X_train_padded, 'y': y_train, 'cv_indices': train_val_indices}
    results = train_model_cv(data, hparams, model, tokenizer, MODELS_DATA_PATH, [EarlyStopping(patience=3, verbose=1)], RANDOM_STATE, USE_TENSORBOARD)
    print('\nExperiment', results[0], 'finished.')

