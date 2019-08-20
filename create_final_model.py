import numpy as np
import os, json, pickle
from main import load_train_data, load_test_data, fit_tokenizer
from main import TRAIN_DATA_PATH, TEST_DATA_PATH, MODELS_DATA_PATH, RANDOM_STATE, LABEL_COLUMNS
from utils import load_hparams_and_model, train_model_from_experiment
from evaluate_model import load_test_labels, create_padded_sequences, TEST_DATA_LABELS_PATH

'''
This script can be used for training a model on all available data
(train set + test set). Test set labels should be available for that. Final model 
will be saved to final_models directory, together with other model-related files.
'''

experiment_id = input('Enter experiment id: ')

X_train_texts, y_train = load_train_data(file_path=TRAIN_DATA_PATH, 
                                            train_column='comment_text',
                                            label_columns=LABEL_COLUMNS)
                                            
X_test_texts, _ = load_test_data(TEST_DATA_PATH, train_column='comment_text')
y_test = load_test_labels(TEST_DATA_LABELS_PATH, label_columns=LABEL_COLUMNS)

# test instances labeled with -1 weren't used for final scoring, so they will be masked out
# because we don't have their true labels
mask = y_test[:,0] != -1 
y_test_masked = y_test[mask]
X_test_texts_masked = []
for i in range(len(mask)):
    if mask[i]:
        X_test_texts_masked.append(X_test_texts[i])

# train and test set will be concatenated to single dataset
X_texts = X_train_texts + X_test_texts_masked
y_data = np.concatenate((y_train, y_test_masked), axis=0)

# hyperparameters and model architecture are loaded from experiment id folder
hparams, model = load_hparams_and_model(experiment_id, MODELS_DATA_PATH)

# tokenizer will be fitted on whole dataset
tokenizer = fit_tokenizer(X_texts, hparams)
print('Found', len(tokenizer.word_index), 'unique train tokens.')
print('MAX WORDS:', hparams['max_words'])

X_data = create_padded_sequences(X_texts, tokenizer, hparams)

data = {'X': X_data, 'y': y_data}

model = train_model_from_experiment(data, hparams, model, random_state=RANDOM_STATE)

# saving model, hparams and tokenizer to experiment id folder in final_models folder
final_model_path = MODELS_DATA_PATH + 'final_models/' + experiment_id + '/'
if not os.path.exists(final_model_path):
    os.makedirs(final_model_path)

model_json = model.to_json()
model_json_file_name = final_model_path + 'model.json'
with open(model_json_file_name, 'w') as json_file:
    json_file.write(model_json)
print('Model architecture saved to', model_json_file_name)

model_weights_file_name = final_model_path + 'weights.h5'
model.save_weights(model_weights_file_name)
print('Model weights saved to', model_weights_file_name)

tokenizer_file_name = final_model_path + 'tokenizer.p'
with open(tokenizer_file_name, 'wb') as p_file:
    pickle.dump(tokenizer, p_file)
print('Tokenizer saved to', tokenizer_file_name)

hparams_json = json.dumps(hparams)
hparams_json_file_name = final_model_path + 'hparams.json'
with open(hparams_json_file_name, 'w') as json_file:
    json_file.write(hparams_json)
    print('Model hparams saved to', hparams_json_file_name)

build_model_file_name = MODELS_DATA_PATH + 'experiments/' + experiment_id + '/' + 'build_model.py'
if os.path.isfile(build_model_file_name):
    with open(build_model_file_name, 'r') as f:
        file_lines = f.readlines()
    new_build_model_file_name = final_model_path + 'build_model.py'
    with open(new_build_model_file_name, 'w') as f:
        f.writelines(file_lines)
        print('build_model.py file saved to', new_build_model_file_name)









