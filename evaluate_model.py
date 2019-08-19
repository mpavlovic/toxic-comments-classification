import os
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import roc_auc_score
from main import load_train_data, load_test_data, fit_tokenizer, create_padded_sequences 
from main import TRAIN_DATA_PATH, TEST_DATA_PATH, MODELS_DATA_PATH, RANDOM_STATE, LABEL_COLUMNS
from utils import load_hparams_and_model, train_model_from_experiment

'''
This script should be run after an experiment is done in main.py file. After entering the experiment id,
all hyperparameters and model architecture will be loaded from experiment directory and model will be 
retrained on the wholde training set. The submission file will be created and saved to submissions directory. 
If available, test set labels will be used for model evaluation, i.e. for estimating model performance on the test set.
Evaluation results will be saved to evaluations directory.
'''

SAVE_TRAINED_MODEL = True # set to True if you want to save a model from which a submission file was created

TEST_DATA_LABELS_PATH = 'test_data/test_labels.csv' # set this variable to None if test labels aren't available


def load_test_labels(file_path, label_columns):
    data_frame = pd.read_csv(file_path)
    return data_frame[label_columns].values

def create_submission_file(ids, label_columns, y_pred, experiment_id, submissions_path):
    submission = pd.DataFrame()
    submission['id'] = ids
    for i in range(len(label_columns)):
        label = label_columns[i]
        submission[label] = y_pred[:,i]
    submission_file_name = submissions_path + 'submission_' + experiment_id + '.csv'
    submission.to_csv(submission_file_name, sep=',', index=False)
    print('\nSubmission file saved to', submission_file_name)


if __name__ == '__main__':
    experiment_id = input('Enter experiment id: ')

    X_train_texts, y_train = load_train_data(file_path=TRAIN_DATA_PATH, 
                                                train_column='comment_text',
                                                label_columns=LABEL_COLUMNS)

    X_test_texts, ids = load_test_data(TEST_DATA_PATH, train_column='comment_text')

    hparams, model = load_hparams_and_model(experiment_id, MODELS_DATA_PATH)

    tokenizer = fit_tokenizer(X_train_texts + X_test_texts, hparams)

    print('Preparing train data:')
    X_train = create_padded_sequences(X_train_texts, tokenizer, hparams)

    data = {'X': X_train, 'y': y_train}
    model = train_model_from_experiment(data, hparams, model, random_state=RANDOM_STATE)

    print('Preparing test data:')
    X_test = create_padded_sequences(X_test_texts, tokenizer, hparams)
    y_test_true = load_test_labels(TEST_DATA_LABELS_PATH, label_columns=LABEL_COLUMNS)

    print('Predicting on test set:')
    y_test_pred = model.predict(X_test, batch_size=2048)

    if TEST_DATA_LABELS_PATH is not None:
        mask = y_test_true[:,0] != -1 # test instances labeled with -1 weren't used for final scoring
        y_test_pred_masked = y_test_pred[mask]
        y_test_true_masked = y_test_true[mask]
        X_test_masked = X_test[mask]

        evaluation_results = []
        roc_auc_test = roc_auc_score(y_test_true_masked, y_test_pred_masked, average='macro') # based on 100% of test data; may be different with private and public LB score
        evaluation_results.append('Test set ROC AUC score: {:.5f}\n'.format(roc_auc_test))

        results = model.evaluate(X_test_masked, y_test_true_masked, batch_size=2048)
        evaluation_results.append('Test set loss: {:.5f}\n'.format(results[0]))
        evaluation_results.append('Test set accuracy: {:.5f}\n'.format(results[1]))
        
        print()
        for r in evaluation_results:
            print(r)

        evaluations_path = MODELS_DATA_PATH + 'evaluations/'
        if not os.path.exists(evaluations_path):
            os.makedirs(evaluations_path)
        evaluations_file_name = evaluations_path + experiment_id + '_evaluation.txt'
        with open(evaluations_file_name, 'w') as f:
            f.writelines(evaluation_results)
            print('Evaluation results saved to', evaluations_file_name)

    # submission file generation and model saving
    print()
    submissions_path = MODELS_DATA_PATH + 'submissions/' + experiment_id + '/'
    if not os.path.exists(submissions_path):
        os.makedirs(submissions_path)

    if SAVE_TRAINED_MODEL:
        model_json = model.to_json()
        model_json_file_name = submissions_path + 'model.json'
        with open(model_json_file_name, 'w') as json_file:
            json_file.write(model_json)
            print('Model architecture saved to', model_json_file_name)

        model_weights_file_name = submissions_path + 'weights.h5'
        model.save_weights(model_weights_file_name)
        print('Model weights saved to', model_weights_file_name)

    create_submission_file(ids, LABEL_COLUMNS, y_test_pred, experiment_id, submissions_path)



