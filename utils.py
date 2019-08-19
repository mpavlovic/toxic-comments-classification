import keras
import numpy as np
import os, csv, time, json, random, pickle
from sklearn.metrics import roc_auc_score
from tensorflow import set_random_seed
from keras.callbacks import TensorBoard
from keras.utils.generic_utils import serialize_keras_object
from keras.models import model_from_json
from keras import optimizers

'''
This file contains utility functions for training, saving and loading models, 
as well as custom metrics / callbacks and other stuff. 
'''

class ROCCallback(keras.callbacks.Callback):
    def __init__(self, training_data, validation_data):
        x = training_data[0]
        y = training_data[1]
        x_val = validation_data[0]
        y_val = validation_data[1]

        self.x_all = np.concatenate((x, x_val))
        self.y_all = np.concatenate((y, y_val))
        self.val_set_start_index = x.shape[0]
        assert self.x_all.shape[0] == self.val_set_start_index + len(x_val) 

        self.val_roc_aucs = []

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_all = self.model.predict(self.x_all, batch_size=2048) # prediction probablities
        
        y_pred_train = y_pred_all[0:self.val_set_start_index]
        y_true_train = self.y_all[0:self.val_set_start_index]

        y_pred_val = y_pred_all[self.val_set_start_index:]
        y_true_val = self.y_all[self.val_set_start_index:]

        roc_train = roc_auc_score(y_true_train, y_pred_train, average='macro')
        roc_val = roc_auc_score(y_true_val, y_pred_val, average='macro')

        self.val_roc_aucs.append(roc_val)
        print('roc-auc: {:.4f} - roc-auc_val: {:.4f}'.format(roc_train, roc_val))
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def train_model_cv(data, hparams, model, tokenizer, models_data_path, callbacks_=[], random_state=42, use_tensorboard=False):
    np.random.seed(random_state)
    set_random_seed(random_state)
    random.seed(random_state)

    X = data['X']
    y = data['y']
    
    cv_indices = data['cv_indices']
    
    fold = 1
    init_weights = model.get_weights() # we will use them for initializing weights in each fold

    min_val_losses = []
    train_losses_of_min_val_losses = []
    epochs_of_min_val_losses = []
    val_aucs_of_min_val_losses = []

    experiment_id = str(time.time())
    experiment_path = models_data_path + 'experiments/' + experiment_id + '/'
    tensorboard_path =  models_data_path + 'tb_logs/' + experiment_id + '/'

    print()
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)
        print(experiment_path, 'created')

    if use_tensorboard and not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
        print(tensorboard_path, 'created')

    for train_indices, val_indices in cv_indices:
        print('\n\nFold', fold)
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        
        X_val = X[val_indices]
        y_val = y[val_indices]
        
        model.set_weights(init_weights) # model weights are reinitialized in each fold

        if use_tensorboard:
            tensorboard_callback = TensorBoard(log_dir=tensorboard_path + "Fold_" + str(fold) + "/")
            if fold > 1:
                callbacks_[-1] = tensorboard_callback
            else:
                callbacks_.append(tensorboard_callback)
        
        roc_auc_callback = ROCCallback((X_train, y_train), (X_val, y_val))

        history = model.fit(X_train , y_train, 
                            epochs=hparams['epochs'], batch_size=hparams['batch_size'], 
                            validation_data=(X_val, y_val), 
                            callbacks=callbacks_+[roc_auc_callback], shuffle=True)

        history = history.history
        
        val_losses = np.array(history['val_loss'])
        val_losses_argmin = np.argmin(val_losses)
        min_val_losses.append(val_losses[val_losses_argmin])
        epochs_of_min_val_losses.append(val_losses_argmin+1)
        train_losses_of_min_val_losses.append(history['loss'][val_losses_argmin])
        val_aucs_of_min_val_losses.append(roc_auc_callback.val_roc_aucs[val_losses_argmin])

        fold += 1

    min_val_losses = np.array(min_val_losses)
    epochs_of_min_val_losses = np.array(epochs_of_min_val_losses)
    train_losses_of_min_val_losses = np.array(train_losses_of_min_val_losses)
    val_aucs_of_min_val_losses = np.array(val_aucs_of_min_val_losses)

    avg_min_val_loss = min_val_losses.mean()
    stddev_min_val_loss = min_val_losses.std()

    avg_epoch = epochs_of_min_val_losses.mean()
    stddev_epoch = epochs_of_min_val_losses.std()
    
    # average is rounded to int and replaced in hparams 
    avg_epoch_rounded = int(round(avg_epoch,0))
    hparams['epochs'] = avg_epoch_rounded

    avg_train_loss = train_losses_of_min_val_losses.mean()
    stddev_train_loss = train_losses_of_min_val_losses.std()
    
    avg_val_auc = val_aucs_of_min_val_losses.mean()
    stddev_val_auc = val_aucs_of_min_val_losses.std()

    text = []
    text.append('Average min val loss is {:.5f} with std of {:.5f}\n'.format(avg_min_val_loss, stddev_min_val_loss))
    text.append('Average train loss of min val loss is {:.5f} with std of {:.5f}\n'.format(avg_train_loss, stddev_train_loss))
    text.append('Average val ROC AUC score is {:.5f} with std of {:.5f}\n'.format(avg_val_auc, stddev_val_auc))
    text.append('Average epoch of min val loss is {:.2f} with std of {:.2f} - rounded to {}\n'.format(avg_epoch, stddev_epoch, avg_epoch_rounded))
    
    print()
    for t in text:
        print(t)

    # saving results to txt file
    experiment_file_name = experiment_path + 'results.txt'
    with open(experiment_file_name, 'w') as f:
        f.writelines(text)
    print('Results saved to', experiment_file_name)

    # saving model architecture to json file
    model_json = model.to_json()
    model_json_file_name = experiment_path + 'model.json'
    with open(model_json_file_name, 'w') as json_file:
        json_file.write(model_json)
    print('Model architecture saved to', model_json_file_name)

    # saving hyperparameters
    if 'optimizer' in hparams.keys():
        optimizer = hparams['optimizer']
        if not isinstance(optimizer, str):
            hparams['optimizer'] = serialize_keras_object(optimizer)
        hparams_json = json.dumps(hparams)
        hparams_json_file_name = experiment_path + 'hparams.json'
        with open(hparams_json_file_name, 'w') as json_file:
            json_file.write(hparams_json)
        print('Model hparams saved to', hparams_json_file_name)

    # saving model summary
    summary_file_name = experiment_path + 'model_summary.txt'
    def save_summary(summary_line):
        summary_line += '\n'
        with open(summary_file_name, 'a') as f:
            f.write(summary_line)
    model.summary(print_fn=save_summary)
    print('Model summary saved to', summary_file_name)

    # saving/copying build_model.py file
    build_model_file_name = 'build_model.py'
    if os.path.isfile(build_model_file_name):
        with open(build_model_file_name, 'r') as f:
            file_lines = f.readlines()
        build_model_file_path = experiment_path + build_model_file_name
        with open(build_model_file_path, 'w') as f:
            f.writelines(file_lines)
            print(build_model_file_name, 'file saved to', build_model_file_path)

    # saving experiment id, and average validation metrics to csv file
    results_csv_file_name = 'tcc_val_results.txt'
    with open(results_csv_file_name, 'a', newline='') as f:
        field_names = ['experiment_id','avg_val_auc', 'avg_min_val_loss']
        writer = csv.DictWriter(f, fieldnames=field_names, delimiter=';')
        writer.writerow({'experiment_id': str(experiment_id), 'avg_val_auc': str(avg_val_auc), 'avg_min_val_loss': str(avg_min_val_loss)})
        print('Validation metrics appended to', results_csv_file_name)

    return([experiment_id, avg_val_auc, avg_min_val_loss])

def load_hparams_and_model(experiment_id, models_data_path):
    print('Loading model hyperparameters from experiment', experiment_id, '...')
    experiment_path = models_data_path + 'experiments/' + experiment_id + '/'
    
    hparams_file_path = experiment_path + 'hparams.json'
    with open(hparams_file_path, 'r') as f:
        data = f.read()
        hparams = json.loads(data)
        
    model_file_path = experiment_path + 'model.json'
    with open(model_file_path, 'r') as f:
        model_json = f.read()
    model = model_from_json(model_json)
        
    return hparams, model

def train_model_from_experiment(data, hparams, model, callbacks_=[], random_state=42):
    np.random.seed(random_state)
    set_random_seed(random_state)
    random.seed(random_state)
        
    X = data['X']
    y = data['y']
    
    print('\nContinuing training with following hparams:')
    print(hparams)
    
    # optimizer setup
    optimizer = hparams['optimizer']
    optimizer = optimizers.get(optimizer)
    
    print('Compiling model...')
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    print('Fitting model...')
    model.fit(X, y, epochs=hparams['epochs'], batch_size=hparams['batch_size'], callbacks=callbacks_, shuffle=True)
    
    return model


