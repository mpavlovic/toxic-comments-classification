# toxic-comments-classification
Sample deep learning model and code for Toxic Comments Classification Challenge on Kaggle. The code can be used as mini framework for similar competitions. It allows and automates experimentation with different models, saving experiments, model evaluation, Tensorboard visualization and final model training. 

## Built with:
* Python 3.6.6
* Tensorflow 1.10.0 (GPU version)
* Keras 2.2.0
* Scikit-learn 0.19.1
* Iterative-stratification 0.1.6
* Pandas 0.24.2
* Numpy 1.15.1


## Workflow

### Making Experiments on Training Set
To start an experiment, edit your model in `build_model.py` file and run `main.py`. This file loads train and test data, evaluates all-zeros baseline model, fits `Tokenizer` instance on the whole dataset and runs a single cross validation experiment on training set, with the model from `build_model.py` file. The number of CV folds can be set in `N_FOLDS` variable (5 by default). You can modify different hyperparameters in `hparams` dictionary:

    hparams = {
        'max_words': 50000, # for Tokenizer
        'max_length': 180,
        'batch_size': 512,
        'epochs': 100,
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

Since this is a multilabel classification problem, the [`iterative-stratification`](https://github.com/trent-b/iterative-stratification) package is used for creating cross validation folds. For measuring model's performance, three main metrics are used: cross entropy loss, accuracy and ROC AUC (Receiver Operating Characteristic Area Under The Curve) score - official competition's performance metric. The training procedure minimizes validation loss and uses early stopping with `patience=3`. 

#### Saving Experiment Results
`main.py` file uses a function `train_model_cv` which trains a model through folds and saves results. This function assigns the unique ID to each experiment. After an experiment is finished, the results are saved to `experiments/<experiment_id>` directory. This directory contains following files:
* `build_model.py` - a copy of original `build_model.py` file so you can modify it for different experiments
* `hparams.json` - JSON dumped `hparams` dictionary from `main.py` file
* `model_summary.txt` - the output from `model.summary()` method of `Keras` model
* `model.json` - the output from `model.to_json()` method of `Keras` model - contains model architeccture
* `results.txt` - textual description with averaged metrics and standard deviations across folds:
    * _average minimal validation loss_
    * _average train loss of minimal validation loss_ - might be useful for estimating overfitting impact
    * _average validation ROC AUC score_ - averaged from epochs of minimal validation losses
    * _average epoch of minimal validation loss_ - averaged number of epochs in which minimal validation loss is measured (through all folds)

__IMPORTANT NOTICE:__ `hparams['epochs']` will be updated with the average number of epochs before dumping `hparams` to `hparams.json` file. It's OK to set the `hparams['epochs']` to some big number before experiment because the early stopping is used.

### Evaluating a Model on Test Set

### Training a Model on the whole Dataset
