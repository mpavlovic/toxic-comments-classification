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
For experimenting with different models, start with `main.py` file. It loads train and test data, evaluates all-zeros baseline model, fits `Tokenizer` instance on the whole dataset and runs a single cross validation experiment on training set, with the model from `build_model.py` file. The number of CV folds can be set in `N_FOLDS` variable. You can modify different hyperparameters in `hparams` dictionary:

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

### Evaluating a Model on Test Set
### Training a Model on the whole Dataset
