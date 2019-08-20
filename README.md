# Toxic Comments Classification
Single deep learning model and code for Toxic Comments Classification Challenge on Kaggle. The code is written as a simple framework for similar competitions. It allows and automates experimentation with different models, experiments saving, model evaluation, TensorBoard visualization and final model training.

## Built with:
* Python 3.6.6
* Keras 2.2.0 
* Tensorflow 1.10.0 (GPU version)
* Scikit-learn 0.19.1
* Iterative-stratification 0.1.6
* Pandas 0.24.2
* Numpy 1.15.1

## Provided Model
The convolutional-only model provided here achieves __0.97704__ ROC-AUC score on the private leaderbord after a late submission (`models/submissions/1566218974.9314115/`) and __0.97736__ on the whole test set. Better performance can be further achieved with advanced LSTM or pre-trained language models. The experiment with default setup takes around 27 minutes on GTX 1060 6 GB GPU. Final training takes 3-5 minutes. Below are instructions on how to experiment with different Keras model architectures.   

## Workflow

### Making Experiments on the Training Set
To start an experiment, edit your model in `build_model.py` file and run `main.py` with `python main.py`. This file loads train and test data, evaluates all-zeros baseline model, fits `Tokenizer` instance on the whole dataset and runs a single cross validation experiment on training set, with the model from `build_model.py` file. The number of CV folds can be set in `N_FOLDS` variable (5 by default). You can modify different hyperparameters in `hparams` dictionary:

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
`main.py` file uses a function `train_model_cv` which trains a model through folds and saves results. This function assigns the unique ID to each experiment. After an experiment is finished, the results are saved to `models/experiments/<experiment_id>/` directory. This directory contains following files:
* `build_model.py` - a copy of original `build_model.py` file so you can modify it for different experiments
* `hparams.json` - JSON dumped `hparams` dictionary from `main.py` file
* `model_summary.txt` - the output from `model.summary()` method of `Keras` model
* `model.json` - the output from `model.to_json()` method of `Keras` model - contains model architeccture
* `results.txt` - textual description with averaged metrics and standard deviations across folds:
    * _average minimal validation loss_
    * _average train loss of minimal validation loss_ - might be useful for estimating overfitting impact
    * _average validation ROC AUC score_ - averaged from epochs of minimal validation losses
    * _average epoch of minimal validation loss_ - averaged number of epochs in which minimal validation loss is measured (through all folds)

__IMPORTANT NOTICE:__ `hparams['epochs']` will be updated with the rounded average number of epochs (from `results.txt`) before dumping `hparams` to `hparams.json` file. It's OK to set the `hparams['epochs']` to some big number before experiment because the early stopping is used.

#### Using TensorBoard
If you want to visualize different apects of learning with TensorBoard, please set the `USE_TENSORBOARD` variable in `main.py` to `True`. `tb_logs` folder will be used for storing TensorBoard log files. Each experiemnt's files will be stored to `models/tb_logs/<experiment_id>/` directory. Before running the `main.py`, please run `tensorboard --logdir=path/to/log-directory` command where `path/to/log-directory` is the path to `tb_logs` directory or `tb_logs` itself. This command will work if it is called from activated Python environment where TensorBoard is installed. The last step is to open `localhost:6006` address with your browser. More details are available [here](https://www.tensorflow.org/guide/summaries_and_tensorboard#launching_tensorboard).

### Model Selection
Exept it stores results after each experiment, the `main.py` script also appends a results line to the `tcc_val_results.txt` file. This file is CSV formatted, so each line has the following form: <br><br>`<experiment_id>;<average_val_auc>;<average_min_val_loss>`.<br><br>Therefore, with experiment ID, each line contains average validation ROC AUC score and minimal validation loss. You can read this file with `pandas` or import it to Excel and sort the table by desired column to select the best performing model on validation folds.

### Evaluation of the Selected Model on the Test Set
Once you selected your model by using cross validation, run the `evaluate_model.py` file to create the submission file. You will have to enter the `<experiment_id>` and the script will automatically load saved hyperparameters and model architecture from the corresponding experiment directory and train the model on the whole training set. The submission file will be named `submission_<experiment_id>.csv` and saved to `models/submissions/<experiment_id>/` directory. Together with submission file, model architecture (`model.json`) and weights (`weights.h5`) will be saved too. You can set the variable `SAVE_TRAINED_MODEL` in `evaluate_model.py` to `False` to disable model saving.
<br><br> If the test set labels are available (they might be after the competition, as they are in this case), the `evaluate_model.py` script will measure the model performance on the test set, print it out and save it to the `models/evaluations/` directory, under the name `<experiment_id>_evaluation.txt`. If you don't have the test set labels, please set the `TEST_DATA_LABELS_PATH` variable to `None`.

### Training the Selected Model on the whole Dataset
The `create_final_model.py` script trains the selected model on the whole dataset (train + test). The test set labels should be available for that. If the test set contains some unused examples labeled with -1, they will be masked out. After running the script, you need to enter `<experiment_id>` again and the corresponding model will start to train. When finished, the model architecture (`model.json`), weights (`weights.h5`), `Tokenizer` instance (`tokenizer.p`), hyperparameters (`hparams.json`) and `build_model.py` will be saved to `models/final_models/<experiment_id>/` directory. 
