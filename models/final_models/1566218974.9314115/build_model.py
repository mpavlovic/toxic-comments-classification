from keras.models import Model
from keras.layers import Embedding, Dense, Input, Conv1D, GlobalMaxPool1D, Dropout, GlobalAvgPool1D, concatenate

'''
This script contains a function for building and compiling a model.
'''

def build_model(hparams):
    print('Building model...')
    
    input_ = Input(shape=(hparams['max_length'],))

    x = Embedding(input_dim=hparams['max_words'], output_dim=96)(input_)
    x = Dropout(0.1)(x)

    x_1 = Conv1D(filters=256, kernel_size=7, strides=1, activation='elu')(x)
    max_1 = GlobalMaxPool1D()(x_1)

    x_2 = Conv1D(filters=256, kernel_size=4, strides=1, activation='elu')(x)
    max_2 = GlobalMaxPool1D()(x_2)

    x_3 = Conv1D(filters=256, kernel_size=1, strides=1, activation='elu')(x)
    max_3 = GlobalMaxPool1D()(x_3)

    x_4 = Conv1D(filters=256, kernel_size=3, strides=1, activation='elu')(x)
    max_4 = GlobalMaxPool1D()(x_4)

    x_5 = Conv1D(filters=256, kernel_size=2, strides=1, activation='elu')(x)
    max_5 = GlobalMaxPool1D()(x_5)

    c = concatenate([max_1, max_2, max_3, max_4, max_5])

    output = Dense(hparams['n_classes'], activation='sigmoid')(c)

    model = Model(inputs=[input_], outputs=[output])

    model.compile(optimizer=hparams['optimizer'], loss='binary_crossentropy', metrics=['acc'])
    print('Parameters:', model.count_params())
    # print(model.summary())
    return model