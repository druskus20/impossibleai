from tensorflow.keras import Sequential, optimizers
from tensorflow.keras import initializers
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, TimeDistributed, Flatten, GRU, Dense
from tensorflow.python.keras.engine.input_layer import InputLayer


def get_model(img_res, num_img_per_seq, num_cls, selected_model='CNN+RNN'):
    if selected_model == 'CNN+RNN':
        model = Sequential()

        model.add(InputLayer(input_shape=(num_img_per_seq, img_res[0], img_res[1], 3)))

        model.add(TimeDistributed(Convolution2D(32, (4, 4), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(5, 5))))
        model.add(TimeDistributed(Convolution2D(16, (4, 4), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(5, 5))))
        model.add(TimeDistributed(Dropout(0.25)))
        model.add(TimeDistributed(Flatten()))

        model.add(GRU(128, kernel_initializer=initializers.RandomNormal(stddev=0.001)))  # 128
        model.add(Dropout(0.25))

        model.add(Dense(60))
        model.add(Dense(40))
        model.add(Dense(num_cls, activation='sigmoid'))

        opt = optimizers.RMSprop(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
        return model

    elif selected_model == 'CNN+MLP':
        model = Sequential()

        model.add(InputLayer(input_shape=(num_img_per_seq, img_res[0], img_res[1], 3)))
        model.add(TimeDistributed(Convolution2D(32, (4, 8), activation='relu')))
        model.add(TimeDistributed(Convolution2D(16, (4, 4), activation='relu')))
        model.add(TimeDistributed(MaxPooling2D(pool_size=(5, 5))))
        model.add(TimeDistributed(Dropout(0.25)))
        model.add(Flatten())

        model.add(Dense(60))
        model.add(Dense(80))
        model.add(Dense(num_cls, activation='sigmoid'))

        opt = optimizers.RMSprop(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['accuracy'])
        return model
