# Encoder
import keras
import keras_contrib
import numpy as np
import time
from utils.utils import save_logs
from keras_contrib.layers import InstanceNormalization
from utils.custom_callbacks import My_ModelCheckpoint, LogPredictions



class Classifier_HTNET:

    def __init__(self, output_directory, input_shape, nb_classes, compile= True, verbose=False):
        self.output_directory = output_directory
        self.compile = compile
        self.model = self.build_model_used_in_paper(input_shape, nb_classes)
        if (verbose == True):
            self.model.summary()
        self.verbose = verbose
        self.model.save_weights(self.output_directory + 'model_init.hdf5')

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)


        # conv block -1
        conv1 = keras.layers.Conv1D(filters=128, kernel_size=9, strides=1, padding='same')(input_layer)
        # conv1 = keras_contrib.layers.InstanceNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=0.2)(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        # conv block -2
        conv2 = keras.layers.Conv1D(filters=256, kernel_size=19, strides=1, padding='same')(conv1)
        conv2 = keras_contrib.layers.InstanceNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=0.2)(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        # conv block -3
        # conv3 = keras.layers.Conv1D(filters=512, kernel_size=21, strides=1, padding='same')(conv2)
        # conv3 = keras_contrib.layers.InstanceNormalization()(conv3)
        # conv3 = keras.layers.PReLU(shared_axes=[1])(conv3)
        # conv3 = keras.layers.Dropout(rate=0.2)(conv3)
        # split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:, :, :128])(conv2)
        attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 128:])(conv2)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])
        # last layers
        dense_layer = keras.layers.Dense(units=128, activation='sigmoid')(multiply_layer)
        dense_layer = keras_contrib.layers.InstanceNormalization()(dense_layer)
        dense_layer = keras.layers.Dense(units=128, activation='sigmoid')(dense_layer)
        dense_layer = keras_contrib.layers.InstanceNormalization()(dense_layer)
        # output layer
        flatten_layer = keras.layers.Flatten()(dense_layer)
        output_layer = keras.layers.Dense(units=nb_classes, activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.00001),
                      metrics=['accuracy'])

        return model

    def build_model_used_in_paper(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)


        # conv block -1
        conv1 = keras.layers.Conv1D(filters=128, kernel_size=9, strides=1, padding='same')(input_layer)
        # conv1 = keras_contrib.layers.InstanceNormalization()(conv1)
        conv1 = keras.layers.PReLU(shared_axes=[1])(conv1)
        conv1 = keras.layers.Dropout(rate=0.2)(conv1)
        conv1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        # conv block -2
        conv2 = keras.layers.Conv1D(filters=256, kernel_size=19, strides=1, padding='same')(conv1)
        conv2 = keras_contrib.layers.InstanceNormalization()(conv2)
        conv2 = keras.layers.PReLU(shared_axes=[1])(conv2)
        conv2 = keras.layers.Dropout(rate=0.2)(conv2)
        conv2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        # split for attention
        attention_data = keras.layers.Lambda(lambda x: x[:, :, :128])(conv2)
        attention_softmax = keras.layers.Lambda(lambda x: x[:, :, 128:])(conv2)
        # attention mechanism
        attention_softmax = keras.layers.Softmax()(attention_softmax)
        multiply_layer = keras.layers.Multiply()([attention_softmax, attention_data])
        # last layer
        dense_layer = keras.layers.Dense(units=128, activation='sigmoid')(multiply_layer)
        dense_layer = keras_contrib.layers.InstanceNormalization()(dense_layer)
        # output layer
        flatten_layer = keras.layers.Flatten()(dense_layer)
        output_layer = keras.layers.Dense(units=nb_classes, activation='softmax')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        if self.compile:
            model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.001), # 0.00001
                      metrics=['accuracy'])

        return model

    def fit_genrator(self, train_generator, val_generator, val_true, workers=20, nb_epochs = 100,
                     save_results_after_each_epoch=True, patience=30):

        file_path = self.output_directory + 'best_model.hdf5'
        log_predictions = LogPredictions(self.output_directory, val_generator,val_true,workers)
        model_checkpoint = My_ModelCheckpoint(filepath=file_path,
                                                           monitor='loss', save_best_only=True)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=patience,
                                      verbose=0, mode='auto')

        tensorboard = keras.callbacks.TensorBoard(log_dir=self.output_directory + 'logs/' + str(time))
        cbks = [model_checkpoint, log_predictions, tensorboard, early_stop]


        start_time = time.time()

        if save_results_after_each_epoch:
            hist = self.model.fit_generator(train_generator, epochs=nb_epochs,
                                            verbose=self.verbose, workers=workers, validation_data=val_generator,
                                            callbacks=cbks, use_multiprocessing=True)
        else:
            hist = self.model.fit_generator(train_generator, epochs=nb_epochs,
                                            verbose=self.verbose, workers=workers, validation_data=val_generator, callbacks=[model_checkpoint], use_multiprocessing=True)
        duration = time.time() - start_time

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5', custom_objects={'InstanceNormalization':InstanceNormalization})

        val_pred = model.predict_generator(val_generator, workers=workers, use_multiprocessing=True)
        # convert the predicted from binary to integer
        val_pred = np.argmax(val_pred, axis=1)
        val_true = np.argmax(val_true, axis=1)

        save_logs(self.output_directory, hist, val_pred, val_true, duration, lr=False)

        keras.backend.clear_session()

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        # x_val and y_val are only used to monitor the test loss and NOT for training
        batch_size = 12
        nb_epochs = 1

        mini_batch_size = batch_size
        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path,
                                                           monitor='loss', save_best_only=True)

        cbks = [model_checkpoint]#, log_predictioms]

        start_time = time.time()

        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=cbks)

        duration = time.time() - start_time

        model = keras.models.load_model(self.output_directory + 'best_model.hdf5',custom_objects={'InstanceNormalization':InstanceNormalization})

        y_pred = model.predict(x_val)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        save_logs(self.output_directory, hist, y_pred, y_true, duration, lr=False)

        keras.backend.clear_session()