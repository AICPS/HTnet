import numpy as np
import time

from utils.utils import save_logs, silentremove,create_directory, save_logs_per_batch,extract_and_save_best
from keras import callbacks
import os


class My_ModelCheckpoint(callbacks.ModelCheckpoint):
    # def __init__(self, filepath, monitor='val_loss', verbose=0,
    #              save_best_only=False, save_weights_only=False,
    #              mode='auto', period=1):
    #     super(My_ModelCheckpoint, self).__init__(filepath=filepath, monitor=monitor, verbose=verbose,
    #              save_best_only=save_best_only, save_weights_only=save_weights_only,
    #              mode=mode, period=period)
    #
    #
    #
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            try:
                if self.save_best_only:
                    current = logs.get(self.monitor)
                    if current is None:
                        print('Warning: Can save best model only with %s available, '
                                      'skipping.' % (self.monitor))
                    else:
                        if self.monitor_op(current, self.best):
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                      ' saving model to %s'
                                      % (epoch + 1, self.monitor, self.best,
                                         current, filepath))
                            self.best = current
                            while( os.path.exists(filepath) ):
                                silentremove(filepath)
                            if self.save_weights_only:
                                self.model.save_weights(filepath, overwrite=True)
                            else:
                                self.model.save(filepath, overwrite=True)
                        else:
                            if self.verbose > 0:
                                print('\nEpoch %05d: %s did not improve from %0.5f' %
                                      (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))

                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(filepath, overwrite=True)
            except:
                print('not able to save the file')


class LogPredictions(callbacks.Callback):
    def __init__(self, output_directory,val_generator,val_true,workers,real_val_generator=None,real_val_true=None,
                 base_val_generator=None, base_val_true=None, save_log_per_batch=False, expected_loss=None, patience=10):
        super(LogPredictions, self).__init__()
        self.targets = []  # collect y_true batches
        self.outputs = []  # collect y_pred batches
        self.output_directory = output_directory
        self.epoch_start_time = time.time()
        self.val_true = val_true
        self.val_generator = val_generator
        self.workers = workers
        self.real_val_generator = real_val_generator
        self.real_val_true = real_val_true
        self.base_val_generator = base_val_generator
        self.base_val_true = base_val_true
        self.real_val_output_directory=output_directory+'/real_prediction_result/'
        self.base_val_output_directory = output_directory + '/base_prediction_result/'
        self.expected_loss = expected_loss
        self.patience = patience
        self.number_of_overfit_loss = 0
        self.save_log_per_batch = save_log_per_batch
        self.epoch_number =0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time= time.time()

    def on_epoch_end(self, epoch, logs):

        if epoch > 0:
            # 5.4.1 For each validation batch
            val_pred = self.model.predict_generator(self.val_generator, workers=self.workers, use_multiprocessing=True)
            # convert the predicted from binary to integer
            val_pred = np.argmax(val_pred, axis=1)
            val_true = np.argmax(self.val_true, axis=1)

            hist = self.model.history
            duration = time.time() - self.epoch_start_time
            save_logs(self.output_directory, hist, val_pred, val_true, duration, lr=False)

        if self.expected_loss is not None:
            current_loss = logs.get('loss')
            if (current_loss < self.expected_loss):
                self.number_of_overfit_loss = self.number_of_overfit_loss + 1
            else:
                self.number_of_overfit_loss = 0

            if self.number_of_overfit_loss > self.patience:
                self.model.stop_training = True

        self.epoch_number = self.epoch_number+1

        # Real
        if(self.real_val_generator!=None):
            val_pred = self.model.predict_generator(self.real_val_generator, workers=self.workers,
                                                    use_multiprocessing=True)
            # convert the predicted from binary to integer
            val_pred = np.argmax(val_pred, axis=1)
            val_true = np.argmax(self.real_val_true, axis=1)

            hist = self.model.history
            duration = time.time() - self.epoch_start_time
            df_metrics=save_logs(self.real_val_output_directory, hist, val_pred, val_true, duration, lr=False, only_metrics=True)
            print("Real Validation: ")
            print(df_metrics)

        # Base
        if(self.base_val_generator!=None):
            val_pred = self.model.predict_generator(self.base_val_generator, workers=self.workers,
                                                    use_multiprocessing=True)
            # convert the predicted from binary to integer
            val_pred = np.argmax(val_pred, axis=1)
            val_true = np.argmax(self.base_val_true, axis=1)

            hist = self.model.history
            duration = time.time() - self.epoch_start_time
            df_metrics =save_logs(self.base_val_output_directory, hist, val_pred, val_true, duration, lr=False, only_metrics=True)
            print("Base Validation: ")
            print(df_metrics)




    def on_train_begin(self, logs=None):
        if (self.real_val_generator != None):
            create_directory(self.real_val_output_directory)
        if (self.base_val_generator != None):
            create_directory(self.base_val_output_directory)

    def on_train_end(self, logs=None):
        if (self.real_val_generator != None):
            extract_and_save_best(self.real_val_output_directory)
        if (self.base_val_generator != None):
            extract_and_save_best(self.base_val_output_directory)

    def on_batch_end(self, batch, logs=None):
        if self.save_log_per_batch:
            currentLoss = logs.get('loss')
            if (currentLoss<self.expected_loss):
                self.number_of_overfit_loss=self.number_of_overfit_loss+1
            else:
                self.number_of_overfit_loss=0

            if self.number_of_overfit_loss>self.patience:
                self.model.stop_training = True

            # epoch_num=0
            # if hasattr(self.model.history, "__getitem__"):
            #     epoch_num=len(self.model.history['loss'])
            #     print("My Calculated epoch num"+str(epoch_num))

            epoch_num = self.epoch_number

            if (self.real_val_generator != None):
                metrics = self.model.evaluate_generator(self.real_val_generator, workers=self.workers,use_multiprocessing=True)
                save_logs_per_batch(self.real_val_output_directory, metrics, metrics_names=self.model.metrics_names, epoch_num= epoch_num)

            if (self.base_val_generator != None):
                metrics = self.model.evaluate_generator(self.base_val_generator, workers=self.workers, use_multiprocessing=True)
                save_logs_per_batch(self.base_val_output_directory, metrics, metrics_names=self.model.metrics_names, epoch_num=epoch_num)