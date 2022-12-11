# Encoder
import keras
import keras_contrib
import numpy as np
import time
import pickle
import os
import pandas as pd
from keras.layers import Dense
from keras.models import Model
from utils.utils import save_logs, save_logs_per_batch, plot_and_save, calculate_metrics, write_dict, create_directory
from keras_contrib.layers import InstanceNormalization
import sklearn
from utils.custom_callbacks import My_ModelCheckpoint, LogPredictions
import keras.backend as K
from keras.regularizers import Regularizer
from keras.engine.network import Network
from keras.optimizers import SGD, Adam
import pickle
import classifiers.htnet
from datetime import datetime
import copy
# Import all models
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.lscp import LSCP

from pyod.models.auto_encoder import AutoEncoder
from pyod.models.cof import COF
from pyod.models.lmdd import LMDD
from pyod.models.loda import LODA
from pyod.models.loci import LOCI
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.sod import SOD
from pyod.models.so_gaal import SO_GAAL
from pyod.models.sos import SOS
from pyod.models.vae import VAE
from pyod.models.xgbod import XGBOD




import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


class EWC_reg(Regularizer):
    def __init__(self, fisher, prior_weights, Lambda=0.001):
        self.fisher = fisher
        self.prior_weights = prior_weights
        self.Lambda = Lambda

    def __call__(self, x):
        regularization = 0.
        regularization += self.Lambda * K.sum(self.fisher * K.square(x - self.prior_weights))
        # print('ewc_reg Regularizer is working')
        return regularization

    def get_config(self):
        return {'Lambda': float(self.Lambda)}



class MyMinMaxScaler():
    def fit_transform(self,x):
        return x
    def fit(self,x):
        pass
    def transform(self,x):
        return x

class MixedAnomalyDetector:
    def __init__(self, feature_extractor_model):
        self.feature_extractor_model = feature_extractor_model
        self.min_max_scaler = MinMaxScaler()
        self.min_max_scaler_base =  MinMaxScaler()
        outliers_fraction = 0.05#'auto'
        random_state =42
        # initialize a set of detectors for LSCP
        detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                         LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                         LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                         LOF(n_neighbors=50)]

        self.anomaly_algorithms = [
            ### ('Auto Encoder', AutoEncoder(hidden_neurons=[32, 16, 16, 32], batch_size= 12, random_state=random_state, contamination=outliers_fraction)),
            ###  ('single-Objective Generative Adversarial Active Learning(SO_GAAL)',SO_GAAL(contamination=outliers_fraction)),
            # ('Stochastic Outlier Selection (SOS)', SOS(contamination=outliers_fraction)), low accuracy
            ###  ('Variational Auto Encoder (VAE)', VAE(contamination=outliers_fraction, random_state=random_state)),
            ###  ('XGBOD: Improving Supervised Outlier Detection with Unsupervised Representation Learning',
            ###   XGBOD(random_state=random_state)), #Works but very slow
            ###  ('XGBOD with estimatiors list', XGBOD(estimator_list=detector_list, random_state=random_state)),
            ###  ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),#Works but very slow
            # ("One-Class SVM", svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)), ##################################################### SHOULD BE THERE!
            ###  ("Isolation Forest", IsolationForest(contamination=outliers_fraction, behaviour="new",
            ###                                       random_state=random_state)),
            ###  ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction, novelty=True)),
            # ('Angle-based Outlier Detector (ABOD)', ABOD(contamination=outliers_fraction)), ##################################################### SHOULD BE THERE!
            # ('Cluster-based Local Outlier Factor (CBLOF)', CBLOF(contamination=outliers_fraction,
            #                                                      check_estimator=False, random_state=random_state)),
            ### ('Feature Bagging', FeatureBagging(LOF(n_neighbors=35),  contamination=outliers_fraction,
            ###                                     random_state=random_state)), Takes too much time
            ###  ('Histogram-base Outlier Detection (HBOS)', HBOS(contamination=outliers_fraction)), I dont know why!
            ###  ('Isolation Forest', IForest(contamination=outliers_fraction,random_state=random_state)),
            # ('K Nearest Neighbors (KNN)', KNN(contamination=outliers_fraction)), ##################################################### SHOULD BE THERE!
            ('Average KNN', KNN(method='mean',contamination=outliers_fraction)),
            ('Local Outlier Factor (LOF) From PYOD', LOF(n_neighbors=35, contamination=outliers_fraction)),
            ###  ('Minimum Covariance Determinant (MCD)', MCD(contamination=outliers_fraction, random_state=random_state)),
            ('One-class SVM (OCSVM) from PYOD', OCSVM(contamination=outliers_fraction)),
            # ('Principal Component Analysis (PCA)',PCA( contamination=outliers_fraction, random_state=random_state)), ##################################################### SHOULD BE THERE!
            ###  ('Locally Selective Combination (LSCP)', LSCP(detector_list, contamination=outliers_fraction,
            ###                                                random_state=random_state)),
            ###  ('Connectivity-Based Outlier Factor (COF)',COF(contamination=outliers_fraction, n_neighbors=20)),
            ###  ('Linear Model Deviation-base outlier detection (LMDD)',LMDD(contamination=outliers_fraction,random_state=random_state)),
            # ('Loda: Lightweight on-line detector of anomalies',LODA(contamination=outliers_fraction, n_bins=10, n_random_cuts=100)), ##################################################### SHOULD BE THERE!
            ###  ('Subspace Outlier Detection(SOD)', SOD(contamination=outliers_fraction))
            ###  ('Local Correlation Integral (LOCI)', LOCI(contamination=outliers_fraction)), takes too much time
            ###  ('Locally Selective Combination of Parallel Outlier Ensembles (LSCP)',LSCP(detector_list)) Not Responsive at all!!!!
            ###  ('Multiple-Objective Generative Adversarial Active Learning.(MO_GAAL)', MO_GAAL(k=10, stop_epochs=20, lr_d=0.01, lr_g=0.0001, decay=1e-06, momentum=0.9, contamination=outliers_fraction)), # takes too myuh memory
        ]

        self.anomaly_algorithms_base = copy.deepcopy(self.anomaly_algorithms) #[("One-Class SVM", svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)) ]#self.anomaly_algorithms.copy()

    def get_anomaly_algorithms_names(self):
        l=[]
        for name, algorithm in self.anomaly_algorithms:
            l.append(name)
        return l

    def predict(self,x):
        x = self.feature_extractor_model.predict(x)
        x = x.reshape((len(x)),-1)
        x = self.min_max_scaler.transform(x)
        y_dict ={}
        for name, algorithm in self.anomaly_algorithms:
            try:
                y = algorithm.predict(x)
            except Exception as e:
                print('##############',name)
                print('Was not able to predict with: ',name)
                y = np.ones([len(x)])
                print(e)
            enc = sklearn.preprocessing.OneHotEncoder(categories=[np.asarray([-1, 1])]) # -1 for normal, 1 for anomaly
            if "pyod" in str(type(algorithm)):
                y = np.asarray(y)*2 - 1
            else:  # pyod gives 1 for anomolus, 0 for normal
                y = - np.asarray(y) # toggle the results for proper encoding for sklearn

            y = enc.fit_transform(y.reshape(-1, 1)).toarray()
            y_dict[name]=y
        return y_dict

    def predict_base(self,x):
        x = x.reshape((len(x)),-1)
        x = self.min_max_scaler_base.transform(x)
        y_dict ={}
        for name, algorithm in self.anomaly_algorithms_base:
            try:
                y = algorithm.predict(x)
            except Exception as e:
                print('##############',name)
                print('Was not able to predict with  for base: ',name)
                y = np.ones([len(x)])
                print(e)
            enc = sklearn.preprocessing.OneHotEncoder(categories=[np.asarray([-1, 1])])
            if "pyod" in str(type(algorithm)):
                y = np.asarray(y)*2 - 1
            else:  # pyod gives 1 for anomolus, 0 for normal
                y = - np.asarray(y) # toggle the results for proper encoding for sklearn

            y = enc.fit_transform(y.reshape(-1, 1)).toarray()
            y_dict[name] = y
        return y_dict

    def fit(self, x_train, tain_base_clf=False):
        timing = {}

        x_train_feature = self.feature_extractor_model.predict(x_train)  # use the model for feature extraction
        number_of_samples= x_train.shape[0]

        x_train_feature = x_train_feature.reshape((len(x_train_feature)),-1)
        tick = datetime.now()
        x_train_feature = self.min_max_scaler.fit_transform(x_train_feature)
        tock = datetime.now()
        timing['MinMaxScaler']  = str(tock-tick)

        for name, algorithm in self.anomaly_algorithms:
            tick = datetime.now()
            if 'XGBOD' in name:
                y=np.zeros([number_of_samples])
               # print(x_train_feature.shape)
                algorithm.fit(x_train_feature,y)
            else:
                algorithm.fit(x_train_feature)
            tock = datetime.now()
            timing[name] = str(tock - tick)
            print('finished training: ', name, ' - time:',datetime.now(), 'duration:',timing[name])


        print('All training is finished: ', datetime.now())
        timing_base={}
        if tain_base_clf:
            print('started training base anomaly detectors', datetime.now())
            x_train = x_train.reshape((len(x_train)),-1)
            x_train = self.min_max_scaler_base.fit_transform(x_train)
            for name, algorithm in self.anomaly_algorithms_base:
                tick = datetime.now()
                if 'XGBOD' in name:
                    y = np.zeros([number_of_samples])
                    algorithm.fit(x_train, y)
                else:
                    algorithm.fit(x_train)
                tock = datetime.now()
                timing_base[name] = str(tock - tick)
                print('finished training base: ', name, ' - time:', datetime.now(), 'duration:', timing_base[name])

            print('All training is finished for base: ', datetime.now())

        return timing,timing_base

    def fit_limited_generator(self, target_generator, number_of_batches_used=5,tain_base_clf=False):
        target_idxs = np.arange(len(target_generator))[:-1]
        np.random.shuffle(target_idxs)

        x_train = []
        for i in range(min(number_of_batches_used,len(target_idxs))):
            batch_target_x, _ = target_generator[target_idxs[i]]
            x_train.append(batch_target_x)

        x_train = np.concatenate(x_train)

        return self.fit(x_train,tain_base_clf)



class TLClassifierDouble:
    def __init__(self, base_model_dir, output_directory, nb_classes=2,
                 fine_tune_only=True, batch_size=12, lambda_ =0.1, verbose=False):
        self.output_directory = output_directory
        self.base_model_dir = base_model_dir
        self.fine_tune_only = fine_tune_only
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.verbose = verbose
        self.anomaly_detector = None # it will be initialized after training
        base_trained_model = self.get_trained_base_model()
        self.lambda_ = lambda_


        if verbose:
            print('base trained model summary')
            base_trained_model.summary()

        ######### Prepare the double networks ###############
        # Delete last layer frome base model
        base_trained_model.layers.pop()


        # Fixed weight
        for layer in base_trained_model.layers:
            if layer.name =="conv1d_2" :  # "dense_1":original htnet |||| "conv1d_4" for more layers |||| conv1d_3 for narrowed
                break
            else:
                layer.trainable = False


        self.model_t = Model(inputs= base_trained_model.input,outputs=base_trained_model.layers[-1].output)
        self.model_t_feature_out = base_trained_model.layers[-1].output_shape[-1]
        print('Number of features: ', self.model_t_feature_out)

        # R network　S and Weight sharing
        self.model_r = Network(inputs=self.model_t.input,
                          outputs=self.model_t.output,
                          name="shared_layer")

        print('adding a new dense layer')
        prediction = Dense(nb_classes, activation='softmax',name= 'new_dense_layer')(self.model_t.output)
        self.model_r = Model(inputs=self.model_r.input, outputs=prediction)
        # self.model_r = Model(inputs=self.model_r.input, outputs=prediction)
        #Compile
        # optimizer = SGD(lr=5e-5, decay=0.00005)
        # optimizer = SGD(lr=5e-4, decay=0.00005)
        optimizer = Adam(5e-4)
        self.model_r.compile(optimizer=optimizer, loss="categorical_crossentropy")
        self.model_t.compile(optimizer=optimizer, loss=self.original_loss)

        if verbose:
            print('model_t and model_r summary: ')
            self.model_t.summary()
            self.model_r.summary()

    def original_loss(self, y_true, y_pred):
        lc = 1 / (self.nb_classes * self.batch_size) * self.batch_size ** 2 * K.sum((y_pred - K.mean(y_pred, axis=0)) ** 2, axis=[1]) / ((self.batch_size - 1) ** 2)
        return lc*self.lambda_

    def get_trained_base_model(self):
        model = keras.models.load_model(self.base_model_dir + 'best_model.hdf5',
                                        custom_objects={'InstanceNormalization': InstanceNormalization})
        # model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(0.0000001),
        #                   metrics=['accuracy'])
        return model

    def fit_anomaly_detector(self,target_generator, number_of_batches_used=5,tain_base_clf=False):
        self.anomaly_detector = MixedAnomalyDetector(self.model_t)
        return self.anomaly_detector.fit_limited_generator(target_generator,number_of_batches_used,tain_base_clf)

    def save_current_models(self):
        self.model_t.save(filepath=self.output_directory+'model_t.h5', overwrite=True)
        self.model_r.save(filepath=self.output_directory+'model_r.h5', overwrite=True)
        # if self.anomaly_detector is not None:
        #     pickle.dump(self.anomaly_detector,  open(self.output_directory+'anomaly_detector.pkl',"wb"))

    def load_models(self):
        model_t =  keras.models.load_model(self.output_directory+'model_t.h5',
                                           custom_objects={'InstanceNormalization': InstanceNormalization,
                                                           'original_loss': self.original_loss})
        model_r =  keras.models.load_model(self.output_directory+'model_r.h5',
                                           custom_objects={'InstanceNormalization': InstanceNormalization,
                                                           'original_loss': self.original_loss})
        anomaly_detector = None
        try:
            anomaly_detector= pickle.load(open(self.output_directory+'anomaly_detector.pkl',"rb") )
        except Exception as e:
            print(e)
        return model_t, model_r, anomaly_detector

    # def save_loss(self, loss,loss_c):
    #     pickle.dump(self.anomaly_detector, open(self.output_directory + 'loss.pkl', "w"))
    #     pickle.dump(self.anomaly_detector, open(self.output_directory + 'loss_c.pkl', "w"))

    def evaluate_generator(self,generator,epoch_number=None, is_base=False):
        idxs = np.arange(len(generator))[:-1]
        np.random.shuffle(idxs)
        # idxs = idxs [0:5]
        y_true_ls = []

        anomaly_algorithm_names = self.anomaly_detector.get_anomaly_algorithms_names()
        y_pred_dict ={}
        for name in anomaly_algorithm_names:
            y_pred_dict[name]=[]

        for idx in idxs:
            x, y_true = generator[idx]
            if is_base:
                y_pred_tmp = self.anomaly_detector.predict_base(x)
            else:
                y_pred_tmp = self.anomaly_detector.predict(x)
            for name in anomaly_algorithm_names:
                y_pred_dict[name].append(y_pred_tmp[name])
            y_true_ls.append(y_true)

        for name in anomaly_algorithm_names:
            y_pred_dict[name] = np.concatenate(y_pred_dict[name])
        y_true= np.concatenate(y_true_ls)

        df_metrics = None
        for name in anomaly_algorithm_names:
            df_metrics_tmp = calculate_metrics(y_true, y_pred_dict[name])
            df_metrics_tmp ['anomaly detector'] = [name]
            if epoch_number is not None:
                df_metrics_tmp['epoch_number'] = [epoch_number]
            if df_metrics is None:
                df_metrics = df_metrics_tmp
            else:
                df_metrics = pd.concat([df_metrics_tmp, df_metrics])

        if is_base:
            df_metrics.to_csv(self.output_directory + 'df_metrics_base.csv', index=False)
        else:
            df_metrics.to_csv(self.output_directory + 'df_metrics.csv', index=False)
            if os.path.isfile(self.output_directory + 'df_metrics_all_epochs.csv'):
                df_metrics_all = pd.read_csv(self.output_directory +'df_metrics_all_epochs.csv')
                df_metrics = pd.concat([df_metrics_all, df_metrics])
            df_metrics.to_csv(self.output_directory + 'df_metrics_all_epochs.csv',index=False)

    def evaluate_generator_base(self, generator): #does no have feature extractor
        self.evaluate_generator( generator, is_base=True)
        # idxs = np.arange(len(generator))[:-1]
        # np.random.shuffle(idxs)
        # # idxs = idxs [0:5]
        # y_true_ls, y_pred_ls = [], []
        # for idx in idxs:
        #     x, y_true = generator[idx]
        #     y_pred = self.anomaly_detector.predict_base(x)
        #     y_true_ls.append(y_true)
        #     y_pred_ls.append(y_pred)
        #
        # y_pred = np.concatenate(y_pred_ls)
        # y_true = np.concatenate(y_true_ls)
        # df_metrics = calculate_metrics(y_true, y_pred)
        # df_metrics.to_csv(self.output_directory + 'df_metrics_base.csv', index=False)

    def fit_generator_base(self,target_generator,val_generator,number_of_batches_used_for_anomaly_detector_training):
        if os.path.isfile(self.output_directory + 'model_t.h5'):
            self.model_t, self.model_r, self.anomaly_detector = self.load_models()
        if not self.anomaly_detector:
            print('started training anomaly detector time:', datetime.now().time())

            self.fit_anomaly_detector(target_generator,
                                      number_of_batches_used=number_of_batches_used_for_anomaly_detector_training,
                                      tain_base_clf=True)

            print('finished training anomaly detector time:', datetime.now().time())

        print('initial base model is loaded and being evaluated without feature extraction.')
        self.evaluate_generator_base(val_generator)

    def evaluate_trained_feature_extractor(self, target_generator, val_generator,
                                           number_of_batches_used_for_anomaly_detector_training=5,
                                           evaluate_base_anomaly_detectors=True,  epoch_number=0):
        if os.path.isfile(self.output_directory+'model_t.h5'):
            try:
                self.model_t, self.model_r, self.anomaly_detector = self.load_models()
            except Exception as e:
                print(e)

        if not self.anomaly_detector:
            print('started training anomaly detector time:', datetime.now().time())

            timing, timing_base = self.fit_anomaly_detector(target_generator,
                                      number_of_batches_used=number_of_batches_used_for_anomaly_detector_training,
                                      tain_base_clf=evaluate_base_anomaly_detectors)

            write_dict(timing,self.output_directory+'anomaly_detectors_training_times.csv')
            write_dict(timing_base,self.output_directory+'anomaly_detectors_base_training_times.csv')

            print('finished training anomaly detector time:', datetime.now().time())

        # print('initial model is loaded and being evaluated')
        self.evaluate_generator(val_generator,epoch_number=epoch_number)
        if evaluate_base_anomaly_detectors:
            self.evaluate_generator_base(val_generator)


    def fit_generator(self, ref_generator, target_generator, val_generator = None, val_not_triggered_generator = None,
                      load_saved_model = True,
                      number_of_batches_used_for_anomaly_detector_training=5, fit_for_anomaly_detection= True,
                      nb_epochs=100, workers=10, save_results_after_each_epoch=True,
                      save_log_per_batch =True, expected_loss=None, patience_for_expected_loss=10):

        # print(load_saved_model, os.path.isfile(self.output_directory+'model_t.h5'),self.output_directory+'model_t.h5')
        self.save_current_models()


        if load_saved_model and os.path.isfile(self.output_directory+'model_t.h5'):
            self.evaluate_trained_feature_extractor(target_generator, val_generator,
                                                    number_of_batches_used_for_anomaly_detector_training,epoch_number=-1)

        # loss, loss_c = [], []
        hist = {'loss' : [],'loss_c' : [],'val_loss_c' : []}
        print("training...")

        ref_idxs = np.arange(len(ref_generator))[:-2]  # last batch might not be in correct size
        target_idxs = np.arange(len(target_generator))[:-2]

        # Learning
        for epochnumber in range(nb_epochs):
            lc, ld = [], []

            # Shuffle ref and target data indexes
            np.random.shuffle(ref_idxs)
            np.random.shuffle(target_idxs)

            for i in range(min(len(target_idxs)-1,len(ref_idxs)-1)): # the traget and ref len might be different
                ref_idx, target_idx =ref_idxs[i],target_idxs[i]

                # Load a batch of data
                batch_ref_x, batch_ref_y = ref_generator[ref_idx]
                batch_target_x, _ = target_generator[target_idx] # batch target y is dummy and it should be always the same
                # print(batch_target_x.shape)

                # target data
                # Get loss while learning
                lc.append(self.model_t.train_on_batch(batch_target_x, np.zeros((self.batch_size, self.model_t_feature_out))))

                # reference data
                # Get loss while learning
                ld.append(self.model_r.train_on_batch(batch_ref_x, batch_ref_y))

            hist['loss'].append(np.mean(ld))
            hist['loss_c'].append(np.mean(lc))

            print("epoch:", epochnumber + 1, "Descriptive loss:", hist['loss'][-1], "Compact loss", hist['loss'][-1])
            self.save_sample_extracted_features(val_generator, epoch_number=epochnumber)

            # print('started training anomaly detector time:',datetime.now().time())
            # self.fit_anomaly_detector(target_generator, number_of_batches_used=number_of_batches_used_for_anomaly_detector_training)
            # print('finished training anomaly detector time:',datetime.now().time())

            if val_not_triggered_generator:
                val_lc = []
                val_not_triggered_idxs = np.arange(len(val_not_triggered_generator))[:-1]
                val_not_triggered_idxs = val_not_triggered_idxs[:min(round(len(val_not_triggered_idxs)/10),2)]
                for idx in val_not_triggered_idxs:
                    batch_val_x, _ = val_not_triggered_generator[idx]
                    val_lc.append(self.model_t.test_on_batch(batch_val_x,
                                                             np.zeros((self.batch_size, self.model_t_feature_out))))
                hist['val_loss_c'].append(np.mean(val_lc))
                print('Not triggered val loss: ',hist['val_loss_c'][-1])


            self.save_current_models()


            plot_and_save(hist, self.output_directory+'epochs_loss.png', metric='loss')
            plot_and_save(hist, self.output_directory+'epochs_loss_c.png', metric='loss_c')
            self.evaluate_trained_feature_extractor(target_generator, val_generator,
                                                    number_of_batches_used_for_anomaly_detector_training,
                                                    evaluate_base_anomaly_detectors=False, epoch_number=epochnumber)


        self.evaluate_trained_feature_extractor(target_generator, val_generator,
                                                number_of_batches_used_for_anomaly_detector_training,
                                                epoch_number=epochnumber)



    def save_sample_extracted_features(self, val_generator, epoch_number):
        o_dir= self.output_directory+'sample_extracted_feature/'
        create_directory(o_dir)

        idxs = np.arange(len(val_generator))[:-1]
        np.random.shuffle(idxs)
        # idx = idxs[0]
        idx= 4
        x, y_true = val_generator[idx]
        np.savetxt(o_dir+"y_idx_"+str(idx)+".csv", y_true.T, delimiter=",")
        # print(x.shape)
        # np.savetxt(o_dir+str(epoch_number)+"_rawdata_idx_"+str(idx)+".csv", x.reshape((len(x)), -1).T, delimiter=",")
        np.savetxt(o_dir+"_rawdata_idx_"+str(idx)+".csv", x.reshape((len(x)), -1).T, delimiter=",")

        x = self.model_t.predict(x)
        x = x.reshape((len(x)), -1)
        # print(x.shape)
        np.savetxt(o_dir+str(epoch_number)+"_features_idx_"+str(idx)+".csv", x.T, delimiter=",")

        # if val_generator:
        #     self.evaluate_generator(val_generator)



        # def evaluate_generator():

        # Result graph
        # plt.plot(loss, label="Descriptive loss")
        # plt.xlabel("epoch")
        # plt.legend()
        # plt.show()
        #
        # plt.plot(loss_c, label="Compact loss")
        # plt.xlabel("epoch")
        # plt.legend()
        # plt.show()
        #

    # def evaluate_generator(self,  train_target_generator, val_genrator):
    #     train = self.model_t.predict(x_train_s)
    #     test_s = model.predict(X_test_s)
    #     test_b = model.predict(X_test_b)



    # log_predictions = LogPredictions(self.output_directory, val_generator, val_true, workers,
        #                                  real_val_generator=real_val_generator, real_val_true=real_val_true,
        #                                  base_val_generator=base_val_generator, base_val_true=base_val_true,
        #                                  save_log_per_batch=save_log_per_batch, expected_loss=expected_loss,
        #                                  patience=patience_for_expected_loss)
        #
        # model_checkpoint = My_ModelCheckpoint(filepath=file_path,
        #                                                    monitor='loss', save_best_only=True)
        # early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
        #                               min_delta=0,
        #                               patience=2,
        #                               verbose=0, mode='auto')
        #
        # tensorboard = keras.callbacks.TensorBoard(log_dir=self.output_directory + 'logs/' + str(time.time()))
        # cbks = [model_checkpoint, log_predictions, tensorboard, early_stop]
        #
        #
        #
        #
        #
        #
        # start_time = time.time()
        # if save_results_after_each_epoch:
        #     hist = self.model.fit_generator(train_generator, epochs=nb_epochs,
        #                                     verbose=self.verbose, workers=workers, validation_data=val_generator,
        #                                     callbacks=cbks, use_multiprocessing=True)
        # else:
        #     hist = self.model.fit_generator(train_generator, epochs=nb_epochs,
        #                                     verbose=self.verbose, workers=workers, validation_data=val_generator,
        #                                     callbacks=[model_checkpoint], use_multiprocessing=True)
        # duration = time.time() - start_time
        #
        # # model = keras.models.load_model(self.output_directory + 'best_model.hdf5',
        # #                                 custom_objects={'InstanceNormalization': InstanceNormalization,
        # #                                                 'EWC_reg': EWC_reg})
        #
        # val_pred = self.model.predict_generator(val_generator, workers=workers, use_multiprocessing=True,)
        # # convert the predicted from binary to integer
        # val_pred = np.argmax(val_pred, axis=1)
        # val_true = np.argmax(val_true, axis=1)
        #
        # save_logs(self.output_directory, hist, val_pred, val_true, duration, lr=False)
