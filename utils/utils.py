from builtins import print
from typing import List

import numpy as np
import pandas as pd 
import matplotlib
import pickle
import errno

matplotlib.use('agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
import os
import operator
import utils.utils_HT as ht
import utils
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import Sequence
import sklearn
import keras
from keras.utils import np_utils
import csv
import copy
from sklearn.preprocessing import LabelEncoder

from itertools import compress
from utils.constants import UNIVARIATE_DATASET_NAMES as DATASET_NAMES
from utils.constants import ARCHIVE_NAMES  as ARCHIVE_NAMES
from utils.constants import CLASSIFIERS 
from utils.constants import ITERATIONS
from utils.constants import MTS_DATASET_NAMES
import random as rd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder

from scipy.interpolate import spline
from scipy.io import loadmat

import re

import os
import zipfile
import shutil


def zipdir(path,output_path=None, delete_original_path = False):
    if output_path is None:
        trimmed_path= path
        while trimmed_path.endswith('/') or trimmed_path.endswith("\\"):
            trimmed_path = trimmed_path[:-1]
        output_path = trimmed_path + '.zip'
    # ziph is zipfile handle
    zipf = zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(path):
        for file in files:
            zipf.write(os.path.join(root, file))
    zipf.close()

    if delete_original_path:
        remove(path)

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        print(exc)
        shutil.copy(src, dst)




def remove(path):
    """ param <path> could either be relative or absolute. """
    if os.path.isfile(path) or os.path.islink(path):
        os.remove(path)  # remove the file
    elif os.path.isdir(path):
        shutil.rmtree(path)  # remove dir and all contains
    else:
        raise ValueError("file {} is not a file or dir.".format(path))

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


def write_dict(dictionary, path):
    with open(path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dictionary.items():
            writer.writerow([key, value])



def comibne_base_and_orginal_summeries(results_dir,input_files=['all_metrics.csv','all_metrics_base.csv'],output_name='combined.csv',
                                       list_of_benchmarks=[],list_of_anomally_detectors=[]):

    for input_file in input_files:
        dir_2_res = results_dir + input_file
        data = pd.read_csv(dir_2_res, index_col=None, header=0)


def summrize(results_dir, input_res='df_best_model.csv', output_res='all_best_model.csv',add_extra_columns=False):
    list_of_folders = ht._get_immediate_subdirectories(results_dir)
    list_of_folders.sort(key=natural_keys)
    # print(list_of_folders)
    results = []
    # final_list_of_folders = list_of_folders.copy()
    final_list_of_folders=[]
    for folder in list_of_folders:
        dir_2_res = results_dir + folder + '/' + input_res
        # print(dir_2_res)
        if os.path.isfile(dir_2_res):
            data = pd.read_csv(dir_2_res, index_col=None, header=0)
            results.append(data)
            for i in range(data.shape[0]):
                final_list_of_folders.append(folder)
        else:
            pass
            # final_list_of_folders.remove(folder)

    results = pd.concat(results, axis=0, ignore_index=True)
    results.loc['mean'] = results.mean()
    results['benchmark_name'] = final_list_of_folders + ['mean']

    if add_extra_columns:
        mu_list = []
        sigmal_list = []
        for folder_name in final_list_of_folders:
            try:
                s = folder_name.split('_')
                mu_list.append(s[s.index('mu')+1])
                sigmal_list.append(s[s.index('sigma') + 1])
            except Exception as e:
                mu_list.append(0)
                sigmal_list.append(0)
        results['mu'] = mu_list + [0]
        results['sigma'] = sigmal_list + [0]

    results.to_csv(results_dir + output_res, index=False)

    print(results)


def readucr(filename):
    # print(filename)
    data = np.loadtxt(filename, delimiter = ',')

    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def create_directory(directory_path): 
    if os.path.exists(directory_path): 
        return None
    else: 
        try: 
            os.makedirs(directory_path)
        except: 
            # in case another machine created the path meanwhile !:(
            return None 
        return directory_path
def create_path(root_dir,classifier_name, archive_name):
    output_directory = root_dir+'/results/'+classifier_name+'/'+archive_name+'/'
    if os.path.exists(output_directory): 
        return None
    else: 
        os.makedirs(output_directory)
        return output_directory

def save_obj(obj, name ):
    with open(name , 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name , 'rb') as f:
        return pickle.load(f)

def get_ht_specific_scalers(dir2bms_folder, dataset_names,number_of_training_for_scaler=2000):
    output_file_path=ht._fixThePathForTheOS(dir2bms_folder)+'scaler_{}_sample_used.pkl'.format(number_of_training_for_scaler)
    if os.path.isdir(ht._fixThePathForTheOS(output_file_path)):
        scalers=load_obj(output_file_path)
    else:
        scalers = {}
        for ht_folder in dataset_names:
            _, _, _, _, scalers[ht_folder], _ = \
                get_splited_list_of_files_and_scaler_HT(
                    dir2bms_folder=dir2bms_folder,
                    name_bms=[ht_folder], number_of_training_for_scaler=number_of_training_for_scaler)
                    # ,number_of_samples_per_folder=number_of_training_for_scaler)

        save_obj(scalers,output_file_path)
    return scalers


class Data_Generator_Combiner(keras.utils.Sequence):
    def __init__(self, data_generators_list, added_noise_mu=0, added_noise_sigma=0):
        self.data_generators_list = copy.deepcopy(data_generators_list)
        fixed_seed = rd.random()
        for dg in self.data_generators_list:
            dg.set_noise_parms(added_noise_mu, added_noise_sigma)
            dirs_to_files, labels = dg.dirs_to_files, dg.labels
            decoded_one_hot =list([np.where(r==1)[0][0] for r in labels])
            # print(decoded_one_hot)
            dg.labels = [x for _,x, _ in sorted(zip(decoded_one_hot,labels, dirs_to_files), key=lambda pair: pair[0])]
            dg.dirs_to_files = [x for _,_,x in sorted(zip(decoded_one_hot, labels, dirs_to_files), key=lambda pair: pair[0])]
            rd.Random(fixed_seed).shuffle(dg.dirs_to_files)
            rd.Random(fixed_seed).shuffle(dg.labels)

    def __len__(self):
        return len(self.data_generators_list[0])

    def __getitem__(self, idx, fix_shape=True):

        outputs = []
        Y = None

        for dg in self.data_generators_list:
            x,y = dg.__getitem__(idx)
            outputs.append(x)
            if Y is None:
                Y = y
            elif (Y != y).any():
                raise ValueError('The labels in two data generators do not match!',
                                 str(list([np.where(r==1)[0][0] for r in Y])),
                                 str(list([np.where(r == 1)[0][0] for r in y])))


        # print(outputs[0].shape)
        # print(outputs[1].shape)

        X = np.concatenate(outputs, axis=1)

        # raise ValueError(str(outputs[0].shape),'  ', str(outputs[1].shape), '    ', str(X.shape), str(Y.shape))

        # print(outputs)
        return X,Y


class Data_Generator(keras.utils.Sequence):

    def __init__(self, dirs_to_files, labels, batch_size, dir2bms_folder=None, scaler=None, print_names=False,
                 circular_shift = None, added_noise_mu=0, added_noise_sigma=0):
        self.dirs_to_files, self.labels = dirs_to_files, labels
        self.batch_size = batch_size
        self.scaler= scaler
        self.dir2bms_folder = dir2bms_folder
        self.print_names = print_names
        self.circular_shift = circular_shift
        self.mu =added_noise_mu
        self.sigma= added_noise_sigma

        if dir2bms_folder is not None:
            print(dir2bms_folder)
            self.dir2bms_folder_len = len(ht._fixThePathForTheOS(dir2bms_folder))


    def set_noise_parms(self,added_noise_mu=0, added_noise_sigma=0):
        self.mu = added_noise_mu
        self.sigma= added_noise_sigma

    def __len__(self):
        return np.int(np.ceil(len(self.dirs_to_files) / float(self.batch_size)))

    def __getitem__(self, idx, fix_shape=True):
        batch_x = self.dirs_to_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.print_names:
            print(batch_x, batch_y,scaler2txt(self.scaler))

        if self.scaler==None:
            output = np.array([
                np.loadtxt(dir_to_file, delimiter='\n')
                   for dir_to_file in batch_x])
        elif type(self.scaler) is dict:
            # dataset_forder_names = [dir_to_file[self.dir2bms_folder_len:].split('/')[0] for dir_to_file in batch_x]
            dataset_forder_names=[]
            for dir_to_file in batch_x:
                dataset_forder_names.append(dir_to_file[self.dir2bms_folder_len:].split('/')[0])
            output = np.array([
                np.loadtxt(dir_to_file, delimiter='\n')
                   for dir_to_file in batch_x])

            for i in range(len(dataset_forder_names)):
                output[i] = self.scaler[dataset_forder_names[i]].transform(output[i].reshape(1, -1))
        else:
            # print(batch_x)
            output = self.scaler.transform(np.array([
                np.loadtxt(dir_to_file, delimiter='\n')
                   for dir_to_file in batch_x]))

        if fix_shape and len(output.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension
            output = output.reshape((output.shape[0], output.shape[1], 1))

        if self.circular_shift:
            output = np.roll(output,[0, self.circular_shift], axis=(0, 1))

        if self.mu>0 or self.sigma>0:
            noise = np.random.normal(self.mu, self.sigma, output.shape)
            output = output + noise #noisy_signal
        return output, np.array(batch_y)

    def get_more_number_of_batches_compact(self,idx, number_of_batches):
        idx = round(idx/number_of_batches)
        original_batch_size = self.batch_size
        self.batch_size = number_of_batches * self.batch_size
        X, y = self.__getitem__(idx)
        self.batch_size = original_batch_size
        return X, y

def _get_list_of_files_HT(dir2bms_folder, name_bms, folder_numbers,use_enabled_trojan_folder, get_categorical_labels_y=False,
                          number_of_samples_per_folder = None):
    dir2bms_folder = ht._fixThePathForTheOS(dir2bms_folder)
    dirs_to_files = []
    y = []
    y_categorical =[]
    for name_bm in name_bms:
        list_of_folders, types = ht._get_immediate_subdirectories_and_set_of_types(dir2bms_folder + name_bm)

        d_cnt, t_cnt,e_cnt =0,0,0

        for className in types:
            if not use_enabled_trojan_folder and 'Enabled' in className:
                continue
            for folder_number in folder_numbers:
                folder_name = className + '_' + str(folder_number)
                dir2data = dir2bms_folder + name_bm + '/' + folder_name + '/' + folder_name + '/'
                print(dir2data, '(', number_of_samples_per_folder, ' samples)')
                sample_names = ht._get_csv_files_sorted_list(dir2data)
                for sample_name in sample_names:
                    if 'Disabled' in className:
                        d_cnt = d_cnt+1
                        if (number_of_samples_per_folder is None) or d_cnt < number_of_samples_per_folder:
                            dirs_to_files.append(sample_name)
                        else:
                            continue
                        y_categorical.append(name_bm+'_Disabled')
                        y.append(0)

                    elif 'Triggered' in className:
                        t_cnt = t_cnt+1
                        if (number_of_samples_per_folder is None) or t_cnt < number_of_samples_per_folder:
                            dirs_to_files.append(sample_name)
                        else:
                            continue
                        y_categorical.append(name_bm+'_Triggered')
                        y.append(1)

                    elif 'Enabled' in className:
                        e_cnt = e_cnt+1
                        if (number_of_samples_per_folder is None) or e_cnt < number_of_samples_per_folder:
                            dirs_to_files.append(sample_name)
                        else:
                            continue
                        y_categorical.append(name_bm + '_Enabled')
                        y.append(2)
                    else:
                        print('bad folder name: {}'.format(className))
                        raise Exception('x should not exceed 5. The value of x was: {}'.format(className))




        #one hot encode labels
    enc = sklearn.preprocessing.OneHotEncoder(categories=[np.asarray([0, 1])])
    if use_enabled_trojan_folder:
        enc = sklearn.preprocessing.OneHotEncoder(categories=[np.asarray([0, 1, 2])])

    y = np.asarray(y)
    y = enc.fit_transform(y .reshape(-1, 1)).toarray()

    if get_categorical_labels_y:
        y_categorical = LabelEncoder().fit_transform(y_categorical)
        return dirs_to_files, y, y_categorical

    return dirs_to_files, y

def scaler2txt(scaler):
   return 'Scaler Mean: {} - Scaler STD: {}'.format(scaler.mean_.mean(), scaler.scale_.mean())


def get_splited_list_of_files_and_scaler_HT(dir2bms_folder='HT_Data/AES_withTrojan_Set2/',
                                            name_bms=['AES-T400'], folder_numbers= [1],
                                            use_enabled_trojan_folder=False, number_of_training_for_scaler=1000,
                                            random_state=11, use_trigerd_data_for_scale_training=True, #10
                                            get_not_triggered_training_only=False, get_triggered_training_only=False,
                                            number_of_samples= None, number_of_samples_per_folder= None , get_categorical_labels_y=False,
                                            get_not_triggered_validation_as_well=False):


    if get_categorical_labels_y:
        dirs_to_files, y, y_categorical = _get_list_of_files_HT(dir2bms_folder, name_bms, folder_numbers,
                                                                use_enabled_trojan_folder, get_categorical_labels_y,
                                                                number_of_samples_per_folder=number_of_samples_per_folder)

        dirs_to_files_train, dirs_to_files_test, y_train, y_test, y_categorical_train, y_categorical_test = \
            train_test_split(dirs_to_files, y, y_categorical, test_size=0.2, random_state=random_state)
    else:
        dirs_to_files, y = _get_list_of_files_HT(dir2bms_folder, name_bms, folder_numbers, use_enabled_trojan_folder,
                                                 number_of_samples_per_folder=number_of_samples_per_folder)
        dirs_to_files_train, dirs_to_files_test, y_train, y_test = train_test_split(dirs_to_files, y, test_size=0.2, random_state=random_state)

    if number_of_samples:
        num_test = int(0.2*number_of_samples)
        num_train = int(0.8*number_of_samples)

        dirs_to_files_train, dirs_to_files_test, y_train, y_test = dirs_to_files_train[0:num_train], dirs_to_files_test[0:num_test],\
                                                                   y_train[0:num_train], y_test[0:num_test]
        if get_categorical_labels_y:
            y_categorical_train, y_categorical_test = y_categorical_train[0:num_train], y_categorical_test[0:num_test]

    # print(dirs_to_files_train[0], '  ', y_train [0], np.argmax(y_train[0:2], axis=1) == 0)

    x_train_tmp, y_train_tmp = Data_Generator(dirs_to_files_train, y_train, batch_size=number_of_training_for_scaler).__getitem__(1, fix_shape=False)
    if not use_trigerd_data_for_scale_training:
        x_train_tmp = x_train_tmp[np.argmax(y_train_tmp, axis=1) == 0]
        y_train_tmp = y_train_tmp[np.argmax(y_train_tmp, axis=1) == 0]
    scaler = preprocessing.StandardScaler()
    scaler.fit(x_train_tmp)

    if get_not_triggered_training_only:
        dirs_to_files_train = list(compress(dirs_to_files_train, (np.argmax(y_train, axis=1) == 0)))
        if get_categorical_labels_y:
            y_categorical_train = y_categorical_train[np.argmax(y_train, axis=1) == 0]
        y_train = y_train[np.argmax(y_train, axis=1) == 0]

    if get_triggered_training_only:
        dirs_to_files_train = list(compress(dirs_to_files_train, (np.argmax(y_train, axis=1) != 0)))
        if get_categorical_labels_y:
            y_categorical_train = y_categorical_train[np.argmax(y_train, axis=1) != 0]
        y_train = y_train[np.argmax(y_train, axis=1) != 0]

    # print(y_train)
    # Print Thing related to scaler
    unique, counts = np.unique(y_train_tmp, return_counts=True,axis=0)
    print('Scaler Mean: {} - Scaler STD: {} - Len of scaler: {} - number of each label count for Scaler: {}:{} '.format(scaler.mean_.mean(), scaler.scale_.mean(), len(scaler.mean_), str(unique[0]), counts[0]))
    if len(x_train_tmp.shape) == 2:  # if univariate, add a dimension to make it multivariate with one dimension
        x_train_tmp = x_train_tmp.reshape((x_train_tmp.shape[0], x_train_tmp.shape[1], 1))

    input_shape = x_train_tmp.shape[1:]
    # print('input shape: ' + str(input_shape))

    if get_categorical_labels_y:
        y_categorical_train = np_utils.to_categorical(y_categorical_train)
        y_categorical_test = np_utils.to_categorical(y_categorical_test)
        return dirs_to_files_train, dirs_to_files_test, y_train, y_test, y_categorical_train, y_categorical_test,\
               scaler, input_shape

    if get_not_triggered_validation_as_well:
        dirs_to_files_test_not_triggered = list(compress(dirs_to_files_test, (np.argmax(y_test, axis=1) == 0)))
        y_test_not_triggered = y_test[np.argmax(y_test, axis=1) == 0]
        return dirs_to_files_train, dirs_to_files_test, dirs_to_files_test_not_triggered, y_train, y_test, \
               y_test_not_triggered, scaler, input_shape

    return dirs_to_files_train, dirs_to_files_test, y_train, y_test, scaler, input_shape

def read_dataset(root_dir,archive_name,dataset_name,fileNameRange=None):
    datasets_dict = {}

    if archive_name == 'mts_archive':
        file_name = root_dir+'/archives/'+archive_name+'/'+dataset_name+'/'
        x_train = np.load(file_name + 'x_train.npy')
        y_train = np.load(file_name + 'y_train.npy')
        x_test = np.load(file_name + 'x_test.npy')
        y_test = np.load(file_name + 'y_test.npy')

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())
    if archive_name == 'HT_archive':
        X, y=ht.getData(name_bm=dataset_name, dir2bms_folder='HT_Data/AES_withTrojan_Set1/',
                    fileNameRange=fileNameRange, sampleInRow=True, use_enabled_trojan=False, fs_expected=2000000000)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
        scaler = preprocessing.StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                       y_test.copy())
    else:
        file_name = root_dir+'/archives/'+archive_name+'/'+dataset_name+'/'+dataset_name
        x_train, y_train = readucr(file_name+'_TRAIN.txt')
        x_test, y_test = readucr(file_name+'_TEST.txt')
        datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
            y_test.copy())

    return datasets_dict


def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e: # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT: # errno.ENOENT = no such file or directory
            raise # re-raise exception if a different error occurred

def read_all_datasets(root_dir,archive_name, split_val = False): 
    datasets_dict = {}

    dataset_names_to_sort = []

    if archive_name == 'mts_archive':

        for dataset_name in MTS_DATASET_NAMES:
            root_dir_dataset = root_dir + '/archives/' + archive_name + '/' + dataset_name + '/'

            x_train = np.load(root_dir_dataset+'x_train.npy')
            y_train = np.load(root_dir_dataset+'y_train.npy')
            x_test = np.load(root_dir_dataset+'x_test.npy')
            y_test = np.load(root_dir_dataset+'y_test.npy')

            datasets_dict[dataset_name] = (x_train.copy(), y_train.copy(), x_test.copy(),
                                           y_test.copy())

    else:
        for dataset_name in DATASET_NAMES:
            root_dir_dataset =root_dir+'/archives/'+archive_name+'/'+dataset_name+'/'
            file_name = root_dir_dataset+dataset_name
            x_train, y_train = readucr(file_name+'_TRAIN')
            x_test, y_test = readucr(file_name+'_TEST')

            datasets_dict[dataset_name] = (x_train.copy(),y_train.copy(),x_test.copy(),
                y_test.copy())

            dataset_names_to_sort.append((dataset_name,len(x_train)))

        dataset_names_to_sort.sort(key=operator.itemgetter(1))

        for i in range(len(DATASET_NAMES)):
            DATASET_NAMES[i] = dataset_names_to_sort[i][0]

    return datasets_dict

def get_func_length(x_train,x_test,func):
    if func==min:
        func_length = np.inf
    else:
        func_length = 0

    n=x_train.shape[0]
    for i in range(n):
        func_length = func(func_length, x_train[i].shape[1])


    n=x_test.shape[0]
    for i in range(n):
        func_length = func(func_length, x_test[i].shape[1])

    return func_length

def transform_to_same_length(x,n_var,max_length):
    n = x.shape[0]

    # the new set in ucr form np array
    ucr_x = np.zeros((n,max_length,n_var),dtype=np.float64)

    # loop through each time series
    for i in range(n):
        mts = x[i]
        curr_length = mts.shape[1]
        idx= np.array(range(curr_length))
        idx_new = np.linspace(0,idx.max(),max_length)
        for j in range(n_var):
            ts = mts[j]
            # linear interpolation
            new_ts = spline(idx,ts,idx_new)
            ucr_x[i,:,j] = new_ts

    return ucr_x

def transform_mts_to_ucr_format():
    mts_root_dir = '/mnt/Other/mtsdata/'
    mts_out_dir = '/mnt/nfs/casimir/archives/mts_archive/'
    for dataset_name in MTS_DATASET_NAMES:
        # print('dataset_name',dataset_name)

        out_dir = mts_out_dir+dataset_name+'/'

        # if create_directory(out_dir) is None:
        #     print('Already_done')
        #     continue

        a = loadmat(mts_root_dir+dataset_name+'/'+dataset_name+'.mat')
        a = a['mts']
        a = a[0,0]

        dt = a.dtype.fields.keys()
        dt = list(dt)

        for i in range(len(dt)):
            if dt[i] == 'train':
                x_train=a[i].reshape(max(a[i].shape))
            elif dt[i] == 'test':
                x_test = a[i].reshape(max(a[i].shape))
            elif dt[i]=='trainlabels':
                y_train = a[i].reshape(max(a[i].shape))
            elif dt[i]=='testlabels':
                y_test = a[i].reshape(max(a[i].shape))

        # x_train = a[1][0]
        # y_train = a[0][:,0]
        # x_test = a[3][0]
        # y_test = a[2][:,0]

        n_var = x_train[0].shape[0]

        max_length = get_func_length(x_train,x_test,func=max)
        min_length = get_func_length(x_train,x_test,func=min)

        print(dataset_name, 'max',max_length,'min', min_length)
        print()
        continue

        x_train = transform_to_same_length(x_train,n_var,max_length)
        x_test = transform_to_same_length(x_test,n_var,max_length)

        # save them
        np.save(out_dir+'x_train.npy',x_train)
        np.save(out_dir+'y_train.npy',y_train)
        np.save(out_dir+'x_test.npy',x_test)
        np.save(out_dir+'y_test.npy',y_test)

        print('Done')

def calculate_metrics(y_true, y_pred,duration = 0,y_true_val=None,y_pred_val=None):
    res = pd.DataFrame(data = np.zeros((1,4),dtype=np.float), index=[0], 
        columns=['precision','accuracy','recall','duration'])
    res['precision'] = precision_score(y_true,y_pred,average='macro')
    res['accuracy'] = accuracy_score(y_true,y_pred)
    
    if not y_true_val is None:
        # this is useful when transfer learning is used with cross validation
        res['accuracy_val'] = accuracy_score(y_true_val,y_pred_val)

    res['recall'] = recall_score(y_true,y_pred,average='macro')
    res['duration'] = duration
    return res

def transform_labels(y_train,y_test,y_val=None):
    """
    Transform label to min equal zero and continuous 
    For example if we have [1,3,4] --->  [0,1,2]
    """
    if not y_val is None : 
        # index for when resplitting the concatenation 
        idx_y_val = len(y_train)
        idx_y_test = idx_y_val + len(y_val)
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit
        y_train_val_test = np.concatenate((y_train,y_val,y_test),axis =0)
        # fit the encoder 
        encoder.fit(y_train_val_test)
        # transform to min zero and continuous labels 
        new_y_train_val_test = encoder.transform(y_train_val_test)
        # resplit the train and test
        new_y_train = new_y_train_val_test[0:idx_y_val]
        new_y_val = new_y_train_val_test[idx_y_val:idx_y_test]
        new_y_test = new_y_train_val_test[idx_y_test:]
        return new_y_train, new_y_val,new_y_test 
    else: 
        # no validation split 
        # init the encoder
        encoder = LabelEncoder()
        # concat train and test to fit 
        y_train_test = np.concatenate((y_train,y_test),axis =0)
        # fit the encoder 
        encoder.fit(y_train_test)
        # transform to min zero and continuous labels 
        new_y_train_test = encoder.transform(y_train_test)
        # resplit the train and test
        new_y_train = new_y_train_test[0:len(y_train)]
        new_y_test = new_y_train_test[len(y_train):]
        return new_y_train, new_y_test   

def generate_results_csv(output_file_name, root_dir):
    res = pd.DataFrame(data = np.zeros((0,7),dtype=np.float), index=[],
        columns=['classifier_name','archive_name','dataset_name',
        'precision','accuracy','recall','duration'])
    for classifier_name in CLASSIFIERS:
        for archive_name in ARCHIVE_NAMES:
            datasets_dict = read_all_datasets(root_dir,archive_name)
            for it in range(ITERATIONS):
                curr_archive_name = archive_name
                if it != 0 :
                    curr_archive_name = curr_archive_name +'_itr_'+str(it)
                for dataset_name in datasets_dict.keys():
                    output_dir = root_dir+'/results/'+classifier_name+'/'\
                    +curr_archive_name+'/'+dataset_name+'/'+'df_metrics.csv'
                    if not os.path.exists(output_dir):
                        continue
                    df_metrics = pd.read_csv(output_dir)
                    df_metrics['classifier_name'] = classifier_name
                    df_metrics['archive_name'] = archive_name
                    df_metrics['dataset_name'] = dataset_name
                    res = pd.concat( (res,df_metrics) ,axis=0,sort=False)

    res.to_csv(root_dir+output_file_name, index = False)
    # aggreagte the accuracy for iterations on same dataset 
    res = pd.DataFrame({
        'accuracy' : res.groupby(
            ['classifier_name','archive_name','dataset_name'])['accuracy'].mean()
        }).reset_index()

    return res 

def plot_and_save(data, file_name,metric='loss'):
    plt.figure()
    plt.plot(data[metric])
    if 'val_' + metric in data:
        plt.plot(data ['val_' + metric])
    plt.title(metric)
    plt.ylabel(metric, fontsize='large')
    plt.xlabel('epoch', fontsize='large')
    if 'val_' + metric in data:
        plt.legend(['train', 'val'], loc='upper left')
    else:
        plt.legend(['train'], loc='upper left')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

def plot_epochs_metric(hist, file_name, metric='loss'):
    plt.figure()
    plt.plot(hist.history[metric])
    plt.plot(hist.history['val_'+metric])
    plt.title('model '+metric)
    plt.ylabel(metric,fontsize='large')
    plt.xlabel('epoch',fontsize='large')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(file_name,bbox_inches='tight')
    plt.close()

def save_logs_t_leNet(output_directory, hist, y_pred, y_true,duration ):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory+'history.csv', index=False)

    df_metrics = calculate_metrics(y_true,y_pred, duration)
    df_metrics.to_csv(output_directory+'df_metrics.csv', index=False)

    index_best_model = hist_df['loss'].idxmin() 
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data = np.zeros((1,6),dtype=np.float) , index = [0], 
        columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc', 
        'best_model_val_acc', 'best_model_learning_rate','best_model_nb_epoch'])
    
    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['acc']
    df_best_model['best_model_val_acc'] = row_best_model['val_acc']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory+'df_best_model.csv', index=False)

    # plot losses 
    plot_epochs_metric(hist, output_directory+'epochs_loss.png')

def save_logs_per_batch(output_directory,metrics, metrics_names=None,epoch_num=0):
    if not os.path.isfile(output_directory+'history_df_metrics_per_batch.csv'):
        with open(output_directory + 'history_df_metrics_per_batch.csv', 'w') as fd:
            fd.write(''.join(str(e) + ',' for e in metrics_names) + 'epoch_num\n')
    with open(output_directory+'history_df_metrics_per_batch.csv', 'a') as fd:
        fd.write(''.join(str(e) + ',' for e in metrics)+str(epoch_num)+'\n')


def extract_and_save_best(output_directory, file_name='history_df_metrics.csv'):
    df_metrics = pd.read_csv(output_directory+file_name)
    max_accuracy_ind = df_metrics[['accuracy']].idxmax()
    df_metrics = df_metrics.iloc[max_accuracy_ind]
    df_metrics.to_csv(output_directory+'best_'+file_name)


def save_logs(output_directory, hist, y_pred, y_true,duration,lr=True,y_true_val=None,y_pred_val=None, epoch_num=None,
              only_metrics=False):
    df_metrics = calculate_metrics(y_true,y_pred, duration,y_true_val, y_pred_val)

    if os.path.isfile(output_directory+'history_df_metrics.csv'):
        with open(output_directory + 'history_df_metrics.csv', 'a') as fd:
            fd.write(str(epoch_num)+','+df_metrics.to_csv(index=False, header=False))
    else:
        with open(output_directory + 'history_df_metrics.csv', 'a') as fd:
            fd.write('epoch_num,' + df_metrics.to_csv(index=False, header=True))

    df_metrics.to_csv(output_directory+'df_metrics.csv', index=False)

    if not only_metrics and hist is not None:
        hist_df = pd.DataFrame(hist.history)
        hist_df.to_csv(output_directory+'history.csv', index=False)

        index_best_model = hist_df['loss'].idxmin()
        row_best_model = hist_df.loc[index_best_model]

        df_best_model = pd.DataFrame(data = np.zeros((1,6),dtype=np.float) , index = [0],
            columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
            'best_model_val_acc', 'best_model_learning_rate','best_model_nb_epoch'])

        df_best_model['best_model_train_loss'] = row_best_model['loss']
        df_best_model['best_model_val_loss'] = row_best_model['val_loss']
        df_best_model['best_model_train_acc'] = row_best_model['acc']
        df_best_model['best_model_val_acc'] = row_best_model['val_acc']
        if lr == True:
            df_best_model['best_model_learning_rate'] = row_best_model['lr']
        df_best_model['best_model_nb_epoch'] = index_best_model

        if os.path.isfile(output_directory+'df_best_model.csv'):
            with open(output_directory+'history_df_best_model.csv', 'a') as fd:
                fd.write(df_best_model.to_csv(index=False,header=not os.path.isfile(output_directory+'history_df_best_model.csv')))

        df_best_model.to_csv(output_directory+'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code 

    # plot losses 
        plot_epochs_metric(hist, output_directory+'epochs_loss.png')

    return df_metrics

def visualize_filter(root_dir):

    import keras
    classifier = 'fcn'
    archive_name = 'UCR_TS_Archive_2015'
    dataset_name = 'Gun_Point'
    datasets_dict = read_dataset(root_dir,archive_name,dataset_name)

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]

    x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)

    model = keras.models.load_model(root_dir+'results/'+classifier+'/'+archive_name+'/'+dataset_name+'/best_model.hdf5')

    # filters
    filters = model.layers[1].get_weights()[0]


    new_input_layer = model.inputs
    new_output_layer = [model.layers[1].output]

    new_feed_forward = keras.backend.function(new_input_layer,new_output_layer)

    classes = np.unique(y_train)

    colors = [(255/255,160/255,14/255),(181/255,87/255,181/255)]
    colors_conv = [(210/255,0/255,0/255),(27/255,32/255,101/255)]

    idx = 10
    idx_filter = 1

    filter = filters[:, 0, idx_filter]

    plt.figure(1)
    plt.plot(filter+0.5, color='gray', label='filter')
    for c in classes:

        c_x_train = x_train[np.where(y_train==c)]
        convolved_filter_1 = new_feed_forward([c_x_train])[0]

        idx_c = int(c)-1

        plt.plot(c_x_train[idx],color=colors[idx_c],label='class'+str(idx_c)+'-raw')
        plt.plot(convolved_filter_1[idx,:,idx_filter],color=colors_conv[idx_c],label='class'+str(idx_c)+'-conv')
        plt.legend()

    plt.savefig('convolution-'+dataset_name+'.pdf')

    return 1

def viz_perf_themes(root_dir,df):
    df_themes = df.copy()
    themes_index = []
    # add the themes
    for dataset_name in df.index:
        themes_index.append(utils.constants.dataset_types[dataset_name])

    themes_index = np.array(themes_index)
    themes, themes_counts = np.unique(themes_index, return_counts=True)
    df_themes.index = themes_index
    df_themes = df_themes.rank(axis=1, method='min', ascending=False)
    df_themes = df_themes.where(df_themes.values == 1)
    df_themes = df_themes.groupby(level=0).sum(axis=1)
    df_themes['#'] = themes_counts

    for classifier in CLASSIFIERS:
        df_themes[classifier] = df_themes[classifier] / df_themes['#'] * 100
    df_themes = df_themes.round(decimals=1)
    df_themes.to_csv(root_dir + 'tab-perf-theme.csv')

def viz_perf_train_size(root_dir,df):
    df_size = df.copy()
    train_sizes = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        train_size = len(datasets_dict[dataset_name][0])
        train_sizes.append(train_size)

    train_sizes = np.array(train_sizes)
    bins = np.array([0, 100, 400, 800, 99999])
    train_size_index = np.digitize(train_sizes, bins)
    train_size_index = bins[train_size_index]

    df_size.index = train_size_index
    df_size = df_size.rank(axis=1, method='min', ascending=False)
    df_size = df_size.groupby(level=0, axis=0).mean()
    df_size = df_size.round(decimals=2)

    print(df_size.to_string())
    df_size.to_csv(root_dir + 'tab-perf-train-size.csv')

def viz_perf_classes(root_dir,df):
    df_classes = df.copy()
    class_numbers = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        train_size = len(np.unique(datasets_dict[dataset_name][1]))
        class_numbers.append(train_size)

    class_numbers = np.array(class_numbers)
    bins = np.array([0, 3, 4, 6, 8, 13, 9999])
    class_numbers_index = np.digitize(class_numbers, bins)
    class_numbers_index = bins[class_numbers_index]

    df_classes.index = class_numbers_index
    df_classes = df_classes.rank(axis=1, method='min', ascending=False)
    df_classes = df_classes.groupby(level=0, axis=0).mean()
    df_classes = df_classes.round(decimals=2)

    print(df_classes.to_string())
    df_classes.to_csv(root_dir + 'tab-perf-classes.csv')

def viz_perf_length(root_dir,df):
    df_lengths = df.copy()
    lengths = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        length = datasets_dict[dataset_name][0].shape[1]
        lengths.append(length)

    lengths = np.array(lengths)
    bins = np.array([0, 81, 251, 451, 700, 1001, 9999])
    lengths_index = np.digitize(lengths, bins)
    lengths_index = bins[lengths_index]

    df_lengths.index = lengths_index
    df_lengths = df_lengths.rank(axis=1, method='min', ascending=False)
    df_lengths = df_lengths.groupby(level=0, axis=0).mean()
    df_lengths = df_lengths.round(decimals=2)

    print(df_lengths.to_string())
    df_lengths.to_csv(root_dir + 'tab-perf-lengths.csv')

def viz_plot(root_dir,df):
    df_lengths = df.copy()
    lengths = []
    datasets_dict_ucr = read_all_datasets(root_dir, archive_name='UCR_TS_Archive_2015')
    datasets_dict_mts = read_all_datasets(root_dir, archive_name='mts_archive')
    datasets_dict = dict(datasets_dict_ucr, **datasets_dict_mts)

    for dataset_name in df.index:
        length = datasets_dict[dataset_name][0].shape[1]
        lengths.append(length)

    lengths_index = np.array(lengths)

    df_lengths.index = lengths_index

    plt.scatter(x=df_lengths['fcn'], y=df_lengths['resnet'])
    plt.ylim(ymin=0,ymax=1.05)
    plt.xlim(xmin=0,xmax=1.05)
    # df_lengths['fcn']
    plt.savefig(root_dir+'plot.pdf')

def viz_for_survey_paper(root_dir, filename='results-ucr-mts.csv'):
    df = pd.read_csv(root_dir+filename, index_col=0)
    df = df.T
    df = df.round(decimals=2)

    # get table performance per themes
    # viz_perf_themes(root_dir,df)

    # get table performance per train size
    # viz_perf_train_size(root_dir,df)

    # get table performance per classes
    # viz_perf_classes(root_dir,df)

    # get table performance per length
    # viz_perf_length(root_dir,df)

    # get plot
    viz_plot(root_dir,df)


def viz_cam(root_dir):
    import keras
    import sklearn
    classifier = 'fcn'
    archive_name = 'UCR_TS_Archive_2015'
    dataset_name = 'Meat'


    if dataset_name == 'Gun_Point':
        save_name = 'GunPoint'
    else:
        save_name = dataset_name
    max_length = 2000
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    y_test  = datasets_dict[dataset_name][3]

    # transform to binary labels
    enc = sklearn.preprocessing.OneHotEncoder()
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train_binary = enc.transform(y_train.reshape(-1, 1)).toarray()

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)

    model = keras.models.load_model(
        root_dir + 'results/' + classifier + '/' + archive_name + '/' + dataset_name + '/best_model.hdf5')

    # filters
    w_k_c = model.layers[-1].get_weights()[0] #  weights for each filter k for each class c

    # the same input
    new_input_layer = model.inputs
    # output is both the original as well as the before last layer
    new_output_layer = [model.layers[-3].output, model.layers[-1].output]

    new_feed_forward = keras.backend.function(new_input_layer, new_output_layer)

    classes = np.unique(y_train)

    for c in classes:
        plt.figure()
        count =0
        c_x_train = x_train[np.where(y_train==c)]
        for ts in c_x_train:
            ts = ts.reshape(1,-1,1)
            [conv_out, predicted] = new_feed_forward([ts])
            pred_label = np.argmax(predicted)
            orig_label = np.argmax(enc.transform([[c]]))
            if pred_label == orig_label:
                cas = np.zeros(dtype=np.float, shape=(conv_out.shape[1]))
                for k, w in enumerate(w_k_c[:, orig_label]):
                    cas += w * conv_out[0,:, k]

                minimum = np.min(cas)

                cas = cas - minimum

                cas = cas / max(cas)
                cas = cas * 100

                x = np.linspace(0,ts.shape[1]-1,max_length,endpoint=True)
                # linear interpolation to smooth
                y = spline(range(ts.shape[1]),ts[0,:,0],x)
                if any(y<-2.2  ):
                    continue
                cas = spline(range(ts.shape[1]),cas,x)
                cas = cas.astype(int)
                plt.scatter(x=x,y=y,c=cas,cmap='jet', marker='.',s=1,vmin=0,vmax = 100)
                if dataset_name == 'Gun_Point':
                    if c ==1:
                        plt.yticks([-1.0,0.0,1.0,2.0])
                    else:
                        plt.yticks([-2,-1.0,0.0,1.0,2.0])
                count += 1

        cbar = plt.colorbar()
        # cbar.ax.set_yticklabels([100,75,50,25,0])
        plt.savefig(root_dir+'/temp/'+classifier+'-cam-'+save_name+'-class-'+str(int(c))+'.png',bbox_inches='tight',dpi=1080)


if __name__ == "__main__":
    summrize(results_dir='Y:/Google-Drive/HT_results/htm/numenta',input_res='finding_best_threshold_results.csv',output_res='all_finding_best_threshold_results')