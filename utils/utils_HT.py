# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 15:08:03 2018

@author: Sina
"""
import pandas as pd
import os
import numpy as np
import scipy.io.wavfile as wavefile
import re
import sys
import platform
import pickle
import sklearn


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def _fixThePathForTheOS(d):
    if platform.system() != 'Windows':
        if '/mnt/NAS/Members/root/' not in d:
            d = '/mnt/NAS/Members/root/' + d
    else:
        if 'Y:/' not in d:
            d = 'Y:/' + d
    return d



def _getDataInRows_np_array(name_bm=None, dir2bms_folder=None,
                       fileNameRange=None, use_enabled_trojan=None, fs_expected=None):
        fs = fs_expected

        dir2bms_folder = _fixThePathForTheOS(dir2bms_folder)
        label_num = 0
        X = []
        y = []
        y_name = []
        list_of_folders, types = _get_immediate_subdirectories_and_set_of_types(dir2bms_folder + name_bm)
        num_of_folders = 1
        print(str(types) + ' len of iterations: ' + str(num_of_folders))

        for className in types:
            if not use_enabled_trojan and 'Enabled' in className:
                continue
            for folder_number in range(1, num_of_folders + 1):
                foldername = className + '_' + str(folder_number)
                dir2data = dir2bms_folder + name_bm + '/' + foldername + '/' + foldername + '/'
                print(dir2data)

                sample_names = _get_csv_files_sorted_list(dir2data)
                for sample_name in sample_names:
                    # print(sample_name)
                    if fileNameRange is None or (int(_get_numbers_in_file_name(sample_name)[-1]) in fileNameRange):
                        x_tmp = np.loadtxt(sample_name, delimiter = '\n')
                        if len(x_tmp.shape) > 1:
                            print('bad csv data shape')
                        #                        return x_tmp,y
                        # x_tmp = x_tmp / max([abs(min(x_tmp)), abs(max(x_tmp))])
                        # x_tmp = (x_tmp - x_tmp.mean)/
                        X.append(x_tmp)
                        y_name.append(className)
                        y.append(label_num)

            label_num = label_num + 1


        y = np.asarray(y)
        X = np.asarray(X)

        return X, y


def _getDataInRows(name_bm=None, dir2bms_folder=None,
            fileNameRange=None, use_enabled_trojan=None, fs_expected=None):
    fs = fs_expected
    dir2bms_folder = _fixThePathForTheOS(dir2bms_folder)
    label_num=0
    X = []
    y = []
    y_name=[]

    list_of_folders, types = _get_immediate_subdirectories_and_set_of_types(dir2bms_folder + name_bm)
    num_of_folders = 1
    print(str(types) + ' len of iterations: ' + str(num_of_folders))

    for className in types:
        if not use_enabled_trojan and 'Enabled' in className:
            continue
        for folder_number in range(1, num_of_folders + 1):
            foldername = className + '_' + str(folder_number)
            dir2data = dir2bms_folder + name_bm + '/' + foldername + '/' + foldername + '/'
            print(dir2data)

            sample_names = _get_csv_files_sorted_list(dir2data)
            for sample_name in sample_names:
                #            print(wavFileName)
                if fileNameRange is None or (int(_get_numbers_in_file_name(sample_name)[-1]) in fileNameRange):
                    x_tmp = pd.read_csv(sample_name, names=['data'], dtype='float32')
                    x_tmp = pd.Series( x_tmp['data'])
                    if len(x_tmp.shape) > 1:
                        print('bad csv data shape')
                    x_tmp = x_tmp.astype('float32')
                    #                        return x_tmp,y
                    # x_tmp = x_tmp / max([abs(min(x_tmp)), abs(max(x_tmp))])
                    # x_tmp = (x_tmp - x_tmp.mean)/
                    X.append(list(x_tmp))
                    y_name.append(className)
                    y.append(label_num)

        label_num = label_num+1
    y = pd.Series(y)
    X = pd.DataFrame(X)

    return X, y

def getData(name_bm='AES-T1000', dir2bms_folder='HT_Data/AES_withTrojan_Set2/',
            fileNameRange=None, sampleInRow=True, use_enabled_trojan=False, np_array_format=True,fs_expected=2000000000):
    fs = fs_expected
    if sampleInRow and np_array_format:
        return _getDataInRows_np_array(name_bm=name_bm, dir2bms_folder=dir2bms_folder, fileNameRange=fileNameRange,
                              use_enabled_trojan=use_enabled_trojan, fs_expected=fs_expected)
    if sampleInRow:
        return _getDataInRows(name_bm=name_bm, dir2bms_folder=dir2bms_folder, fileNameRange=fileNameRange,
                              use_enabled_trojan=use_enabled_trojan, fs_expected=fs_expected)

    dir2bms_folder = _fixThePathForTheOS(dir2bms_folder)
    # id_num=0
    X = pd.DataFrame(columns=['id', 'time', 'sig_power'])
    y = []
    ID = 0

    list_of_folders, types = _get_immediate_subdirectories_and_set_of_types(dir2bms_folder + name_bm)
    num_of_folders = 1  # int(len(list_of_folders)/3)
    print(str(types) + ' len of iterations: ' + str(num_of_folders))

    for className in types:
        if not use_enabled_trojan and 'Enabled' in className:
            continue
        for folder_number in range(1, num_of_folders + 1):
            foldername = className + '_' + str(folder_number)
            dir2data = dir2bms_folder + name_bm + '/' + foldername + '/' + foldername + '/'
            print(dir2data)

            sample_names = _get_csv_files_sorted_list(dir2data)
            for sample_name in sample_names:
                #            print(wavFileName)
                if fileNameRange is None or (int(_get_numbers_in_file_name(sample_name)[-1]) in fileNameRange):
                    x_tmp = pd.read_csv(sample_name, names=['data'], dtype='float32')
                    x_tmp = x_tmp['data']
                    if len(x_tmp.shape) > 1:
                        print('bad csv data shape')
                    x_tmp = x_tmp.astype('float32')
                    #                        return x_tmp,y
                    x_tmp = x_tmp / max([abs(min(x_tmp)), abs(max(x_tmp))])

                    t = np.linspace(0, round(len(x_tmp) / fs), num=len(x_tmp))
                    d = {'id': pd.Series(ID, index=list(range(len(x_tmp))), dtype='int64'),
                         'time': pd.Series(t),
                         'sig_power': pd.Series(x_tmp)}
                    d = pd.DataFrame(d)
                    X = X.append(d, ignore_index=True)
                    ID = ID + 1
                    y.append(className)

    y = pd.Series(y)
    return X, y


def _get_immediate_subdirectories_and_set_of_types(a_dir):
    list_of_folders = _get_immediate_subdirectories(a_dir)
    if 'not valid' in list_of_folders:
        list_of_folders.remove('not valid')
    types = []
    for f in list_of_folders:
        types.append(f.split('_')[0])

    types = set(types)
    return list_of_folders, types


def _get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def _get_csv_files_sorted_list(a_dir):
    l = [a_dir + name for name in os.listdir(a_dir) if name.endswith('.csv')]
    #    l.sort(key=_natural_keys)
    return l


def _get_numbers_in_file_name(s):
    return [int(s) for s in re.split('_|\s+|\.', s) if s.isdigit()]


if __name__ == "__main__":
    X, y = getData(fileNameRange=range(5, 1000))
    print(X.shape)
