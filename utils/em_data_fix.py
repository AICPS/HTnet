import os, platform,csv,re
import pandas as pd
import numpy as np
from pathlib import Path

def _fix_the_path_for_the_os(d):
    if platform.system() != 'Windows':
        if '/mnt/NAS/Members/root/' not in d:
            d = '/mnt/NAS/Members/root/' + d
    else:
        if 'Y:/' not in d:
            d = 'Y:/' + d
    return d


def _get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def _get_immediate_subdirectories_and_set_of_types(a_dir):
    list_of_folders = _get_immediate_subdirectories(a_dir)
    if 'not valid' in list_of_folders:
        list_of_folders.remove('not valid')
    types = []
    for f in list_of_folders:
        types.append(f.split('_')[0])

    types = set(types)
    return list_of_folders, types


def _atoi(text):
    return int(text) if text.isdigit() else text


def _natural_keys(text):
    return [_atoi(c) for c in re.split(r'(\d+)', text) ]


def _get_csv_files_sorted_list(a_dir):
    l = [a_dir + name for name in os.listdir(a_dir) if name.endswith('.csv')]
    l.sort(key=_natural_keys)
    return l


def get_list_of_files_HT(dir2bms_folder='HT_Data/AES_withTrojan_Set1/',
                          name_bms=['AES-T400'], folder_numbers=[1], with_anomaly_data= False):
    dir2bms_folder = _fix_the_path_for_the_os(dir2bms_folder)
    dirs_to_files = []
    for name_bm in name_bms:
        list_of_folders, types = _get_immediate_subdirectories_and_set_of_types(dir2bms_folder + name_bm)
        for className in types:
            if (not with_anomaly_data and 'Disabled' in className)or (with_anomaly_data and 'Triggered' in className):
                for folder_number in folder_numbers:
                    folder_name = className + '_' + str(folder_number)
                    dir2data = dir2bms_folder + name_bm + '/' + folder_name + '/' + folder_name + '/'
                    print(dir2data)
                    sample_names = _get_csv_files_sorted_list(dir2data)
                    for sample_name in sample_names:
                        dirs_to_files.append(sample_name)

    return dirs_to_files



def fix_em_file(input_file,output_file):
    df= pd.read_csv(input_file,header=None)
    # print(df)
    df=df.drop(columns=[0])
    df = df.drop(df.index[0:16])
    # print(df)
    df.to_csv(output_file,header=False,index=False)


def fix_em_files(dir2bms_folder= 'HT_Data/AES_withTrojan_Set8/',
              output_folder='HT_Data/AES_withTrojan_Set10/', name_bms=['AES-T400'],
              folder_numbers=[1]):

    output_folder = _fix_the_path_for_the_os(output_folder)
    for folder_number in folder_numbers:
        for name_bm in name_bms:

            output_bm_folder = output_folder + name_bm + '/'
            # For the Disabled folder
            if os.path.isdir(output_bm_folder):
                print(output_bm_folder, 'exist')
                continue
            file_names = get_list_of_files_HT(dir2bms_folder,name_bms =[name_bm], folder_numbers=folder_numbers,
                                              with_anomaly_data = False)
            inner_folder_name = name_bm.split('_')[0]+'+TrojanDisabled_'+str(folder_number)
            output_folder_final = output_bm_folder +inner_folder_name+'/'+inner_folder_name+'/'
            path = Path(output_folder_final)
            path.mkdir(parents=True, exist_ok=True)
            print('Output folder: ',path)
            for input_file in file_names:
                output_file= output_folder_final +input_file.split('/')[-1]
                fix_em_file(input_file,output_file)

            # For the Triggered folder
            file_names = get_list_of_files_HT(dir2bms_folder,name_bms =[name_bm], folder_numbers=folder_numbers,
                                              with_anomaly_data = True)
            inner_folder_name = name_bm.split('_')[0]+'+TrojanTriggered_'+str(folder_number)
            output_folder_final = output_bm_folder+inner_folder_name+'/'+inner_folder_name+'/'
            path = Path(output_folder_final)
            path.mkdir(parents=True, exist_ok=True)
            for input_file in file_names:
                output_file= output_folder_final +input_file.split('/')[-1]
                fix_em_file(input_file,output_file)




if __name__ == '__main__':

    dir2bms_folder = 'HT_Data/AES_withTrojan_Set4/'
    output_folder = 'HT_Data/AES_withTrojan_Set7/'
    #
    # dataset_names = [ 'AES-T500', 'AES-T400', 'AES-T700', 'AES-T800',
    #                           'AES-T1000', 'AES-T1100','AES-T2000','AES-T1800','AES-T1600','AES-T1400_1000p','AES-T1400_1000p','AES-T600_1000p']

    # dataset_names = ['AES-T800+Temp25','AES-T800+Temp35_1000p', 'AES-T800+Temp45_1000p','AES-T800+Temp55_1000p', 'AES-T800+Temp65_1000p',
    #                  'AES-T800+Temp75_1000p']
    # dataset_names = ['AES-T700+SQRT2bits_1000p','AES-T700+SQRT4bits_1000p', 'AES-T700+SQRT8bits_1000p','AES-T700+SQRT16bits_1000p', 'AES-T700+SQRT32bits_1000p',
    #                  'AES-T700+SQRT64bits_1000p','AES-T700+SQRT128bits_1000p']
    dataset_names = ['AES-T1300_1000p']
    #, 'AES-T800+Temp25']
    fix_em_files(dir2bms_folder=dir2bms_folder, output_folder=output_folder, name_bms=dataset_names, folder_numbers=[1])

