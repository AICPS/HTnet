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



def INITIALIZE(func):
    setattr(func,"sample_number",0)
    setattr(func,"second",0)
    setattr(func,"minute",0)
    setattr(func,"hour",0)
    setattr(func,"day",1)
    setattr(func,"month",1)
    setattr(func,"year",2019)
    return func

"""
my_clock () is tick based clock, with every tick it increments the sample_number varible and so on
"""
@INITIALIZE
def my_clock():

    month_names = ['January','February','March','April','May','June','July','August','September','October','November','December']
    month_days = [31,27,31,30,31,30,31,31,30,31,30,31]

    my_clock.sample_number = my_clock.sample_number + 1
    #print(my_clock.sample_number)
    if my_clock.sample_number == 1:
        my_clock.sample_number = 0
        my_clock.second = my_clock.second + 1
        #print('second',my_clock.second)

    if my_clock.second == 60:
        my_clock.second = 0
        my_clock.minute = my_clock.minute + 1
        #print('minute',my_clock.minute)

    if my_clock.minute == 60:
        my_clock.minute = 0
        my_clock.hour = my_clock.hour + 1
        print('hour',my_clock.hour)

    if my_clock.hour == 24:
        my_clock.hour = 0
        my_clock.day = my_clock.day + 1
        print('day',my_clock.day)

    if (my_clock.day == (month_days [my_clock.month-1]+1)):
        my_clock.day = 1
        my_clock.month = my_clock.month + 1
        print('month',my_clock.month)
        #exit()

    if (my_clock.month == 13):
        my_clock.month = 1
        my_clock.year = my_clock.year + 1
        print('year',my_clock.year)

    str_year = str(my_clock.year)

    if my_clock.month < 10:
        str_month = '0'+str(my_clock.month)
    elif my_clock.month >= 10:
        str_month = str(my_clock.month)

    if my_clock.day < 10:
        str_day = '0'+str(my_clock.day)
    elif my_clock.day >= 10:
        str_day = str(my_clock.day)

    if my_clock.hour < 10:
        str_hour = '0'+str(my_clock.hour)
    elif my_clock.hour >= 10:
        str_hour = str(my_clock.hour)

    if my_clock.minute < 10:
        str_minute = '0'+str(my_clock.minute)
    elif my_clock.minute >= 10:
        str_minute = str(my_clock.minute)

    if my_clock.second < 10:
        str_second = '0'+str(my_clock.second)
    elif my_clock.second >= 10:
        str_second = str(my_clock.second)

    return str_month + '/' + str_day + '/' + str_year + ' ' + str_hour + ':' + str_minute + ':' + str_second


def _merge(start_range=101, sample_no=4, output_file_name='"./combined_HT-free.csv"',
           dir2bms_folder='HT_Data/AES_withTrojan_Set1/',
           name_bms=['AES-T400'], folder_numbers=[1], with_anomaly_data=False):
    file_names = get_list_of_files_HT(dir2bms_folder,name_bms, folder_numbers, with_anomaly_data)
    file_names = file_names[start_range:start_range + sample_no]
    print(file_names)
    for file_name in file_names:
        with open(file_name) as original_file:
            original_file_reader = csv.reader(original_file, delimiter=',')
            with open(output_file_name,'a',newline='') as new_file:
                writer = csv.writer(new_file, delimiter=',')
                for row in original_file_reader:
                    writer.writerow([float(row[0])])


def add_timestamp_csv(input_file_name, output_file_name):
    with open(input_file_name) as old_file:
        old_file_object = csv.reader(old_file, delimiter=',')
        add_row=['timestamp','value']
        with open(output_file_name,'w',newline='') as new_file:
            new_file_object = csv.writer(new_file, delimiter=',')
            new_file_object.writerow(add_row)
            time = '01/01/2019 00:00:00'
            tick = 0
            #row_number = 0
            for row in old_file_object:
                if 'timestamp' not in row:
                    time = my_clock()
                    add_row = [time, row[0]]
                    new_file_object.writerow(add_row)


def merge_samples(dir2bms_folder,name_bms,
                  uninfected_start_range=1,uninfected_samples_no=10, infected_start_range=1,infected_samples_no=10,
                  combined_file_name='./combined_HT.csv',
                  ht_free_file_name="./combined_HT-free.csv",
                  ht_infected_file_name = "./combined_HT-infected.csv",
                  two_seperate_files=True,
                  folder_numbers=[1]):

    if not two_seperate_files:
        ht_free_file_name, ht_infected_file_name = combined_file_name, combined_file_name

    for output_file_name in [combined_file_name,ht_free_file_name,ht_infected_file_name]:
        try:
            os.remove(output_file_name)
            print(output_file_name+' DELETED')
        except:
            print(output_file_name+' NOT PRESENT')

    # merge_uninfected_csv
    _merge(start_range=uninfected_start_range, sample_no=uninfected_samples_no, output_file_name=ht_free_file_name,
           with_anomaly_data=False,dir2bms_folder=dir2bms_folder, name_bms=name_bms, folder_numbers=folder_numbers)
    # merge_infected_csv
    _merge(start_range=infected_start_range, sample_no=infected_samples_no, output_file_name=ht_infected_file_name,
           with_anomaly_data=True,dir2bms_folder=dir2bms_folder, name_bms=name_bms, folder_numbers=folder_numbers)

    for file_name in set([ht_free_file_name,ht_infected_file_name]):
        add_timestamp_csv(file_name,'tmp.csv')
        os.remove(file_name)
        os.rename('tmp.csv',file_name)


def _add_noise_single_file(input_file,output_file, mu, sigma):
    clean_signal = pd.read_csv(input_file)
    noise = np.random.normal(mu, sigma, clean_signal.shape)
    noisy_signal = clean_signal + noise
    noisy_signal.to_csv(output_file,header= False, index= False)


def add_noise(dir2bms_folder= 'HT_Data/AES_withTrojan_Set1/',
              output_folder='HT_Data/AES_withTrojan_Set5/', name_bms=['AES-T400'],
              number_of_samples =1000,
              mu = 0, sigma = 0.1, do_not_add_mu_sigma_to_name= False,
              folder_numbers=[1]):

    output_folder = _fix_the_path_for_the_os(output_folder)
    for folder_number in folder_numbers:
        for name_bm in name_bms:
            if do_not_add_mu_sigma_to_name:
                output_bm_folder = output_folder + name_bm.split('_')[0] + '_' + str(number_of_samples) + 'p' + '/'
            else:
                output_bm_folder = output_folder + name_bm.split('_')[0] + '_mu_'+str(mu) + '_sigma_'+str(sigma) +'_'+str(number_of_samples) + 'p' + '/'


            # For the Disabled folder
            if os.path.isdir(output_bm_folder):
                print(output_bm_folder, 'exist')
                continue
            file_names = get_list_of_files_HT(dir2bms_folder,name_bms =[name_bm], folder_numbers=folder_numbers,
                                              with_anomaly_data = False)
            file_names = file_names [0:number_of_samples]
            inner_folder_name = name_bm.split('_')[0]+'+TrojanDisabled_'+str(folder_number)
            output_folder_final = output_bm_folder +inner_folder_name+'/'+inner_folder_name+'/'
            path = Path(output_folder_final)
            path.mkdir(parents=True, exist_ok=True)
            print('Output folder: ',path)
            for input_file in file_names:
                output_file= output_folder_final +input_file.split('/')[-1]
                _add_noise_single_file(input_file,output_file,mu,sigma)

            # For the Triggered folder
            file_names = get_list_of_files_HT(dir2bms_folder,name_bms =[name_bm], folder_numbers=folder_numbers,
                                              with_anomaly_data = True)
            file_names = file_names[0:number_of_samples]
            inner_folder_name = name_bm.split('_')[0]+'+TrojanTriggered_'+str(folder_number)
            output_folder_final = output_bm_folder+inner_folder_name+'/'+inner_folder_name+'/'
            path = Path(output_folder_final)
            path.mkdir(parents=True, exist_ok=True)
            for input_file in file_names:
                output_file= output_folder_final +input_file.split('/')[-1]
                _add_noise_single_file(input_file,output_file,mu,sigma)




if __name__ == '__main__':

    dir2bms_folder = 'HT_Data/AES_withTrojan_Set2/'
    output_folder = 'HT_Data/AES_withTrojan_Set11/'

    dataset_name = 'AES-T500'
    dataset_names = [dataset_name]
    #
    # dataset_names = ['AES-T400', 'AES-T500', 'AES-T600', 'AES-T700', 'AES-T800',
    #                  'AES-T1000', 'AES-T1100', 'AES-T1300',  'AES-T1400',
    #                  'AES-T1600','AES-T1800', 'AES-T2000']

    # dataset_names = ['AES-T800+Temp25','AES-T800+Temp35','AES-T800+Temp45','AES-T800+Temp55','AES-T800+Temp65',
    #                  'AES-T800+Temp75','AES-T800+Temp85']

    # for mu in [0,5.25,5.5,5.75,6,6.25,6.5,6.75,7,7.25,7.5,7.75,8,8.25,8.5,8.75,9,9.25,9.5,9.75,10]:
        # for sigma in [10,15,20,5,25,0, 0.2, 0.4, 0.6, 0.8, 10]:
        # for sigma in [10, 15, 20, 5, 25, 0, 0.2, 0.4, 0.6, 0.8,1,2,3,4,5,6,7,8,9,10]:
     #   for sigma in [0, 0.2, 0.4, 0.6, 0.8,1]:

    dataset_names = ['AES-T700+SQRT2bits_1000p','AES-T700+SQRT4bits_1000p','AES-T700+SQRT8bits_1000p',
                     'AES-T700+SQRT16bits_1000p','AES-T700+SQRT32bits_1000p','AES-T700+SQRT64bits_1000p',
                     'AES-T700+SQRT128bits_1000p','AES-T700+SQRT256bits_1000p']
    add_noise(dir2bms_folder=dir2bms_folder,
              output_folder=output_folder, name_bms=dataset_names, number_of_samples =1000,
              mu=0, sigma=0, do_not_add_mu_sigma_to_name = True,
              folder_numbers=[1])

    # for mu in [0,0.4,0.8,1.2,1.6]:
    #     for sigma in [0,0.4,0.8,1.2,1.6]:
    #         add_noise(dir2bms_folder=dir2bms_folder,
    #                       output_folder=output_folder, name_bms=dataset_names, number_of_samples =1000,
    #                       mu=mu, sigma=sigma, do_not_add_mu_sigma_to_name = False,
    #                       folder_numbers=[1])
    #
    # for mu in [0,0.4,0.8,1.2,1.6,0.2,0.6,1,1.4,1.8,2,2.4,2.8,3.2,3.6,4]:
    #     for sigma in [0,0.4,0.8,1.2,1.6,0.2,0.6,1,1.4,1.8,2,2.4,2.8,3.2,3.6,4]:
    #         add_noise(dir2bms_folder=dir2bms_folder,
    #                       output_folder=output_folder, name_bms=dataset_names, number_of_samples =1000,
    #                       mu=mu, sigma=sigma, do_not_add_mu_sigma_to_name = False,
    #                       folder_numbers=[1])

    # for mu in [0]:
    #     for sigma in [0,1,2,3,4,5,6,7,8,9,10,15,20,25]: # 1.25, 1.5,1.75, 2, 2.25, 2.5,2.75,3, 3.25, 3.5,3.75,4,4.25,4.5,
    #         add_noise(dir2bms_folder=dir2bms_folder,
    #                       output_folder=output_folder, name_bms=dataset_names, number_of_samples =1000,
    #                       mu=mu, sigma=sigma,
    #                       folder_numbers=[1])
    #
    # for mu in [0,1,2,3,4,5,6,7,8,9,10,15,20,25]:
    #     for sigma in [0]: # 1.25, 1.5,1.75, 2, 2.25, 2.5,2.75,3, 3.25, 3.5,3.75,4,4.25,4.5,
    #         add_noise(dir2bms_folder=dir2bms_folder,
    #                       output_folder=output_folder, name_bms=dataset_names, number_of_samples =1000,
    #                       mu=mu, sigma=sigma,
    #                       folder_numbers=[1])

    # merge_samples(dir2bms_folder, name_bms=dataset_names,
    #               uninfected_start_range=1, uninfected_samples_no=10, infected_start_range=1,infected_samples_no=3,
    #               ht_free_file_name="./combined_HT-free.csv",
    #               ht_infected_file_name="./combined_HT-infected.csv",
    #               two_seperate_files=False)


    # add_noise()

