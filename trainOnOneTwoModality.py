from utils.utils import create_directory
from utils.utils import get_splited_list_of_files_and_scaler_HT
from utils.utils import Data_Generator, summrize, Data_Generator_Combiner

import numpy as np
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
# os.environ["CUDA_VISIBLE_DEVICES"]="1";

import tensorflow as tf
import keras
from random import shuffle
import utils.utils_HT as ht
from classifiers import htnetNarrowed as htnet

if __name__ == '__main__':
    root_dir = 'HT_results/'
    # dir2bms_folder = 'HT_Data/AES_withTrojan_Set7/'
    dir2bms_folder_ls = ['HT_Data/AES_withTrojan_Set1/',
                         'HT_Data/AES_withTrojan_Set7/']
    classifier_name = 'htnetNarrowed'
    base_folder = '/transfer_learning_PWEM/'  #'/data_generator_results/'
    archive_name = 'HT_trained_and_validated_on_one'
    # dataset_names = ['AES-T1300', 'AES-T500', 'AES-T600', 'AES-T2000', 'AES-T400',
    #                      'AES-T700', 'AES-T800','AES-T1000', 'AES-T1100', 'AES-T1400',
    #                      'AES-T1600', 'AES-T1800']

    dataset_names = ['AES-T1600','AES-T1300','AES-T1000', 'AES-T1100', 'AES-T1400', 'AES-T1800',
                         'AES-T500', 'AES-T600', 'AES-T2000', 'AES-T400',
                         'AES-T700', 'AES-T800']
    number_of_samples_per_folder = 1000


    batch_size = 12
    nb_epochs = 80
    workers = 6
    patience = 10


    # fix data set names for added circuit
    # dataset_names=[]
    # added_bits = [2, 4, 8, 16, 32, 64, 128, 256, 35, 45, 55, 65, 75, 85]
    # for dataset_name in dataset_names_base:
    #     for added_bit in added_bits:
    #         dataset_names
    #         dataset_names.append(dataset_name.split('_')[0]+'+Temp'+str(added_bit)+'_'+dataset_name.split('_')[1])
    #         dataset_names.append(dataset_name.split('_')[0] + '+SQRT' + str(added_bit) + 'bits_' + dataset_name.split('_')[1])

    for dataset_name in dataset_names:
        output_directory = root_dir + base_folder + classifier_name + '/' + archive_name + '/' + dataset_name + '/'
        if os.path.isdir(output_directory):
            print('The results folder exists: ' + output_directory)
            continue
        input_shapes = 0
        train_generators = []
        val_generators = []
        for dir2bms_folder in dir2bms_folder_ls:
            if not os.path.isdir(ht._fixThePathForTheOS(dir2bms_folder) + dataset_name):
                print('There is no folder with name: ' + ht._fixThePathForTheOS(dir2bms_folder) + dataset_name)
                continue
            create_directory(output_directory)
            dirs_to_files_train, dirs_to_files_test, y_train, y_test, scaler, input_shape = get_splited_list_of_files_and_scaler_HT(
                dir2bms_folder=dir2bms_folder, name_bms=[dataset_name], number_of_training_for_scaler=300 ,
                number_of_samples_per_folder =number_of_samples_per_folder)

            train_generators.append(Data_Generator(dirs_to_files_train, y_train, batch_size=batch_size,
                                             dir2bms_folder=dir2bms_folder, scaler=scaler))
            val_generators.append(Data_Generator(dirs_to_files_test, y_test, batch_size=batch_size,
                                           dir2bms_folder=dir2bms_folder, scaler=scaler))

            input_shapes = input_shape[0] +input_shapes
            print(input_shape)
            print('number of training samples from: ', dir2bms_folder,len(dirs_to_files_train))


        input_shape = (input_shapes,1)
        train_generator = Data_Generator_Combiner(train_generators)
        val_generator = Data_Generator_Combiner(val_generators)

        print('data generators are made')

        nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

        classifier = htnet.Classifier_HTNET(output_directory, input_shape, nb_classes, verbose=1)

        readme_file = open(output_directory + "/training_readme.txt", "w")
        readme_file.write("batch_size: {}, nb_epochs: {}, workers: {}\n"
                          "Training Datasets: {}\n"
                          "number of training files: {} , number of test files: {}\n".format(batch_size, nb_epochs,
                                                                                             workers, str(dataset_name),
                                                                                             len(dirs_to_files_train),
                                                                                             len(dirs_to_files_test)))
        classifier.model.summary(print_fn=lambda x: readme_file.write(x + '\n'))
        readme_file.close()

        print('started training for: ', dataset_name)
        classifier.fit_genrator(train_generator, val_generator, y_test, nb_epochs=nb_epochs, workers=workers, patience=patience)

        output_directory = root_dir + base_folder + classifier_name + '/' + archive_name + '/'
        summrize(output_directory, input_res='df_best_model.csv', output_res='all_best_model.csv')
        summrize(output_directory, input_res='df_metrics.csv', output_res='all_metrics.csv')