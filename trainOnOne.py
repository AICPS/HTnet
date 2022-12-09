from utils.utils import create_directory
from utils.utils import get_splited_list_of_files_and_scaler_HT
from utils.utils import Data_Generator, summrize
from classifiers import htnet
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";

import tensorflow as tf
import keras
from random import shuffle
import utils.utils_HT as ht

if __name__ == '__main__':
    root_dir = 'HT_results/'
    dir2bms_folder = 'HT_Data/AES_withTrojan_Set1/'
    top_folder= '/base/'
    classifier_name = "htnet"
    archive_name = 'with_retraining'
    # dataset_names_base = ['AES-T500_1000p']
    number_of_samples = 1000
    dataset_names = []
    dataset_names_base = ['AES-T700_1000p','AES-T800_1000p','AES-T500', 'AES-T400', 'AES-T700',
                          'AES-T600', 'AES-T800', 'AES-T1000', 'AES-T1100',
                          'AES-T1300', 'AES-T1400', 'AES-T1600', 'AES-T1800',
                          'AES-T2000']

    dataset_names.extend(dataset_names_base)
    for dataset_name in dataset_names:
        if not os.path.isdir(ht._fixThePathForTheOS(dir2bms_folder)+dataset_name):
            print('There is no folder with name: ' + ht._fixThePathForTheOS(dir2bms_folder)+dataset_name)
            continue

        output_directory = root_dir + top_folder + classifier_name + '/' + archive_name + '/' + dataset_name+ '/'
        if os.path.isdir(output_directory):
            print('The results folder exists: ' + output_directory)
            continue
        create_directory(output_directory)

        dirs_to_files_train, dirs_to_files_test, y_train, y_test, scaler, input_shape = \
            get_splited_list_of_files_and_scaler_HT(dir2bms_folder=dir2bms_folder, name_bms=[dataset_name],
                                                    number_of_training_for_scaler=100, number_of_samples = number_of_samples)

        batch_size = 12
        nb_epochs = 50
        workers = 6

        print('scaler:', scaler)

        train_generator = Data_Generator(dirs_to_files_train, y_train, batch_size=batch_size,
                                         dir2bms_folder=dir2bms_folder, scaler=scaler)
        val_generator = Data_Generator(dirs_to_files_test, y_test, batch_size=batch_size,
                                       dir2bms_folder=dir2bms_folder, scaler=scaler)


        nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        keras.backend.tensorflow_backend.set_session(tf.Session(config=config))



        classifier = htnet.Classifier_HTNET(output_directory, input_shape, nb_classes, verbose=1)
        readme_file = open(output_directory + "/training_readme.txt", "w")
        readme_file.write("batch_size: {}, nb_epochs: {}, workers: {}\n"
                          "Training Datasets: {}\n".format(batch_size, nb_epochs, workers, str(dataset_name)))
        classifier.model.summary(print_fn=lambda x: readme_file.write(x + '\n'))
        readme_file.close()

        classifier.fit_genrator(train_generator, val_generator, y_test, nb_epochs=nb_epochs, workers=workers)
        output_directory = root_dir + top_folder + classifier_name + '/' + archive_name + '/'
        summrize(output_directory, input_res='df_best_model.csv', output_res='all_best_model.csv')
        summrize(output_directory, input_res='df_metrics.csv', output_res='all_metrics.csv')