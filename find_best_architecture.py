
from utils.utils import create_directory,summrize
from utils.utils import get_splited_list_of_files_and_scaler_HT
from utils.utils import  Data_Generator


import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";

import tensorflow as tf
import keras


if __name__ == '__main__':

    output_root_dir = 'HT_results/'
    classifier_name='htnet'
    archive_name = 'HT_trained_and_validated_on_all'
    dataset_names = ['AES-T1000', 'AES-T1400', 'AES-T1100',  'AES-T1600', 'AES-T500',
                     'AES-T600', 'AES-T1800', 'AES-T1300', 'AES-T700', 'AES-T2000',
                     'AES-T400', 'AES-T800']

    for dataset_name in dataset_names:
        output_directory = output_root_dir  + classifier_name + '/' + archive_name + '/'+dataset_name+'/'
        create_directory(output_directory)
        print(output_directory)

        dirs_to_files_train, dirs_to_files_test, y_train, y_test, scaler, input_shape = \
            get_splited_list_of_files_and_scaler_HT(dir2bms_folder='HT_Data/AES_withTrojan_Set1/'
                                                    , name_bms=[dataset_name], number_of_training_for_scaler=1000)
        batch_size = 1
        nb_epochs = 500
        workers = 7

        train_generator = Data_Generator(dirs_to_files_train, y_train, batch_size=batch_size, scaler=scaler)
        val_generator = Data_Generator(dirs_to_files_test, y_test, batch_size=batch_size, scaler=scaler)

        nb_classes = len(np.unique(np.concatenate((y_train,y_test),axis =0)))
        # classifier = transferLearning.Classifier_tlEncoder(output_directory, input_shape, nb_classes, verbose=1)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.8
        keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

        from classifiers import htnet
        classifier = htnet.Classifier_HTNET(output_directory, input_shape, nb_classes, verbose=1)

        # Write some readme file
        readme_file = open(output_directory + "/training_readme.txt", "w")
        readme_file.write("batch_size: {}, nb_epochs: {}, workers: {}\n"
                        "Training Datasets: {}\n".format(batch_size, nb_epochs, workers, str(dataset_names)))
        classifier.model.summary(print_fn=lambda x: readme_file.write(x + '\n'))
        readme_file.close()

        classifier.fit_genrator(train_generator, val_generator, y_test, nb_epochs = nb_epochs,workers=workers)

        output_directory = output_root_dir + classifier_name + '/' + archive_name + '/'
        summrize(output_directory, input_res='df_best_model.csv', output_res='all_best_model.csv')
        summrize(output_directory, input_res='df_metrics.csv', output_res='all_metrics.csv')