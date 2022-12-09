
from utils.utils import create_directory, summrize

from utils.utils import get_splited_list_of_files_and_scaler_HT
from utils.utils import Data_Generator, Data_Generator_Combiner
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";

import tensorflow as tf
import keras
from classifiers import htnetNarrowed as htnet


if __name__ == '__main__':
    dir2bms_folder_ls = ['HT_Data/AES_withTrojan_Set1/',
                         'HT_Data/AES_withTrojan_Set7/']
    root_dir = 'HT_results/'
    base_folder='/transfer_learning_PWEM/'
    number_of_samples_per_folder = 1000
    classifier_name = 'htnetNarrowed'
    archive_name = 'HT_trained_and_validated_on_all_EXCEPT_folder_name_different_classes'

    dataset_names_all = ['AES-T1600','AES-T1300', 'AES-T500', 'AES-T600', 'AES-T2000', 'AES-T400',
                         'AES-T700', 'AES-T800','AES-T1000', 'AES-T1100', 'AES-T1400', 'AES-T1800']

    batch_size = 12
    nb_epochs = 150
    workers = 8
    patience = 30

    for exception_dataset_name in dataset_names_all:
        output_directory = root_dir + base_folder + classifier_name + '/' + archive_name + '/' + exception_dataset_name + '/'
        if os.path.isdir(output_directory):
            print('skipping: ' + output_directory)
            output_directory = root_dir + base_folder + classifier_name + '/' + archive_name + '/'
            summrize(output_directory, input_res='df_best_model.csv', output_res='all_best_model.csv')
            summrize(output_directory, input_res='df_metrics.csv', output_res='all_metrics.csv')
            continue
        create_directory(output_directory)
        dataset_names = dataset_names_all.copy()
        dataset_names.remove(exception_dataset_name)
        train_generators =[]
        val_generators = []
        input_shapes = 0
        for dir2bms_folder in dir2bms_folder_ls:
            dirs_to_files_train, dirs_to_files_test, y_train, y_test,y_categorical_train, y_categorical_test, scaler,\
            input_shape = get_splited_list_of_files_and_scaler_HT(dir2bms_folder=dir2bms_folder,
                                                                  name_bms=dataset_names, number_of_training_for_scaler=100,
                                                                  get_categorical_labels_y=True ,
                                                                  number_of_samples_per_folder= number_of_samples_per_folder)
            input_shapes =input_shapes+input_shape[0]
            print('total number of samples used for training',len(dirs_to_files_train))
            print('total number of samples used for testing',len(dirs_to_files_test))

            scalers = {}
            for ht_folder in dataset_names:
                _, _, _, _, scalers[ht_folder], _ = \
                    get_splited_list_of_files_and_scaler_HT(
                        dir2bms_folder=dir2bms_folder,
                        name_bms=[ht_folder], number_of_training_for_scaler=100,  number_of_samples_per_folder= number_of_samples_per_folder)

            # Write some readme file
            train_generators.append(Data_Generator(dirs_to_files_train, y_categorical_train, batch_size=batch_size, dir2bms_folder=dir2bms_folder, scaler=scalers))
            val_generators.append( Data_Generator(dirs_to_files_test, y_categorical_test, batch_size=batch_size, dir2bms_folder=dir2bms_folder, scaler=scalers))

        input_shape = (input_shapes,1)
        train_generator = Data_Generator_Combiner(train_generators)
        val_generator = Data_Generator_Combiner(val_generators)

        nb_classes = len (y_categorical_train[0])
        print('number of classes: ', nb_classes)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

        classifier = htnet.Classifier_HTNET(output_directory, input_shape, nb_classes, verbose=1)

        readme_file = open(output_directory + "/training_readme.txt", "w")
        readme_file.write("nb_classes: {}, batch_size: {}, nb_epochs: {}, workers: {}\n"
                          "Training Datasets: {}\n".format(nb_classes, batch_size, nb_epochs, workers, str(dataset_names)))
        classifier.model.summary(print_fn=lambda x: readme_file.write(x + '\n'))
        readme_file.close()

        classifier.fit_genrator(train_generator, val_generator, y_categorical_test, nb_epochs=nb_epochs, workers=workers, patience=patience)

        output_directory = root_dir + base_folder+ classifier_name + '/' + archive_name + '/'
        summrize(output_directory, input_res='df_best_model.csv', output_res='all_best_model.csv')
        summrize(output_directory, input_res='df_metrics.csv', output_res='all_metrics.csv')

