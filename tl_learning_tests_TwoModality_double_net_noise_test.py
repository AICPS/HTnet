# https://askubuntu.com/questions/8653/how-to-keep-processes-running-after-ending-ssh-session
# import warnings
# warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')
# warnings.simplefilter(action='ignore', category=FutureWarning)
from utils.utils import create_directory
from utils.utils import get_splited_list_of_files_and_scaler_HT, get_ht_specific_scalers
from utils.utils import  Data_Generator, summrize, Data_Generator_Combiner, copyanything, remove
from classifiers.transferLearningDoubleNetAnomalyDetectorGroup import TLClassifierDouble
from sys import platform

import numpy as np
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";

import tensorflow as tf
import keras


if __name__ == '__main__':


    root_dir = 'HT_results/'
    dir2bms_folder_refs = ['HT_Data/AES_withTrojan_Set1/',
                           'HT_Data/AES_withTrojan_Set7/']
    dir2bms_folder_targets = ['HT_Data/AES_withTrojan_Set1/',
                              'HT_Data/AES_withTrojan_Set7/']

    random_state = 10
    classifier_name = 'htnetNarrowed'
    base_folder='/transfer_learning_PWEM/'
    use_ht_specific_scaler = True
    use_refrence_scaler_for_traget_data= True
    trained_models_folder= 'HT_trained_and_validated_on_all_EXCEPT_folder_name_different_classes'
    tl_results_folder = 'tl_results_with_double_net_noise_test_700'

    number_of_training_for_scaler = 100
    number_of_batches_used_for_anomaly_detector_training = 50
    lambda_ = 1
    batch_size = 60
    nb_epochs = 200
    number_of_samples_per_folder = 1000

    #Changes in the tests
    changes_in_the_tests=True
    circular_shift = 0
    added_noise_mus = [0]
    added_noise_sigmas = [x / 1000.0 for x in range(0, 200, 5)]



    all_dataset_names = ['AES-T700', 'AES-T800']

    dataset_name_to_be_tested = all_dataset_names.copy()

    for dataset_name in dataset_name_to_be_tested:
        trained_model_dir = root_dir +base_folder+ classifier_name + '/'+trained_models_folder + '/' + dataset_name+ '/'

        ref_generators, target_generators, val_generators, val_not_triggered_generators = [], [], [], []
        for dir2bms_folder_ref, dir2bms_folder_target in zip(dir2bms_folder_refs, dir2bms_folder_targets):
            dataset_names = all_dataset_names.copy()
            # get not_triggered data training data from current chip
            dirs_to_files_train, dirs_to_files_test, dirs_to_files_test_not_triggered, y_train, y_test, y_test_not_triggered, \
            scaler, input_shape = get_splited_list_of_files_and_scaler_HT(dir2bms_folder= dir2bms_folder_target,
                                                        name_bms=[dataset_name], number_of_training_for_scaler= number_of_training_for_scaler,
                                                        use_trigerd_data_for_scale_training=False, random_state=random_state,
                                                        get_not_triggered_training_only=True, get_triggered_training_only=False,
                                                        get_not_triggered_validation_as_well = True,
                                                                          number_of_samples_per_folder=number_of_samples_per_folder)

            # get triggered and not triggered data from others
            dataset_names.remove(dataset_name)
            dirs_to_files_train_others, dirs_to_files_test_others, y_train_others, y_test_others, y_categorical_train_others, \
            y_categorical_test_others, scaler_others, input_shape_others = \
                get_splited_list_of_files_and_scaler_HT(dir2bms_folder= dir2bms_folder_ref, name_bms=dataset_names, number_of_training_for_scaler= number_of_training_for_scaler,
                                                        use_trigerd_data_for_scale_training= True,random_state=random_state,
                                                        get_not_triggered_training_only= False, get_triggered_training_only= False,
                                                        get_categorical_labels_y = True,number_of_samples_per_folder=number_of_samples_per_folder)

            if use_refrence_scaler_for_traget_data:
                scaler =scaler_others
            if use_ht_specific_scaler:
                scaler_others = get_ht_specific_scalers(dir2bms_folder_ref, dataset_names,
                                                        number_of_training_for_scaler=number_of_training_for_scaler)



            ref_generators.append(Data_Generator(dirs_to_files_train_others, y_categorical_train_others,
                                           dir2bms_folder = dir2bms_folder_ref, batch_size=batch_size, scaler=scaler_others))
            target_generators.append(Data_Generator(dirs_to_files_train, y_train, batch_size=batch_size, scaler=scaler))

            val_generators.append(Data_Generator(dirs_to_files_test, y_test, batch_size=batch_size, scaler=scaler,
                                                 circular_shift=circular_shift))
            val_not_triggered_generators.append(Data_Generator(dirs_to_files_test_not_triggered, y_test_not_triggered, batch_size=batch_size, scaler=scaler))

        for added_noise_mu in added_noise_mus:
            for added_noise_sigma in added_noise_sigmas:
                added_txt = ''

                added_txt=added_txt+'_mu_'+str(added_noise_mu)+'_sigma_'+str(added_noise_sigma)
                src = root_dir + base_folder + classifier_name + '/' + tl_results_folder + '/' + dataset_name + '/'
                output_directory = root_dir + base_folder + classifier_name + '/' + tl_results_folder + '/' + dataset_name +added_txt+ '/'
                if os.path.isdir(output_directory):
                    print('The results folder exists: ' + output_directory)
                    continue

                copyanything(src,output_directory)
                ref_generator = Data_Generator_Combiner(ref_generators)
                target_generator = Data_Generator_Combiner(target_generators)

                val_generator = Data_Generator_Combiner(val_generators,added_noise_mu=added_noise_mu,
                                                        added_noise_sigma=added_noise_sigma)
                val_not_triggered_generator = Data_Generator_Combiner(val_not_triggered_generators)

                nb_classes = len(y_categorical_train_others[0])

                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = 0.8
                keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

                classifier = TLClassifierDouble(base_model_dir=trained_model_dir, output_directory=output_directory,
                                                nb_classes=nb_classes, batch_size=batch_size, lambda_=lambda_, verbose=1)

                readme_file = open(output_directory + "/training_readme.txt", "w")
                readme_file.write("batch_size: {}, nb_epochs: {},\n"
                                  "Training Datasets: {}\n".format(batch_size, nb_epochs, str(dataset_names)))
                readme_file.write("model_t: (target model)\n")
                classifier.model_t.summary(print_fn=lambda x: readme_file.write(x + '\n'))
                readme_file.write("\nmodel_r: (reference model)\n")
                classifier.model_r.summary(print_fn=lambda x: readme_file.write(x + '\n'))
                readme_file.close()


                classifier.evaluate_trained_feature_extractor(target_generator, val_generator,
                                                              number_of_batches_used_for_anomaly_detector_training)
                unwanted_files= ['anomaly_detectors_base_training_times.csv','anomaly_detectors_training_times.csv',
                                 'df_metrics_all_epochs.csv','epochs_loss.png','epochs_loss_c.png','model_r.h5',
                                 'model_t.h5','training_readme.txt']
                for f in unwanted_files:
                    remove(output_directory+f)

                keras.backend.clear_session()

                output_directory = root_dir + base_folder+ classifier_name + '/' + tl_results_folder + '/'

                summrize(output_directory, input_res='df_metrics.csv', output_res='all_metrics'+'.csv',add_extra_columns=True)
                print('Without feature extraction')
                summrize(output_directory, input_res='df_metrics_base.csv', output_res='all_metrics_base'+'.csv',add_extra_columns=True)


