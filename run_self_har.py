import os
import gc
import pickle
import argparse
import datetime
import time
import json
import distutils.util
import pprint

import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow import keras
import scipy.constants
import sklearn

import data_pre_processing
import self_har_models
import self_har_utilities
import self_har_trainers
import transformations

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2021 C. I. Tang"

"""
Complementing the work of Tang et al.: SelfHAR: Improving Human Activity Recognition through Self-training with Unlabeled Data
@article{10.1145/3448112,
  author = {Tang, Chi Ian and Perez-Pozuelo, Ignacio and Spathis, Dimitris and Brage, Soren and Wareham, Nick and Mascolo, Cecilia},
  title = {SelfHAR: Improving Human Activity Recognition through Self-Training with Unlabeled Data},
  year = {2021},
  issue_date = {March 2021},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  volume = {5},
  number = {1},
  url = {https://doi.org/10.1145/3448112},
  doi = {10.1145/3448112},
  abstract = {Machine learning and deep learning have shown great promise in mobile sensing applications, including Human Activity Recognition. However, the performance of such models in real-world settings largely depends on the availability of large datasets that captures diverse behaviors. Recently, studies in computer vision and natural language processing have shown that leveraging massive amounts of unlabeled data enables performance on par with state-of-the-art supervised models.In this work, we present SelfHAR, a semi-supervised model that effectively learns to leverage unlabeled mobile sensing datasets to complement small labeled datasets. Our approach combines teacher-student self-training, which distills the knowledge of unlabeled and labeled datasets while allowing for data augmentation, and multi-task self-supervision, which learns robust signal-level representations by predicting distorted versions of the input.We evaluated SelfHAR on various HAR datasets and showed state-of-the-art performance over supervised and previous semi-supervised approaches, with up to 12% increase in F1 score using the same number of model parameters at inference. Furthermore, SelfHAR is data-efficient, reaching similar performance using up to 10 times less labeled data compared to supervised approaches. Our work not only achieves state-of-the-art performance in a diverse set of HAR datasets, but also sheds light on how pre-training tasks may affect downstream performance.},
  journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
  month = mar,
  articleno = {36},
  numpages = {30},
  keywords = {semi-supervised training, human activity recognition, unlabeled data, self-supervised training, self-training, deep learning}
}

Access to Article:
    https://doi.org/10.1145/3448112
    https://dl.acm.org/doi/abs/10.1145/3448112

Contact: cit27@cl.cam.ac.uk

Copyright (C) 2021 C. I. Tang

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

"""

LOGS_SUB_DIRECTORY = 'logs'
MODELS_SUB_DIRECTORY = 'models'


def testing():
    x1 = np.random.rand(400,3)
    x2 = np.random.rand(500,3)
    multi_in = self_har_models.attach_multihead_input([x1,x2], conv_layers=2)
    return multi_in
def get_parser():
    def strtobool(v):
        return bool(distutils.util.strtobool(v))

    parser = argparse.ArgumentParser(
        description='SelfHAR Training')
        
    parser.add_argument('--working_directory', default='run',
                        help='directory containing datasets, trained models and training logs')
    parser.add_argument('--config', default='sample_configs/self_har.json',
                        help='')
    
    parser.add_argument('--labelled_dataset_path', default='run/processed_datasets/motionsense_processed.pkl', type=str, 
                        help='name of the labelled dataset for training and fine-tuning')
    parser.add_argument('--unlabelled_dataset_path', default='run/processed_datasets/hhar_processed.pkl', type=str, 
                        help='name of the unlabelled dataset to self-training and self-supervised training, ignored if only supervised training is performed.')
    
    parser.add_argument('--window_size', default=400, type=int,
                        help='the size of the sliding window')
    parser.add_argument('--max_unlabelled_windows', default=40000, type=int,
                        help='')

    parser.add_argument('--use_tensor_board_logging', default=True, type=strtobool,
                        help='')
    parser.add_argument('--verbose', default=1, type=int,
                        help='verbosity level')

    return parser

#get_train_test_users: tells which users are in the training vs the testing split, in this code every 5 users are in the taraining
def prepare_dataset(dataset_path, window_size, get_train_test_users, validation_split_proportion=0.1, verbose=1):
    if verbose > 0: # if verbose logging is enabled
        print(f"Loading dataset at {dataset_path}") # print the path of the selected dataset

    with open(dataset_path, 'rb') as f: # open the dataset path
        dataset_dict = pickle.load(f) # load the dataset that was stored in the format that pickel uses
        user_datasets = dataset_dict['user_split'] # this is where you get the user the users associated with all their activityies from the loadded model format: {1:[[([[.. associates a user with a series that has a label
        label_list = dataset_dict['label_list'] #There are the types of activities being performed, the labels will be 1D list [sit, std, wlk,ups,...

    label_map = dict([(l, i) for i, l in enumerate(label_list)]) # This will create a dictionary of a the activitiy name to integer ids
    output_shape = len(label_list) # The number of output values found by just getting number of labels

    har_users = list(user_datasets.keys()) #Get the key value in the user dict to get all the numeric user ids then turn that into a 1D list
    train_users, test_users = get_train_test_users(har_users) # get the split of training and testing users using the passed in function this will do an 80-20 split where 1 in 5 users will be in the testing set
    #NOTE: They make sure that you don't shuffle the users so that users will always have their activities that they performed associated only with themselves to avoid data leak
    if verbose > 0:
        print(f'Testing users: {test_users}, Training users: {train_users}')
    # this stage will do the data preprocessing
    """
    The stages are as follows:
    1) The Data is partitioned into windows, the windowing of data allows for better pattern recognition in timeseries data
    This takes a user and will window their own data. Users will have (by default) X many windows each with 400 entries per window with 3 entries per column
    a 50% overlap between these windows
    Each user will have 293,400,3, 293 being the ENTER and 400 being the window size and 3 being the number of columns
    dimensions of data after: Num_Usersx7x400x3
    2) The windowed Data is combined   
    """
    np_train, np_val, np_test = data_pre_processing.pre_process_dataset_composite(
        user_datasets=user_datasets, 
        label_map=label_map, 
        output_shape=output_shape, 
        train_users=train_users, 
        test_users=test_users, 
        window_size=window_size, 
        shift=window_size//2, #divide the window into two and divide down this will be how much the window moves each time so 200 in the case of 400 meaning 50% intersect
        normalise_dataset=True, 
        validation_split_proportion=validation_split_proportion, # The validation set is used to determine how good our training set is performing it is .1 by default
        verbose=verbose
    )
    ### This is the stage where the preprocessing occurs  ^^^^^
    #return a dictionary of the  train, val and test sets
    return {
        'train': np_train,
        'val': np_val,
        'test': np_test,
        'label_map': label_map,
        'input_shape': np_train[0].shape[1:], #this will be the #this will be the shape of inputs 400 rows with 3 cols each
        'output_shape': output_shape,
    }

def generate_unlabelled_datasets_variations(unlabelled_data_x, labelled_data_x, labelled_repeat=1, verbose=1):
    if verbose > 0:
        print("Unlabeled data shape: ", unlabelled_data_x.shape)
    
    labelled_data_repeat = np.repeat(labelled_data_x, labelled_repeat, axis=0)
    np_unlabelled_combined = np.concatenate([unlabelled_data_x, labelled_data_repeat])
    if verbose > 0:
        print(f"Unlabelled Combined shape: {np_unlabelled_combined.shape}")
    gc.collect()

    return {
        'labelled_x_repeat': labelled_data_repeat,
        'unlabelled_combined': np_unlabelled_combined
    }

def load_unlabelled_dataset(prepared_datasets, unlabelled_dataset_path, window_size, labelled_repeat, max_unlabelled_windows=None, verbose=1):
    def get_empty_test_users(har_users):
        return (har_users, [])

    prepared_datasets['unlabelled'] = prepare_dataset(unlabelled_dataset_path, window_size, get_empty_test_users, validation_split_proportion=0, verbose=verbose)['train'][0]
    if max_unlabelled_windows is not None:
        prepared_datasets['unlabelled'] = prepared_datasets['unlabelled'][:max_unlabelled_windows]
    prepared_datasets = {
        **prepared_datasets,
        **generate_unlabelled_datasets_variations( #manually generates unlabelled and combined with the lablled training data
            prepared_datasets['unlabelled'], 
            prepared_datasets['labelled']['train'][0],
            labelled_repeat=labelled_repeat
    )}
    return prepared_datasets

def get_config_default_value_if_none(experiment_config, entry, set_value=True):
    if entry in experiment_config:
        return experiment_config[entry]
    
    if entry == 'type':
        default_value = 'none'
    elif entry == 'tag':
        default_value = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    elif entry == 'previous_config_offset':
        default_value = 0
    elif entry == 'initial_learning_rate':
        default_value = 0.0003
    elif entry == 'epochs':
        default_value = 30
    elif entry == 'batch_size':
        default_value = 300
    elif entry == 'optimizer':
        default_value = 'adam'
    elif entry == 'self_training_samples_per_class':
        default_value = 10000
    elif entry == 'self_training_minimum_confidence':
        default_value = 0.0
    elif entry == 'self_training_plurality_only':
        default_value = True
    elif entry == 'trained_model_path':
        default_value = ''
    elif entry == 'trained_model_type':
        default_value = 'unknown'
    elif entry == 'eval_results':
        default_value = {}
    elif entry == 'eval_har':
        default_value = False

    if set_value:
        experiment_config[entry] = default_value
        print(f"INFO: configuration {entry} set to default value: {default_value}.")
    
    return default_value


if __name__ == '__main__':
    parser = get_parser() # get users args
    args = parser.parse_args() # parse the args that were passed in

    current_time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # get the current time
    working_directory = args.working_directory # get the current working directory from the args
    verbose = args.verbose # get if verbose logging is enabled
    use_tensor_board_logging = args.use_tensor_board_logging # get if tensorboard logging is enabled, tensor board is a tool that allows visualizations
    window_size = args.window_size # the data window size in this case the example size is 400

    if use_tensor_board_logging: # check if tensorboard logging is enabled
        logs_directory = os.path.join(working_directory, LOGS_SUB_DIRECTORY) # combine the current wd with the logs sub directory name
        if not os.path.exists(logs_directory): # check if the logs directory already exists
            print(os.getcwd())
            os.mkdir(logs_directory) # create the directory if it does not exist
    models_directory = os.path.join(working_directory, MODELS_SUB_DIRECTORY) # creating relative path to wehre the models are stored
    if not os.path.exists(models_directory): # if there is not a path to the models directory
        os.mkdir(models_directory) #create the models directory
    transform_funcs_vectorized = [
        transformations.noise_transform_vectorized,
        transformations.scaling_transform_vectorized, 
        transformations.rotation_transform_vectorized, 
        transformations.negate_transform_vectorized, 
        transformations.time_flip_transform_vectorized, 
        transformations.time_segment_permutation_transform_improved,  # this creates an arraye of the transformation functions
        transformations.time_warp_transform_low_cost, 
        transformations.channel_shuffle_transform_vectorized
    ]
    transform_funcs_names = ['noised', 'scaled', 'rotated', 'negated', 'time_flipped', 'permuted', 'time_warped', 'channel_shuffled'] # add lables for all the transformation functions

    prepared_datasets = {} #create a dictionary for the prepared dataset
    labelled_repeat = 1             # TODO: improve flexibility transformation_multiple
    

    """
    Allocates the training and testing users
    """
    def get_fixed_split_users(har_users):   # TODO: improve flexibility
        test_users = har_users[0::5] # the test users are every 5 users [0,5,10,15,...] so 0 till the end jump by 5 these will go into the testing set so 1/5 = 20% user split
        train_users = [u for u in har_users if u not in test_users] # the training users are everything not ine the testing set 80%
        return (train_users, test_users) # you now a training user and testing user split in a tuple of arrays


    """
        The prepared dataset at labelled will be a tuple containing the ndarray of activities that have been windowed and joined with all the users
        the other entry will be an ndarray of one hot encoded categories that relate to the sequence in the large training sequence array
    """
    """ 
        below will set the prepared data in the dict to be the dataset that was passed in with a passed in window size (usually 400)
        what also is passed in is the function to split the data into testing and training 
        it is important to note that 80-20 test train split is hardcoded
        the validation split proportion is passed in with 10% of the training 
        for the inner works go to line 104
    """
    # wes = True
    # if wes:
    #     multi_in = testing()
    #     # get the other data
    #
    #     exit(1)
    gpu_device = 0  # Replace with the index of the GPU you want to use
    print(tf.test.gpu_device_name(), 'testing new')

    # me testing the wisdm
    #wisdm  = 'test_run/processed_datasets/wisdm_processed.pkl'
    # test = 'test_run/processed_datasets/motionsense_processed.pkl'
    print(keras.backend.backend())
    prepared_datasets['labelled'] = prepare_dataset(args.labelled_dataset_path, window_size, get_fixed_split_users, validation_split_proportion=0.1, verbose=verbose)
    #prepared_datasets['labelled'] = prepare_dataset(wisdm, window_size, get_fixed_split_users, validation_split_proportion=0.1, verbose=verbose)
    #print(tf.configlist_logical_devices('GPU'))
    input_shape = prepared_datasets['labelled']['input_shape'] #  (window_size, 3) the window size in my case was 400
    output_shape = prepared_datasets['labelled']['output_shape'] # the number of classificcations

    with open(args.config, 'r') as f:  # open the jsonified model from the parameter config
        config_file = json.load(f)  # load the json file
        file_tag = config_file['tag']  # get the file name tag from the json
        experiment_configs = config_file[
            'experiment_configs']  # get the configurations for the model, this will be an array

    if verbose > 0:  # print the configurations for the experiment
        print("Experiment Settings:")
        for i, config in enumerate(experiment_configs):
            print(f"Experiment {i}:")  # the current experment
            print(config)  # json config
            print("------------")
        time.sleep(5)  # sleep at the end

    for i, experiment_config in enumerate(experiment_configs):
        if verbose > 0:  # printing that you are starting with this configuration
            print("---------------------")
            print(f"Starting Experiment {i}: {experiment_config}")  # the current configuration to be run
            print("---------------------")
            time.sleep(5)  # sleep for 5 seconds before starting
        gc.collect()  # tell the garbage colllector to run a full collection
        tf.keras.backend.clear_session()  # clear the global state of keras to save storage if you are building a lot of models

        experiment_type = get_config_default_value_if_none(experiment_config,
                                                           'type')  # get the type associated with the experiment
        if experiment_type == 'none':  # skip this model configuration if this has a none type
            continue

        if get_config_default_value_if_none(experiment_config,
                                            'previous_config_offset') == 0:  # check if there was a previous configuration 0 means there was none
            previous_config = None  # set the previous to none if offset is 0
        else:  # this is the case where you trained a model in the previous stage and you will need to load it back from memory (ex second stage of teacher training)
            previous_config = experiment_configs[i - experiment_config[
                'previous_config_offset']]  # access the previous config from the configs by using the offset to subtract it from i
            # if verbose > 0:
            #     print("Previous config", previous_config)

        tag = f"{current_time_string}_{file_tag}_{get_config_default_value_if_none(experiment_config, 'tag')}"  # create a formatted string that has the time the file tag the config
        # on the default config this will result in the teacher being trained first
        if experiment_type == 'eval_har':  # if the experiment type is the eval har where you evaluate the models

            if previous_config is None or get_config_default_value_if_none(previous_config, 'trained_model_path',
                                                                           set_value=False) == '':
                print("ERROR Evaluation model does not exist")
                continue

            if get_config_default_value_if_none(previous_config, 'trained_model_type') == 'har_model':
                previous_model = tf.keras.models.load_model(previous_config['trained_model_path'])
                model = previous_model
            elif get_config_default_value_if_none(previous_config, 'trained_model_type') == 'transform_with_har_model':
                previous_model = tf.keras.models.load_model(previous_config['trained_model_path'])
                model = self_har_models.extract_har_model(previous_model, optimizer=optimizer, model_name=tag)

            pred = model.predict(prepared_datasets['labelled']['test'][0])
            eval_results = self_har_utilities.evaluate_model_simple(pred, prepared_datasets['labelled']['test'][1])
            if verbose > 0:
                print(eval_results)
            experiment_config['eval_results'] = eval_results

            continue

        # you build your model from the joson config loadded
        initial_learning_rate = get_config_default_value_if_none(experiment_config,
                                                                 'initial_learning_rate')  # get the learning rate for the model from the config
        epochs = get_config_default_value_if_none(experiment_config,
                                                  'epochs')  # get the number of epochs from the config
        batch_size = get_config_default_value_if_none(experiment_config,
                                                      'batch_size')  # get the batch size form the configuration
        optimizer_type = get_config_default_value_if_none(experiment_config,
                                                          'optimizer')  # get the optimizer from the config
        if optimizer_type == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)

        if experiment_type == 'transform_train':  # if this is a transform train experiment NOT APPLICABLE FOR BASE MODEL this would be used for the transformation training described in the overall structure
            """
                The transform training would be where the unlabelled data would have transformations applied and the model would have to discriminate the task and the transformation 
            """
            if 'unlabelled' not in prepared_datasets:  # if there is not a unlablled dataset
                prepared_datasets = load_unlabelled_dataset(prepared_datasets, args.unlabelled_dataset_path,
                                                            window_size, labelled_repeat,
                                                            max_unlabelled_windows=args.max_unlabelled_windows,
                                                            verbose=verbose)  # create the unlabelled data

            if previous_config is None or get_config_default_value_if_none(previous_config, 'trained_model_path',
                                                                           set_value=False) == '':  # if there was not a previous model or there is not a stored model
                if verbose > 0:  # creating a new layer
                    print("Creating new model...")
                    # set the core model to to be a multilayer convolutional model with a pooling layer at the end
                core_model = self_har_models.create_1d_conv_core_model(
                    input_shape)  # create a new model that is a 1D convolutional  model
                """
                    Above will create a convolution layer 
                    the core conv layer will then have multiple heads on top for the HAR categorization as well as transformation dicrimination  
                """
            else:  # if there are previous layers
                if verbose > 0:  # print the previous model
                    print(f"Loading previous model {previous_config['trained_model_path']}")
                previous_model = tf.keras.models.load_model(
                    previous_config['trained_model_path'])  # get the previous model loaded from memory
                # the core model
                core_model = self_har_models.extract_core_model(
                    previous_model)  # get the core model from the previous model this will return the layer of the model at position [1]
            should_show_training = get_config_default_value_if_none(experiment_config, "single_train_eval",
                                                                    set_value=False)  # if you want to show the training

            features = [64, 128, 256, 512, 1024]  # the number of features in the output
            epoch_list = [20, 30, 35, 40, 50]
            model_holder = {}
            if should_show_training:  # if you want to show the training
                if get_config_default_value_if_none(experiment_config, 'test_num_features', set_value=False):
                    model_holder['test_num_features'] = self_har_models.test_number_features(input_shape,
                                                                                             transform_funcs_names,
                                                                                             features,
                                                                                             initial_learning_rate)
                # then create a list of models with the different features being returned
                if get_config_default_value_if_none(experiment_config, 'test_epochs', set_value=False):
                    model_holder['test_epochs'] = self_har_models.test_epochs(len(epoch_list), input_shape, transform_funcs_names,
                                                                              initial_learning_rate)
                # now for each of the models train them and find the best one
            else:
                transform_model = self_har_models.attach_multitask_transform_head(core_model,
                                                                                  output_tasks=transform_funcs_names,
                                                                                  optimizer=optimizer)
                transform_model.summary()  # put a summary of the model

            """
                Above will add to the core model heads for Multi task learning which allows for the model to better learn the features of the data
                this will have one categorical HAR head and the rest will be categorical bce 
                I am unsure why the transformation model is not used in all examples
            """
            if verbose > 0:
                print(
                    f"Dataset for transformation discrimination shape: {prepared_datasets['unlabelled_combined'].shape}")

            multitask_transform_dataset = self_har_utilities.create_individual_transform_dataset(
                prepared_datasets['unlabelled_combined'],
                transform_funcs_vectorized)  # this will add the unlabelled combined with the labelled with the labels removed

            multitask_transform_train = (multitask_transform_dataset[0],
                                         self_har_utilities.map_multitask_y(multitask_transform_dataset[1],
                                                                            transform_funcs_names))  # get the multitask inputs
            multitask_split = self_har_utilities.multitask_train_test_split(multitask_transform_train, test_size=0.10,
                                                                            random_seed=42)  # create the split
            multitask_train = (multitask_split[0], multitask_split[1])  # training
            multitask_val = (multitask_split[2], multitask_split[3])  # validation


            def training_rate_schedule(
                    epoch):  # this is the schedule of which the model learning should be updated  per batch
                rate = initial_learning_rate * (0.5 ** (epoch // 15))  # go down per epoch
                if verbose > 0:
                    print(f"RATE: {rate}")
                return rate


            training_schedule_callback = tf.keras.callbacks.LearningRateScheduler(
                training_rate_schedule)  # create a callback of the above that can be used
            if should_show_training:
                # go through the models and train each of them
                for configuration in model_holder:
                    for i,model in enumerate(model_holder[configuration]):
                        # now you will train the models in this configuration
                        self_har_trainers.composite_train_model(
                            full_model=model,
                            training_set=multitask_train,
                            validation_set=multitask_val,
                            working_directory=working_directory,
                            callbacks=[training_schedule_callback],
                            epochs=epochs if configuration != "test_epochs" else epoch_list[i],
                            batch_size=batch_size,
                            tag=tag,
                            use_tensor_board_logging=use_tensor_board_logging,
                            verbose=verbose,
                            single_train=should_show_training,
                            name=f"transform_model_{features[i]}" if configuration != "test_epochs" else f"transform_model_epochs_{epoch_list[i]}"
                        )
                    # self_har_trainers.composite_train_model(
                    #     full_model=model,
                    #     training_set=multitask_train,
                    #     validation_set=multitask_val,
                    #     working_directory=working_directory,
                    #     callbacks=[training_schedule_callback],
                    #     epochs=epochs,
                    #     batch_size=batch_size,
                    #     tag=tag,
                    #     use_tensor_board_logging=use_tensor_board_logging,
                    #     verbose=verbose,
                    #     single_train=should_show_training,
                    #     name=f"transform_model_{features[i]}"
                    # )
                    gc.collect()

                exit(1)

            best_transform_model_file_name, last_transform_pre_train_model_file_name = self_har_trainers.composite_train_model(
                full_model=transform_model,
                training_set=multitask_train,
                validation_set=multitask_val,
                working_directory=working_directory,
                callbacks=[training_schedule_callback],  # this will train the model and save it
                epochs=epochs,
                batch_size=batch_size,
                tag=tag,
                use_tensor_board_logging=use_tensor_board_logging,
                verbose=verbose,
                single_train=should_show_training
            )

            experiment_config['trained_model_path'] = best_transform_model_file_name  # set the model path in the dict
            experiment_config['trained_model_type'] = 'transform_model'  # set the type
            # the har full fine tune will be after the student model has been trained on the transformed data and now you want to use the original data
        if experiment_type == 'har_full_train' or experiment_type == 'har_full_fine_tune' or experiment_type == 'har_linear_train':  # if the model is a full train (what the default is)
            # create the core code that will be the teacher to start
            # the teacher is not the core convolutional model
            is_core_model = False
            if previous_config is None or get_config_default_value_if_none(previous_config, 'trained_model_path',
                                                                           set_value=False) == '':  # if this is the first model being trained, no previous models to use
                if verbose > 0:
                    print("Creating new model...")
                # here I will create an additional model
                core_model = self_har_models.create_1d_conv_core_model(input_shape) # create the convolution model that has input shape (400,3) this is the core model and everything else will go on top
                is_core_model = True # say the core model has been set
            else:
                if verbose > 0:
                    print(f"Loading previous model {previous_config['trained_model_path']}") # load the past model
                previous_model = tf.keras.models.load_model(previous_config['trained_model_path']) #get the past model

                if experiment_type == 'har_linear_train': #after you have evaluated the student model
                    core_model = self_har_models.extract_core_model(previous_model) #get the core cnn model from the past stage of student fine tunining this will be the core CNN model
                    is_core_model = True
                elif get_config_default_value_if_none(previous_config, 'trained_model_type') == 'har_model':
                    har_model = previous_model
                    is_core_model = False
                elif previous_config['trained_model_type'] == 'transform_with_har_model': # if your model has been trained on the transformation data
                    har_model = self_har_models.extract_har_model(previous_model, optimizer=optimizer, model_name=tag) # this will create a model that has the same inputs as the last model but the outputs will be only the HAR head's
                    is_core_model = False
                else:
                    core_model = self_har_models.extract_core_model(previous_model)
                    is_core_model = True

            if is_core_model:#if you have a core model
                if experiment_type == 'har_linear_train': #after the fine tuning stage
                    self_har_models.set_freeze_layers(core_model, num_freeze_layer_index=None) #freeze all the layers from the core model
                    har_model = self_har_models.attach_linear_classification_head(core_model, output_shape, optimizer=optimizer, model_name="Linear") # add a linear classifier this allows testing the performance of trained core model

                elif experiment_type == 'har_full_train':
                    self_har_models.set_freeze_layers(core_model, num_freeze_layer_index=0) #say if some layers can be trained or not this enables all layers to train and learn
                    # for the case of the teacher model you don't freeze any layers
                    har_model = self_har_models.attach_full_har_classification_head(core_model, output_shape, optimizer=optimizer, num_units=1024, model_name="HAR") #add the classifier to the end of the model it has 6 outputs in 1D
                elif experiment_type == 'har_full_fine_tune':
                    self_har_models.set_freeze_layers(core_model, num_freeze_layer_index=5)
                    har_model = self_har_models.attach_full_har_classification_head(core_model, output_shape, optimizer=optimizer, num_units=1024, model_name="HAR")
            else:
                if experiment_type == 'har_full_train':
                    self_har_models.set_freeze_layers(self_har_models.extract_core_model(har_model), num_freeze_layer_index=0)
                elif experiment_type == 'har_full_fine_tune': #if you are about to fine tune the student mode
                    self_har_models.set_freeze_layers(self_har_models.extract_core_model(har_model), num_freeze_layer_index=5) #this takes the functional layer that comes after the iunput layer and freeze the 5th layer
            
            def training_rate_schedule(epoch): #define the training rate schedule while is setting to the starting rate
                rate = initial_learning_rate
                if verbose > 0:
                    print(f"RATE: {rate}")
                return rate
            training_schedule_callback = tf.keras.callbacks.LearningRateScheduler(training_rate_schedule) # this creates the training schedule
            #TRAINING START
            # this is where trianing occurs
            best_har_model_file_name, last_har_model_file_name = self_har_trainers.composite_train_model(
                full_model=har_model, 
                training_set=prepared_datasets['labelled']['train'],
                validation_set=prepared_datasets['labelled']['val'], #train the model
                working_directory=working_directory, 
                callbacks=[training_schedule_callback], #in the case of the linear train (after the model has been fine tuned, this freezes the layers and creates a linear classifier this serves as a final eval
                epochs=epochs, 
                batch_size=batch_size,
                tag=tag, 
                use_tensor_board_logging=use_tensor_board_logging, #this will train with the base training set
                verbose=verbose
            )

            experiment_config['trained_model_path'] = best_har_model_file_name #set the file path for the best configuration model
            experiment_config['trained_model_type'] = 'har_model' #set the experiement config type in the dict

            
        
        if experiment_type == 'self_training' or experiment_type == 'self_har': #this will not be run for the teacher model
            if 'unlabelled' not in prepared_datasets: # If you do not have the unlablled dataset prepared
                prepared_datasets = load_unlabelled_dataset(prepared_datasets, args.unlabelled_dataset_path, window_size, labelled_repeat, max_unlabelled_windows=args.max_unlabelled_windows) # load  the unlabelled data
                # load in the unlablled data and combine it with the training data without labels add it to the datasets dicts
            if previous_config is None or get_config_default_value_if_none(previous_config, 'trained_model_path', set_value=False) == '': #if the previous config does not exist or the config was not saved give up
                print("ERROR No previous model for self-training") #error case
                break #stop
            else:
                if verbose > 0: #printing
                    print(f"Loading previous model {previous_config['trained_model_path']}")
                teacher_model = tf.keras.models.load_model(previous_config['trained_model_path']) # load the previously trained teacher set that was trained on training data
            if verbose > 0: #printing of dataset
                print("Unlabelled Datasete Shape", prepared_datasets['unlabelled_combined'].shape)
            unlabelled_pred_prob = teacher_model.predict(prepared_datasets['unlabelled_combined'], batch_size=batch_size) # teacher model will predict on the unlabelled data generating its own labels for what it thinks NOTE: this will be a probabiotiy distribution
            np_self_labelled = self_har_utilities.pick_top_samples_per_class_np( #this will pick the predictions with the highest confidence from the teacher for each class
                prepared_datasets['unlabelled_combined'],  #the combined unlablled datase
                unlabelled_pred_prob, #this will be an array of predictions for all the sequences from the the teacher
                num_samples_per_class=get_config_default_value_if_none(experiment_config, 'self_training_samples_per_class'),
                minimum_threshold=get_config_default_value_if_none(experiment_config, 'self_training_minimum_confidence'), 
                plurality_only=get_config_default_value_if_none(experiment_config, 'self_training_plurality_only')
            )
            
            #you will create 8 x teacher predictions to correspond with each transformation that will serve as har labels
            multitask_X, multitask_transform_y, multitask_har_y = self_har_utilities.create_individual_transform_dataset(
                np_self_labelled[0],  # the unlabelled sample data
                transform_funcs_vectorized,  #the transformation functions
                other_labels=np_self_labelled[1] # the probability the sample is a given class from the teacher
            )
            # above returns ((274914, 400 ,3 transformation x), (274914,8 corresponds to the transformation applied), (274914, 6 the activity) )

            core_model = self_har_models.create_1d_conv_core_model(input_shape) # create the core model
            def training_rate_schedule(epoch): #define a trainng rate schedule that updates and goes down with the epoch
                rate = 0.0003 * (0.5 ** (epoch // 15))
                if verbose > 0:
                    print(f"RATE: {rate}")
                return rate
            training_schedule_callback = tf.keras.callbacks.LearningRateScheduler(training_rate_schedule) #create a learning rate scheduler

            
            if experiment_type == 'self_training':
                student_pre_train_dataset = np_self_labelled

                student_model = self_har_models.attach_full_har_classification_head(core_model, output_shape, optimizer=optimizer, model_name="StudentPreTrain")
                student_model.summary()

                pre_train_split = sklearn.model_selection.train_test_split(student_pre_train_dataset[0], student_pre_train_dataset[1], test_size=0.10, random_state=42)
                student_pre_train_split_train = (pre_train_split[0], pre_train_split[2])
                student_pre_train_split_val = (pre_train_split[1], pre_train_split[3])

            else:

                multitask_transform_y_mapped = self_har_utilities.map_multitask_y(multitask_transform_y, transform_funcs_names) # using the one hot encoding of the transformations applied to each x create a map
                multitask_transform_y_mapped['har'] = multitask_har_y # set the mapping for har to be the teacher's probabilities
                self_har_train = (multitask_X, multitask_transform_y_mapped) # the training set is the transformed data along with the mappings of what transformation was applied
                student_pre_train_dataset = self_har_train\
                # set the student train data to be the now labelled for MTL dataset
                student_model = self_har_models.attach_multitask_transform_head(core_model, output_tasks=transform_funcs_names, optimizer=optimizer, with_har_head=True, har_output_shape=output_shape, num_units_har=1024, model_name="StudentPreTrain") # add the multitask learning head now to the core model for the student
                student_model.summary()
                #create the treaining split
                pre_train_split = self_har_utilities.multitask_train_test_split(student_pre_train_dataset, test_size=0.10, random_seed=42) # create a split in the training set that is used for training and validation

                student_pre_train_split_train = (pre_train_split[0], pre_train_split[1]) # get the training data and labels
                student_pre_train_split_val = (pre_train_split[2], pre_train_split[3]) #create the validation set split

            
            best_student_pre_train_file_name, last_student_pre_train_file_name = self_har_trainers.composite_train_model(
                full_model=student_model,
                training_set=student_pre_train_split_train,
                validation_set=student_pre_train_split_val, 
                working_directory=working_directory, 
                callbacks=[training_schedule_callback], # this will train the student on the teacher labelled data that has been transformed
                epochs=epochs, 
                batch_size=batch_size, 
                tag=tag, 
                use_tensor_board_logging=use_tensor_board_logging, 
                verbose=verbose
            )
            

            experiment_config['trained_model_path'] = best_student_pre_train_file_name #put the now trained stuednt model filepath in the dictionary
            if experiment_type == 'self_training':
                experiment_config['trained_model_type'] = 'har_model'
            else:
                experiment_config['trained_model_type'] = 'transform_with_har_model' # tset the trained model type to be the model trained on transformation and har


        if get_config_default_value_if_none(experiment_config, 'eval_har', set_value=False): #the evaluation stage
            if get_config_default_value_if_none(experiment_config, 'trained_model_type') == 'har_model':
                best_har_model = tf.keras.models.load_model(experiment_config['trained_model_path']) # load the fine-tuned student model
            elif get_config_default_value_if_none(experiment_config, 'trained_model_type') == 'transform_with_har_model':
                previous_model = tf.keras.models.load_model(experiment_config['trained_model_path'])
                best_har_model = self_har_models.extract_har_model(previous_model, optimizer=optimizer, model_name=tag)
            else:
                continue

            pred = best_har_model.predict(prepared_datasets['labelled']['test'][0]) # have the fine tuned student model predit on the labelled testing dasta  EVALUATION or if the linear train
            eval_results = self_har_utilities.evaluate_model_simple(pred, prepared_datasets['labelled']['test'][1]) # evaluate the model accuracy using the labels to see how accurate
            if verbose > 0:
                print(eval_results)
            experiment_config['eval_results'] = eval_results #store the evaluation results in the dict
    
    if verbose > 0:
        print("Finshed running all experiments.")
        print("Summary:")
        for i, config in enumerate(experiment_configs):
            print(f"Experiment {i}:")
            print(config)
            print("------------")

    result_summary_path = os.path.join(working_directory, f"{current_time_string}_{file_tag}_results_summary.txt")
    with open(result_summary_path, 'w') as f:
        structured = pprint.pformat(experiment_configs, indent=4)
        f.write(structured)
    if verbose > 0:
        print("Saved results summary to ", result_summary_path)


