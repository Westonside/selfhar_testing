import os
import re
from datetime import datetime

import hickle as hkl
import numpy as np
import tensorflow as tf
import self_har_models
import self_har_trainers
import self_har_utilities
import transformations
from feature_extraction.data_loading import load_datasets


def train_data_fn(path):
    # todo add in proper function to load the data
    train_data = np.random.rand((3000,128,3))
    train_data_labels = np.full((3000), fill_value=1)
    test_data = np.random.rand((2000,128,3))
    test_labels = np.full((2000), fill_value=1)
    return train_data, train_data_labels, test_data, test_labels
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
def extract_features(data, data_labels, modal):
    transform_funcs_names = ['noised', 'scaled', 'rotated', 'negated', 'time_flipped', 'permuted', 'time_warped', 'channel_shuffled'] # add lables for all the transformation functions
    # use the processing used in the sensor based impl
    initial_learning_rate = 0.003
    num_features = 1000
    optim = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    multitask_transform_dataset = self_har_utilities.create_individual_transform_dataset(data, transform_funcs_vectorized) #todo add in arguments
    input_shape = (128,3)
    core = self_har_models.create_1d_conv_core_model(input_shape)
    model = self_har_models.attach_multitask_transform_head(core, output_tasks=transform_funcs_names,optimizer=optim, num_features= num_features, with_har_head=False)


    model.summary()


    multitask_transform_train = (multitask_transform_dataset[0],
                             self_har_utilities.map_multitask_y(multitask_transform_dataset[1],
                                                                # also applies transformations
                                                                transform_funcs_names))  # get the multitask inputs
    multitask_split = self_har_utilities.multitask_train_test_split(multitask_transform_train, test_size=0.10,
                                                                    random_seed=42)  # create the split
    multitask_train = (multitask_split[0], multitask_split[1])  # training
    multitask_val = (multitask_split[2], multitask_split[3])  # validation


    def training_rate_schedule(
            epoch, verbose=False):  # this is the schedule of which the model learning should be updated  per batch
        rate = initial_learning_rate * (0.5 ** (epoch // 15))  # go down per epoch
        if verbose > 0:
            print(f"RATE: {rate}")
        return rate

    training_rate_scheduler_cb = tf.keras.callbacks.LearningRateScheduler(training_rate_schedule)

    current_time_string = datetime.now().strftime("%Y%m%d-%H%M%S") # get the current time
    tag = f"{current_time_string}_feature_extraction"
    best_transform_model_file_name, last_transform_pre_train_model_file_name = self_har_trainers.composite_train_model( #train the model
        full_model=model,
        training_set=multitask_train,
        validation_set=multitask_val,
        working_directory='.',
        callbacks=[training_rate_scheduler_cb],  # this will train the model and save it
        epochs=200,
        batch_size=64,
        tag=tag,
        use_tensor_board_logging=False,
        verbose=True,
        single_train=False
    )


    #now that the model has been trained you can extract the heads and then take the extracted features
    core_model = self_har_models.extract_core_model(model)


    unlabelled_pred_prob = core_model.predict(data,
                                                 batch_size=64)  # teacher model will predict on the unlabelled data generating its own labels for what it thinks NOTE: this will be a probabiotiy distribution
    # now that you have the features you can save the labels and the features
    # TODO ADD ADDITIONAL LOGIC TO SAVE THE FOLDER TO SPECIFIC LOCATION
    current_time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_name = f"{current_time_string}_MotionSense_features_{modal}.pkl"
    hkl.dump({
        "features": unlabelled_pred_prob,
        "labels": data_labels
    }, save_name)






if __name__ == "__main__":
    files = ["../../SensorBasedTransformerTorch/datasets/processed/MotionSense/"]
    modals = 2
    for file in files:
        position = (0,3)
        data = load_datasets('MotionSense', path='../../SensorBasedTransformerTorch/datasets/processed')
        train_data,train_labels = data[0] # there is no need for training and test because it is just a pretext task
        test_data,test_labels = data[1]
        data = np.vstack((train_data, test_data))
        labels = np.concatenate([train_labels, test_labels])
        for modal in range(modals):
            modal_data = data[:,:,position[0]: position[1]]
            extract_features(modal_data,labels, 'accel' if modal == 0 else 'gyro')
            position = position[1], position[1]+3