import os.path
import re
from datetime import datetime
import sys
print("Import successful")

import hickle as hkl
import numpy as np
import tensorflow as tf
import self_har_models
import self_har_trainers
import self_har_utilities
import transformations
from keras.models import load_model
# from feature_extraction.data_loading import load_datasets
from preprocess.dataset_loading import load_datasets
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
def extract_features(data, data_labels, is_accel, extract_dataset, save_path=None): # extracting features
    transform_funcs_names = ['noised', 'scaled', 'rotated', 'negated', 'time_flipped', 'permuted', 'time_warped', 'channel_shuffled'] # add lables for all the transformation functions
    # # use the processing used in the sensor based impl
    initial_learning_rate = 0.003
    num_features = 1000
    optim = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    input_shape = (128,3)
    core = self_har_models.create_1d_conv_core_model(input_shape) # create cnn core
    model = self_har_models.attach_multitask_transform_head(core, output_tasks=transform_funcs_names,optimizer=optim, num_features= num_features, with_har_head=False)


    model.summary()
    # create the multitask dataset for the transformation classification note: the har head is not added to make it SSL
    multitask_transform_dataset = self_har_utilities.create_individual_transform_dataset(
        data,
        transform_funcs_vectorized)
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
        callbacks=[training_rate_scheduler_cb],  # this will train the model and save ig
        epochs=100,
        batch_size=64,
        tag=tag,
        use_tensor_board_logging=False,
        verbose=True,
        single_train=False,

    )


    #now that the model has been trained you can extract the heads and then take the extracted features
    core_model = self_har_models.extract_core_model(model)
    if save_path is not None:
        core_model.save(save_path)
    ext_dataset = extract_dataset
    extract_stage(core_model, ext_dataset, is_accel)



def extract_stage(core_model, ext_dataset, is_accel):
    core_model.summary()
    train_data, train_labels = ext_dataset.train, ext_dataset.train_label
    test_data, test_labels = ext_dataset.test, ext_dataset.test_label
    if is_accel:
        train_data = train_data[:,:,:3] # for the extraction of the downstream dataset choose which modailty to remove
        test_data = test_data[:,:,:3]

    else:
        train_data = train_data[:,:,3:] # gyroscope is the last three columns
        test_data = test_data[:,:,3:]



    unlabelled_pred_prob = core_model.predict(train_data)  # teacher model will predict on the unlabelled data generating its own labels for what it thinks NOTE: this will be a probabiotiy distribution

    unlablled_pred_prob_test = core_model.predict(test_data, batch_size=64)
    # now that you have the features you can save the labels and the features
    # TODO ADD ADDITIONAL LOGIC TO SAVE THE FOLDER TO SPECIFIC LOCATION
    current_time_string = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_name = f"{current_time_string}_SHL_features_{'accel' if is_accel else 'gyro'}.hkl"
    hkl.dump({
        "train_features": unlabelled_pred_prob,
        "train_labels": np.argmax(train_labels, axis=1),
        "test_features": unlablled_pred_prob_test,
        "test_labels": np.argmax(test_labels, axis=1)
    }, save_name)


"""
    For running feature extraction if pretrained state is true then it will search the current working directory for a pretrained extractor for that modality
    and this feature extraction can speed up the process
"""
def running(target_ds, pretrained_state=True):
    position = (0, 3)
    data = load_datasets(['MotionSense', "UCI", "WISDM"], path='../../SensorBasedTransformerTorch/datasets/processed')
    # data = load_datasets(['MotionSense'], path='../../SensorBasedTransformerTorch/datasets/processed')
    train_data, train_labels = data.train, data.train_label  # there is no need for training and test because it is just a pretext task
    test_data, test_labels = data.test, data.test_label
    data = np.vstack((train_data, test_data))
    labels = np.concatenate([train_labels, test_labels])



    if pretrained_state:
        for modal in range(2):
            is_accel = modal == 0
            if is_accel:
                model = load_model('saved_accel.h5')
            else:
                model = load_model('saved_gyro.h5')

            extract_stage(model,target_ds, is_accel)
    else:
        for modal in range(2):
            modal_data = data[:, :, position[0]: position[1]]
            extract_features(modal_data, labels, modal == 0, target_ds, save_path=f"./saved_{'accel' if modal == 0 else 'gyro'}.h5")
            position = position[1], position[1] + 3


if __name__ == "__main__":
    files = ["../../SensorBasedTransformerTorch/datasets/processed/MotionSense/"]
    extract_dataset = ("SHL", "../../SensorBasedTransformerTorch/datasets/processed/")
    extract_target = load_datasets([extract_dataset[0]], path=extract_dataset[1])
    # now extract shl and save the model
    running(extract_target, pretrained_state=False)
    extract_dataset = ("PAMAP", "../../SensorBasedTransformerTorch/datasets/processed/")
    extract_target = load_datasets([extract_dataset[0]], path=extract_dataset[1])
    running(extract_target, pretrained_state=True)