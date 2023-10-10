import glob
import re
import os
import pandas as pd
import numpy as np

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

def process_motion_sense_accelerometer_files(accelerometer_data_folder_path):
    """
    Preprocess the accelerometer files of the MotionSense dataset into the 'user-list' format
    Data files can be found at https://github.com/mmalekzadeh/motion-sense/tree/master/data
    Parameters:
        accelerometer_data_folder_path (str):
            the path to the folder containing the data files (unzipped)
            e.g. motionSense/B_Accelerometer_data/
            the trial folders should be directly inside it (e.g. motionSense/B_Accelerometer_data/dws_1/)
    Return:
        
        user_datsets (dict of {user_id: [(sensor_values, activity_labels)]})
            the processed dataset in a dictionary, of type {user_id: [(sensor_values, activity_labels)]}
            the keys of the dictionary is the user_id (participant id)
            the values of the dictionary are lists of (sensor_values, activity_labels) pairs
                sensor_values are 2D numpy array of shape (length, channels=3)
                activity_labels are 1D numpy array of shape (length)
                each pair corresponds to a separate trial 
                    (i.e. time is not contiguous between pairs, which is useful for making sliding windows, where it is easy to separate trials)
    """

    # label_set = {}
    user_datasets = {}
    all_trials_folders = sorted(glob.glob(accelerometer_data_folder_path + "/*"))

    # Loop through every trial folder
    for trial_folder in all_trials_folders:
        trial_name = os.path.split(trial_folder)[-1]

        # label of the trial is given in the folder name, separated by underscore
        label = trial_name.split("_")[0]
        # label_set[label] = True
        print(trial_folder)
        
        # Loop through files for every user of the trail
        for trial_user_file in sorted(glob.glob(trial_folder + "/*.csv")):

            # use regex to match the user id
            user_id_match = re.search(r'(?P<user_id>[0-9]+)\.csv', os.path.split(trial_user_file)[-1])
            if user_id_match is not None:
                user_id = int(user_id_match.group('user_id'))

                # Read file
                user_trial_dataset = pd.read_csv(trial_user_file)
                user_trial_dataset.dropna(how = "any", inplace = True)

                # Extract the x, y, z channels
                values = user_trial_dataset[["x", "y", "z"]].values

                # the label is the same during the entire trial, so it is repeated here to pad to the same length as the values
                labels = np.repeat(label, values.shape[0])

                if user_id not in user_datasets:
                    user_datasets[user_id] = []
                user_datasets[user_id].append((values, labels))
            else:
                print("[ERR] User id not found", trial_user_file)
    
    return user_datasets


def process_hhar_accelerometer_files(data_folder_path):
    """
    Preprocess the accelerometer files of the HHAR dataset into the 'user-list' format
    Data files can be found at http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition
    
    """
    # print(data_folder_path)

    har_dataset = pd.read_csv(os.path.join(data_folder_path, 'Phones_accelerometer.csv')) # "<PATH_TO_HHAR_DATASET>/Phones_accelerometer.csv"
    har_dataset.dropna(how = "any", inplace = True)
    har_dataset = har_dataset[["x", "y", "z", "gt","User"]]
    har_dataset.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    har_users = har_dataset["user-id"].unique()

    user_datasets = {}
    for user in har_users:
        user_extract = har_dataset[har_dataset["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].values
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        user_datasets[user] = [(data,labels)]
    
    return user_datasets


def process_hhar(data_folder_path: str, folder_name:str) -> dict:
    har_dataset = pd.read_csv(os.path.join(data_folder_path, folder_name))
    har_dataset.dropna(how="any", inplace=True)
    har_dataset = har_dataset[["x", "y", "z", "gt", "User"]]
    har_dataset.columns = ["x-axis", "y-axis", "z-axis", "activity", "user-id"]
    har_users = har_dataset["user-id"].unique()

    user_datasets = {}
    for user in har_users:
        user_extract = har_dataset[har_dataset["user-id"] == user]
        data = user_extract[["x-axis", "y-axis", "z-axis"]].values
        labels = user_extract["activity"].values
        print(f"{user} {data.shape}")
        user_datasets[user] = [(data, labels)]

    return user_datasets



def get_files_from_dir(directory) -> list:
    files = []
    for file in os.listdir(directory):
        # print(file, "here we are")
        if file.endswith(".txt"):
            files.append(os.path.join(directory,file))
    return files

def get_labels_from_file(file_path) -> dict:
    labels = {}
    with open(file_path, 'r') as f:
        lines = f.read().splitlines()[:-1]
        # print(lines)
        for line in lines:
            split = line.split(" ", 1)
            end = line[line.index('= ') + 2:]
            labels[split[0]] = end
            # print(labels)
    print(labels)
    return labels


def process_wisdm(data_folder_path: str, labels) -> dict:
    res = {}
    files = get_files_from_dir(data_folder_path)
    mapped = process_user_data(files) # now you have the data for each user
    res['label_list'] = list(labels.values() )
    res['label_list_full_name'] = list(labels.keys())
    res['has_null_class'] = False
    res['user_split'] = mapped

    # print(list(labels.values()))
    # print(res)
    print(res['label_list'], res['label_list_full_name'])
    return res


def process_user_data(files: list) -> dict:
    user_data = {}
    short_key = []
    for i, file in enumerate(files):

        user_tup, symbol = user_activity_split(file, i == 1, False)
        if i == 0:
            mini_label = symbol

        user_data[i + 1] = user_tup
    return user_data


def user_activity_split(file, should_print, apply_fft) -> tuple:
    transform = {'A': 'wlk', 'B': 'jog', 'D': 'sit', 'E': 'std'}

    opened = pd.read_csv(file, header=None, names=["user_id", "activity", "timestamp", "x", "y", "z"], sep=",")
    opened["z"] = opened["z"].str.replace(";", "")


    res = []
    # get all the sensor data for an activity
    # the issue is the different activities so you need to have all the same activities as the other classes
    for activity in np.unique(opened["activity"]):
        transformed_activity = transform.get(activity, activity)
        seq = (opened[opened["activity"] == activity][["x", "y", "z"]].values.astype(np.float32))
        lab = np.array([transformed_activity] * len(seq))
        # print(lab)
        res.append((seq, lab))
        # print(sequences, labels)
    return res, np.unique(opened["activity"])

