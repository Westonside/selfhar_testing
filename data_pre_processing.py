import numpy as np
import scipy.stats
import sklearn.model_selection
import tensorflow as tf

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

def get_mode(np_array):
    """
    Get the mode (majority/most frequent value) from a 1D array
    """
    return scipy.stats.mode(np_array)[0] # this gets the most occuring value in a 1D 400 entry label array
    # values, counts = np.unique(np_array, return_counts=True)
    # return values[counts.argmax()]
def sliding_window_np(X, window_size, shift, stride, offset=0, flatten=None):
    """
    Create sliding windows from an ndarray
    These sliding windows do increase the dimensionality

    Good example: you take temperate readings every day for 30 days there are 24 readings a day (24 hours)
    if your window is a day then you would have 30 windows, taking your dataset from
    Before windowing your data was 1D with 24 datapoints per window after your dasta was 2D 2(time,window)
    #for the case of the 1d label array it will return a 1D nparray
    Parameters:
    
        X (numpy-array)
            The numpy array to be windowed
            This array will have readings in the format:
            [[r1,r2,r3], [r4,r5,r6],...]
            the windows will get elements
            [0-400],[200-600]
            making the output shape of the format
            [[[r1,r2,r3], [r4,r5,r6], - 400], ...]]]
        
        shift (int)
            number of timestamps to shift for each window
            (200 here refers to 50% overlap, no overlap if =400) if you have a shift of 200 and the window size is 400 you get [0-400], [200-600]
            50% overlap
        stride (int)
            stride of the window (dilation)
        offset (int)
            starting index of the first window

        flatten (function (array) -> (value or array) )
            the function to be applied to a window after it is extracted
            can be used with get_mode (see above) for extracting the label by majority voting
            ignored if is None
    Return:
        Windowed ndarray
            shape[0] is the number of windows
    """

    overall_window_size = (window_size - 1) * stride + 1 #get the total window size which will combine the window size and  stride
    num_windows = (X.shape[0] - offset - (overall_window_size)) // shift + 1 #get the number of windows based off the number of vectors in the first dimension ex: (1781, 3) would return 1781 then remove the offset  and the window size and divide by shift this tells how mnay windows there will be for this user
    windows = [] # array to hold the windows
    for i in range(num_windows): # for the total number of windows
        start_index = i * shift + offset # get the starting index which uses i, the shift and the offset
        this_window = X[start_index : start_index + overall_window_size : stride] # stride in this case is how much it will jump by in default case it just jumps by 1 this will get elements [0:400], [200: 600]
        if flatten is not None: # in the base case it does not flatten and has dimensions (400,3), meaning 400 entries with 3 columns each flattening is not used for user data
            this_window = flatten(this_window) #call the flatten function on 1D label array and this_window will be one value which was the most occuring value in the array and this will act as the main label for the window
        windows.append(this_window) # add the new window
    return np.array(windows) # this goes [0-400], [200-600], returning an ndarray of shape (7,400,3) 7 windows len 400 with 3 cols

#This is where the preprocessing starts windowing allows better handling of larger data andhandle sequences. This can help find
#information that spans multiple windows
# For the example of stock data you can use a window of the past week's data to forcast the next day's price
# to predict you would do days[0-7] the next window would be days[1-8] and you would keep going
"""
The user dataset is formatted as an array of tuples that holds a sequences of readings and then it will hold the label for each of the sequences
[(array([[ 0.525208,  0.584137, -0.381042],
       [ 0.373001,  0.577896, -0.201477],
       [ 0.430847,  0.844421,  0.284988],
       ...,
       [ 0.732651,  0.480606,  0.309097],
       [ 0.8004  ,  0.495239,  0.24295 ],
       [ 1.051361,  0.53241 ,  0.152451]]), array(['dws', 'dws', 'dws', ..., 'dws', 'dws', 'dws'], dtype='<U3')), 
"""
def get_windows_dataset_from_user_list_format(user_datasets, window_size=400, shift=200, stride=1, verbose=0):
    """
    Create windows dataset in 'user-list' format using sliding windows
    Parameters:
        user_datasets
            dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}
        
        window_size = 400
            size of the window (output)
        shift = 200
            number of timestamps to shift for each window
            (200 here refers to 50% overlap, no overlap if =400)
        stride = 1
            stride of the window (dilation)
        verbose = 0
            debug messages are printed if > 0
    
    Return:
        user_dataset_windowed
            Windowed version of the user_datasets
            Windows from different trials are combined into one array
            type: {user_id: ( windowed_sensor_values, windowed_activity_labels)}
            windowed_sensor_values have shape (num_window, window_size, channels)
            windowed_activity_labels have shape (num_window)
            Labels are decided by majority vote
    """                                                                             #0        #1            #0      #1 here the label will correspond like below
    # this will get the sequences that correspond with the user for each user ex: ([[1,3,4], [2,2,1],...], ['dws',''dws')
    user_dataset_windowed = {}

    """
        what this does is it will take a user (based off their id that corresponds to their index)
        and then create windowed preprocessing format
        You will go through the users' data where there will be a list of many sequences with corresponding labels
        you will window each of these nested sequence (around 15 sequences) and flatten their labels 
    """
    for user_id in user_datasets: # go through all user id
        if verbose > 0: # if verbose print
            print(f"Processing {user_id}")
        x = []
        y = []

        # Loop through each trail of each user
        for v,l in user_datasets[user_id]:
            v_windowed = sliding_window_np(v, window_size, shift, stride) #turn the sensor data into windows (adds dimension)[[[2 2 2] [2 2 2]]] becomes [[2 2 2] [2 2 2]] the shape will be (7,400,3) 7 windows with 400 entries of arrays of size 3
            
            # flatten the window by majority vote (1 value for each window)
            l_flattened = sliding_window_np(l, window_size, shift, stride, flatten=get_mode) #this will flatten the label array to have one label for each window will be 1D
            if len(v_windowed) > 0: #if there were entries then:
                x.append(v_windowed) #add the windowed sensor data
                y.append(l_flattened) #add the flattened labels that correspond with the windowed data
            if verbose > 0: # if verbose printing
                print(f"Data: {v_windowed.shape}, Labels: {l_flattened.shape}") #print the shape

        # combine all trials of that user where this is a list of activities that are windowed this this will contain 15 entries in a list of the windoweded sequence data 15 X (windows x 400 x 3) this will result in an array of form: (15 x window) x 400 x 3 so  join the array on the first dimension
        user_dataset_windowed[user_id] = (np.concatenate(x), np.concatenate(y).squeeze()) # join the arrays along axis = 0 and squeeze remove entries of length one from the axes so
        # the concat will join all the windoweded sequences into one continuous grouping
        # the concat will join the arrays on the first dimension and will the joined data into a dict at their  corresponding position
        #squeeze will ([[1], [2], [3]]) -> [1 2 3] join items of size one collapses the dimensions allowingh for the labels to be in a similar position to the sequences

    return user_dataset_windowed # return all the windowed data  for all users

def combine_windowed_dataset(user_datasets_windowed, train_users, test_users=None, verbose=0):
    """
    Combine a windowed 'user-list' dataset into training and test sets
    Parameters:
        user_dataset_windowed
            dataset in the windowed 'user-list' format {user_id: ( windowed_sensor_values, windowed_activity_labels)}
        
        train_users
            list or set of users (corresponding to the user_id) to be used as training data
        test_users = None
            list or set of users (corresponding to the user_id) to be used as testing data
            if is None, then all users not in train_users will be treated as test users 
        verbose = 0
            debug messages are printed if > 0
    Return:
        (train_x, train_y, test_x, test_y)
            train_x, train_y
                the resulting training/test input values as a single numpy array
            test_x, test_y
                the resulting training/test labels as a single (1D) numpy array
    """
    
    train_x = [] # training set
    train_y = [] # training labels
    test_x = [] # testing set
    test_y = [] # testing labels
    for user_id in user_datasets_windowed: #go through the windowed dataset by user
        
        v,l = user_datasets_windowed[user_id] # get the windowdwd data for the user
        if user_id in train_users: # if the user is in the training set
            if verbose > 0:
                print(f"{user_id} Train") #print train if verbose
            train_x.append(v) # asdd that value to the training set
            train_y.append(l) # add the label to the training set
        elif test_users is None or user_id in test_users:  #They are in the testing set
            if verbose > 0:
                print(f"{user_id} Test")
            test_x.append(v) #put the user in the testing set
            test_y.append(l) #put their labels in the testing set
    # Above will check the test users or train and will divide up the inputs in to testing and training labels and window sequences

    if len(train_x) == 0: #if there is no training data j
        train_x = np.array([])
        train_y = np.array([])
    else:
        #train_x = [array([1, 2, 3]), array([4, 5, 6]), array([7, 8, 9])]
        # array([1, 2, 3, 4, 5, 6, 7, 8, 9]) to this using concatenate

        #squeeze wiould turn train_y = [array([1, 2]), array([3, 4]), array([5, 6])]
        # into array([1, 2],
        #       [3, 4],
        #       [5, 6])
        train_x = np.concatenate(train_x) # if train_x is a list of arrays then concatenate will combine them into a singale array along the default axis so this will be all user sequences in a long series
        # the above results int the form Array( [[[1,2,3], -> [[[1,2,3] ,[2,3,4], ...
        # It is important to note that they loop over the user and ensure values are added into the training set in order so that user sequences stay in the correct order
        train_y = np.concatenate(train_y).squeeze() # join all the labels together in the same fashion and the squeeze will get the format to be a 1d array
    
    if len(test_x) == 0: # if the testing is empty
        test_x = np.array([]) # add empty arraey
        test_y = np.array([]) # add empty lables
    else:
        test_x = np.concatenate(test_x) # otherwise do the same as you did to the training set
        test_y = np.concatenate(test_y).squeeze() # do the same to the labels

    return train_x, train_y, test_x, test_y # return the sets with their labels

def  get_mean_std_from_user_list_format(user_datasets, train_users):
    """
    Obtain and means and standard deviations from a 'user-list' dataset (channel-wise)
    from training users only
    Parameters:
        user_datasets
            dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}
        
        train_users
            list or set of users (corresponding to the user_ids) from which the mean and std are extracted
    Return:
        (means, stds)
            means and stds of the particular users (channel-wise)
            shape: (num_channels)
    """
    
    mean_std_data = []
    for u in train_users: #get the user id in the training users
        for data, _ in user_datasets[u]: #get the data corresponding to the user
            mean_std_data.append(data) #Add the data at that user sequence, this will hold all the sequences for the user add it to the mean std data
    mean_std_data_combined = np.concatenate(mean_std_data) # concatenate all the user sequence data and combine into a large array with arrays of size 3 (1126576,3)
    means = np.mean(mean_std_data_combined, axis=0) # take the mean of all the user data
    stds = np.std(mean_std_data_combined, axis=0) # take the standard deviation of all the data
    return (means, stds) # return the mean and std

def normalise(data, mean, std):
    """
    Normalise data (Z-normalisation)
    """

    return ((data - mean) / std)

def apply_label_map(y, label_map):
    """
    Apply a dictionary mapping to an array of labels
    Can be used to convert str labels to int labels
    Parameters:
        y
            1D array of labels
        label_map
            a label dictionary of (label_original -> label_new)
    Return:
        y_mapped
            1D array of mapped labels
            None values are present if there is no entry in the dictionary
    """

    y_mapped = []
    for l in y:
        y_mapped.append(label_map.get(l))
    return np.array(y_mapped)


def filter_none_label(X, y):
    """
    Filter samples of the value None
    Can be used to exclude non-mapped values from apply_label_map
    Parameters:
        X
            data values
        y
            labels (1D)
    Return:
        (X_filtered, y_filtered)
            X_filtered
                filtered data values
            
            y_filtered
                filtered labels (of type int)
    """

    valid_mask = np.where(y != None)
    return (np.array(X[valid_mask]), np.array(y[valid_mask], dtype=int))

def pre_process_dataset_composite(user_datasets, label_map, output_shape, train_users, test_users, window_size, shift, normalise_dataset=True, validation_split_proportion=0.2, verbose=0):
    """
    A composite function to process a dataset
    Steps
        1: Use sliding window to make a windowed dataset (see get_windows_dataset_from_user_list_format)
        2: Split the dataset into training and test set (see combine_windowed_dataset)
        3: Normalise the datasets (see get_mean_std_from_user_list_format)
        4: Apply the label map and filter labels (see apply_label_map, filter_none_label)
        5: One-hot encode the labels (see tf.keras.utils.to_categorical)
        6: Split the training set into training and validation sets (see sklearn.model_selection.train_test_split)
    
    Parameters:
        user_datasets
            dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}
        label_map
            a mapping of the labels
            can be used to filter labels
            (see apply_label_map and filter_none_label)
        output_shape
            number of output classifiction categories
            used in one hot encoding of the labels
            (see tf.keras.utils.to_categorical)
        train_users
            list or set of users (corresponding to the user_id) to be used as training data
        test_users
            list or set of users (corresponding to the user_id) to be used as testing data
        window_size
            size of the data windows
            (see get_windows_dataset_from_user_list_format)
        shift
            number of timestamps to shift for each window
            (see get_windows_dataset_from_user_list_format)
        normalise_dataset = True
            applies Z-normalisation if True
        validation_split_proportion = 0.2
            if not None, the proportion for splitting the full training set further into training and validation set using random sampling
            (see sklearn.model_selection.train_test_split)
            if is None, the training set will not be split - the return value np_val will also be none
        verbose = 0
            debug messages are printed if > 0
    
    Return:
        (np_train, np_val, np_test)
            three pairs of (X, y)
            X is a windowed set of data points
            y is an array of one-hot encoded labels
            if validation_split_proportion is None, np_val is None
    """

    # Step 1: This is where the data is partitioned
    #windowing helps capture local patterns if you have sequential data. This is where you divide the data into windows
    user_datasets_windowed = get_windows_dataset_from_user_list_format(user_datasets, window_size=window_size, shift=shift) #

    # Step 2: This will take the windowded data and generate the testing and training sets using the tuple of sequence to label
    train_x, train_y, test_x, test_y = combine_windowed_dataset(user_datasets_windowed, train_users, test_users)

    # Step 3 this is where the normalization occurs
    if normalise_dataset:
        #NOTE that they are getting the mean and std from the user dataset
        means, stds = get_mean_std_from_user_list_format(user_datasets, train_users) #take the mean and standard deviation of the data sets only for the training
        train_x = normalise(train_x, means, stds) # apply z normalization to the data set
        if len(test_users) > 0: # if there are testing useres
            test_x = normalise(test_x, means, stds) # appy z score normalization to the test set
        else:
            test_x = np.empty((0,0)) # if no testing users just put in empty tuple

    # Step 4: This takes the training labels and uses a label map that associates an activity with a numeric id and converts the training labels to integer ids
    # ['dws', 'dws',..], {'dw': 1, 'sit' : 2} -> [1,1,..]
    train_y_mapped = apply_label_map(train_y, label_map) # create mapping from integer representation to string representation
    test_y_mapped = apply_label_map(test_y, label_map) # map the test set too

    # Remvoe the none label
    train_x, train_y_mapped = filter_none_label(train_x, train_y_mapped) # rove any none values
    test_x, test_y_mapped = filter_none_label(test_x, test_y_mapped) # remove the none label

    if verbose > 0:
        print("Test")
        print(np.unique(test_y, return_counts=True))
        print(np.unique(test_y_mapped, return_counts=True))
        print("-----------------")

        print("Train")
        print(np.unique(train_y, return_counts=True))
        print(np.unique(train_y_mapped, return_counts=True))
        print("-----------------")

    # Step 5: Generate a one hot encoding for each of the training and the testing data, one hot will use a [0,0,0,1,0] to indicate what category it pertains to out of all the total categories
    train_y_one_hot = tf.keras.utils.to_categorical(train_y_mapped, num_classes=output_shape) # get one hot encoding forr the training
    test_y_one_hot = tf.keras.utils.to_categorical(test_y_mapped, num_classes=output_shape) # get the one hot encoding for the testing

    r = np.random.randint(len(train_y_mapped))  # check if mappings done correctly
    assert train_y_one_hot[r].argmax() == train_y_mapped[r] # makes sure the one hot encoding occurred correctly
    if len(test_users) > 0: # if you have testing users j
        r = np.random.randint(len(test_y_mapped))
        assert test_y_one_hot[r].argmax() == test_y_mapped[r] #check your mapping

    # Step 6: If there is a validation set then it will split the training set into the validation set
    if validation_split_proportion is not None and validation_split_proportion > 0: # create a validation set using the training set
        train_x_split, val_x_split, train_y_split, val_y_split = sklearn.model_selection.train_test_split(train_x, train_y_one_hot, test_size=validation_split_proportion, random_state=42)
    else:
        train_x_split = train_x #if there is not a validation set  just have training and testing
        train_y_split = train_y_one_hot # set to be the one hot encoding
        val_x_split = None # set validation to none
        val_y_split = None
        

    if verbose > 0: # printing of shapes
        print("Training data shape:", train_x_split.shape)
        print("Validation data shape:", val_x_split.shape if val_x_split is not None else "None")
        print("Testing data shape:", test_x.shape)

    np_train = (train_x_split, train_y_split) #training set  tuple
    np_val = (val_x_split, val_y_split) if val_x_split is not None else None # validation tuple
    np_test = (test_x, test_y_one_hot) #testing tuple

    # original_np_train = np_train
    # original_np_val = np_val
    # original_np_test = np_test

    return (np_train, np_val, np_test) #return the sets

def pre_process_dataset_composite_in_user_format(user_datasets, label_map, output_shape, train_users, window_size, shift, normalise_dataset=True, verbose=0):
    """
    A composite function to process a dataset which outputs processed datasets separately for each user (of type: {user_id: ( windowed_sensor_values, windowed_activity_labels)}).
    This is different from pre_process_dataset_composite where the data from the training and testing users are not combined into one object.
    Steps
        1: Use sliding window to make a windowed dataset (see get_windows_dataset_from_user_list_format)
        For each user:
            2: Apply the label map and filter labels (see apply_label_map, filter_none_label)
            3: One-hot encode the labels (see tf.keras.utils.to_categorical)
            4: Normalise the data (see get_mean_std_from_user_list_format)
    
    Parameters:
        user_datasets
            dataset in the 'user-list' format {user_id: [(sensor_values, activity_labels)]}
        label_map
            a mapping of the labels
            can be used to filter labels
            (see apply_label_map and filter_none_label)
        output_shape
            number of output classifiction categories
            used in one hot encoding of the labels
            (see tf.keras.utils.to_categorical)
        train_users
            list or set of users (corresponding to the user_id) to be used for normalising the dataset
        window_size
            size of the data windows
            (see get_windows_dataset_from_user_list_format)
        shift
            number of timestamps to shift for each window
            (see get_windows_dataset_from_user_list_format)
        normalise_dataset = True
            applies Z-normalisation if True
        verbose = 0
            debug messages are printed if > 0
    
    Return:
        user_datasets_processed
            Processed version of the user_datasets in the windowed format
            type: {user_id: (windowed_sensor_values, windowed_activity_labels)}
    """

    # Preparation for step 2
    if normalise_dataset:
        means, stds = get_mean_std_from_user_list_format(user_datasets, train_users)

    # Step 1 this will  window the data for each of the users and then join these windows into one continuous sequence of vectors
    user_datasets_windowed = get_windows_dataset_from_user_list_format(user_datasets, window_size=window_size, shift=shift)

    
    user_datasets_processed = {}
    for user, user_dataset in user_datasets_windowed.items():
        data, labels = user_dataset

        # Step 2
        labels_mapped = apply_label_map(labels, label_map)
        data_filtered, labels_filtered = filter_none_label(data, labels_mapped)

        # Step 3
        labels_one_hot = tf.keras.utils.to_categorical(labels_filtered, num_classes=output_shape)

        # random check
        r = np.random.randint(len(labels_filtered))
        assert labels_one_hot[r].argmax() == labels_filtered[r]

        # Step 4
        if normalise_dataset:
            data_filtered = normalise(data_filtered, means, stds)

        user_datasets_processed[user] = (data_filtered, labels_one_hot)

        if verbose > 0:
            print("Data shape of user", user, ":", data_filtered.shape)
    
    return user_datasets_processed

def add_user_id_to_windowed_dataset(user_datasets_windowed, encode_user_id=True, as_feature=False, as_label=True, verbose=0):
    """
    Add user ids as features or labels to a windowed dataset
    The user ids are appended to the last dimension of the arrays
    E.g. sensor values of shape (100, 400, 3) will become (100, 400, 4), and data[:, :, -1] will contain the user id
    Similarly labels of shape (100, 5) will become (100, 6), and labels[:, -1] will contain the user id
    
    Parameters:
        user_datasets_windowed
            dataset in the 'windowed-user' format type: {user_id: (windowed_sensor_values, windowed_activity_labels)}
        encode_user_id = True
            whether to encode the user ids as integers
            if True: 
                encode all user ids as integers when being appended to the np arrays
                return the map from user id to integer as an output
                note that the dtype of the output np arrays will be kept as float if they are originally of type float
            if False:
                user ids will be kept as is when being appended to the np arrays
                WARNING: if the user id is of type string, the output arrays will also be converted to type string, which might be difficult to work with
        as_feature = False
            user ids will be added to the windowed_sensor_values arrays as extra features if True
        as_label = False
            user ids will be added to the windowed_activity_labels arrays as extra labels if True
        verbose = 0
            debug messages are printed if > 0
    Return:
        user_datasets_modified, user_id_encoder
            user_datasets_modified
                the modified version of the input (user_datasets_windowed)
                with the same type {user_id: ( windowed_sensor_values, windowed_activity_labels)}
            user_id_encoder
                the encoder which maps user ids to integers
                type: {user_id: encoded_user_id}
                None if encode_user_id is False
    """

    # Create the mapping from user_id to integers
    if encode_user_id:
        all_users = sorted(list(user_datasets_windowed.keys()))
        user_id_encoder = dict([(u, i) for i, u in enumerate(all_users)])
    else:
        user_id_encoder = None

    # if none of the options are enabled, return the input
    if not as_feature and not as_label:
        return user_datasets_windowed, user_id_encoder

    user_datasets_modified = {}
    for user, user_dataset in user_datasets_windowed.items():
        data, labels = user_dataset

        # Get the encoded user_id
        if encode_user_id:
            user_id = user_id_encoder[user]
        else:
            user_id = user

        # Add user_id as an extra feature
        if as_feature:
            user_feature = np.expand_dims(np.full(data.shape[:-1], user_id), axis=-1)
            data_modified = np.append(data, user_feature, axis=-1)
        else:
            data_modified = data
        
        # Add user_id as an extra label
        if as_label:
            user_labels = np.expand_dims(np.full(labels.shape[:-1], user_id), axis=-1)
            labels_modified = np.append(labels, user_labels, axis=-1)
        else:
            labels_modified = labels

        if verbose > 0:
            print(f"User {user}: id {repr(user)} -> {repr(user_id)}, data shape {data.shape} -> {data_modified.shape}, labels shape {labels.shape} -> {labels_modified.shape}")

        user_datasets_modified[user] = (data_modified, labels_modified)
    
    return user_datasets_modified, user_id_encoder

def make_batches_reshape(data, batch_size):
    """
    Make a batched dataset from a windowed time-series by simple reshaping
    Note that the last batch is dropped if incomplete
    Parameters:
        data
            A 3D numpy array in the shape (num_windows, window_size, num_channels)
        batch_size
            the (maximum) size of the batches
    Returns:
        batched_data
            A 4D numpy array in the shape (num_batches, batch_size, window_size, num_channels)
    """

    max_len = (data.shape[0]) // batch_size * batch_size
    return data[:max_len].reshape((-1, batch_size, data.shape[-2], data.shape[-1]))

def np_random_shuffle_index(length):
    """
    Get a list of randomly shuffled indices
    """
    indices = np.arange(length)
    np.random.shuffle(indices)
    return indices

def ceiling_division(n, d):
    """
    Ceiling integer division
    """
    return -(n // -d)

def get_batched_dataset_generator(data, batch_size):
    """
    Create a data batch generator
    Note that the last batch might not be full
    Parameters:
        data
            A numpy array of data
        batch_size
            the (maximum) size of the batches
    Returns:
        generator<numpy array>
            a batch of the data with the same shape except the first dimension, which is now the batch size
    """

    num_bathes = ceiling_division(data.shape[0], batch_size)
    for i in range(num_bathes):
        yield data[i * batch_size : (i + 1) * batch_size]

    # return data[:max_len].reshape((-1, batch_size, data.shape[-2], data.shape[-1]))
