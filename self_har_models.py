import keras
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


def create_1d_conv_core_model(input_shape, multi_modalities=None, model_name="base_model",
                              use_standard_max_pooling=False):
    """
    Create the base model for activity recognition
    Reference (TPN model):
        Saeed, A., Ozcelebi, T., & Lukkien, J. (2019). Multi-task self-supervised learning for human activity detection. Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 3(2), 1-30.
    Architecture:
        Input
        -> Conv 1D: 32 filters, 24 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 64 filters, 16 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Conv 1D: 96 filters, 8 kernel_size, relu, L2 regularizer
        -> Dropout: 10%
        -> Global Maximum Pooling 1D
    
    Parameters:
        input_shape
            the input shape for the model, should be (window_size, num_channels)
    
    Returns:
        model (tf.keras.Model)
    """
    """
    this model wil
    create convolution layers in the form
     conv 1d (32 filters 24 x 24) 
     dropout layer
     conv 1d (64 filters with 16x16 filter)
     droupout
     conv 1d (96 with 8x8 filter)
     dropout
     Pooling layer(either max pool or global max pooling)
    """
    if multi_modalities is not None:
        inputs = tf.keras.Input(shape=input_shape,
                                name='input')  # create the input leayer which takes a windowed size amount of data (400,3)
    else:
        inputs = tf.keras.Input(shape=input_shape,
                                name='input')  # create the input leayer which takes a windowed size amount of data (400,3)
    x = inputs  # put the input layer
    x = tf.keras.layers.Conv1D(
        # the first param is the number of filers, 32 filters with a kernel size of 24  you are adding now a convolutional layer
        32, 24,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)
        # regularizers allow applying a penalty to the layer's kernel these penatlties are summed into the loss function and these penalties are applied on a per layer basis
    )(x)
    x = tf.keras.layers.Dropout(0.1)(
        x)  # now have a dropout layer where there is a 10% dropout to input layers, avoiding overfitting only applied when training is true

    x = tf.keras.layers.Conv1D(  # another conv 1d layer with 64 filter and a 16 sized filter
        64, 16,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    x = tf.keras.layers.Conv1D(
        96, 8,
        activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
    )(x)
    x = tf.keras.layers.Dropout(0.1)(x)

    if use_standard_max_pooling:
        x = tf.keras.layers.MaxPool1D(pool_size=x.shape[1], padding='valid', data_format='channels_last',
                                      name='max_pooling1d')(x)
        x = tf.keras.layers.Reshape([x.shape[-1]], name='reshape_squeeze')(
            x)  # the ouput of the 1d max pooling will be generally 3D so this will reshapoe the tensor to be [x.shape[-1]] specifies the target shape which would be to 1D last dimension of the shape
    else:
        x = tf.keras.layers.GlobalMaxPool1D(data_format='channels_last', name='global_max_pooling1d')(
            x)  # add a global max pooling layer

    return tf.keras.Model(inputs, x, name=model_name)  # return the model


def extract_core_model(composite_model):
    return composite_model.layers[1]  # this will get the functional layer from the model


def extract_har_model(multitask_model, optimizer, output_index=-1, model_name="har"):
    model = tf.keras.Model(inputs=multitask_model.inputs, outputs=multitask_model.outputs[output_index],
                           name=model_name)  # if the output index is -1 then you will get the HAR classifier output and ignore the other transformation classifiers

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model


def set_freeze_layers(model, num_freeze_layer_index=None):
    if num_freeze_layer_index is None:  # for the fine tune this takes the trained early conv layers are to be frozen
        for layer in model.layers:  # if you are in this if statement this means that you have evaluated the model and now you shall freeze all the layers of the core model
            layer.trainable = False
    else:
        for layer in model.layers[
                     :num_freeze_layer_index]:  # freeze up to the freeze laye (only freeze first 2 conv layers), this allows the model to fine tune to the final task dataset while retaining previously learned knowledge
            layer.trainable = False  # disable the layer you are freezing the first 2 convolution layers
        for layer in model.layers[
                     num_freeze_layer_index:]:  # keep the other layers after the frozen ones trainable so this will be one convolutional layer
            layer.trainable = True


def attach_full_har_classification_head(core_model, output_shape,
                                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), num_units=1024,
                                        model_name="HAR"):
    """
    Create a full 2-layer classification model from the base mode, using activitations from an intermediate layer with partial freezing
    Architecture:
        base_model-intermediate_layer
        -> Dense: 1024 units
        -> ReLU
        -> Dense: output_shape units
        -> Softmax
    
    Optimizer: Adam
    Loss: CategoricalCrossentropy
    Parameters:
        base_model
            the base model from which the activations are extracted, in our case this will be the 1d convolution model
        this add a classification layer as the head
        output_shape
            number of output classifiction categories
        model_name
            name of the output model
        intermediate_layer
            the index of the intermediate layer from which the activations are extracted
        last_freeze_layer
            the index of the last layer to be frozen for fine-tuning (including the layer with the index)
    
    Returns:
        trainable_model (tf.keras.Model)
    """
    # for the teacher network this will add the input head and the har classifier that uses the softmax to have the probability of the classification
    inputs = tf.keras.Input(shape=core_model.input.shape[1:],
                            name='input')  # create the input layer to the classifier this will expect (400, 3) take inputs from the layer before
    intermediate_x = core_model(inputs)  # add the inputs to the core model the output of the core model

    x = tf.keras.layers.Dense(num_units, activation='relu')(intermediate_x)
    x = tf.keras.layers.Dense(output_shape)(x)
    outputs = tf.keras.layers.Softmax()(
        x)  # softmax will distribute the values of the vector to a proability distribution

    model = tf.keras.Model(inputs=inputs, outputs=outputs,
                           name=model_name)  # create a model with those inputs and those outputs the ouput being the probabilities

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    return model


def attach_linear_classification_head(core_model, output_shape, optimizer=tf.keras.optimizers.SGD(learning_rate=0.03),
                                      model_name="Linear"):
    """
    Create a linear classification model from the base mode, using activitations from an intermediate layer
    Architecture:
        base_model-intermediate_layer
        -> Dense: output_shape units
        -> Softmax
    
    Optimizer: SGD
    Loss: CategoricalCrossentropy
    Parameters:
        base_model
            the base model from which the activations are extracted
        
        output_shape
            number of output classifiction categories
        intermediate_layer
            the index of the intermediate layer from which the activations are extracted
    
    Returns:
        trainable_model (tf.keras.Model)
    """

    inputs = tf.keras.Input(shape=core_model.input.shape[1:],
                            name='input')  # this will create an input layer of shape (400,3) you do the [1:] to skip the batch will generally be (None, 400,3)
    intermediate_x = core_model(inputs)  # have the input layer take the input of the cnn core
    # inputs will be passed through this inputs layer then through the cnn core and other layers the outputs will then feed the output out through the softmax
    x = tf.keras.layers.Dense(output_shape, kernel_initializer=tf.random_normal_initializer(stddev=.01))(
        intermediate_x)  # add a dense layer with a random kernel init giving random weights to start with with a std of 0.01 and this will take the inputs of the input layer
    outputs = tf.keras.layers.Softmax()(x)  # add a softmax output layer that will return probability distribution

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)  # create the model

    model.compile(
        optimizer=optimizer,  # compile the model
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"), tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )
    return model


def attach_multitask_transform_head(core_model, output_tasks, optimizer, with_har_head=False, har_output_shape=None,
                                    num_units_har=1024, model_name="multitask_transform", num_features=256):
    """
    Note: core_model is also modified after training this model (i.e. the weights are updated)
    """
    # multi-task learning allows sharing information from training signals of related tasks
    # there are two types of MTL hard or soft parameter sharing of hidden parmams
    # hard param sharing wil share hiddent layers between tasks while keeping several task-specific output layers
    # multitask learning is where learn a nubmer of inputs at once
    # the core model will be the feature extractor  this will attach additional layers to itself
    # the output tasks are the strings of the tasks that you want the multi task model to perform each task is one output head in the multi-task model
    # optimizer used for training the multi-task model
    # the inputs will have the same dimenisions except for the first dimension (whicih usually corresponds to batch size??) this rmeove the first none label in (None, 400,3)
    inputs = tf.keras.Input(shape=core_model.input.shape[1:],
                            name='input')  # set an input layer coming from the base inputs that has a defined in the base convolutional layer
    intermediate_x = core_model(
        inputs)  # adding an input layer to the core model that will hold the output of the core model
    outputs = []
    losses = [tf.keras.losses.BinaryCrossentropy() for _ in
              output_tasks]  # create an array of bCE loss for all output tasks related to the teransformation binary classificaiton task and then you will add an additional loss for the har classification loss

    # this will add for the HAR head, the NoisedTask Head
    for task in output_tasks:  # create dense layers for each of the output tasks which are the transformation bin classification tasks
        # TODO look at 256 value 256 will be the features the cnn will generate these  this line tells cnn generate 256 features justify why 256
        # Look to see if enlarging the 256 will improve the performance of the transformation classification
        x = tf.keras.layers.Dense(256, activation='relu')(
            intermediate_x)  # add a dense layer to the intermediate: NOTE THIS IS WHERE THE MUILTI HEAD CONFIG STARTS you are adding a bin classification head to the intermediate layer for each transformation
        pred = tf.keras.layers.Dense(1, activation='sigmoid', name=task)(
            x)  # this function will be the output for the binary classification, sigmoid will between 0 and 1 and it will have one output give a name for the predictor so you know which prediction task goes with it
        outputs.append(pred)  # add the binary classificaiton task head output the outputs tasks

    if with_har_head:  # if there is a HAR head there should only be one har head this will predict the human activity
        x = tf.keras.layers.Dense(num_units_har, activation='relu')(
            intermediate_x)  # add a dense layer connected to the intermediate layer
        x = tf.keras.layers.Dense(har_output_shape)(x)  # add another dense layer with the har output shape
        har_pred = tf.keras.layers.Softmax(name='har')(
            x)  # apply the softamx which iwll convert a vector of numbers into a probability distribution  this will be the prediction layer that classifices the activity

        outputs.append(har_pred)  # for each of these tasks append the har prediction layer
        losses.append(
            tf.keras.losses.CategoricalCrossentropy())  # to the losses add the loss fn for Categorical CE used for the HAR classification task

    # TODO: allow combining another models's outputs and take the weighted average of the outputs
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=model_name)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics=['accuracy']
    )

    return model


# inspiration for the mulithead input taken from hart
# this will fuse the two sensor modalities together
def attach_multihead_input(inputs, dropout_rate=0.0, conv_layers=2):
    ins = []
    outputs = []
    for i in range(len(inputs)):
        # for each of the sensor modalities, create an input_layer layer
        input_layer = tf.keras.Input(shape=inputs[i].shape, name='input_layer' + str(i))

        x = input_layer
        for _ in range(conv_layers):
            x = tf.keras.layers.Conv1D(
                32, 24,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l=1e-4)
            )(x)
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        # flatten the oupput of the cnn
        x = tf.keras.layers.Flatten()(x)  # flatten out the inputs
        output = tf.keras.layers.Dense(64, activation='softmax')(x)
        ins.append(input_layer)
        outputs.append(output)

    if len(inputs) > 1:
        combined = tf.keras.layers.concatenate(outputs)
    else:
        combined = outputs[0]

    output_layer = tf.keras.layers.Dense(64, activation='relu')(combined)
    return inputs, output_layer


def test_number_features(input_shape, transform_funcs_names, features, initial_learning_rate=0.001):
    model_holder = []
    for feature in features:
        core = create_1d_conv_core_model(input_shape)
        model_holder.append(attach_multitask_transform_head(core, output_tasks=transform_funcs_names,
                                                            optimizer=tf.keras.optimizers.Adam(
                                                                learning_rate=initial_learning_rate),
                                                            num_features=feature))  # add the model with the feature to the list
    for model in model_holder:
        model.summary()
    return model_holder


def test_epochs(num_tests, input_shape, transform_funcs_names, initial_learning_rate=0.001):
    model_holder = []
    for i in range(num_tests):
        core = create_1d_conv_core_model(input_shape)
        model_holder.append(attach_multitask_transform_head(core, output_tasks=transform_funcs_names,
                                                            optimizer=tf.keras.optimizers.Adam(
                                                                learning_rate=initial_learning_rate),
                                                            num_features=1024))  # add the model with the feature to the list
    return model_holder
