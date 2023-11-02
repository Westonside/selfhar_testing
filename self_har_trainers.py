import tensorflow as tf
import os

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2021 C. I. Tang"

import self_har_training_visualization

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

def composite_train_model(
    full_model, 
    training_set,
    working_directory, 
    callbacks, 
    epochs,
    validation_set=None,
    batch_size=200,
    tag="har", 
    use_tensor_board_logging=True,
    steps_per_epoch=None,
    verbose=1,
    single_train=False,
    name=None
):
    # This function will be called for the teacher model and when after you have frozen layers in the student model to fine tune
    best_model_file_name = os.path.join(working_directory, "models", f"{tag}_best.hdf5") #write out the path for the best model
    last_model_file_name = os.path.join(working_directory, "models", f"{tag}_last.hdf5") #write the file name for the last model trained
    best_model_callback = tf.keras.callbacks.ModelCheckpoint(
        best_model_file_name, #create a check point for the model by saving the file, saves the best model
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        save_weights_only=False,
        verbose=verbose
    )
    #the test[1] will have 'noised': ndarraay that is the label for the noised head
    local_callbacks = [best_model_callback] #store the callback

    if use_tensor_board_logging: #if tensor board
        logdir = os.path.join(working_directory, "logs", tag) #the logging directory
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir) # the callback for the tensor board
        local_callbacks.append(tensorboard_callback) #add the callback to the tenasorboard calback to log if tensorboard logging is allowed

    local_callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, verbose=verbose)) #add the early stopping callback
    #each head gets the same input and they will each predict on one of the transformation
    training_history = full_model.fit(
        x=training_set[0],
        y=training_set[1],
        validation_data=(validation_set[0], validation_set[1]), # this will train the model on the teacher labelled data

        batch_size=batch_size,
        shuffle=True,
        epochs=epochs,
        callbacks = callbacks + local_callbacks,

        steps_per_epoch=steps_per_epoch,
        verbose=verbose
    ) #this is where the traiing will occur
    if single_train:
        self_har_training_visualization.plot_training_history(training_history, name) #plot the training history
    
    full_model.save(last_model_file_name) # at the end save the model
    
    return best_model_file_name, last_model_file_name # return the best model file anme and the last trained model filename



