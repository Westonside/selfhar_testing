



## MultiTask Learning
373 main file uses multitask learning teach model to do multiple things at a time 
Use single model to do multiple things ex: given an image do object detection, scene classification, ...
This allows for more efficieny and better generalization so you would share some layers with different but related tasks
Use shared backbone with multiple heads, each head will be dedicated to one task
When give more capacity to shared layers and less to the individual heads then the tasks would be more tightly coupled since they are sharing more features 
So the amount they share depends on how close the tasks are
Seemingly related things can hurt the performance of others if they share training 
Can also define soft parameter sharing you determine how much of the features should be shared
Learning multiple tasks together in MTL can help a model  learn a better representation 
The loss function will be a combination of all the head losses
You need to accomodate on losses importance how they change, ...

The second layer of the selfHAR in the json file is the pretraining it selects the values that the teacher predicted and it specifies the minimum confidence in the predictions


## Transfer Learning
#freeze up to the freeze laye (only freeze first 2 conv layers), this allows the model to fine tune to the final task dataset while retaining previously learned knowledge
# Summary of the SelfHAR paper
How this model works is as follows:
## Preprocessing
## Teacher model
The teacher model will be configured as follows: A core CNN model will be created and then on top of that model a HAR classification head will be added. A softmax 
layer will be applied meaning that the output will be a probability distribution will be outputted. The Teacher model will be trained on the labelled data. 
Once the teacher model has been trained it will then predict upon the large unlabelled dataset. Using the prediction spread, the samples that have the highest confidence
from the teacher will then be filtered and added to a new dataset. The training set for the teacher will also be added into the teacher labelled dataset.

## Teacher labelled dataset
The teacher labelled dataset will then have a number of transformations applied to the samples(8 transformations generating 8 different versions of the data).
Now the teacher labeled data will have the following labels: the transformation label and the HAR label from the teacher.


### Student Model
The student model will have a CNN core like the teacher model but it will also have a HAR head and a nubmer of transformation heads (corresponding to the number of transformation functions)
The multiple heads allow for Multi-Task Learning where the model will try to predict the HAR activity and a binary classification for what transformation is applied,
the MTL approach allows for the model to better learn about the dataset and ignore any additional noise, it allows for better convergence (reach point where loss and learning slows)
After the student has been trained on the teacher classified data, the early layers of the student model(CNN core) will be frozen. The HAR classification branch will be fine tuned 
using the original input labelled data.

### window size 
 They say : "Each dataset is then segmented into sliding windows with size 400 × 3, where 400 is the number of time stamps and 3 being the number of channels of a triaxial accelerometer"

## New Terms
### Confusion Matrix
Performance measure for ML classification prooblems where there are two or more output classes.
Useful for measuring Recall, Precision, Specificity, Accuracy and AUC ROC(most important)