import matplotlib.pyplot as plt


def plot_training_history(history):
    # this is a function that will plot the training history and plot the accuracy and loss
    plt.figure(figsize=(20,10)) #set the figure size
    plt.subplot(1,2,1) #plot the accuracy
    plt.plot(history.history['accuracy']) #plot the accuracy
    plt.plot(history.history['val_accuracy']) #plot the validation accuracy
    plt.title('model accuracy') #set the title
    plt.ylabel('accuracy') #set the y label
    plt.xlabel('epoch') #set the x label
    plt.legend(['train', 'validation'], loc='upper left') #set the legend
    plt.subplot(1,2,2) #plot the loss
    plt.plot(history.history['loss']) #plot the loss
    plt.plot(history.history['val_loss']) #plot the validation loss
    plt.title('model loss') #set the title
    plt.ylabel('loss') #set the y label
    plt.xlabel('epoch') #set the x label
    plt.legend(['train', 'validation'], loc='upper left') #set the legend
    plt.show() #show the plot
    # save the plots to a file
    plt.savefig('model_accuracy_loss.png') #save the plot
    plt.close() #close the plot


