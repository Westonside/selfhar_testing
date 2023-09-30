import matplotlib.pyplot as plt


def plot_training_history(history):

    # this is a function that will plot the training history and plot the accuracy and loss
    plt.figure(figsize=(20,10)) #set the figure size


    plt.subplot(1,2,1) #plot the accuracy
    # get all values in the history that have accuracy in the name
    accuracy = [k for k in history.history.keys() if "accuracy" in k and "val" not in k]
    # now plot the accuracy for both training and validation data
    for acc in accuracy:
        plt.plot(history.history[acc], label=acc)
    plt.legend()
    plt.title("Accuracy")
    # add labels for each line in the plot
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")



    plt.subplot(1,2,2) #plot the loss
    # get all the keys from history that contain the word loss
    loss_keys = [k for k in history.history.keys() if "loss" in k and "val" not in k]
    # now plot each of these keys and their loss on a graph with the x as epochs
    for k in loss_keys:
        plt.plot(history.history[k], label=k)


    plt.legend()
    plt.title("Loss")
    # save the figure
    plt.savefig("loss.png")
    plt.show()




