import numpy as np
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import readdata as rd

def training_neural(x_train,y_tain,epoch,label):
    features = len(x_train[0])
    weights = np.random.rand(features, label)
    # print(np.shape(weights))
    epoch_count = []
    accuracy_each_epoch = []
    for epo in range(0, epoch):
        print("Epoch %d/%d" % (epo + 1, epoch))
        epoch_count.append(epo + 1)
        error = 0
        for i in range(0, len(x_train)):
            curr_label = y_train[i]
            temp_numpy = np.zeros((features, 1))
            for j in range(0, features):
                temp_numpy[j] = x_train[i][j]
            # print(np.shape(temp_numpy))
            dot_product = np.dot(weights.T, temp_numpy)
            # print(dot_product)
            predicted_label = np.argmax(dot_product)
            # print(predicted_label)
            # print(curr_label)
            if predicted_label != curr_label:
                error = error + 1
                weight_Diff = (weights[:, predicted_label] - weights[:, curr_label])
                dotProduct_feature = np.dot(temp_numpy.T, temp_numpy)

                tau_numerator = np.dot(weight_Diff.T, temp_numpy) + 1
                tau_denom = 2 * dotProduct_feature
                tau = tau_numerator / tau_denom

                weights[:, curr_label] = weights[:, curr_label] + (tau * temp_numpy[:, 0])
                weights[:, predicted_label] = weights[:, predicted_label] - (tau * temp_numpy[:, 0])
        accuracy = 100 - ((error / len(x_train)) * 100)
        accuracy_each_epoch.append(accuracy)
        print("Accuracy: ", accuracy)
        if (accuracy == 100.0):
            break
    return weights, epoch_count, accuracy_each_epoch

def testing_neural(x_test,y_test,weights_learned):
    features = len(x_test[0])
    error = 0
    for i in range(0,len(x_test)):
        curr_label = y_test[i]
        temp_numpy = np.zeros((features,1))
        for j in range(0,features):
            temp_numpy[j] = x_test[i][j]
        dot_product = np.dot(weights_learned.T, temp_numpy)
        predicted_label = np.argmax(dot_product)
        if (predicted_label != curr_label):
            error +=1
    return (100 - (error/len(y_test))*100)



source_train_images = "/Users/jainipatel/Downloads/data/facedata/facedatatrain"
source_train_labels = "/Users/jainipatel/Downloads/data/facedata/facedatatrainlabels"
source_test_images = "/Users/jainipatel/Downloads/data/facedata/facedatatest"
source_test_labels = "/Users/jainipatel/Downloads/data/facedata/facedatatestlabels"


fetch_data_train = rd.load_data(source_train_images, 451, 70, 60)
fetch_data_test = rd.load_data(source_test_images, 150, 70, 60)
Y_train_labels = labels = rd.load_label(source_train_labels)
X_train = rd.matrix_transformation(fetch_data_train, 70, 60)
X_test = rd.matrix_transformation(fetch_data_test, 70, 60)
Y_test_labels = rd.load_label(source_test_labels)


tem = 0.99
accuracy_array = []
percent_training = [10,20,30,40,50,60,70,80,90,100]
total_training_time = 0
start1 = time.time()

new_x_test = np.zeros((len(X_test), 14, 12))
feature_list_test = [0] * len(new_x_test)

for i in range(0, len(X_test)):
    for n_r in range(0, 14):
        for n_c in range(0, 12):
            for row in range(5 * n_r, 5 * (n_r + 1)):
                for col in range(5 * n_c, 5 * (n_c + 1)):
                    new_x_test[i][n_r][n_c] += X_test[i][row][col]
    feature_list_test[i] = new_x_test[i].flatten()

X_feat = feature_list_test
Y_labels = list(map(int,Y_test_labels))

for i in range(0,10):
    start = time.time()
    tem -= 0.10
    if tem < 0:
        tem = 0.001
    x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train_labels,test_size=tem, random_state=45)
    # print(len(x_train))
    label_count = [0] *2 #list of count of labels
    new_x = np.zeros((len(x_train),14, 12))
    feature_list = [0]* len(new_x)

    for i in range(0,len(x_train)):
        label = int(y_train[i])
        label_count[label] += 1
        for n_r in range(0,14):
            for n_c in range(0,12):
                for row in range(5 * n_r, 5 * (n_r + 1)):
                    for col in range(5 * n_c, 5 * (n_c + 1)):
                        new_x[i][n_r][n_c] += x_train[i][row][col]
        feature_list[i] = new_x[i].flatten()

    y_train = list(map(int,y_train))
    y_test = list(map(int,y_test))
    x_train = feature_list

    weights_learned,epoch_count,counter = training_neural(x_train,y_train,170,2)
    end = time.time()
    total_training_time += end - start
    predict = testing_neural(X_feat,Y_labels,weights_learned)
    accuracy_array.append(predict)


fig, axs = plt.subplots(2)
axs[0].plot(percent_training,accuracy_array,'-ko', linewidth=1, markersize=3)
axs[0].set_xlabel("Partition Percentage")
axs[0].set_ylabel("Accuracy")

axs[1].plot(epoch_count,counter,'-b')
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Accuracy")
end1 = time.time()
plt.show()

print(accuracy_array)
print("Total training time: ",total_training_time," seconds")
print("Total time taken: ",end1-start1," seconds")