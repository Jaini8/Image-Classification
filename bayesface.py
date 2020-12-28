import numpy as np
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import readdata as rd

def Predictor(pixel_count,prior_prob,test_img):
    prob_face = [1]*2
    pre_prob_image = [0]*168
    for i in range(0,2):
        for j in range(0,168):
            pre_prob_image[j] = pixel_count[i,j,int(test_img[j])]
            if pre_prob_image[j] == 0:
                pre_prob_image[j] = 0.001
        for k in pre_prob_image:
            prob_face[i] = prob_face[i]*k
    for i in range(0,2):
        prob_face[i] = prob_face[i]*prior_prob[i]
    return prob_face.index(max(prob_face))


# source = "D:/project2 AI/facedata/facedatatrain"
# source1 = "D:/project2 AI/facedata/facedatatrainlabels"
# source2 = "D:/project2 AI/facedata/facedatatest"
# source3 = "D:/project2 AI/facedata/facedatatestlabels"

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

for i in range(0,10):
    start = time.time()
    tem -= 0.10
    if tem < 0:
        tem = 0.001
    x_train,x_test,y_train,y_test = train_test_split(X_train,Y_train_labels,test_size=tem, random_state=45)
    new_x = np.zeros((len(x_train),14, 12))
    new_X = []
    for i in range(0,len(x_train)):
        for n_r in range(0,14):
            for n_c in range(0,12):
                for row in range(5 * n_r, 5 * (n_r + 1)):
                    for col in range(5 * n_c, 5 * (n_c + 1)):
                        new_x[i][n_r][n_c] += x_train[i][row][col]
        new_X.append(new_x[i].flatten())

    new_test_x = np.zeros((len(X_test),14, 12))
    new_test_X = []
    for i in range(0,len(X_test)):
        for n_r in range(0,14):
            for n_c in range(0,12):
                for row in range(5 * n_r, 5 * (n_r + 1)):
                    for col in range(5 * n_c, 5 * (n_c + 1)):
                        new_test_x[i][n_r][n_c] += X_test[i][row][col]
        new_test_X.append(new_test_x[i].flatten())

    pixel_count = np.zeros((2,168,18))
    image_count = [0,0]
    for w in range(0,len(new_X)):
        k = int(y_train[w])
        image_count[k] += 1
        for i in range(0,len(new_X[w])):
            pixel_count[k, i, int(new_X[w][i])] += 1

    for i in range(0,len(image_count)):
       pixel_count[i,:,:] = (pixel_count[i,:,:])/image_count[i]

    prior = [0]*2
    for i in range(0,len(image_count)):
        prior[i] = image_count[i]/len(y_train)

    # print(prior)
    end = time.time()
    total_training_time += end - start
    predicted_val = []
    for w in range(0, len(new_test_X)):
        pred_lab = Predictor(pixel_count,prior,new_test_X[w])
        predicted_val.append(pred_lab)
        # print(y_test[0])
        # print(predicted_val[0])

    print(predicted_val)
    print(Y_test_labels)
    final_cnt = 0
    for i in range(0,len(Y_test_labels)):
        if int(Y_test_labels[i]) == predicted_val[i]:
            final_cnt +=1

    print(final_cnt/len(Y_test_labels))
    accuracy_array.append(final_cnt / len(Y_test_labels) * 100)

plt.plot(percent_training,accuracy_array,'-ko', linewidth=1, markersize=3)
plt.xlabel("Partition Percentage")
plt.ylabel("Accuracy")
end1 = time.time()
plt.show()

print("Total training time: ",total_training_time," seconds")
print("Total time taken: ",end1-start1," seconds")