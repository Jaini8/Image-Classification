import readdata as rd
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split


def Predictor(pixel_count, prior_prob,test_img):
    prob_digit = [1] * 10
    pre_prob_image = [0] * 28
    for i in range(0, 10):
        for j in range(0, 28):
            pre_prob_image[j] = pixel_count[i, j, test_img[j]]
            if pre_prob_image[j] == 0:
                pre_prob_image[j] = 0.00001
        for k in pre_prob_image:
            prob_digit[i] = prob_digit[i] * k
    for i in range(0, 10):
        prob_digit[i] = prob_digit[i] * prior_prob[i]
    return prob_digit.index(max(prob_digit))


source_train_images = "/Users/jainipatel/Downloads/data/digitdata/trainingimages"
source_train_labels = "/Users/jainipatel/Downloads/data/digitdata/traininglabels"
source_test_images = "/Users/jainipatel/Downloads/data/digitdata/testimages"
source_test_labels = "/Users/jainipatel/Downloads/data/digitdata/testlabels"

# source_train_images = "D:/project2 AI/digitdata/trainingimages"
# source_train_labels = "D:/project2 AI/digitdata/traininglabels"
# source_test_images = "D:/project2 AI/digitdata/testimages"
# source_test_labels = "D:/project2 AI/digitdata/testlabels"

fetch_data_train = rd.load_data(source_train_images, 5000, 28, 28)
fetch_data_test = rd.load_data(source_test_images, 1000, 28, 28)
Y_train_labels = labels = rd.load_label(source_train_labels)
X_train = rd.matrix_transformation(fetch_data_train, 28, 28)
X_test = rd.matrix_transformation(fetch_data_test, 28, 28)
Y_test_labels = rd.load_label(source_test_labels)
# print(len(Y1))

tem = 0.99
accuracy_array = []
percent_training = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
total_training_time = 0
start1 = time.time()

for i in range(0, 10):
    start = time.time()
    tem -= 0.10
    if tem < 0:
        tem = 0.001
    x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train_labels, test_size=tem, random_state=45)

    pixel_count = np.zeros((10, 28, 22))
    image_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for w in range(0, len(x_train)):
        temp = -1
        k = int(y_train[w])
        image_count[k] += 1
        for i in x_train[w]:
            temp += 1
            count = 0
            for j in i:
                if j == 1:
                    count += 1
            pixel_count[k, temp, count] += 1

    for i in range(0, len(image_count)):
        pixel_count[i, :, :] = (pixel_count[i, :, :]) / image_count[i]

    prior = [0] * 10

    for i in range(0, len(image_count)):
        prior[i] = image_count[i] / len(y_train)

    end = time.time()
    total_training_time += end - start
    img_di = [0] * 28
    predicted_val = []

    for w in range(0, len(X_test)):
        count = 0
        temp = -1
        for i in X_test[w]:
            temp += 1
            count = 0
            for j in i:
                if j == 1:
                    count += 1
            img_di[temp] = count
        pred_lab = Predictor(pixel_count, prior, img_di)
        predicted_val.append(pred_lab)

    print(predicted_val)
    print(Y_test_labels)
    final_cnt = 0

    for i in range(0, len(Y_test_labels)):
        if int(Y_test_labels[i]) == predicted_val[i]:
            final_cnt += 1

    print(final_cnt/len(Y_test_labels)*100)
    accuracy_array.append(final_cnt / len(Y_test_labels) * 100)

# print(res)

plt.plot(percent_training, accuracy_array, '-ko', linewidth=1, markersize=3)
plt.xlabel("Partition Percentage")
plt.ylabel("Accuracy")
end1 = time.time()
plt.show()

print("Total training time: ", total_training_time, " seconds")
print("Total time taken: ", end1 - start1, " seconds")