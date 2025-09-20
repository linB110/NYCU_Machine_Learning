# import necessary lib such as calculation ,plotting...
import pandas as pd
import matplotlib.pyplot as plt
from HW1_KNN import *

# loading dataset
from sklearn import datasets
iris = datasets.load_iris()

# check data content
#print("Iris dataset attributes : ", iris.keys())
#print("data shape : ", iris.data.shape)

# transfer to pandas dataframe
import pandas as pd
data_frame = pd.DataFrame(iris.data, columns=iris.feature_names) # construct a data file based on feature names
data_frame["target"] = iris.target  # label
data_frame["target_name"] = iris.target_names[iris.target] # class name

#print(data_frame.head(n))   n => represent how many data to print, default value is 5
#print(data_frame.head())


# ====== visualization of dataset ======
# plot figure function
def plot_figure(x_label, y_label,data_frame):
    colors = ['red', 'orange', 'green']
    species = iris.target_names
    
    plt.figure(figsize=(8, 6))
    for i, sp in enumerate(species):
        mask = data_frame["target"] == i
        subset = data_frame[mask]
        
        plt.scatter(
            subset[x_label],
            subset[y_label],
            color=colors[i],
            label=sp,
            alpha=0.8, edgecolors="k", linewidths=0.5
        )
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"Scatter Plot of {x_label} vs {y_label}")
    plt.legend(title="Species")


setosa_data = data_frame[:50]
versicolor_data = data_frame[50:100]
virginica_data = data_frame[100:]
#print(len(setosa_data))
#print(len(versicolor_data))
#print(len(virginica_data))

# plotting 2-D scatter 
#plot_figure("sepal length (cm)", "petal width (cm)", data_frame)
#plot_figure("sepal length (cm)", "petal width (cm)", setosa_data)
#plot_figure("sepal length (cm)", "petal width (cm)")
plt.show()

# ====== feature selection ======
feature1 = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
feature2 = ["petal length (cm)", "petal width (cm)"]
feature3 = ["sepal length (cm)", "sepal width (cm)"]

# change used feature for training dataset
training_feature = [feature1, feature2, feature3]

# ====== splitting dataset (binary classification) =======
data_train_set_ver = pd.concat([setosa_data[0:25], versicolor_data[0:25]], ignore_index=True)
data_train_set_vir = pd.concat([setosa_data[0:25], virginica_data[0:25]], ignore_index=True)
data_train_ver_vir = pd.concat([versicolor_data[0:25], virginica_data[0:25]], ignore_index=True)

label_train_set_ver = data_train_set_ver["target"].to_numpy()
label_train_set_vir = data_train_set_vir["target"].to_numpy()
label_train_ver_vir = data_train_ver_vir["target"].to_numpy()

data_test_set_ver = pd.concat([setosa_data[25:50], versicolor_data[25:50]], ignore_index=True)
data_test_set_vir = pd.concat([setosa_data[25:50], virginica_data[25:50]], ignore_index=True)
data_test_ver_vir = pd.concat([versicolor_data[25:50], virginica_data[25:50]], ignore_index=True)

label_test_set_ver = data_test_set_ver["target"].to_numpy()
label_test_set_vir = data_test_set_vir["target"].to_numpy()
label_test_ver_vir = data_test_ver_vir["target"].to_numpy()



# ====== splitting dataset (trinary classification) =======
# training dataset
data_train = pd.concat([setosa_data[0:25], versicolor_data[0:25], virginica_data[0:25]], ignore_index=True)
label_train = data_train["target"].to_numpy()
# print("=========")
# print(data_train)
# print("=========")

# print("=========")
# print(label_train)
# print("=========")

# testing dataset
data_test = pd.concat([setosa_data[25:50], versicolor_data[25:50], virginica_data[25:50]], ignore_index=True)
label_test = data_test["target"].to_numpy()

def classifier_rate(ground_truth, predictions) :
    counter = 0
    for i in range(len(ground_truth)) :
        counter += predictions[i] == ground_truth[i] 
        
    return counter / len(ground_truth)

def two_fold_cv_knn(k, X_train, y_train, X_test, y_test):
    # train -> test
    knn = KNN_classifier(k=k)
    knn.training_data_setting(X_train, y_train)
    y_pred1 = knn.knn_classifier(X_test)
    acc1 = classifier_rate(y_test, y_pred1)

    # test -> train
    knn_ = KNN_classifier(k=k)
    knn_.training_data_setting(X_test, y_test)
    y_pred2 = knn_.knn_classifier(X_train)
    acc2 = classifier_rate(y_train, y_pred2)

    return acc1, acc2, (acc1 + acc2) / 2.0

for feature in training_feature : 
    X_train_set_ver = data_train_set_ver.loc[:, feature].to_numpy()
    X_train_set_vir = data_train_set_vir.loc[:, feature].to_numpy()
    X_train_ver_vir = data_train_ver_vir.loc[:, feature].to_numpy()
    
    X_test_set_ver = data_test_set_ver.loc[:, feature].to_numpy()
    X_test_set_vir = data_test_set_vir.loc[:, feature].to_numpy()
    X_test_ver_vir = data_test_ver_vir.loc[:, feature].to_numpy()
    
    X_train = data_train.loc[:, feature].to_numpy()
    X_test = data_test.loc[:, feature].to_numpy()
    
    print("============= Binary classification ============")

    # ---------- setosa vs versicolor ----------
    print("[setosa vs versicolor]  features =", feature)
    # 1-NN
    acc1, acc2, avg = two_fold_cv_knn(1, X_train_set_ver, label_train_set_ver, X_test_set_ver, label_test_set_ver)
    print(f"1-NN  CR1: {acc1:.4f}  CR2: {acc2:.4f}  2-fold avg: {avg:.4f}")
    # 3-NN
    acc1, acc2, avg = two_fold_cv_knn(3, X_train_set_ver, label_train_set_ver, X_test_set_ver, label_test_set_ver)
    print(f"3-NN  CR1: {acc1:.4f}  CR2: {acc2:.4f}  2-fold avg: {avg:.4f}")
    print("-" * 55)

    # ---------- setosa vs virginica ----------
    print("[setosa vs virginica]  features =", feature)
    # 1-NN
    acc1, acc2, avg = two_fold_cv_knn(1, X_train_set_vir, label_train_set_vir, X_test_set_vir, label_test_set_vir)
    print(f"1-NN  CR1: {acc1:.4f}  CR2: {acc2:.4f}  2-fold avg: {avg:.4f}")
    # 3-NN
    acc1, acc2, avg = two_fold_cv_knn(3, X_train_set_vir, label_train_set_vir, X_test_set_vir, label_test_set_vir)
    print(f"3-NN  CR1: {acc1:.4f}  CR2: {acc2:.4f}  2-fold avg: {avg:.4f}")
    print("-" * 55)

    # ---------- versicolor vs virginica ----------
    print("[versicolor vs virginica]  features =", feature)
    # 1-NN
    acc1, acc2, avg = two_fold_cv_knn(1, X_train_ver_vir, label_train_ver_vir, X_test_ver_vir, label_test_ver_vir)
    print(f"1-NN  CR1: {acc1:.4f}  CR2: {acc2:.4f}  2-fold avg: {avg:.4f}")
    # 3-NN
    acc1, acc2, avg = two_fold_cv_knn(3, X_train_ver_vir, label_train_ver_vir, X_test_ver_vir, label_test_ver_vir)
    print(f"3-NN  CR1: {acc1:.4f}  CR2: {acc2:.4f}  2-fold avg: {avg:.4f}")
    print("==================================================\n")

    print("============= Trinary classification ============")
    print("features =", feature)
    # 1-NN
    acc1, acc2, avg = two_fold_cv_knn(1, X_train, label_train, X_test, label_test)
    print(f"1-NN  CR1: {acc1:.4f}  CR2: {acc2:.4f}  2-fold avg: {avg:.4f}")
    # 3-NN
    acc1, acc2, avg = two_fold_cv_knn(3, X_train, label_train, X_test, label_test)
    print(f"3-NN  CR1: {acc1:.4f}  CR2: {acc2:.4f}  2-fold avg: {avg:.4f}")
    print("==================================================\n")
