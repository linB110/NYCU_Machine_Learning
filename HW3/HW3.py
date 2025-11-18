# import necessary lib such as calculation ,plotting...
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as m
from collections import Counter
from HW2_SVM import SVM_classifier

# ignore sparse matrix warning
import warnings
warnings.filterwarnings("ignore", message="Converted matrix")

# loading dataset
from sklearn import datasets
iris = datasets.load_iris()

# transfer to pandas dataframe
import pandas as pd
data_frame = pd.DataFrame(iris.data, columns=iris.feature_names) # construct a data file based on feature names
data_frame["target"] = iris.target  # label
data_frame["target_name"] = iris.target_names[iris.target] # class name

# set presicion and representation format
np.set_printoptions(precision=4, suppress=True, floatmode='fixed')

# construct training and testing dataset with label
setosa_data = data_frame[:50]
versicolor_data = data_frame[50:100]
virginica_data = data_frame[100:]

feature = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

# ====== splitting dataset (binary) ======
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

# set labels to binary labels
label_train_set_ver = np.where(label_train_set_ver == 0, 1, -1)
label_test_set_ver = np.where(label_test_set_ver == 0, 1, -1)

label_train_set_vir = np.where(label_train_set_vir == 0, 1, -1)
label_test_set_vir = np.where(label_test_set_vir == 0, 1, -1)

label_train_ver_vir = np.where(label_train_ver_vir == 1, 1, -1)
label_test_ver_vir = np.where(label_test_ver_vir == 1, 1, -1)


# ====== splitting dataset (mixed) ======
data_train = pd.concat([setosa_data[0:25], versicolor_data[0:25], virginica_data[0:25]], ignore_index=True)
label_train = data_train["target"].to_numpy()
data_test = pd.concat([setosa_data[25:50], versicolor_data[25:50], virginica_data[25:50]], ignore_index=True)
label_test = data_test["target"].to_numpy()
         
def SVM_OAO(results, y_true):
    y_true = np.asarray(y_true).ravel()
    N = len(y_true)

    def _norm_pred(pred, N):
        arr = np.asarray(pred)
        if arr.ndim == 0:                   
            arr = np.repeat(arr, N)
        else:
            arr = arr.ravel()
            if len(arr) != N:               
                arr = np.repeat(arr[0], N)
                
        return arr.astype(int)

    res12 = _norm_pred(results[0], N)
    res13 = _norm_pred(results[1], N)
    res23 = _norm_pred(results[2], N)

    # vote matrix
    votes = np.zeros((N, 3), dtype=int)

    # 12: +1→0(setosa), -1→1(versicolor)
    votes[:, 0] += (res12 == 1)
    votes[:, 1] += (res12 == -1)

    # 13: +1→0(setosa), -1→2(virginica)
    votes[:, 0] += (res13 == 1)
    votes[:, 2] += (res13 == -1)

    # 23: +1→1(versicolor), -1→2(virginica)
    votes[:, 1] += (res23 == 1)
    votes[:, 2] += (res23 == -1)

    pred = votes.argmax(axis=1)
    ties = (votes.max(axis=1, keepdims=True) == votes).sum(axis=1) > 1  # tie situation -> wrong classification

    correct = ((pred == y_true) & (~ties)).sum()
    return correct / N

# RBF-SVM grid search parameter 
penalty_weight = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
rbf_sigma = []

for i in range(-100, 101, 5):
    rbf_sigma.append( (1.05)**i )
    
# construct data frame for storing results        
CR1_table = pd.DataFrame(index=rbf_sigma, columns=penalty_weight, dtype=float)
CR2_table = pd.DataFrame(index=rbf_sigma, columns=penalty_weight, dtype=float)
CRavg_table = pd.DataFrame(index=rbf_sigma, columns=penalty_weight, dtype=float)

rbf_grid_search = []
best_score = 0.0
best_param = None

for c in range( len(penalty_weight) ):
    for s in range( len(rbf_sigma) ):
        # create 3 SVM using same penalty weight and sigma and 3 SVM with testing dataset
        SVM_12_1 = SVM_classifier(penalty_weight=penalty_weight[c], kernel_function="RBF", sigma=rbf_sigma[s])
        SVM_13_1 = SVM_classifier(penalty_weight=penalty_weight[c], kernel_function="RBF", sigma=rbf_sigma[s])
        SVM_23_1 = SVM_classifier(penalty_weight=penalty_weight[c], kernel_function="RBF", sigma=rbf_sigma[s])
        SVM_12_2 = SVM_classifier(penalty_weight=penalty_weight[c], kernel_function="RBF", sigma=rbf_sigma[s])
        SVM_13_2 = SVM_classifier(penalty_weight=penalty_weight[c], kernel_function="RBF", sigma=rbf_sigma[s])
        SVM_23_2 = SVM_classifier(penalty_weight=penalty_weight[c], kernel_function="RBF", sigma=rbf_sigma[s])

        # train individual SVM
        SVM_12_1.set_training_data(data_train_set_ver[feature], label_train_set_ver)
        SVM_13_1.set_training_data(data_train_set_vir[feature], label_train_set_vir)
        SVM_23_1.set_training_data(data_train_ver_vir[feature], label_train_ver_vir)
        
        SVM_12_2.set_training_data(data_test_set_ver[feature], label_test_set_ver)
        SVM_13_2.set_training_data(data_test_set_vir[feature], label_test_set_vir)
        SVM_23_2.set_training_data(data_test_ver_vir[feature], label_test_ver_vir)
        
        # using last 75 data as testing
        re1 = SVM_12_1.model_fit(data_test[feature])
        re2 = SVM_13_1.model_fit(data_test[feature])
        re3 = SVM_23_1.model_fit(data_test[feature])
        
        # using first 75 data as testing
        re4 = SVM_12_2.model_fit(data_train[feature])
        re5 = SVM_13_2.model_fit(data_train[feature])
        re6 = SVM_23_2.model_fit(data_train[feature])
                
        cr1 = SVM_OAO([re1, re2, re3], label_test)
        cr2 = SVM_OAO([re4, re5, re6], label_train)
        avg_cr = (cr1 + cr2) / 2.0
        
        # restore result to data frame for visualization
        log_sigma = m.log(rbf_sigma[s], 1.05)
        CR1_table.loc[log_sigma, penalty_weight[c]] = cr1
        CR2_table.loc[log_sigma, penalty_weight[c]] = cr2
        CRavg_table.loc[log_sigma, penalty_weight[c]] = avg_cr
        
        rbf_grid_search.append(avg_cr)
        print(f"C={penalty_weight[c]:<6.1f}, sigma={rbf_sigma[s]:<10.4e}, CR={avg_cr:.2f}")

        if avg_cr > best_score:
            best_score = avg_cr
            best_param = (penalty_weight[c], rbf_sigma[s])
 
# restore result to csv files       
CR1_table.to_csv("CR1.csv", float_format="%.4f")
CR2_table.to_csv("CR2.csv", float_format="%.4f")
CRavg_table.to_csv("CR_avg.csv", float_format="%.4f")
       
print("\n=== Best Result ===")
print(f"Best CR: {best_score:.4f}")
print(f"Best parameters: penalty_weight={best_param[0]}, sigma=1.05^{m.log(best_param[1], 1.05):.2f}")
