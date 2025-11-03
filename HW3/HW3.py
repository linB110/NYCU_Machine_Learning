# import necessary lib such as calculation ,plotting...
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from HW2_SVM import SVM_classifier

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

# assign versicolor as positive class and virginica as negative class with label
positive_cls = data_frame[data_frame["target_name"] == "versicolor"]
positive_lbl = np.ones(len(positive_cls)) 
negative_cls = data_frame[data_frame["target_name"] == "virginica"]
negative_lbl = -np.ones(len(negative_cls))

feature = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]

# construct training and testing dataset with label
training_data = pd.concat([positive_cls[feature].iloc[:25], negative_cls[feature].iloc[:25]], axis=0, ignore_index=True)
testing_data = pd.concat([positive_cls[feature].iloc[25:], negative_cls[feature].iloc[25:]], axis=0, ignore_index=True)

training_lbl = np.concatenate([positive_lbl[:25], negative_lbl[:25]])
testing_lbl  = np.concatenate([positive_lbl[25:], negative_lbl[25:]])

# RBF-SVM grid search parameter 
penalty_weight = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
rbf_sigma = []

for i in range(-100, 101, 5):
    rbf_sigma.append( (1.05)**i )
    
    
def two_fold_cv(X_train, y_train, X_test, y_test, penalty_weight, sigma):
    rbf_svm_1 = SVM_classifier(penalty_weight=penalty_weight, kernel_function="RBF", sigma=sigma)
    rbf_svm_1.set_training_data(X_train, y_train)
    
    rbf_svm_2 = SVM_classifier(penalty_weight=penalty_weight, kernel_function="RBF", sigma=sigma)
    rbf_svm_2.set_training_data(X_test, y_test)
    
    groundtruth_1 = y_test
    groundtruth_2 = y_train
    result_1 = rbf_svm_1.model_fit(X_test)
    result_2 = rbf_svm_2.model_fit(X_train)
    
    classification_rate_1 = 0
    classification_rate_2 = 0
    
    for i in range( len(groundtruth_1) ):
        if (result_1[i] == groundtruth_1[i]) : 
            classification_rate_1 += 1
    
    for i in range( len(groundtruth_2) ):
        if (result_2[i] == groundtruth_2[i]) : 
            classification_rate_2 += 1
    
    classification_rate_1 /= len(groundtruth_1)
    classification_rate_2 /= len(groundtruth_2)
    
    return (classification_rate_1 + classification_rate_2) / 2.0

rbf_grid_search = []
best_score = 0.0
best_param = None
for c in range( len(penalty_weight) ):
    for s in range( len(rbf_sigma) ):
        score = two_fold_cv(training_data, training_lbl, testing_data, testing_lbl, penalty_weight=penalty_weight[c], sigma=rbf_sigma[s])
        rbf_grid_search.append(score)
        print(f"C={penalty_weight[c]:<6.1f}, sigma={rbf_sigma[s]:<10.4e}, accuracy={score:.2f}")
        
        if (score > best_score):
            best_score = score
            best_param = (penalty_weight[c], rbf_sigma[s])
        
print("\n=== Best Result ===")
print(f"Best CR: {best_score:.2f}")
print(f"Best parameters: penalty_weight={best_param[0]}, sigma={best_param[1]:.4e}")