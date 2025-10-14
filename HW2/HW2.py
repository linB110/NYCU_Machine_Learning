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

# assign versicolor as positive class and virginica as negative class with label
positive_cls = data_frame[data_frame["target_name"] == "versicolor"]
positive_lbl = np.ones(len(positive_cls)) 
negative_cls = data_frame[data_frame["target_name"] == "virginica"]
negative_lbl = -np.ones(len(negative_cls))

# print(positive_cls)
# print(negative_cls)

# assign designated feature combination
feature = ["petal length (cm)", "petal width (cm)"]

# construct training and testing dataset with label
training_data = pd.concat([positive_cls[feature].iloc[:25], negative_cls[feature].iloc[0:25]], axis=0, ignore_index=True)
testing_data = pd.concat([positive_cls[feature].iloc[25:], negative_cls[feature].iloc[25:]], axis=0, ignore_index=True)

training_lbl = np.concatenate([positive_lbl[:25], negative_lbl[:25]])
testing_lbl  = np.concatenate([positive_lbl[25:], negative_lbl[25:]])

def compute_classification_rate(ground_truth, prediction):
    ground_truth = np.asarray(ground_truth)
    prediction = np.asarray(prediction)
    
    cr = 0
    for i in range(len(ground_truth)):
        if prediction[i] == ground_truth[i]:
            cr += 1 
    
    return cr/len(ground_truth)

# analysis parameter
linear_penalty_weight = [1.0, 10.0, 100.0]
rbf_sigma = [5, 1, 0.5, 0.1, 0.05]
poly_power = [1, 2, 3, 4, 5]
    
# ====== linear SVM ====== 
# penalty_weight = 1.0
for i in range(len(linear_penalty_weight)):
    linear_svm = SVM_classifier(penalty_weight=linear_penalty_weight[i], kernel_function="linear")
    linear_svm.set_training_data(train_data=training_data, train_label=training_lbl)
    linear_result = compute_classification_rate(testing_lbl, linear_svm.model_fit(testing_data))
    print("classification rate of linear SVM with penalty weight of " + str(linear_penalty_weight[i]) + ": ", linear_result)
    linear_svm.plot_2d(X_test=testing_data.values, y_test=testing_lbl,
                    title="Linear SVM: OSH ", show_margins=True, feature=feature, description="penalty weight=" + str(linear_penalty_weight[i]))

# ======  RBF SVM ====== 
for i in range(len(rbf_sigma)):
    rbf_svm = SVM_classifier(penalty_weight=10.0, kernel_function="RBF", sigma=rbf_sigma[i])
    rbf_svm.set_training_data(train_data=training_data, train_label=training_lbl)
    rbf_result = compute_classification_rate(testing_lbl, rbf_svm.model_fit(testing_data))
    print("classification rate of RBF SVM with sigma of " + str(rbf_sigma[i]) + ": ", rbf_result)
    rbf_svm.plot_2d(X_test=testing_data.values, y_test=testing_lbl,
                    title="RBF SVM: decision regions", show_margins=False, feature=feature, description="sigma=" + str(rbf_sigma[i]))

# # ====== polynomial SVM ====== 
for i in range(len(poly_power)):
    poly_svm = SVM_classifier(penalty_weight=1.0, kernel_function="polynomial", power=poly_power[i])
    poly_svm.set_training_data(train_data=training_data, train_label=training_lbl)
    poly_result = compute_classification_rate(testing_lbl, poly_svm.model_fit(testing_data))
    print("classification rate of polynomial SVM with power of " + str(poly_power[i]) + ": ", poly_result)
    poly_svm.plot_2d(X_test=testing_data.values, y_test=testing_lbl,
                    title="Polynomial SVM: decision regions", show_margins=False, feature=feature, description="power=" + str(poly_power[i]))

# # test
# # t1 = testing_data[feature].iloc[0].values.reshape(1, -1)
# # l1 = testing_lbl[0]
# # print("Sample:", t1, "Label:", l1)
# # res = svm.model_fit(t1)
# # print("Predicted:", res)



