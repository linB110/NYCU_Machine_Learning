# import necessary lib such as calculation ,plotting...
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from HW1_KNN import KNN_classifier
from HW1_LDA import LDA_classifier

# evaluate ROC
from HW1_evaluate import roc_curve_from_scores, auc_trapezoid

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

def two_fold_cv_lda(X_train, y_train, X_test, y_test, pos_penalty=0.5, neg_penalty=0.5):
    # train -> test
    lda = LDA_classifier()
    lda.training_setting(X_train, y_train, pos_penalty=pos_penalty, neg_penalty=neg_penalty)
    y_pred1 = lda.predict(X_test)
    acc1 = classifier_rate(y_test, y_pred1)

    # test -> train
    lda2 = LDA_classifier()
    lda2.training_setting(X_test, y_test, pos_penalty=pos_penalty, neg_penalty=neg_penalty)
    y_pred2 = lda2.predict(X_train)
    acc2 = classifier_rate(y_train, y_pred2)

    return acc1, acc2, (acc1 + acc2) / 2.0

# trinary classification using LDA by voting method
def LDA_oao_train(training_data, training_label, pos_penalty=0.5, neg_penalty=0.5) :
    labels = np.unique(training_label)
    labels.sort()
    models = {}
    
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            mask = (training_label == a) | (training_label == b)

            lda = LDA_classifier()
            lda.training_setting(training_data[mask], training_label[mask],
                                 pos_penalty=pos_penalty, neg_penalty=neg_penalty)
            models[(a, b)] = lda
            
    return models

def LDA_oao_predict(models, X, tie_break="margin"):
    classes = sorted({c for pair in models.keys() for c in pair})
    class_to_idx = {c: i for i, c in enumerate(classes)}

    n = X.shape[0]
    votes = np.zeros((n, len(classes)), dtype=int)
    margins = np.zeros((n, len(classes)), dtype=float)

    for (a, b), lda in models.items():
        w, b0 = lda.model_weight, lda.model_bias
        score = X @ w + b0  # > 0 -> lda.labels[0], <= 0 -> lda.labels[1]

        lab_pos, lab_neg = lda.labels[0], lda.labels[1]
        idx_pos, idx_neg = class_to_idx[lab_pos], class_to_idx[lab_neg]

        votes[score > 0, idx_pos] += 1
        votes[score <= 0, idx_neg] += 1

        # used for "tie situation"
        margins[:, idx_pos] += np.maximum(score, 0)
        margins[:, idx_neg] += np.maximum(-score, 0)
        
    y_pred = np.empty(n, dtype=int)
    
    for i in range(n):
        top = votes[i].max()
        cand = np.where(votes[i] == top)[0]
        if len(cand) == 1:
            y_pred[i] = classes[cand[0]]
        else:
            if tie_break == "min_label":
                y_pred[i] = classes[cand.min()]
            else:  
                j = cand[np.argmax(margins[i, cand])]
                y_pred[i] = classes[j]
                
    return y_pred

def two_fold_cv_lda_oao(X_train, y_train, X_test, y_test,
                        pos_penalty=0.5, neg_penalty=0.5, tie_break="margin"):
    # train -> test
    models = LDA_oao_train(X_train, y_train, pos_penalty=pos_penalty, neg_penalty=neg_penalty)
    y_pred1 = LDA_oao_predict(models, X_test, tie_break=tie_break)
    acc1 = classifier_rate(y_test, y_pred1)

    # test -> train
    models_ = LDA_oao_train(X_test, y_test, pos_penalty=pos_penalty, neg_penalty=neg_penalty)
    y_pred2 = LDA_oao_predict(models_, X_train, tie_break=tie_break)
    acc2 = classifier_rate(y_train, y_pred2)

    return acc1, acc2, (acc1 + acc2) / 2.0
        
### running prediction via multiple feature selections
for feature in training_feature : 
    
    # training dataset (binary class)
    X_train_set_ver = data_train_set_ver.loc[:, feature].to_numpy()
    X_train_set_vir = data_train_set_vir.loc[:, feature].to_numpy()
    X_train_ver_vir = data_train_ver_vir.loc[:, feature].to_numpy()
    
    # testing dataset (binary class)
    X_test_set_ver = data_test_set_ver.loc[:, feature].to_numpy()
    X_test_set_vir = data_test_set_vir.loc[:, feature].to_numpy()
    X_test_ver_vir = data_test_ver_vir.loc[:, feature].to_numpy()
    
    # training and testing dataset (trinary class)
    X_train = data_train.loc[:, feature].to_numpy()
    X_test = data_test.loc[:, feature].to_numpy()
    
    #=====================
    # KNN classifier
    #=====================
    # ====== splitting dataset (trinary classification) =======
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
    
    #=====================
    # LDA classifier
    #=====================

    print("============= Binary classification with LDA ============")

    print("[setosa vs versicolor]  features =", feature)
    acc1, acc2, avg = two_fold_cv_lda(X_train_set_ver, label_train_set_ver, X_test_set_ver, label_test_set_ver)
    print(f"LDA   CR1: {acc1:.4f}  CR2: {acc2:.4f}  2-fold avg: {avg:.4f}")

    print("[setosa vs virginica]  features =", feature)
    acc1, acc2, avg = two_fold_cv_lda(X_train_set_vir, label_train_set_vir, X_test_set_vir, label_test_set_vir)
    print(f"LDA   CR1: {acc1:.4f}  CR2: {acc2:.4f}  2-fold avg: {avg:.4f}")

    print("[versicolor vs virginica]  features =", feature)
    acc1, acc2, avg = two_fold_cv_lda(X_train_ver_vir, label_train_ver_vir, X_test_ver_vir, label_test_ver_vir)
    print(f"LDA   CR1: {acc1:.4f}  CR2: {acc2:.4f}  2-fold avg: {avg:.4f}")
    print("==================================================\n")
    
    print("============= Trinary classification with LDA (OaO, majority vote) ============")
    print("features =", feature)
    acc1, acc2, avg = two_fold_cv_lda_oao(
        X_train, label_train, X_test, label_test,
        pos_penalty=0.5, neg_penalty=0.5, tie_break="margin"
    )
    print(f"LDA OaO  CR1: {acc1:.4f}  CR2: {acc2:.4f}  2-fold avg: {avg:.4f}")
    print("==================================================\n")
    
    #=====================
    # ROC for versicolor (pos=1) vs virginica (neg=2) with LDA 
    #=====================
    y_test_bin = (np.asarray(label_test_ver_vir) == 2).astype(int)

    # train LDA
    lda_tmp = LDA_classifier()
    lda_tmp.training_setting(X_train_ver_vir, label_train_ver_vir, pos_penalty=0.5, neg_penalty=0.5)

    raw_scores = X_test_ver_vir @ lda_tmp.model_weight + lda_tmp.model_bias
    sign = 1 if lda_tmp.labels[0] == 2 else -1
    scores = sign * raw_scores

    # ROC and AUC
    fprs, tprs, ths = roc_curve_from_scores(scores, y_test_bin)
    auc = auc_trapezoid(fprs, tprs)

    # plotting
    plt.figure(figsize=(6, 6))
    plt.plot(fprs, tprs, lw=2, label=f"ROC (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", lw=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"ROC — virginica (pos=2) vs versicolor (neg=1)\nfeatures = {feature}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    thr0 = 0.0
    y_pred0 = (scores >= thr0).astype(int)
    TP = np.sum((y_test_bin == 1) & (y_pred0 == 1))
    FP = np.sum((y_test_bin == 0) & (y_pred0 == 1))
    FN = np.sum((y_test_bin == 1) & (y_pred0 == 0))
    TN = np.sum((y_test_bin == 0) & (y_pred0 == 0))
    TPR0 = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    FPR0 = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    print(f"[ROC@{feature}] thr=0.0  TPR={TPR0:.3f}  FPR={FPR0:.3f}  (TP={TP}, FP={FP}, FN={FN}, TN={TN})")
