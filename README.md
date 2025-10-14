# 🧠 Machine Learning 

Using the classic **Iris Dataset** to implement and compare various classification models.

## 📘 Dataset

The **Iris dataset** contains 150 samples divided into three classes:

* Setosa
* Versicolor
* Virginica

Each sample has 4 features:

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width

---

## 🧩 HW1 — KNN and LDA Classifiers

### Objective

Use the following methods on the Iris dataset and evaluate classification rate (CR) using **2-fold Cross Validation (CV)**.

### Methods

* 1-NN (k = 1)
* 3-NN (k = 3)
* LDA (Linear Discriminant Analysis)

### Procedure

1. Randomly split the dataset into two folds (2-fold CV)
2. Train on one fold and test on the other, then swap
3. Compute the average classification rate (CR)

---

## ⚙️ HW2 — Support Vector Machine (SVM) Models

### Objective

Apply SVM models with different kernel functions on the Iris dataset and compute the classification rate (CR).

### Methods

* Linear SVM
* RBF SVM
* Polynomial SVM

### Procedure

1. Standardize the dataset
2. Use the same data split strategy 
3. Compare CR results across all SVM kernels to analyze performance differences

---

## 📊 Results

* Display classification rate (CR) for each model

---

