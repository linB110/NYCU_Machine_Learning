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


## ⚙️ HW3 — RBF-SVM Parameter Tuning (Grid Search)

### Objective

Perform **grid search** on the **RBF kernel SVM** to determine the best combination of **penalty weight (C)** and **kernel width (σ)** parameters that maximize the classification rate (CR).

### Procedure

1. Use the same **Iris dataset** setup from HW2 (binary classification between *Versicolor* and *Virginica*).  
2. Define:
   - `penalty_weight = [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]`
   - `sigma = 1.05^i` where `i ∈ [-100, 100]` in steps of 5  
3. Perform **2-Fold Cross Validation (CV)** for each `(C, σ)` pair:
   - Train SVM on one fold, test on the other, then swap.
   - Compute the average CR from both folds.
4. Record all results and identify the **best parameter set**.

### Implementation Details

* Custom `SVM_classifier` (from **HW2**) with `RBF` kernel  
* Nested loops iterate through each `C`–`σ` combination  
* Compute accuracy and print results with two-decimal precision  
* Track and display the best configuration with highest CR

---

