import numpy as np
from qpsolvers import solve_qp
import matplotlib.pyplot as plt

# SVM classifier
# W     : model weight vector
# b     : model bias
# C     : penalty parameter (regularization strength)
# ζ_i   : slack variable for sample i
# α : Lagrange multiplier

# Objective function:
#   (1/2) * ||W||^2 + C * Σ ζ_i ,  where i = 1 to N
#   (N is the number of training samples)
#
# Constraints:
#   y_i * (Wᵀx_i + b) ≥ 1 - ζ_i
#   ζ_i ≥ 0 ,  for all i = 1 to N
class SVM_classifier:
    def __init__(self, penalty_weight, kernel_function="linear", sigma=1.0, power=1):
        self.penalty_weight = float(penalty_weight)
        self.kernel_function = kernel_function
        self.sigma = float(sigma)
        self.power = int(power)

        self.train_data = None
        self.train_label = None
        self.alpha = None
        self.K = None
        self.bias = None

    def set_training_data(self, train_data, train_label):
        self.train_data = np.asarray(train_data, dtype=float)
        self.train_label = np.asarray(train_label, dtype=float).flatten()
        
        # clear previous training result
        self.alpha = None
        self.K = None
        self.bias = None

    # generate kernel matrix
    def _kernel_matrix(self, X1, X2):
        if self.kernel_function == "linear":
            return X1 @ X2.T                                   

        elif self.kernel_function == "polynomial":
            return (1.0 + X1 @ X2.T) ** self.power           

        elif self.kernel_function == "RBF":
            X1_sq = np.sum(X1**2, axis=1, keepdims=True)      
            X2_sq = np.sum(X2**2, axis=1, keepdims=True).T    
            dist2 = X1_sq - 2.0 * (X1 @ X2.T) + X2_sq         # avoid 2-layer for loop time consumption operation
            gamma = 1.0 / (2.0 * self.sigma**2)
            
            return np.exp(-gamma * dist2)                     

        else:
            raise ValueError(f"Unknown kernel {self.kernel_function}")

    # solve dual problem for α 
    # φ(x) : feature mapping function
    # K(x_i, x_j) = x_i.T * x_j
    # SumiN : sum of i = 1 to N 
    # minimize SumiN SumjN α_i * α_j * y_i * y_j * x_i.T * x_j
    # subject to SumiN α_i * y_i = 0 and 0 <= α_i <= C for all i
    def solve_Lagrange_multiplier(self):
        X = self.train_data
        y = self.train_label
        N = len(y)
        C = self.penalty_weight

        K = self._kernel_matrix(X, X)         
        self.K = K

        Y = np.outer(y, y)                     
        Q = Y * K
        Q = (Q + Q.T) / 2.0                    

        q = -np.ones(N)
        A = y.reshape(1, -1)                   
        b = np.array([0.0])

        lb = np.zeros(N)
        ub = np.full(N, C)

        alpha = solve_qp(P=Q, q=q, A=A, b=b, lb=lb, ub=ub, solver="osqp")
        if alpha is None:
            raise RuntimeError("QP solver failed to find a solution.")
        self.alpha = alpha
        return alpha

    # optimal model bias is computed follow by formula below
    # b* = 1/y_k - SiN α_i * y_i * K(x_i, x_k)  where x_k is training data for 0 < α_k < K
    def calculate_model_bias(self):
        if self.alpha is None:
            self.solve_Lagrange_multiplier()

        a = self.alpha
        y = self.train_label
        C = self.penalty_weight

        # take 0 < α < C (penalty_weight)
        support_idx = (a > 1e-8) & (a < C - 1e-8)
        if np.any(support_idx):
            idxs = np.where(support_idx)[0]
            b_vals = []
            for k in idxs:
                b_k = y[k] - np.sum(a * y * self.K[:, k])
                b_vals.append(b_k)
            b_opt = float(np.mean(b_vals))
        else:
            k = int(np.argmax(a))
            b_opt = float(y[k] - np.sum(a * y * self.K[:, k]))

        self.bias = b_opt
        return b_opt

    # classification
    def model_fit(self, input_data):
        X_train = self.train_data
        y = self.train_label
        X_test = np.asarray(input_data, dtype=float)

        if self.alpha is None:
            self.solve_Lagrange_multiplier()
        if self.bias is None:
            self.calculate_model_bias()

        result = []

        # D(x) = Σ α_i y_i K(x_i, x_test) + b
        for j in range(len(X_test)):
            K_vec = self._kernel_matrix(X_train, X_test[j:j+1])  # shape = (N_train, 1)

            # SVM decision function
            decision = np.sum(self.alpha * y * K_vec[:, 0]) + self.bias

            result.append(1) if decision >= 0 else result.append(-1)

        return result
    
    ### visualization of hyperplane 
    def _ensure_trained(self):
        if self.alpha is None:
            self.solve_Lagrange_multiplier()
        if getattr(self, "bias", None) is None: 
            self.calculate_model_bias()

    def decision_function(self, X):
        self._ensure_trained()
        K_test = self._kernel_matrix(self.train_data, np.asarray(X, dtype=float))  
        return (self.alpha * self.train_label) @ K_test + self.bias  # (n,)

    def plot_2d(self, X_test=None, y_test=None, title=None, show_margins=True, feature=None, description=None):
        self._ensure_trained()
        X_tr = self.train_data
        y_tr = self.train_label
        alpha = self.alpha

        if X_test is not None:
            X_all = np.vstack([X_tr, np.asarray(X_test, dtype=float)])
        else:
            X_all = X_tr
        x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
        y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 300),
            np.linspace(y_min, y_max, 300)
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        zz = self.decision_function(grid).reshape(xx.shape)

        plt.figure(figsize=(7, 5))
        plt.contour(xx, yy, zz, levels=[0], colors='k', linewidths=2, linestyles='-')
        plt.contour(xx, yy, zz, levels=[-1, 1], colors='gray', linewidths=1, linestyles='--')

        # plotting training data
        plt.scatter(
            X_tr[y_tr == 1, 0], X_tr[y_tr == 1, 1],
            color='blue', label='Train +1', marker='o', edgecolors='k'
        )
        plt.scatter(
            X_tr[y_tr == -1, 0], X_tr[y_tr == -1, 1],
            color='red', label='Train -1', marker='x'
        )

        # plotting test data
        if X_test is not None and y_test is not None:
            X_te = np.asarray(X_test, dtype=float)
            y_te = np.asarray(y_test, dtype=float)
            plt.scatter(
                X_te[y_te == 1, 0], X_te[y_te == 1, 1],
                color='cyan', label='Test +1', marker='o', edgecolors='k'
            )
            plt.scatter(
                X_te[y_te == -1, 0], X_te[y_te == -1, 1],
                color='orange', label='Test -1', marker='x'
            )

        # plotting OSH and region for linear SVM
        if self.kernel_function == "linear" and show_margins:
            W = np.sum((alpha * y_tr)[:, None] * X_tr, axis=0)
            b = self.bias
            xs = np.linspace(x_min, x_max, 400)
            ys = -(W[0] * xs + b) / W[1]
            ys_m1 = -(W[0] * xs + b - 1) / W[1]
            ys_p1 = -(W[0] * xs + b + 1) / W[1]
            plt.plot(xs, ys, 'k-', linewidth=2)
            plt.plot(xs, ys_m1, 'k--', linewidth=1)
            plt.plot(xs, ys_p1, 'k--', linewidth=1)
        
        plt.xlabel(str(feature[0]))
        plt.ylabel(str(feature[1]))
        plt.title(title or f"SVM ({self.kernel_function}) decision boundary")
    
        if description:
            plt.figtext(0.02, 0.98, description, wrap=True, fontsize=10, ha='left', va='top')
        
        plt.legend(loc='best')
        plt.grid(True, linestyle=':', linewidth=0.6)
        plt.subplots_adjust(bottom=0.2)  
        plt.show()
        plt.show()

        

