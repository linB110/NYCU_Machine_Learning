import numpy as np
from qpsolvers import solve_qp

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
    def __init__(self, penalty_weight, kernel_function = "linear", sigma = 1.0, power = 1):
        self.penalty_weight = float(penalty_weight)
        self.kernel_function = kernel_function
        self.sigma = float(sigma)
        self.power = int(power)
        
        self.train_data = None
        self.train_label = None
        self.alpha = None
        self.K = None
    
    def set_training_data(self, train_data, train_label):
        self.train_data = np.asarray(train_data)
        self.train_label = np.asarray(train_label)
    
        # clear old training result
        self.alpha = None
        self.K = None
        self.bias_ = None
        
    def linear_kernel(self, x_i, x_j):
        return float(np.dot(x_i, x_j))
    
    def RBF_kernel(self, x_i, x_j, sigma = 1.0):
        gamma = 1.0 / (2.0 * sigma**2)
        diff = x_i - x_j
        
        return float(np.exp(-gamma * np.dot(diff, diff))) 
    
    def polynomial_kernel(self, x_i, x_j, power):        
        return float((1.0 + np.dot(x_i, x_j)) ** power)
        
    # solve dual problem for α 
    # φ(x) : feature mapping function
    # K(x_i, x_j) = x_i.T * x_j
    # SumiN : sum of i = 1 to N 
    # minimize SumiN SumjN α_i * α_j * y_i * y_j * x_i.T * x_j
    # subject to SumiN α_i * y_i = 0 and 0 <= α_i <= C for all i
    def solve_Lagrange_multiplier(self):
        x = np.asarray(self.train_data)
        y = np.asarray(self.train_label)
        
        N = len(y)
        C = self.penalty_weight
        kernel_type = self.kernel_function
        
        K = 0 # inner product of mapped data
        if (kernel_type == "linear"):
            K = self.linear_kernel(x, x)
        elif (kernel_type == "RBF"):
            K = self.RBF_kernel(x, x, self.sigma)
        elif (kernel_type == "polynomial"):
            K = self.polynomial_kernel(x, x, self.power)
         
        self.K = K # restore K value for further usage
        
        Y = np.outer(y, y)
        Q = Y * K
        Q = (Q + Q.T) / 2 # symmetric matrix
        
        q = -np.ones(len(y))
        A = y.reshape(1, -1)
        b = np.array([0.0])
        
        lower_bound = np.zeros(len(y))
        upper_bound = np.full(len(y), C)
        
        # solve quadratic programming
        alpha = solve_qp(P=Q, q=q, A=A, b=b, lb=lower_bound, ub=upper_bound, solver="osqp")
        
        return alpha        
    
    # optimal model bias is computed follow by formula below
    # b* = 1/y_k - SiN α_i * y_i * K(x_i, x_k)  where x_k is training data for 0 < α_k < K
    def calculate_model_bias(self):
        a = np.asarray(self.solve_Lagrange_multiplier())
        y = np.asarray(self.train_label)
        
        # take 0 < α < C (penalty_weight)
        support_idx = (a > 1e-6) & (a < self.penalty_weight - 1e-6)
        
        if len(support_idx) > 0:
            b_list = []
            for i in np.where(support_idx)[0]:
                b_i = y[i] - np.sum(a * y * self.K[:, i])
                b_list.append(b_i)
            optimal_bias = np.mean(b_list)
        else:
            i = int(np.argmax(a))
            optimal_bias =  y[i] - np.sum(a * y * self.K[:, i])
                    
        return optimal_bias
    
    def model_fit(self, input_data):
        if self.alpha is None:
            alpha = self.solve_Lagrange_multiplier()
        if self.bias_ is None:
            bias = self.calculate_model_bias()
            
        kernel_type = self.kernel_function
        y = np.asarray(self.train_label)
        x_i = np.asarray(self.train_data)
        x_j = np.asarray(input_data)
        
        result = []
        K = 0 # inner product of mapped data
        for k in range(len(x_i)):
            decision = 0

            for n in range(len(input_data)):
                if (kernel_type == "linear"):
                    K = self.linear_kernel(x_i[n], x_j[k])
                elif (kernel_type == "RBF"):
                    K = self.RBF_kernel(x_i[n], x_j[k], self.sigma)
                elif (kernel_type == "polynomial"):
                    K = self.polynomial_kernel(x_i[n], x_j[k], self.power)
                    
                decision += alpha[n] * y[n] * K 
            
            decision += bias
            result.append(1) if decision >= 0 else result.append(-1)
        
        return result
        
            
                
            
        
