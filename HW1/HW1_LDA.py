import numpy as np
import math

# ====== LDA classifier ======
# LDA : D(x) = W^T*X + b 
# step 1 calculate mean vector
# step 2 calculate covariance matrix
# step 3 calculate priori probability of each class
# step 4 setting penalty weight 
# step 5 calculate model's weight
# step 6 calculate model's bias

class LDA_classifier : 
    def __init__(self) :
        pass
    
    def mean_vector(self, data, label) :
        data = np.asarray(data)
        label = np.asarray(label)
        labels = np.unique(label)
        
        self.data = data
        self.label = label
        self.labels = labels
        
        mean_vec = [] 
        for l in labels :
            mean_vec.append(np.mean(data[label == l], axis = 0))
            
        self.mean_vec = np.asarray(mean_vec)    
        return mean_vec
    
    def priori_probability(self) : 
        data = self.data
        label = self.label
        labels = self.labels
        priori_prob = []
        
        for l in labels:
            n_i = data[label == l].shape[0] 
            priori_prob.append(n_i / data.shape[0])
        
        self.priori_prob = priori_prob
        
            
    def cov_mat(self) :
        data = self.data
        label = self.label
        labels = self.labels
        mean_vec = self.mean_vec
        
        covariance_mat = []  # variance matrix for each label
        total_covariance = np.zeros((data.shape[1], data.shape[1]))  # covariance matrix
        
        for l in labels:
            n_i = data[label == l].shape[0] 
            x_i = data[label == l] - mean_vec[l]   
            cov_i = np.dot(x_i.T, x_i) / (n_i - 1)           
            weighted_cov_i = (n_i / data.shape[0]) * cov_i   
            
            covariance_mat.append(weighted_cov_i)            
            total_covariance += weighted_cov_i              
             
        self.covariance_mat = covariance_mat
        self.total_covariance = total_covariance
        
        return total_covariance
            
    def decision_boundary(self, pos_penalty=0.5, neg_penalty=0.5) : 
        mean_vec = self.mean_vec
        covariance = self.total_covariance
        priori_prob = self.priori_prob
        
        weight = np.dot(np.linalg.inv(covariance), ((mean_vec[0] - mean_vec[1]).T))
        #print("weight", weight)

        log_term = (math.log(priori_prob[0]) - math.log(priori_prob[1])) + (math.log(neg_penalty) - math.log(pos_penalty))  
        bias = -0.5 * np.dot(((mean_vec[0] + mean_vec[1])), weight) - log_term
        #print("bias", bias)
        
        self.c1 = pos_penalty
        self.c2 = neg_penalty
        self.model_weight = weight
        self.model_bias = bias
        
        return weight, bias
        
    def predict(self, input_data) : 
        c1, c2 = self.c1 ,self.c2
        input_data = np.asarray(input_data)
        self.priori_probability()
        
        pred_label = []
        w, b = self.decision_boundary(c1, c2)
        
        for x in input_data : 
            score = np.dot(w, x) + b
            
            if (score > 0) : 
                pred_label.append(0)
            else : 
                pred_label.append(1)
                
        return pred_label
