import numpy as np

class Fisher:
    def __init__(self, input_data, input_label):
        self.data = np.asarray(input_data, dtype=float)   
        self.label = np.asarray(input_label)             
        
        self.classes = None           
        self.mean_vec = None          
        self.cls_mean_vec = None      
        self.priori_prob = None       
        
        self.Sb = None                
        self.Sw = None                
    
    # rearrange data by class clustering
    def rearrange_data(self):
        sorted_idx = np.argsort(self.label)
        
        sorted_data = self.data[sorted_idx]
        sorted_label = self.label[sorted_idx]
        
        self.data = sorted_data
        self.label = sorted_label
        
    # mean vector of all data    
    def compute_mean_vector(self):
        mean_vec = np.mean(self.data, axis=0)
        self.mean_vec = mean_vec
        
        return mean_vec
    
    # mean vector of each class
    def compute_cls_mean_vector(self):
        self.classes = classes = np.unique(self.label)
        d = self.data.shape[1]
        cls_mean_vec = np.zeros((len(classes), d))  
        
        for idx, cls in enumerate(classes):
            Xc = self.data[self.label == cls]      
            cls_mean_vec[idx] = np.mean(Xc, axis=0)
        
        self.cls_mean_vec = cls_mean_vec      
             
        return cls_mean_vec
        
    def compute_priori_probability(self):
        if self.classes is None:
            self.classes = np.unique(self.label)
        
        priori_prob = []
        N = len(self.label)
        
        for cls in self.classes:
           
            p = np.sum(self.label == cls) / N
            priori_prob.append(p)
        
        self.priori_prob = np.array(priori_prob)  
        
        return self.priori_prob
    
    # C : number of classses  N : number of data in that class
    # Sw = sum of i = 1 to C (priori_prob * (1/N) * (sum of j = 1 to N)( Xij - mi )( Xij - mi)^T )
    def within_class_scatter(self):
        d = self.data.shape[1]   
        Sw = np.zeros((d, d))
        
        for idx, cls in enumerate(self.classes):
            Xc = self.data[self.label == cls]  
            mc = self.cls_mean_vec[idx]         

            diff = Xc - mc                    
            Sw += diff.T @ diff                 

        self.Sw = Sw
        
        return Sw
            
    # # Sb = sum of i = 1 to C (N * (mi - m)(mi - m)^T)
    def between_class_scatter(self):
        d = self.data.shape[1]   
        Sb = np.zeros((d, d))
        
        for idx, cls in enumerate(self.classes):
            Xc = self.data[self.label == cls]
            N = Xc.shape[0]                     

            
            diff = (self.cls_mean_vec[idx] - self.mean_vec).reshape(-1, 1)
            Sb += N * (diff @ diff.T)
        
        self.Sb = Sb
        
        return Sb
    
    # # Fk = Sb(k) / Sw(k) => diagonal value of the matrix    
    def compute_fisher_score(self):
        diag_Sw = np.diag(self.Sw)      
        diag_Sb = np.diag(self.Sb) 
             
        # element-wise division
        fisher_score = diag_Sb / diag_Sw 

        return fisher_score
    
    # call this function for complete usage and result    
    def apply_fisher_criterion(self):
        # rearrange data for further usage
        self.rearrange_data()
        
        # compute mean vector for Sw, Sb calculation
        self.compute_mean_vector()
        self.compute_cls_mean_vector()
        
        self.compute_priori_probability()
        
        # compute Sw, Sb
        self.within_class_scatter()
        self.between_class_scatter()
        
        f_scores = self.compute_fisher_score()
        
        return f_scores
    