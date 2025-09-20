import numpy as np
from collections import Counter

# ====== KNN classifier ======
# step 1 : input a data point and calculate its distance to all others
# step 2 : input data's label is determined by n-nearst neighbor (majority)
def calculate_euclidean_distance(a, b) : 
    a = np.asarray(a)
    b = np.asarray(b)
    
    return np.linalg.norm(a-b)

#print(calculate_euclidean_distance([4,6], [4,4]))  # unit test
# k => how many nearest neighbor to determine input's label
# data => training data
# input => input data point 
class KNN_classifier:
    def __init__(self, k = 3) :
        self.k = k
    
    def training_data_setting(self, train_data, train_label) :
        self.x_train = train_data
        self.y_train = train_label
        
    # process single prediction
    def predict(self, input_data) :
        # compute distance
        distance = [calculate_euclidean_distance(input_data, training_data) for training_data in self.x_train] 
        
        # sorting distance from nearest
        label_index = np.argsort(distance)[:self.k]
        
        # get corresponding label
        label = [self.y_train[i] for i in label_index]
        
        # majority labels
        prediction_label = Counter(label).most_common(1)[0][0]    
        return prediction_label
        
    # process whole input data
    def knn_classifier(self, input_data) : 
        predictions = [self.predict(unlabeled) for unlabeled in input_data]        
        
        return predictions