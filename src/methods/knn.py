import numpy as np
import math
from src.utils import normalize_fn
class KNN(object):
    """
    kNN classifier object.
    """

    def __init__(self, k=1, task_kind="classification"):
        """
        Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind = task_kind
        self.training_data=None
        self.training_labels=None
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: Since KNN does not really have parameters to train, you can try saving
        the training_data and training_labels as part of the class. This way, when you
        call the "predict" function with the test_data, you will have already stored
        the training_data and training_labels in the object.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): labels of shape (N,)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """

        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        predictions=[]
        for data in training_data:
            distances = self._euclidean_dist(data,training_data)
            nn_indices = self._find_k_nearest_neighbors(self.k,distances)#there is a 0.0 distance
            neighbor_labels = training_labels[nn_indices]
            predictions.append(self._predict_label(neighbor_labels.astype(int)))
        self.training_data=training_data
        self.training_labels=training_labels.astype(int)
        return np.array(predictions)
    def _euclidean_dist(self,example, training_examples):
        """Compute the Euclidean distance between a single example
        vector and all training_examples.

        Inputs:
            example: shape (D,)
            training_examples: shape (NxD) 
        Outputs:
            euclidean distances: shape (N,)
        """
        # WRITE YOUR CODE HERE
        v= [(example-x) for x in training_examples] 
        return np.array([math.sqrt(w@w) for w in v])
    def _predict_label(self,neighbor_labels):
        """Return the most frequent label in the neighbors'.

        Inputs:
            neighbor_labels: shape (N,) 
        Outputs:
            most frequent label
        """
        if self.task_kind == "regression":
            return np.mean(neighbor_labels)
        elif self.task_kind == "classification":
            return np.argmax(np.bincount(neighbor_labels.astype(int)))
        else:
            raise ValueError(f"Unknown task_kind: {self.task_kind}")
    def _find_k_nearest_neighbors(self,k:int, distances):
        """ Find the indices of the k smallest distances from a list of distances.
            Tip: use np.argsort()

        Inputs:
            k: integer
            distances: shape (N,) 
        Outputs:
            indices of the k nearest neighbors: shape (k,)
        """
        indices = np.argsort(distances)[:k]#there is 0.0 distance
        return indices
    def _kNN_one_example(self,unlabeled_example, training_features, training_labels, k):
        """Returns the label of a single unlabelled example.

        Inputs:
            unlabeled_example: shape (D,) 
            training_features: shape (NxD)
            training_labels: shape (N,) 
            k: integer
        Outputs:
            predicted label
        """
        distances = self._euclidean_dist(unlabeled_example,training_features)
        nn_indices = self._find_k_nearest_neighbors(k,distances)
        neighbor_labels = training_labels[nn_indices]
        best_label = self._predict_label(neighbor_labels)
        
        return best_label

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            test_labels (np.array): labels of shape (N,)s
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##
        return np.apply_along_axis(self._kNN_one_example,1,test_data,self.training_data,self.training_labels,self.k)
