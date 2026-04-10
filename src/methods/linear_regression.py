import numpy as np
from utils import normalize_fn
from utils import append_biais_term
class LinearRegression(object):
    """
    Linear regression.
    """

    def __init__(self):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.
        """
        self.mean=0
        self.std=1
        self.weights=None
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Hint: You can use the closed-form solution for linear regression
        (with or without regularization). Remember to handle the bias term.

        Arguments:
            training_data (np.array): training data of shape (N,D)
            training_labels (np.array): regression target of shape (N,)
        Returns:
            pred_labels (np.array): target of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.mean=np.mean(training_data,axis=1,keepdims=True)
        self.std=np.std(training_data,axis=1,keepdims=True)
        #one_colums=np.ones(training_data.shape[0],1)
        biased_training_data=append_biais_term(training_data)#np.concatenate((one_colums,training_data),axis=1)
        self.weights=np.linalg.pinv(biased_training_data)@training_labels
        return self.weights@training_data

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (np.array): test data of shape (N,D)
        Returns:
            pred_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        one_colums=np.ones(test_data.shape[0],1)
        biased_test_data=np.concatenate((one_colums,normalize_fn(test_data,self.mean,self.std)),axis=1)
        return self.weights@biased_test_data
