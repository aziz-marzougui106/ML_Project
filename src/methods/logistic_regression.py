import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label,accuracy_fn


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.weights=None
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

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
        #i can store the weights and keep the maximum in the sense accuracy is maximized
        max_accuracy=0
        weights = np.random.normal(0., 0.1, [training_data.shape[1],])
        predictions=np.empty(training_labels.shape)
        for it in range(self.max_iters):
            ############# WRITE YOUR CODE HERE: find the gradient and do a gradient step
            gradient = self._gradient_cross_entropy(training_data,training_labels,weights)
            weights = weights- self.lr*gradient
            ##################################
            
            # If we reach 100% accuracy, we can stop training immediately
            predictions = self._logistic_regression_predict(training_data, weights)
            accuracy=accuracy_fn(predictions, training_labels)
            if  accuracy== 100:
                self.weights=weights
                break
            if accuracy> max_accuracy:
                self.weights=weights
        #return weights
        return predictions

    def _sigmoid(self,a):
        """
        Apply the sigmoid function to each element of an array.
        
        Args:
            a (array): Input data of shape (N,) 
        Returns:
            sigmoid(a) (array): Probabilites of shape (N,), where each value is in (0, 1).
        """
        ### WRITE YOUR CODE HERE
        return np.array([1.0/(1+np.exp(-x)) for x in a])

    def _gradient_cross_entropy(self,data, labels, w):
        """
        Gradient of the cross-entropy for logistic regression on binary classes.
        
        Args:
            data (array): Dataset of shape (N, D).
            labels (array): Labels of shape (N,).
            w (array): Weights of logistic regression model of shape (D,)
        Returns:
            grad (array): Gradient array of shape (D,)
        """
        ### WRITE YOUR CODE HERE
        return data.T@(self._sigmoid(data@w)-labels)

    def _logistic_regression_predict(self,data, w):
        """ 
        Predict the label of data for binary class logistic regression. 
        
        Args:
            data (array): Dataset of shape (N, D).
            w (array): Weights of logistic regression model of shape (D,)
        Returns:
            predictions (array): Predicted labels of data, of shape (N,)
        """
        ### WRITE YOUR CODE HERE
        arr=self._sigmoid(data@w)
        #predictions= np.array([0 if x < 0.5 else 1 for x in arr]) #correct
        #predictions = np.vectorize(lambda x: 0 if x<0.5 else 1)(arr) #correct
        predictions = np.where(arr<0.5,0,1)
        return predictions.astype(int)
    
    def _f_softmax(self,data, W):
        """
        Softmax function for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            W (array): Weights of shape (D, C) where C is the number of classes
        Returns:
            array of shape (N, C): Probability array where each value is in the
                range [0, 1] and each row sums to 1.
                The row i corresponds to the prediction of the ith data sample, and
                the column j to the jth class. So element [i, j] is P(y_i=k_j | x_i, W)
        """
        ### WRITE YOUR CODE HERE 
        # Hint: try to decompose the above formula in different steps to avoid recomputing the same things.
        new_matrix = np.vectorize(lambda x: np.exp(x))(data@W)
        sums = new_matrix@np.ones(W.shape[1])
        return np.array([x/y for (x,y) in zip(new_matrix,sums)])
    
    def _loss_logistic_multi(self,data, labels, w):
        """ 
        Loss function for multi class logistic regression, i.e., multi-class entropy.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            w (array): Weights of shape (D, C)
        Returns:
            float: Loss value 
        """
        ### WRITE YOUR CODE HERE 
        return -sum([ y*t for (Y,T) in zip(np.vectorize(lambda x: np.log(x))(self._f_softmax(data,w)),labels) for (y,t) in zip(Y,T) ])

    def _gradient_logistic_multi(self,data, labels, W):
        """
        Compute the gradient of the entropy for multi-class logistic regression.
        
        Args:
            data (array): Input data of shape (N, D)
            labels (array): Labels of shape  (N, C)  (in one-hot representation)
            W (array): Weights of shape (D, C)
        Returns:
            grad (np.array): Gradients of shape (D, C)
        """
        predictions=self._f_softmax(data,W)
        res=np.matmul(data.T,(predictions-labels))
        return res
    
    def logistic_regression_predict_multi(self,data, W):
        """
        Prediction the label of data for multi-class logistic regression.
        
        Args:
            data (array): Dataset of shape (N, D).
            W (array): Weights of multi-class logistic regression model of shape (D, C)
        Returns:
            array of shape (N,): Label predictions of data.
        """
        res=np.argmax(self._f_softmax(data,W)).T
        print(res.shape)
        print(res)
        #return np.array([np.argmax(x) for x in f_softmax(data,W)])
        return np.argmax(self._f_softmax(data,W)).T

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
        return test_data@self.weights
