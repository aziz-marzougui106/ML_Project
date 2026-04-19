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
        hot_labels =label_to_onehot(training_labels)
        weights = np.random.normal(0., 0.1, (training_data.shape[1],3))
        predictions=np.empty(training_labels.shape)
        G = np.zeros_like(weights)
        eps = 1e-8
        m = np.zeros_like(weights)
        v = np.zeros_like(weights)
        beta1, beta2 = 0.9, 0.999
        for t in range(1,self.max_iters+1):
            ############# WRITE YOUR CODE HERE: find the gradient and do a gradient step
            gradient = self._gradient_logistic_multi(training_data,hot_labels,weights)
            # G += gradient**2
            # adjusted_lr = self.lr / (np.sqrt(G) + eps)
            #weights = weights- adjusted_lr*gradient
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * (gradient**2)

            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)

            weights = weights - self.lr * m_hat / (np.sqrt(v_hat) + eps)
            ##################################
            
            # If we reach 100% accuracy, we can stop training immediately
            predictions = self._logistic_regression_predict_multi(training_data, weights)
            accuracy=accuracy_fn(predictions, training_labels)
            if  accuracy== 100:
                self.weights=weights
                break
            if accuracy> max_accuracy:
                max_accuracy=accuracy
                self.weights=weights
        return predictions

    
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
        scores = data @ W
        # numerical stability: subtract per-row max before exp
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)  # (N, C)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # (N, C)

        return probs
    
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
        probs = self._f_softmax(data, w)
        return - np.sum(labels * np.log(probs))
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
        res=data.T@(predictions-labels)
        return res
    
    def _logistic_regression_predict_multi(self,data, W):
        """
        Prediction the label of data for multi-class logistic regression.
        
        Args:
            data (array): Dataset of shape (N, D).
            W (array): Weights of multi-class logistic regression model of shape (D, C)
        Returns:
            array of shape (N,): Label predictions of data.
        """
        return np.argmax(self._f_softmax(data,W),axis=1)

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
        return self._logistic_regression_predict_multi(test_data,self.weights)
