import numpy as np


# General utilities
##################

def label_to_onehot(labels, C=None):
    """
    Transform the labels into one-hot representations.

    Arguments:
        labels (np.array): labels as class indices, of shape (N,)
        C (int): total number of classes. Optional, if not given
                 it will be inferred from labels.
    Returns:
        one_hot_labels (np.array): one-hot encoding of the labels, of shape (N,C)
    """
    N = labels.shape[0]
    if C is None:
        C = get_n_classes(labels)
    one_hot_labels = np.zeros([N, C])
    one_hot_labels[np.arange(N), labels.astype(int)] = 1
    return one_hot_labels


def onehot_to_label(onehot):
    """
    Transform the labels from one-hot to class index.

    Arguments:
        onehot (np.array): one-hot encoding of the labels, of shape (N,C)
    Returns:
        (np.array): labels as class indices, of shape (N,)
    """
    return np.argmax(onehot, axis=1)


def append_bias_term(data):
    """
    Append to the data a bias term equal to 1.

    Arguments:
        data (np.array): of shape (N,D)
    Returns:
        (np.array): shape (N,D+1)
    """
    N = data.shape[0]
    data = np.concatenate([np.ones([N, 1]), data], axis=1)
    return data


def normalize_fn(data, means, stds):
    """
    Return the normalized data, based on precomputed means and stds.

    Arguments:
        data (np.array): of shape (N,D)
        means (np.array): of shape (1,D)
        stds (np.array): of shape (1,D)
    Returns:
        (np.array): shape (N,D)
    """
    # return the normalized features
    return (data - means) / stds


def get_n_classes(labels):
    """
    Return the number of classes present in the data labels.

    This is approximated by taking the maximum label + 1 (as we count from 0).
    """
    return int(np.max(labels) + 1)


# Metrics
#########

def accuracy_fn(pred_labels, gt_labels):
    """
    Return the accuracy of the predicted labels.
    """
    return np.mean(pred_labels == gt_labels) * 100.


def macrof1_fn(pred_labels, gt_labels):
    """
    Return the macro F1-score.

    Arguments:
        pred_labels (np.array):
        gt_labels (np.array):
    Returns:
        float: macro F1-score
    """
    class_ids = np.unique(gt_labels)
    macrof1 = 0
    for val in class_ids:
        predpos = (pred_labels == val)
        gtpos = (gt_labels == val)

        tp = sum(predpos * gtpos)
        fp = sum(predpos * ~gtpos)
        fn = sum(~predpos * gtpos)
        if tp == 0:
            continue
        else:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)

        macrof1 += 2 * (precision * recall) / (precision + recall)

    return macrof1 / len(class_ids)


def mse_fn(pred, gt):
    """
    Mean Squared Error

    Arguments:
        pred: Nx1 (or N,) prediction array
        gt: Nx1 (or N,) groundtruth values for each prediction
    Returns:
        returns the computed MSE loss
    """
    loss = (pred - gt) ** 2
    loss = np.mean(loss)
    return loss


def k_fold_cross_validation(X, y, model, k=5, random_state=None):
    """
    Perform k-fold cross-validation and compute average metrics.

    Arguments:
        X (np.array): features, shape (N, D)
        y (np.array): labels, shape (N,)
        model: model instance with fit(X, y) and predict(X) methods
        k (int): number of folds
        random_state (int): seed for reproducibility
    Returns:
        dict: average accuracy, macro F1, and MSE over the k folds
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    N = X.shape[0]
    indices = np.arange(N)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    fold_sizes = np.full(k, N // k, dtype=int)
    fold_sizes[:N % k] += 1
    
    accuracies = []
    macrof1s = []
    mses = []
    
    start = 0
    for fold_size in fold_sizes:
        end = start + fold_size
        test_indices = np.arange(start, end)
        train_indices = np.concatenate([np.arange(0, start), np.arange(end, N)])
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        accuracies.append(accuracy_fn(pred, y_test))
        macrof1s.append(macrof1_fn(pred, y_test))
        mses.append(mse_fn(pred, y_test))
        
        start = end
    
    return {
        'accuracy': np.mean(accuracies),
        'macrof1': np.mean(macrof1s),
        'mse': np.mean(mses)
    }
