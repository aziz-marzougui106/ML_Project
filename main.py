import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, k_fold_cross_validation
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Save full training data for k-fold CV
    full_train_features = train_features.copy()
    full_train_labels_reg = train_labels_reg.copy()
    full_train_labels_classif = train_labels_classif.copy()

    # Make a validation set (it can overwrite xtest, ytest)
    if not args.test:
        ### WRITE YOUR CODE HERE
        n = np.arange(train_features.shape[0])
        np.random.shuffle(n)
        limit = int(0.8 * train_features.shape[0])
        train_indx = n[:limit]
        test_indx = n[limit:]
        # Use copies to avoid modifying original
        original_train_features = train_features.copy()
        original_train_labels_reg = train_labels_reg.copy()
        original_train_labels_classif = train_labels_classif.copy()
        train_features = original_train_features[train_indx]
        test_features = original_train_features[test_indx]
        train_labels_reg = original_train_labels_reg[train_indx]
        test_labels_reg = original_train_labels_reg[test_indx]
        train_labels_classif = original_train_labels_classif[train_indx]
        test_labels_classif = original_train_labels_classif[test_indx]

    ### WRITE YOUR CODE HERE to do any other data processing

    ## 3. Initialize the method you want to use.
    #we can remove some data preparation operations to see their effects
    
    if args.normalize_data:
        mean=np.mean(train_features, axis=0,keepdims=True)
        std=np.std(train_features, axis=0,keepdims=True)
        train_features= normalize_fn(train_features,mean,std)
        test_features= normalize_fn(test_features,mean,std)
        if args.kfold:
            full_train_features = normalize_fn(full_train_features, mean, std)
    # Follow the "DummyClassifier" example for your methods
    if args.add_bias:
        train_features= append_bias_term(train_features)
        test_features=append_bias_term(test_features)
        if args.kfold:
            full_train_features = append_bias_term(full_train_features)
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "knn":
        ### WRITE YOUR CODE HERE
        method_obj = KNN(args.K, args.task)

    elif args.method == "logistic_regression":
        ### WRITE YOUR CODE HERE
        method_obj= LogisticRegression(args.lr,args.max_iters) if args.max_iters else LogisticRegression(args.lr)

    elif args.method == "linear_regression":
        ### WRITE YOUR CODE HERE
        method_obj=LinearRegression()

    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":
        assert args.method != "linear_regression", f"You should use linear regression as a regression method"
        # Fit the method on training data
        preds_train = method_obj.fit(train_features, train_labels_classif)

        # Predict on unseen data
        preds = method_obj.predict(test_features)

        # Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, train_labels_classif)
        macrof1 = macrof1_fn(preds_train, train_labels_classif)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, test_labels_classif)
        macrof1 = macrof1_fn(preds, test_labels_classif)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    elif args.task == "regression":
        assert args.method != "logistic_regression", f"You should use logistic regression as a classification method"
        # Fit the method on training data
        preds_train = method_obj.fit(train_features, train_labels_reg)

        # Predict on unseen data
        preds = method_obj.predict(test_features)

        # Report results: MSE on train and valid/test sets
        train_mse = mse_fn(preds_train, train_labels_reg)
        print(f"\nTrain set: MSE = {train_mse:.6f}")

        test_mse = mse_fn(preds, test_labels_reg)
        print(f"Test set:  MSE = {test_mse:.6f}")

    else:
        raise ValueError(f"Unknown task: {args.task}")

    # Optional k-fold cross-validation
    if args.kfold:
        if args.task == "classification":
            results = k_fold_cross_validation(full_train_features, full_train_labels_classif, method_obj, k=args.kfold)
            print(f"K-fold CV ({args.kfold} folds): accuracy = {results['accuracy']:.3f}% - F1-score = {results['macrof1']:.6f}")
        elif args.task == "regression":
            results = k_fold_cross_validation(full_train_features, full_train_labels_reg, method_obj, k=args.kfold)
            print(f"K-fold CV ({args.kfold} folds): MSE = {results['mse']:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / knn / logistic_regression / linear_regression",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=1,
        help="number of neighboring datapoints used for knn",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    # Feel free to add more arguments here if you need!
    parser.add_argument(
        "--normalize_data",
        action="store_true",
        help="normalize the data"
    )
    parser.add_argument(
        "--add_bias",
        action="store_true",
        help="add a bias term to the input data"
    )
    parser.add_argument(
        "--kfold",
        type=int,
        default=None,
        help="number of folds for k-fold cross-validation (if set, runs CV instead of train/test split)"
    )
    args = parser.parse_args()
    main(args)
