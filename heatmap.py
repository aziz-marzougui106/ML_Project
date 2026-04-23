import matplotlib.pyplot as plt
import numpy as np
from src.methods.dummy_methods import DummyClassifier
from src.methods.logistic_regression import LogisticRegression
from src.methods.linear_regression import LinearRegression
from src.methods.knn import KNN
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, k_fold_cross_validation
import os


def compute_matrices(X_train, y_train, X_val, y_val, learning_rates, max_iterations_list):
    acc_matrix = np.zeros((len(max_iterations_list), len(learning_rates)))
    f1_matrix = np.zeros((len(max_iterations_list), len(learning_rates)))

    for i, max_iter in enumerate(max_iterations_list):
        for j, lr in enumerate(learning_rates):
            print(f'{i},{j}')
            model = LogisticRegression(lr, max_iter)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc_matrix[i, j] = accuracy_fn(y_pred, y_val)
            f1_matrix[i, j] = macrof1_fn(y_pred, y_val)
    return acc_matrix, f1_matrix


def grid_search_LOGREG(X_train, y_train, X_val, y_val, learning_rates, max_iterations_list):
    acc_matrix = np.zeros((len(max_iterations_list), len(learning_rates)))
    f1_matrix = np.zeros((len(max_iterations_list), len(learning_rates)))

    for i, max_iter in enumerate(max_iterations_list):
        for j, lr in enumerate(learning_rates):
            print(f'{i},{j}')
            model = LogisticRegression(lr, max_iter)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            acc_matrix[i, j] = accuracy_fn(y_pred, y_val)
            f1_matrix[i, j] = macrof1_fn(y_pred, y_val)

    best_acc = -1
    best_f1 = -1
    best_learning_rate = None
    best_max_iterations = None


    for i, max_iter in enumerate(max_iterations_list):
        for j, lr in enumerate(learning_rates):
            print("hello")
            acc = acc_matrix[i][j]
            f1 = f1_matrix[i][j]
            if acc > best_acc or (acc == best_acc and f1 > best_f1):
                best_acc = acc
                best_f1 = f1
                best_learning_rate = lr
                best_max_iterations = max_iter

    # Highlight best point on the accuracy curve
    
    
    print("Best hyperparameters found:")
    print("learning_rate =", best_learning_rate)
    print("max_iterations =", best_max_iterations)
    print("accuracy =", best_acc)
    print("f1_score =", best_f1)

    return best_learning_rate, best_max_iterations, best_acc, best_f1


