from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.semi_supervised import LabelPropagation
import pandas as pd

import os, sys

# Get the absolute paths of the directories containing the utils functions and train_one_epoch
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))

# Add these directories to sys.path
sys.path.append(src_path)

from graph import make_graph, prepare_data
from process_data import process_df
from save_and_load import load_graph


def test(A, y, node_idx, X_test, y_test, kernel, gamma, n_neighbors, print_statistics, labels_flag):
    """
    Apply Label Propagation algorithm and predict labels for test nodes.

    Inputs:
    - A: scipy sparse matrix representing the adjacency matrix of the graph
    - y: array of node labels (with -1 for unlabeled nodes)
    - node_idx: dictionary mapping node names to their indices
    - X_test: list of test node names (qids)
    - y_test: list of true labels for test nodes
    - kernel: type of kernel to use ('knn' or 'rbf')
    - gamma: gamma value for 'rbf' kernel
    - n_neighbors: number of neighbors for 'knn' kernel
    - print_statistics: boolean to print evaluation metrics (confusion matrix, classification report)

    Output:
    - final_df: pandas DataFrame with 'qid' and predicted label
    - acc: float, accuracy score on test nodes
    """
    predictions_df = pd.DataFrame({'qid': X_test})

    labels_to_int = {"cultural exclusive": 0, "cultural representative": 1, "cultural agnostic": 2}

    # 6. Applica Label Propagation di sklearn
    lp_model = LabelPropagation(kernel=kernel,  # oppure 'knn'
                                gamma=gamma,  # solo per kernel 'rbf'
                                n_neighbors=n_neighbors,  # solo per kernel 'knn'
                                max_iter=10000,
                                tol=1e-3)
    lp_model.fit(A, y)

    # 7. Predici per i nodi di test
    y_pred = []
    if labels_flag:
        y_true = []
    for node in X_test:
        i = node_idx[node]  # index for current test node
        pred = lp_model.transduction_[i]  # get prediction from index
        y_pred.append(pred)
        # Get the true label from y_test instead of the graph
        # This fixes the issue of having None in y_true
        if labels_flag:
            true_label = y_test[X_test.index(node)]
            y_true.append(true_label)

    predictions_df['y_pred'] = y_pred

    int_to_labels = {v: k for k, v in labels_to_int.items()}

    final_df = predictions_df.copy()
    list_of_labels = predictions_df['y_pred'].map(int_to_labels)
    final_df['predicted_label'] = predictions_df['y_pred'].map(int_to_labels)


    if print_statistics:
        if labels_flag:
            acc = accuracy_score(y_true, y_pred)
            print(f"Accuracy su nodi di test: {acc:.4f}")

            # 9. Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            # print("\nConfusion Matrix:")
            # print(cm)

            # Visualizza confusion matrix con seaborn
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix - Label Propagation')
            plt.tight_layout()
            plt.show()

            # 10. Classification Report
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, digits=4))

    return final_df, list_of_labels


def eval_non_lm(dataset_csv, path = './models', print_statistics = False):
    """
    Evaluate non-language-model-based approach:
    - Parse properties and languages for the input dataframe
    - Expand the training graph with test nodes (without labels)
    - Apply Label Propagation to predict labels

    Inputs:
    - df: pandas DataFrame containing at least 'item', 'category', 'subcategory', and 'label'

    Output:
    - predictions_df: pandas DataFrame with 'qid' and predicted label
    """
    
     # this function downloads the data of the properties and the languages
    if not os.path.exists(dataset_csv):
        print("No dataset provided.\nPlease load it in a csv format into the 'Files' section of colab.\nAfter this, run another time the cells of 'Load of the dataset'.")
        return 0, 0

    # Transform csv of the dataset into a pandas dataframe
    df = pd.read_csv(dataset_csv)

    labels_flag = False
    # check if the test_dataset has labels......... no comment
    if 'label' in df.columns:
        labels_flag = True

    my_test_df = process_df(df = df, path = path, labels_flag = labels_flag)
    G = load_graph('cultural_graph')
    # adds the new elements to the graph already made of the training data, without labels, the labels are set to -1 (to be predicted)
    make_graph(G, my_test_df, False)

    X_test = my_test_df["qid"].tolist()
    if labels_flag:
        # if the test set has labels, we take them from the dataframe
        y_test = my_test_df["label"].tolist()
    else:
        # if the test set has no labels, we create a list of -1
        y_test = [-1] * len(X_test)

    A, y, node_idx = prepare_data(G)


    # returns a df with qid and predictions, the last param indicates to not print the confusion matrix
    predictions_df, column_to_add = test(A, y, node_idx, X_test, y_test, 'knn', 10, 10, print_statistics, labels_flag=labels_flag)
    # Add the column with the labels to the predictions dataframe
    df['qid'] = df['item'].apply(lambda x: x.split('/')[-1])

    qid_to_label = dict(zip(predictions_df['qid'], column_to_add))

    df['label'] = df['qid'].map(qid_to_label)

    df.to_csv('Caponata_Lovers_output_nonLM.csv', index=False)
    return predictions_df, my_test_df
