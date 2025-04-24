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

from graph.src.graph import make_graph, prepare_data
from graph.src.process_data import process_df
from graph.src.save_and_load import load_graph


def test(A, y, node_idx, X_test, y_test, kernel, gamma, n_neighbors, print_statistics):
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
    y_true = []
    for node in X_test:
        i = node_idx[node]  # index for current test node
        pred = lp_model.transduction_[i]  # get prediction from index
        y_pred.append(pred)
        # Get the true label from y_test instead of the graph
        # This fixes the issue of having None in y_true
        true_label = y_test[X_test.index(node)]
        y_true.append(true_label)

    predictions_df['y_pred'] = y_pred

    int_to_labels = {v: k for k, v in labels_to_int.items()}

    final_df = predictions_df[['qid']].copy()
    final_df['label'] = predictions_df['predicted_label'].map(int_to_labels)

    # 8. Accuracy
    acc = accuracy_score(y_true, y_pred)

    if print_statistics:
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

    return final_df, acc


def eval_non_lm(df):
  # this function downloads the data of the properties and the languages
  my_test_df = process_df(df)
  G = load_graph('cultural_graph')
  # adds the new elements to the graph already made of the training data, without labels, the labels are set to -1 (to be predicted)
  make_graph(G, my_test_df, False)

  X_test = my_test_df["qid"].tolist()
  y_test = my_test_df["label"].tolist()

  A, y, node_idx = prepare_data(G)

  # returns a df with qid and predictions, the last param indicates to not print the confusion matrix
  predictions_df, acc = test(A, y, node_idx, X_test, y_test, 'knn', 10, 10, False)
  #print(f'accuracy sul test: {acc}')
  #predictions_df.head()
  predictions_df.to_csv('predictions.csv', index=False)
  return predictions_df
