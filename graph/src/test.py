from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.semi_supervised import LabelPropagation


def test(A, y, node_idx, X_test, y_test, kernel, gamma, n_neighbors, print_statistics):
    predictions_df = pd.DataFrame({'qid': X_test})  # Create DataFrame

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

    return predictions_df, acc

# MIGLIORE NN = 10: 0.73 CON SPLIT_VAL=2000