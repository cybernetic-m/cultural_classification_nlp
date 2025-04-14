from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(y_true_list, y_pred_list, metrics_dict, epoch = None):
  
  # Compute metrics using sklearn.metrics functions
  accuracy = accuracy_score(y_true_list, y_pred_list)
  precision = precision_score(y_true_list, y_pred_list, average='macro', zero_division=0)
  recall = recall_score(y_true_list, y_pred_list, average='macro', zero_division=0)
  f1 = f1_score(y_true_list, y_pred_list, average='macro', zero_division=0)

  # Compute the confusion matrix 
  conf_matrix = confusion_matrix(y_true_list, y_pred_list, labels=[0, 1, 2])

  # Add all the computed values inside the dictionary
  if epoch is not None:
    metrics_dict['epoch'] = epoch + 1

  metrics_dict['accuracy'] = accuracy
  metrics_dict['precision'] = precision
  metrics_dict['recall'] = recall
  metrics_dict['f1'] = f1

                                   
  return metrics_dict, conf_matrix