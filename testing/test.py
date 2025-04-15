from datetime import datetime
import os, sys
import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Get the absolute paths of the directories containing the utils functions and train_one_epoch
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))

# Add these directories to sys.path
sys.path.append(src_path)

# Import section
from calculate_metrics import calculate_metrics
from utils import dict_save_and_load
import pandas as pd

def test(model, model_path, test_dataloader, test_metrics_dict, loss_fn, device):
  
  # Create a new directory for this training with the path ./training_2025-04-11_14-38
  current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
  test_dir = './test_' + current_time
  os.makedirs(test_dir, exist_ok=True)

  # Load the model weights from a pt file
  model.load_state_dict(torch.load(model_path)) 

  # Set the model in evaluation mode
  model.eval()

  # Initialization of loss and lists for y_pred, y_true and the inference time
  test_loss_batch = 0
  y_true_list = []
  y_pred_list = []
  inference_time_list = []
  time_dict = {}

  # Loop for evaluate the model
  for _, data in enumerate(tqdm(test_dataloader, desc="Testing", leave=False)):

    # Read the input_ids and attention_mask from the dataloader
    input_ids_i = data['input_ids'].to(device) 
    attention_mask_i = data['attention_mask'].to(device)

    # Store the y_true and compute y_pred 
    y_true = data['label'].to(device)
    start_time = time.time()

    with torch.no_grad():
      y_pred = model(input_ids_i, attention_mask_i)
      end_time = time.time()
      inference_time_list.append(end_time - start_time)

      # Update the lists
      y_true_list += y_true.tolist()
      y_pred_list += torch.argmax(y_pred,dim=1).tolist()

      # Compute the loss giving both tensors with predictions and true values for this batch
      # it is a pytorch tensor tensor(1.2345, grad_fn=<NllLossBackward0>)
      loss = loss_fn(y_pred, y_true)
      test_loss_batch += loss.item()

  test_loss = test_loss_batch / len(test_dataloader) # average for all the batches

  # Compute also the total training time and the average per epoch 
  time_dict['total_training_time'] = sum(inference_time_list) 
  time_dict ['avg_epoch_training_time'] = sum(inference_time_list) / num_epochs

  test_metrics_dict, confusion_matrix = calculate_metrics(y_true_list, y_pred_list, train_metrics_dict) 
  test_metrics_dict['loss'] = test_loss

  # Save training, validation dictionaries of the whole training 
  # the config file with all the hyperparams of the model
  # the time data (total time and average time)
  dict_save_and_load(test_metrics_dict, test_dir + '/test_metrics_dict.json', todo='save')
  dict_save_and_load(time_dict, test_dir + '/time.json', todo='save')

  # Transform the confusion matrix to a pandas dataframe
  labels = ['C.A.', 'C.R.', 'C.E.']
  cm_df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

  # Save the confusion matrix in a csv file
  cm_df.to_csv(test_dir + "/confusion_matrix.csv")

  print("Test Metrics:")
  print(f"test loss: {test_metrics_dict['loss']}")
  print(f"accuracy: {test_metrics_dict['accuracy']}  precision: {test_metrics_dict['precision']}  recall: {test_metrics_dict['recall']}  f1_score: {test_metrics_dict['f1']}")
  print("Confusion Matrix:")

  plt.figure(figsize=(6, 5))
  sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
  plt.title("Confusion Matrix")
  plt.ylabel("True Label")
  plt.xlabel("Predicted Label")
  plt.tight_layout()
  plt.savefig("confusion_matrix.png") 