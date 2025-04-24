from datetime import datetime
import os, sys
import torch
import time
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# Get the absolute paths of the directories 
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../dataset'))

# Add these directories to sys.path
sys.path.append(src_path)
sys.path.append(dataset_path)

# Import section
from calculate_metrics import calculate_metrics
from utils import dict_save_and_load, add_wikipedia_data
from CulturalDataset import CulturalDataset
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
  time_dict ['avg_epoch_training_time'] = sum(inference_time_list) / len(inference_time_list)

  test_metrics_dict, confusion_matrix = calculate_metrics(y_true_list, y_pred_list, test_metrics_dict) 
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
  sns.heatmap(cm_df, annot=True, fmt="d", cmap="Purples")
  plt.title("Confusion Matrix")
  plt.ylabel("True Label")
  plt.xlabel("Predicted Label")
  plt.tight_layout()
  plt.savefig(test_dir + "/confusion_matrix.png") 


def eval_lm(model, model_path, dataset_csv, tokenizer, batch_size, max_length, test_metrics_dict, loss_fn, device):
  
  # Load the model weights from a pt file
  model.load_state_dict(torch.load(model_path, map_location=device)) 

  # Set the model in evaluation mode
  model.eval()
  
  # Values to Label
  val_to_lab = {0: 'Cultural agnostic', 1: 'Cultural representative.', 2: 'Cultural exclusive'}

  if not os.path.exists(dataset_csv):
    print("No dataset provided.\n Please load it in a csv format into the 'Files' section of colab.\n After this, run another time the cells of 'Load of the dataset'.")
    return
  
  # Transform csv of the dataset into a pandas dataframe
  df = pd.read_csv(dataset_csv)
  
  # Add Wikipedia data
  df_wikipedia = add_wikipedia_data(df, lang='en', max_workers=5)

  # Create the dataloader for the test set
  test_dataset = CulturalDataset(df_wikipedia, tokenizer = tokenizer, max_length = max_length, text_type='NDVS')
  test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

  # Add another column in the dataframe with all the predictions in strings
  df_wikipedia['predicted_label'] = [val_to_lab[i] for i in y_pred_list]

  # Save in csv format the dataframe with the predictions
  df_wikipedia.to_csv("./predictions.csv", index=False)

  # Compute also the total training time and the average per epoch 
  time_dict['total_training_time'] = sum(inference_time_list) 
  time_dict ['avg_epoch_training_time'] = sum(inference_time_list) / len(inference_time_list)

  test_metrics_dict, confusion_matrix = calculate_metrics(y_true_list, y_pred_list, test_metrics_dict) 
  test_metrics_dict['loss'] = test_loss

  # Save training, validation dictionaries of the whole training 
  # the config file with all the hyperparams of the model
  # the time data (total time and average time)
  dict_save_and_load(test_metrics_dict, './test_metrics_dict.json', todo='save')
  dict_save_and_load(time_dict, './time.json', todo='save')

  # Transform the confusion matrix to a pandas dataframe
  labels = ['C.A.', 'C.R.', 'C.E.']
  cm_df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)

  # Save the confusion matrix in a csv file
  cm_df.to_csv("./confusion_matrix.csv")

  print("Test Metrics:")
  print(f"test loss: {test_metrics_dict['loss']}")
  print(f"accuracy: {test_metrics_dict['accuracy']}  precision: {test_metrics_dict['precision']}  recall: {test_metrics_dict['recall']}  f1_score: {test_metrics_dict['f1']}")
  print("Confusion Matrix:")

  plt.figure(figsize=(6, 5))
  sns.heatmap(cm_df, annot=True, fmt="d", cmap="Purples")
  plt.title("Confusion Matrix")
  plt.ylabel("True Label")
  plt.xlabel("Predicted Label")
  plt.tight_layout()
  plt.savefig("/.confusion_matrix.png") 