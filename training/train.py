from datetime import datetime
import os, sys
import torch
import time

# Get the absolute paths of the directories containing the utils functions and train_one_epoch
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../training'))

# Add these directories to sys.path
sys.path.append(src_path)
sys.path.append(training_path)

# Import section
from calculate_metrics import calculate_metrics
from train_one_epoch import train_one_epoch
from utils import dict_save_and_load

def train(num_epochs, model, train_dataloader, val_dataloader, train_metrics_dict, val_metrics_dict, optimizer, loss_fn, device, config_dict):

  # Create a new directory for this training with the path ./training_2025-04-11_14-38
  current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
  training_dir = './training_' + current_time
  os.makedirs(training_dir, exist_ok=True)

  train_dict_list = [] # a list to save all the training metrics dictionaries (1 for each epoch)
  val_dict_list = [] # a list to save all the validation metrics dictionaries (1 for each epoch)
  time_epochs = [] # a list of each epoch training time

  best_vloss = 10000000000 # Set an high validation loss as start to save the best model

  for epoch in range(num_epochs):

    model.train() # Set the model in training mode (abilitates the updating of parameters)

    print(f'\nEPOCH {epoch + 1}/{num_epochs}:')

    start_epoch_time = time.time() # start counting the epoch time
    # Start the epoch
    loss_epoch, y_true_list, y_pred_list = train_one_epoch(model, train_dataloader, optimizer, loss_fn, device)
    end_epoch_time = time.time() # finish counting the epoch time
    epoch_time = end_epoch_time - start_epoch_time # compute the time of this epoch
    time_epochs.append(epoch_time) # append in the list

    # compute accuracy_ precision, recall and f1_score and save them in the train_metrics_dict dictionary
    train_metrics_dict, _ = calculate_metrics(y_true_list, y_pred_list, train_metrics_dict, epoch) 
    train_metrics_dict['loss'] = loss_epoch # add to the dictionary also the training average loss for that epoch
    train_dict_list.append(train_metrics_dict.copy()) # append the dictionary to the list of all the epochs
    
    print("TRAIN:")
    print(f"loss:{loss_epoch}  accuracy: {train_metrics_dict['accuracy']}  precision:{train_metrics_dict['precision']}  recall:{train_metrics_dict['recall']}  f1-score: {train_metrics_dict['f1']}")

    model.eval() # Set the model in evaluation mode (disable the updating of parameters)
    with torch.no_grad():
      # Do the validation pass
      vloss_epoch, vy_true_list, vy_pred_list = train_one_epoch(model, val_dataloader, optimizer, loss_fn, device, train_mode=False)
    
    # Same passages of before but with validation data
    validation_metrics_dict, _ = calculate_metrics(vy_true_list, vy_pred_list, val_metrics_dict, epoch)
    validation_metrics_dict['loss'] = vloss_epoch
    val_dict_list.append(val_metrics_dict.copy())

    # save the best model in terms of loss
    if vloss_epoch < best_vloss:
      best_vloss = vloss_epoch
      torch.save(model.state_dict(), training_dir + '/best_model.pt')

    print("VALIDATION:")
    print(f"loss: {vloss_epoch}  accuracy: {validation_metrics_dict['accuracy']}  precision:{validation_metrics_dict['precision']}  recall:{validation_metrics_dict['recall']}  f1-score: {validation_metrics_dict['f1']}")

  # Save training, validation dictionaries of the whole training and also the config file with all the hyperparams of the model
  dict_save_and_load(train_dict_list, training_dir + '/train_metrics_dict.json', todo='save')
  dict_save_and_load(val_dict_list, training_dir + '/val_metrics_dict.json', todo='save')
  dict_save_and_load(config_dict, training_dir + '/hyperparams.json', todo='save')

  # Compute also the total training time and the average per epoch (returned)
  total_training_time = sum(time_epochs)
  avg_epoch_training_time = total_training_time / num_epochs

  return total_training_time, avg_epoch_training_time