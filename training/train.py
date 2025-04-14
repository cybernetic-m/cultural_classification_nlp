from datetime import datetime
import os, sys

# Get the absolute paths of the directories containing the utils functions and train_one_epoch
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
training_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../training'))

# Add these directories to sys.path
sys.path.append(src_path)
sys.path.append(training_path)

# Import section
from calculate_metrics import calculate_metrics
from train_one_epoch import train_one_epoch

def train(num_epochs, model, train_dataloader, val_dataloader, train_metrics_dict, val_metrics_dict, optimizer, loss_fn):

  # Create a new directory for this training with the path ./training_2025-04-11_14-38
  current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
  training_dir = './training_' + current_time
  os.makedirs(training_dir, exist_ok=True)

  train_dict_list = [] #
  val_dict_list = [] # a list

  best_vloss = 10000000000 # Set an high validation loss as start to save the best model

  for epoch in range(num_epochs):

    model.train() # Set the model in training mode (abilitates the updating of parameters)

    print(f'\nEPOCH {epoch + 1}/{num_epochs}:')

    loss_epoch, y_true_list, y_pred_list = train_one_epoch(model, train_dataloader, optimizer, loss_fn, device)

    train_metrics_dict, _ = calculate_metrics(y_true_list, y_pred_list, train_metrics_dict, epoch)
    train_metrics_dict['loss'] = loss_epoch
    train_dict_list.append(train_metrics_dict.copy())
    
    print("TRAIN:")
    print(f"loss:{loss_epoch}  accuracy: {train_metrics_dict['accuracy']}  precision:{train_metrics_dict['precision']}  recall:{train_metrics_dict['recall']}  f1-score: {train_metrics_dict['f1']}")

    model.eval() # Set the model in evaluation mode (disable the updating of parameters)
    with torch.no_grad():
      vloss_epoch, vy_true_list, vy_pred_list = train_one_epoch(model, val_dataloader, optimizer, loss_fn, device, train_mode=False)
    
    validation_metrics_dict, _ = calculate_metrics(vy_true_list, vy_pred_list, val_metrics_dict, epoch)
    validation_metrics_dict['loss'] = vloss_epoch
    val_dict_list.append(val_metrics_dict.copy())

    if vloss_epoch < best_vloss:
      best_vloss = vloss_epoch
      torch.save(model.state_dict(), training_dir + '/best_model.pt')

    print("VALIDATION:")
    print(f"loss: {vloss_epoch}  accuracy: {validation_metrics_dict['accuracy']}  precision:{validation_metrics_dict['precision']}  recall:{validation_metrics_dict['recall']}  f1-score: {validation_metrics_dict['f1']}")

  dict_save_and_load(train_dict_list, training_dir + '/train_metrics_dict.json', todo='save')
  dict_save_and_load(val_dict_list, training_dir + '/val_metrics_dict.json', todo='save')