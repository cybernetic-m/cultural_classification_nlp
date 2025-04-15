import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, loss_fn, device, train_mode=True):

  loss_epoch = 0 # Initialize the loss of the epoch
  y_true_list = [] # List used for append all the true labels to compute metrics in train function
  y_pred_list = [] # List used for append all the pred labels to compute metrics in train function

  for _, data in enumerate(tqdm(dataloader, desc='batch passing...', leave=False)):
    
    # Extract the input_ids and the attention_mask tensors of the ith batch
    input_ids_i = data['input_ids'].to(device)
    attention_mask_i = data['attention_mask'].to(device)
    
    # Extract y_true tensors [0,1,1,0,2,2,....] of dimension batch size and make the predictions 
    y_true = data['label'].to(device)
    y_pred = model(input_ids_i, attention_mask_i)
    y_true_list += y_true.tolist()
    y_pred_list += torch.argmax(y_pred,dim=1).tolist()

    # Compute the loss giving both tensors with predictions and true values for this batch
    # it is a pytorch tensor tensor(1.2345, grad_fn=<NllLossBackward0>)
    loss = loss_fn(y_pred, y_true)

    if train_mode:
      # Zeroes the gradient, compute the amount of gradient updates per parameters and update the parameters
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # With loss.item() we take the number float inside the tensor tensor(1.2345, grad_fn=<NllLossBackward0>) -> 1.2345
    # Accumulate the loss of the epoch summing the loss of this batch
    loss_epoch += loss.item()

  # Compute the average loss averaging for the number of batches
  avg_loss = loss_epoch / len(dataloader)

  return loss_epoch, y_true_list, y_pred_list