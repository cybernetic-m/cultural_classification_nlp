import torch.nn as nn
import torch

class LMClassifier(nn.Module):
  def __init__(self, encoder, num_labels, mlp_list = None, classifier_type = 'linear', pooling='cls'):
    super(LMClassifier, self).__init__()
    
    """
    A classification model that combines a transformer encoder (e.g., BERT)
    with a linear or MLP classification at the end.

    Args:
    - encoder: a pretrained transformer model
    - num_labels: number of output classes
    - mlp_list: list of hidden dimensions for the MLP classifier (Ex. [50, 20] will create an MLP of the type [hidden_size, 50, 20, num_labels] )
    - classifier_type: 'linear' or 'mlp' (type of classification head)
    - pooling: 'cls' (take the first token embedding), 'mean' (take the mean of all tokens embeddings), 'max' (take the max of all tokens embeddings), 'attention' (use attention mechanism to get the sentence embedding)
    """

    # The encoder (like BERT)
    self.encoder = encoder
    # The classifier output dimension
    self.num_labels = num_labels
    # The pooling type
    self.pooling = pooling

    # hidden_size is the dimension of each embedding returned by the encoder (like BERT)
    # Suppose you have a sequence of tokens ['[CLS]', 'pizza', 'is', 'delicious', '.', '[SEP]']
    # in output you have something like (batch_size, seq_len, hidden_size) -> (1, 6, 768) (each token is a vector of 768 dimensional)
    hidden_size = self.encoder.config.hidden_size

    # If the pooling_type == 'attention', we need to define the attention layer
    if pooling == 'attention':
      self.attention_pooling_layer = nn.Linear(hidden_size, 1) # Linear layer to compute the attention scores

    # Definition of the classifier of the type ('linear', 'mlp')
    if classifier_type == 'linear':
      self.classifier = nn.Linear(hidden_size, self.num_labels)

    # Definition of the mlp classifier having the list
    elif classifier_type == 'mlp':
      layers = [] # definition of the layer list in which we append all the sequence

      # Append the first layer because the input_dim is fixed (depend on BERT hidden_size)
      layers.append(nn.Linear(hidden_size, mlp_list[0]))
      layers.append(nn.ReLU())

      # Append all the hidden_layers
      for i in range(len(mlp_list)-1):
        layers.append(nn.Linear(mlp_list[i], mlp_list[i+1]))
        layers.append(nn.ReLU())

      # Append the output layer without ReLU activation function
      layers.append(nn.Linear(mlp_list[-1], num_labels))

      self.classifier = nn.Sequential(*layers) # create the classifier

    # Print the summary of the model
    self.summary()

  def forward(self, input_ids, attention_mask):

      # Send the input to the encoder (the input_ids from the tokenizer and the attention_mask with 1 and 0 (padding))
      encoder_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

      # The encoder_out containing different things, we take only last_hidden_state that contain all the embeddings for all the tokens
      # passed through all the transformer layers
      # Ex. "Pizza is my love" -> ["[CLS], "pizza", "is", "my", "love", "[SEP]"]
      # input_ids:tensor([[  101, 10733,  2003,  2026,  2293,   102,     0,     0,     0,     0]])
      # attention_mask:tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
      # last_hidden_state = tensor[batch_size, seq_len, hidden_size] => (1, 10, 768) (seq_len is padded to)
      last_hidden_state = encoder_out.last_hidden_state
      
      # Now we want to send to the classifier only one tensor (i.e. the embedding of the sentence) 
      # A tensor of the type tensor[batch_size, seq_len, hidden_size] => (1, 1, 768)
      # We can use different pooling strategies:
      # - cls: We take only the first ([CLS]) token that give the embedding of all the sentence
      # - mean: We take the mean of all the tokens embedding (not tokens that are padding)
      # - max: We take the max of all the tokens embedding (not tokens that are padding)
      # - attention: compute the attention scores for each token (not tokens that are padding) and take the weighted sum of the tokens embedding
      if self.pooling == 'cls':
        # We take only the first ([CLS]) token that give the embedding of all the sentence
        logits = self.classifier(last_hidden_state[:,0,:])
      elif self.pooling == 'mean':
        mask = attention_mask.unsqueeze(-1) # originally (batch_size, seq_len) -> (batch_size, seq_len, 1) for multiplication with last_hidden_state
        masked_hidden_state = mask*last_hidden_state # we take only the vectors of the tokens that are not padding vectors
        sum_hidden_state = torch.sum(masked_hidden_state, dim=1) # Sum all the vectors of the tokens that are not padding
        num_tokens = torch.sum(mask, dim=1) # Count the number of tokens that are not padding
        mean_hidden_state = sum_hidden_state / num_tokens # Take the mean of all the tokens embedding
        logits = self.classifier(mean_hidden_state)
      elif self.pooling == 'max':
        mask = attention_mask.unsqueeze(-1) # originally (batch_size, seq_len) -> (batch_size, seq_len, 1) for multiplication with last_hidden_state
        masked_hidden_state = last_hidden_state.masked_fill(mask == 0, -1e9)  # Set the padding tokens to a very low value 
        max_hidden_state, _ = torch.max(masked_hidden_state, dim=1) # Take the max of all the tokens embedding
        logits = self.classifier(max_hidden_state)
      elif self.pooling == 'attention':
        attention_scores = self.attention_pooling_layer(last_hidden_state).squeeze(-1) # Compute the attention scores
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9) # Mask the attention scores of padding tokens with a very low value
        attention_probs = torch.softmax(attention_scores, dim=1) # Normalize the scores to get the attention probabilities
        context_vector = torch.sum(attention_probs.unsqueeze(-1) * last_hidden_state, dim=1) # Compute a single vector as a weighted sum of the token embeddings
        logits = self.classifier(context_vector)

      return logits

  def summary(self):
    print("Encoder type:\n\n", self.encoder)
    print("\nClassifier:\n\n", self.classifier)
    print("\nPooling type:\n\n", self.pooling)