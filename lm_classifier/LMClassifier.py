import torch.nn as nn

class LMClassifier(nn.Module):
  def __init__(self, encoder, num_labels, mlp_list = None, classifier_type = 'linear'):
    super(LMClassifier, self).__init__()
    
    """
    A classification model that combines a transformer encoder (e.g., BERT)
    with a linear or MLP classification at the end.

    Args:
    - encoder: a pretrained transformer model
    - num_labels: number of output classes
    - mlp_list: list of hidden dimensions for the MLP classifier (Ex. [50, 20] will create an MLP of the type [hidden_size, 50, 20, num_labels] )
    - classifier_type: 'linear' or 'mlp' (type of classification head)
    """

    # The encoder (like BERT)
    self.encoder = encoder
    # The classifier output dimension
    self.num_labels = num_labels

    # hidden_size is the dimension of each embedding returned by the encoder (like BERT)
    # Suppose you have a sequence of tokens ['[CLS]', 'pizza', 'is', 'delicious', '.', '[SEP]']
    # in output you have something like (batch_size, seq_len, hidden_size) -> (1, 6, 768) (each token is a vector of 768 dimensional)
    hidden_size = self.encoder.config.hidden_size

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
      # last_hidden_state = tensor[batch_size, seq_len, hidden_size] => (1, 6, 768)
      last_hidden_state = encoder_out.last_hidden_state

      # We take only the first ([CLS]) token that give the embedding of all the sentence
      logits = self.classifier(last_hidden_state[:,0,:])

      return logits

  def summary(self):
    print("Encoder type:\n\n", self.encoder)
    print("\nClassifier:\n\n", self.classifier)