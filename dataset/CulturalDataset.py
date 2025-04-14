from torch.utils.data import Dataset

class CulturalDataset(Dataset):
  def __init__(self, dataset, tokenizer, max_length):
    super(CulturalDataset, self).__init__()

    self.dataset = dataset # Pandas dataframe with cultural data
    self.tokenizer = tokenizer # the type of tokenizer
    self.max_length = max_length # the length maximum of each tokenized output

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    item = self.dataset.iloc[index] # take the ith row from the dataset pandas dataframe

    # Construct a unique sentence with all the features in the dataset to give to BERT encoder
    #text = f"The item '{item['name']}' is a '{item['type']}' in the '{item['category']}' category. Description: '{item['description']}'"
    text = f"'{item['name']}''{item['description']}'"
    #text = f"The item is '{item['name']}'"

    # Put label as 0,1,2
    if item['label'] == 'cultural agnostic':
      label = 0
    elif item['label'] == 'cultural representative':
      label = 1
    else:
      label = 2

    # Tokenization
    # The tokenizer having a text returns:
    # - input_ids: the indeces in the vocabulary of each token
    # - attention_mask: A mask with 1 or 0 depending on "real" of "padded" tokens
    # Ex. "Pizza is my love" -> ["[CLS], "pizza", "is", "my", "love", "[SEP]"]
    # input_ids:tensor([[  101, 10733,  2003,  2026,  2293,   102,     0,     0,     0,     0]])
    # attention_mask:tensor([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])

    tokens = self.tokenizer(
    text,
    max_length=self.max_length,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
    )

    # The squeeze is needed to transform the tensor [1, 512] -> [512] eliminating the first useless dimension
    return {"input_ids":tokens['input_ids'].squeeze(0), "attention_mask":tokens['attention_mask'].squeeze(0), "label":label}