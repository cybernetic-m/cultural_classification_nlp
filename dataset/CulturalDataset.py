from torch.utils.data import Dataset

class CulturalDataset(Dataset):
  def __init__(self, dataset, tokenizer, max_length, text_type='ND', labels_flag=False):
    super(CulturalDataset, self).__init__()

    self.dataset = dataset # Pandas dataframe with cultural data
    self.tokenizer = tokenizer # the type of tokenizer
    self.max_length = max_length # the length maximum of each tokenized output
    # it is the type of text in output 
    # ('ND' [Name and Description], 'NDV' [Name, Description, Views], 'NDVS' [Name, Description, Views and Summary])
    # Views are the wikipedia page views, while Summary is a Wikipedia page summary
    self.text_type = text_type 
    # Flag to check if the dataset has labels or not
    self.labels_flag = labels_flag

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, index):
    item = self.dataset.iloc[index] # take the ith row from the dataset pandas dataframe
    
    # Construct a unique sentence with all the features in the dataset to give to BERT encoder
    if self.text_type == 'ND':
      text = f"Name: {item['name']}. Description: {item['description']}"
    elif self.text_type == 'NDV':
      text = f"Name: {item['name']}. Description: {item['description']}. Views: {item['en_wikipedia_views']}"
    elif self.text_type == 'NDVS':
      text = f"Name: {item['name']}. Description: {item['description']}. Views: {item['en_wikipedia_views']}. Summary: {item['en_wikipedia_summary']}"

    if self.labels_flag == True:
      # Put label as 0,1,2
      if item['label'] == 'cultural agnostic':
        label = 0
      elif item['label'] == 'cultural representative':
        label = 1
      else:
        label = 2
    else:
      label = -1 # Dummy value for the label 

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
    return {"input_ids":tokens['input_ids'].squeeze(0), "attention_mask":tokens['attention_mask'].squeeze(0), "label":label, "text": text}