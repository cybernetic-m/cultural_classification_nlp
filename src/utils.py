import pandas as pd


def dataset_parser(dataset):
# Function to read the tsv dataset file and returning it as a pandas dataframe
# Args: 
#       - dataset: is the tsv file containing the dataset

    # Read the tsv file using a pandas dataframe
    df = pd.read_csv(dataset, sep='\t')

    print("The dataset is:\n")
    print(df.head(10))

    list_id = [] # A list of all wikidata id of the entities
    urls_col = df['item'] # select the column of each url

    for url in urls_col:
        print(url)

    return df

#df = dataset_parser("./train_set.tsv")