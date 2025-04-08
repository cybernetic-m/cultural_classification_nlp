import pandas as pd


def dataset_parser(dataset):
# Function to read the tsv dataset file and returning it as a pandas dataframe
# Args: 
#       - dataset: is the tsv file containing the dataset

    df = pd.read_csv(dataset, sep='\t')
    print("The dataset is:\n")
    print(df.head(10))
    return df