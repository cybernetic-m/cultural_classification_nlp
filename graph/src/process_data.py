import pandas as pd
import json
from tqdm import tqdm
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import random
import networkx as nx
import numpy as np
import requests
from urllib.parse import quote
from wikidata.client import Client
import os
import sys

# Get the absolute paths of the directories containing the utils functions and train_one_epoch
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))

# Add these directories to sys.path
sys.path.append(src_path)

# Import section
from save_and_load import load_encoder

# Thread-safe cache per client.get()
wikidata_cache = {}
wikidata_lock = threading.Lock()  # Per scrittura concorrente sicura


def cached_id2string(id_, client):
    """
        Retrieve the English label for a given Wikidata ID using a cache.

        Inputs:
        - id_: Wikidata ID string (e.g., 'Q42')
        - client: Wikidata API client object

        Output:
        - label: English label (string) corresponding to the ID
    """
    with wikidata_lock:
        if id_ in wikidata_cache:
            return wikidata_cache[id_]
    # fuori dal lock: esegue chiamata se non era presente
    element_ = client.get(id_, load=True)
    label = element_.label.get('en', '')
    with wikidata_lock:
        wikidata_cache[id_] = label
    return label


def getEP_cached(id_, client):
    """
        Retrieve properties of a Wikidata entity, using caching for property labels.

        Inputs:
        - id_: Wikidata entity ID (string)
        - client: Wikidata API client object

        Outputs:
        - entity: full entity object
        - prop_list: list of property IDs
        - prop_names_list: list of corresponding property labels (strings)
    """
    prop_list = []
    prop_names_list = []
    try:
        entity = client.get(id_, load=True)
    except Exception as e:
        print(f"Error getting entity {id_}: {e}")
        return None, [], []  # Return None and empty lists if entity retrieval fails

    if entity is None:
        return None, [], []  # Return None and empty lists if entity is None

    for prop in entity.data['claims']:
        prop_list.append(prop)
        prop_name = cached_id2string(prop, client)
        prop_names_list.append(prop_name)

    return entity, prop_list, prop_names_list


def dict2pd(list_dict):
    """
        Convert a list of dictionaries into a pandas DataFrame,
        with one-hot encoding of property IDs.

        Inputs:
        - list_dict: list of dictionaries containing entity data

        Output:
        - df: pandas DataFrame with columns for 'qid', properties (as binary flags), and 'label'
    """
    pid_set = set()
    rows = []

    for mydict in list_dict:
        for q in mydict.values():
            pid_set.update(q['properties']['id'])

    ordered_pid = sorted(pid_set, key=lambda x: int(x[1:]))

    for mydict in list_dict:
        for qid, qdata in mydict.items():
            row = OrderedDict()
            row['qid'] = qid

            for pid in ordered_pid:
                row[pid] = 1 if pid in qdata['properties']['id'] else 0

            label = qdata['label']
            row['label'] = label
            rows.append(row)

    df = pd.DataFrame(rows)
    final_columns = ['qid'] + ordered_pid + ['label']
    df = df[final_columns]
    df[ordered_pid] = df[ordered_pid].fillna(0).astype(int)

    return df


def process_item(args, client, label_map):
    """
        Process a single item (URL, name, label) to extract Wikidata properties.

        Inputs:
        - args: tuple (url, name, label) for the item
        - client: Wikidata API client object
        - label_map: dictionary mapping label strings to integer classes

        Output:
        - dictionary with entity data if successful, otherwise None
    """
    time.sleep(random.uniform(0.3, 1.0))  # Sleep fra 300ms e 1s
    url, name_, label_ = args
    single_id = url.split("/")[-1]
    entity, prop_list, prop_names_list = getEP_cached(single_id, client)

    if entity is None:
        print(f"Skipping item {single_id} due to entity retrieval error.")
        return None  # Signal to skip this item

    return {
        single_id: {
            'name': name_,
            'properties': {
                'id': prop_list,
                'name': prop_names_list,
            },
            'label': label_map.get(label_, -1), # default is -1
        }
    }


def parse_df_properties(dataset, client, labels_flag = False):
    """
    Parse a dataset to extract Wikidata properties and convert to a pandas DataFrame.

    Inputs:
    - dataset: pandas DataFrame with columns ['item', 'name', 'label']
    - client: Wikidata API client object

    Output:
    - my_df: pandas DataFrame where each row represents an entity,
             with one-hot encoded properties and class label
    """

    original_df = dataset
    list_dict = []

    label_map = {
        "cultural agnostic": 2,
        "cultural representative": 1,
        "cultural exclusive": 0,
        "": -1
    }

    urls_col = original_df['item']

    if labels_flag:
        label_col = original_df['label']
    else:
        label_col = [-1 for _ in range(len(original_df))] # if no labels

    name_col = original_df['name']

    data_tuples = list(zip(urls_col, name_col, label_col))

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(tqdm(executor.map(lambda args: process_item(args, client, label_map), data_tuples),
                            total=len(data_tuples),
                            desc="Parsing dataset properties..."))

        list_dict = [result for result in results if result is not None]  # Exclude None results

    my_df = dict2pd(list_dict)

    return my_df


# -------------------------------------#
# Function to get views of wikipedia page
# -------------------------------------#
def get_pageviews(lang, title, start='20240101', end='20250101', retries=5, delay=1):
    """
    Retrieve the total pageviews of a Wikipedia article over a given time period.

    Inputs:
    - lang: language code for Wikipedia (e.g., 'en', 'it')
    - title: title of the Wikipedia article (string)
    - start: start date in 'YYYYMMDD' format (default '20240101')
    - end: end date in 'YYYYMMDD' format (default '20250101')
    - retries: number of retry attempts in case of failure (default 5)
    - delay: delay (in seconds) between retry attempts (default 1)

    Output:
    - tuple (lang, total_views) where total_views is an integer or np.nan if failed
    """
    encoded_title = quote(title, safe='')
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{lang}.wikipedia/all-access/user/{encoded_title}/daily/{start}00/{end}00"
    )

    headers = {
        "User-Agent": "MyResearchBot/1.0 (lissalattanzio.2154208@studenti.uniroma1.it)"
    }

    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code == 200:
                data = r.json()
                views = sum(item['views'] for item in data['items'])
                return (lang, views)
            elif r.status_code == 404:
                return (lang, np.nan)
            else:
                print(f"[{lang}, {encoded_title}] Tentativo {attempt + 1}: status code {r.status_code} - url: {url}")
        except Exception as e:
            print(f"[{lang}, {encoded_title}] Tentativo {attempt + 1} fallito con errore: {e}")
        time.sleep(delay)

    return (lang, np.nan)


# -------------------------------------#
# Function to download pageviewels for each language
# -------------------------------------#
def parse_df_languages(df, labels_flag = False):
    """
    Parse a dataset to collect Wikipedia pageviews across multiple languages for each entity.

    Inputs:
    - df: pandas DataFrame with at least the columns ['item', 'label']

    Output:
    - output_df: pandas DataFrame where each row corresponds to a QID,
                 with columns for pageviews in different languages and class label
    """

    client = Client()
    language_pageview_data = {}
    labels_to_int = {"cultural exclusive": 0, "cultural representative": 1, "cultural agnostic": 2}
    all_languages = set()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Parsing languages...'):
        qid = row['item'].split('/')[-1]

        if labels_flag:
            label_from_df = row['label']
            # Initialize the pageview dictionary for the current QID with its class label
            language_pageview_data[qid] = {'label': labels_to_int[label_from_df]}
        else:
            # Initialize the pageview dictionary for the current QID without a class label
            language_pageview_data[qid] = {}
            print(f'made data of {qid}')

        try:
            item = client.get(qid, load=True)
        except Exception as e:
            print(f"Errore con {qid}: {e}")
            continue

        labels = item.data.get("labels", {})  # Get all available language labels

        if not labels:
            print(f" - Warning : {qid} has no languages associated.")
            #continue



        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {
                executor.submit(get_pageviews, lang, label['value']): lang
                for lang, label in list(labels.items())[:100]  # Limit to first 100 languages
            }

            for future in as_completed(futures):
                lang = futures[future]
                try:
                    lang, views = future.result()
                    if views > 0:
                        language_pageview_data[qid][lang] = views
                        all_languages.add(lang)
                except Exception as exc:
                    print(f"Errore durante il recupero di {lang}: {exc}")

    # Crea il DataFrame finale
    output_df = pd.DataFrame.from_dict(language_pageview_data, orient='index')
    output_df = output_df.fillna(0).astype(int)  # Fill missing values with 0
    output_df.index.name = 'qid'
    output_df = output_df.reset_index()

    return output_df


def encode_cols_and_merge(df, output_df, category_encoder, subcategory_encoder):
    """
    Encode 'category' and 'subcategory' columns using pre-trained encoders and merge with output dataframe.

    Inputs:
    - df: pandas DataFrame containing original dataset (must have 'item', 'category', 'subcategory' columns)
    - output_df: pandas DataFrame to merge with (must have 'qid')
    - category_encoder: pre-fitted sklearn encoder for 'category'
    - subcategory_encoder: pre-fitted sklearn encoder for 'subcategory'

    Output:
    - merged_df: pandas DataFrame with encoded category columns merged
    """

    df_encoded = df.copy()
    # get ID from 'item'
    df_encoded['item'] = df_encoded['item'].astype(str).apply(lambda x: x.split('/')[-1])

    # but use the pre-fitted encoders with -1 if class is unknown:
    df_encoded['category_encoded'] = df_encoded['category'].astype(str).apply(
        lambda x: category_encoder.transform([x])[0] if x in category_encoder.classes_ else -1
    )

    df_encoded['subcategory_encoded'] = df_encoded['subcategory'].astype(str).apply(
        lambda x: subcategory_encoder.transform([x])[0] if x in subcategory_encoder.classes_ else -1
    )

    output_df['qid'] = output_df['qid'].astype(str)

    # Merge con encoding
    merged_df = pd.merge(
        output_df,
        df_encoded[['item', 'category_encoded', 'subcategory_encoded']],
        left_on='qid',
        right_on='item',
        how='left',
        suffixes=('_output', '_encoded')  # Specify suffixes to avoid duplicate column names
    )

    return merged_df


def merge_p_language(lang_df, p_df):
    """
    Merge property columns (starting with 'P') from one dataframe into language df.

    Inputs:
    - lang_df: language pandas DataFrame (must have 'qid')
    - p_df: pandas DataFrame containing 'qid' and property columns

    Output:
    - merged DataFrame with property columns added
    """
    p_cols = [col for col in p_df.columns if col.startswith('P')]
    my_df_p = p_df[['qid'] + p_cols]

    return pd.merge(lang_df, my_df_p, on='qid', how='left')

def clean_df(df):
    """
    Cleans the df, this was needed after wikidata removed an element on april 17
    sometimes we may get float values in the df, we also convert them to int here
    """
    # Get the 'qid' values before dropping rows
    original_qids = set(df['qid'])

    # Remove rows with any missing values
    df = df.dropna()

    # Get the 'qid' values after dropping rows
    remaining_qids = set(df['qid'])

    # Find the difference to get the removed 'qid' values
    removed_qids = original_qids - remaining_qids
    print("Removed qids:", removed_qids)

    df = df.copy()  # removes a warning by pandas
    for column in df.select_dtypes(include=['float64']).columns:
        df[column] = df[column].astype(int)

    return df

def process_df(df, path = None, labels_flag = False):
    """
    Complete processing pipeline:
    - Parse Wikidata properties
    - Parse Wikipedia languages and pageviews
    - Encode category and subcategory
    - Merge all features into a final dataset

    Input:
    - df: pandas DataFrame with at least 'item', 'category', 'subcategory', 'label'

    Output:
    - final_df: pandas DataFrame fully processed and ready for modeling
    """

    print('Parsing properties')
    my_df_P = parse_df_properties(df, Client())  # get properties

    print('Parsing languages')
    my_df_lang = parse_df_languages(df, labels_flag)  # get languages

    #my_df_P.to_csv('dataframe_P.csv', index=False)
    #my_df_lang.to_csv('dataframe_lang.csv', index=False)

    if labels_flag:
        my_df_lang['total_views'] = my_df_lang.drop(columns=['label', 'qid']).sum(axis=1)
    else:
        my_df_lang['total_views'] = my_df_lang.drop(columns=['qid']).sum(axis=1)


    category_encoder = load_encoder(path + '/category_encoder.pkl')
    subcategory_encoder = load_encoder(path + '/subcategory_encoder.pkl')

    merged_df = encode_cols_and_merge(df, my_df_lang, category_encoder, subcategory_encoder)
    merged_df = merged_df.drop(columns=['item'])

    final_df = merge_p_language(merged_df, my_df_P)

    return clean_df(final_df)
