import pandas as pd
import json 
import requests
from wikidata.client import Client
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.notebook import tqdm
import numpy as np
import time
import wikipedia

def id2string(id_, client):
    element_ = client.get(id_, load=True)
    label = element_.label.get('en', '')
    return label

def print_labels_counts(my_df, property_id):
  country_df = my_df[my_df[property_id] == 1]  # Select rows where the "country" property is present (1 in one-hot encoding)
  label_counts = country_df.groupby('label')['label'].count()
  print('     -', label_counts)

def dict_save_and_load(mydict, path, todo):

  # This function can save or load dictionaries in json format
  # Args:
  #       - mydict: it is the python dictionary that you want to save (only needed if todo=='save')
  #       - path: it is the wikidata client that get entity and properties
  #       - todo: it can be 'load' in case of loading or 'save' in the case of saving
  # Output:
  #       - mydict: in case of todo=='load', return the loaded dictionary

  if todo=="save":
    # Save it to a JSON file
    with open(path, "w", encoding="utf-8") as f:
      json.dump(mydict, f, ensure_ascii=False, indent=4)

  elif todo=="load":
    # Load the JSON file
    with open(path, "r", encoding="utf-8") as f:
      mydict = json.load(f)
    return mydict


def get_pageviews(title, lang = 'en', start='20240101', end='20250101', retries=5, delay=1):
    
  # This function return the total number of page views given a wikipedia title page
  # Args:
  #       - lang : the language (ex. 'en')
  #       - title : the title of the wikipedia page
  #       - start / end: time range considered in format YYYYMMDD
  #       - retries: number of retry attempts if the request fails
  #       - delay: delay between retries in seconds
  # Output:
  #       - (lang, views) : tuple containing the language and the total views

    encoded_title = quote(title, safe='') # encode the title as URL format (ex. spaces become %20)
    
    # Build an URL for the API request depending on language, title, start and end period
    url = (
        f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"{lang}.wikipedia/all-access/user/{encoded_title}/daily/{start}00/{end}00"
    )

    # This is the header dictionary that we append to authorize our request (research bot)
    headers = {
        "User-Agent": "MyResearchBot/1.0 (lissalattanzio.2154208@studenti.uniroma1.it)"
    }

    # loop in retries 
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=10) # try the request waiting 10 sec
            if r.status_code == 200: # if the status_code in response is 200 the request is succesfull
                data = r.json() # return the data in json format
                # The data is a dictionary in which the 'items' key contain a list of dictionaries with 'timestamp' and 'views'
                # it means that  we have the views for each day in the year, and we sum all iterating in each dictionary and taking
                # the views values
                views = sum(item['views'] for item in data['items'])  
                return (lang, views)  # we return as a tuple
            elif r.status_code == 404:
                return (lang, np.nan) # we return views as NaN if "Not Found"
            else:
                print(f"[{lang}, {encoded_title}] Trial {attempt+1}: status code {r.status_code}") # in the other case retry
        except Exception as e:
            print(f"[{lang}, {encoded_title}] Trial {attempt+1} failed: {e}")
        time.sleep(delay) # wait deaìlay seconds before another retry

    return (lang, np.nan) # return NaN if after tot retry the request does not success


def get_description(wikipedia_title, lang='en'):

  # This function having the wikipedia page title of the wikidata item (Ex. Q102) return the wikipedia summary 
  # Args:
  #       - wikipedia_title: the title of the wikipedia page
  #       - lang: the language of the wikipedia page
  # Output:
  #       - summary: a brief summary of the wikipedia page

  wikipedia.set_lang(lang) # Set the language of the wikipedia page

  try:
    summary = wikipedia.summary(wikipedia_title) # it is a little description of the wikipedia page
    return summary
  except wikipedia.DisambiguationError as e:
    # Error if the title has multipl pages
    tqdm.write(f"\nDisambiguationError for '{wikipedia_title}': options are {e.options[:3]}...")
    return None
  except wikipedia.PageError:
    # Error if the page does not exist
    tqdm.write(f"\nNo page found for '{wikipedia_title}'")
    return None
  except Exception as e:
    # Other errors
    tqdm.write(f"\nOther error: {e}")
    return None

def process_row(row, lang='en', start='20240101', end='20250101', retries=5, delay=1):
    
  # This function return the views and the wikipedia summary having a row of the dataframe
  # Args:
  #       - row : single row from the dataframe (a single wikidata item)
  #       - lang : the language of wikipedia page for taking the views
  # Output:
  #       - (views, summary) : return a tuple of summary of wikipedia page and its views

    try:
        client = Client() # create a Wikidata API client
        qid = row['item'].split('/')[-1] # take the Q-ID (Ex. 'Q207')
        item = client.get(qid, load=True) # get the item 

        # labels are name in different languages
        # labels = {'en': {'language': 'en', 'value': 'Dog'}, 'it': {'language': 'it', 'value': 'Cane'}}
        labels = item.data.get("labels", {}) 
        my_label = labels.get(lang) # take the name in out desired language

        # Check if there is no key 'value' inside the my_label dictionary (it means no wikipedia title) or my_lable is missing (None value, empty dictionary or False)
        if not my_label or 'value' not in my_label:
            return np.nan # return NaN in this case

        wikipedia_title = my_label['value'] # we take the value of my_label (the wikipedia title) 
        _, views = get_pageviews(wikipedia_title, lang=lang, start=start, end=end, retries=retries, delay=delay) # take the view of wikipedia page
        summary = get_description(wikipedia_title=wikipedia_title, lang=lang) # return (if available) the wikipedia summary
        return (views, summary)
    except Exception as e:
        print(f"Error processing QID {qid}: {e}")
        return np.nan


def add_wikipedia_data(df, save_path=None, lang='en', max_workers=10):
    
      # This function take a dataframe and return a dataframe with a column with the wikipedia page views and a column with the wikipedia summary
      # Args:
      #       - df : the input dataframe
      #       - save_path : path where to save the new dataframe
      #       - lang: language of the desired wikipedia page
      #       - max_workers: number of parallel execution of process_row function
      # Output:
      #       - df_wikipedia : the new dataframe with an added column with the views and an added column with the summary

    df_wikipedia = df.copy() # make a copy of the original df to make the original clean
    views_col = [np.nan] * len(df) # create a list of dimension df (the rows) empty (with NaN) to save the views
    summ_col = [np.nan] * len(df) # create a list of dimension df (the rows) empty (with NaN) to save the summary


    # Start a thread pool: it means that we run a collection of threads (multiple path of instructions inside the same program) to run tasks in parallel 
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 
        futures = {
            executor.submit(process_row, row, lang): i
            for i, row in df.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Adding Wikipedia data..."):
            idx = futures[future]
            try:
                views, summary = future.result()
                views_col[idx] = views
                summ_col[idx] = summary
            except Exception as e:
                print(f"[Index {idx}] Error: {e}")

    # Add the two new columns with wikipedia views and summary
    df_wikipedia[f'{lang}_wikipedia_views'] = views_col
    df_wikipedia[f'{lang}_wikipedia_summary'] = summ_col
    if save_path is None:
       print("No save path provided. The dataframe will not be saved.")
    else:
      # Save the new dataframe in csv format
      print("Saving the dataframe in csv format at the path: ", save_path)
      df_wikipedia.to_csv(save_path, index=False)
    return df_wikipedia