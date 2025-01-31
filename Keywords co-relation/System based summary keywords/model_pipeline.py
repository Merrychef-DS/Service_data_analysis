# #
# # import pandas as pd
# # from nltk.tokenize import word_tokenize
# # from sklearn.feature_extraction.text import TfidfVectorizer
# #
# # # Load the processed data
# # df = pd.read_excel('C:/Users/mn1006/Documents/Service Incident Summary.xlsx')
# #
# # # Function to extract significant keywords based on TF-IDF
# # def extract_keywords_tfidf(texts, top_n=10):
# #     vectorizer = TfidfVectorizer(stop_words='english')
# #     tfidf_matrix = vectorizer.fit_transform(texts)
# #     feature_names = vectorizer.get_feature_names_out()
# #     dense = tfidf_matrix.todense()
# #     denselist = dense.tolist()
# #     df_tfidf = pd.DataFrame(denselist, columns=feature_names)
# #     top_keywords = {word: df_tfidf[word].sum() for word in feature_names}
# #     sorted_keywords = sorted(top_keywords.items(), key=lambda x: x[1], reverse=True)
# #     return sorted_keywords[:top_n]
# #
# # # Extract significant keywords from service summaries
# # df['significant_keywords'] = df['Service Incident Summary'].apply(lambda x: extract_keywords_tfidf([x]))
# #
# # # Combine Part Number and Description
# # df['Part Info'] = df['Part Number'].astype(str) + " - " + df['Part Description']
# #
# # # Correlation function to associate keywords with parts, types, and resolutions
# # def correlate_keywords(row):
# #     keywords = [kw[0] for kw in row['significant_keywords']]
# #     part_info = row['Part Info']
# #     service_line_type = row['Service Line Type']
# #     claim_status = row['Claim Status']
# #     return pd.Series([keywords, part_info, service_line_type, claim_status])
# #
# # # Apply correlation
# # df[['Keywords', 'Part Info', 'Service Line Type', 'Claim Status']] = df.apply(correlate_keywords, axis=1)
# #
# # # Save the modified DataFrame
# # output_file = 'Keyword_Part_Correlation_Analysis.csv'
# # df.to_csv(output_file, index=False)
# #
# # print(f"Keyword and part correlation data saved to {output_file}")
# #

#---------------------------------------------------------------------------------------------------------------------------------------------
#end of 1st commenting part
import difflib
import numpy as np
import pandas as pd
import re
import spacy
import json
import string
import nltk
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import os
from langdetect import detect, LangDetectException

# Set necessary parameters
acronyms_json = 'english_acronyms.json'
contraction_json = 'english_contractions.json'
threshold = 5
feature_counts = 50
EIKON_ERROR_LIST = "eikon_error_list.csv"
CONNEX_ERROR_LIST = "connex_error_list.csv"
col_error_code = 'Error Code'

# Ensure necessary NLTK downloads
nltk.download('punkt', quiet=True)

# Load external resources
def read_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON in {file_path}.")
        return {}

# Load acronyms and contractions
acronyms_dict = read_json(acronyms_json)
acronyms_list = list(acronyms_dict.keys())
contraction_dict = read_json(contraction_json)
contraction_list = list(contraction_dict.keys())
nlp = spacy.load('en_core_web_sm')

# # Error List Loading
def read_error_file():
    try:
        eikon_error = pd.read_csv(EIKON_ERROR_LIST)
        connex_error = pd.read_csv(CONNEX_ERROR_LIST)
        print("Error file list loaded successfully.")
        return eikon_error, connex_error
    except FileNotFoundError as fnf_error:
        print(f"{os.getcwd()} : The file was not found in the specified directory. {fnf_error}")
        return pd.DataFrame(), pd.DataFrame()  # Returning empty DataFrame in case of error
    except Exception as e:
        print(f"{os.getcwd()} : An error occurred: {e}")
        return pd.DataFrame(), pd.DataFrame()  # Returning empty DataFrame in case of error

def list_errors():
    try:
        eikon_error, connex_error = read_error_file()
        eikon_error_codes = eikon_error[col_error_code].tolist() if not eikon_error.empty else []
        connex_error_codes = connex_error[col_error_code].tolist() if not connex_error.empty else []
        error_list = set(eikon_error_codes + connex_error_codes)
        error_keywords = [error_code.lower() for error_code in error_list]
        return error_keywords
    except Exception as e:
        print(f"Error in list_errors: {e}")
        return set()

def list_error_code_without_E():
    try:
        list_error = list_errors()
        updated_list_error = set()
        for code in list_error:
            if code.startswith('e') and code[1:].isalnum():
                updated_list_error.add(code[1:])  # Remove 'E' and add the remaining part to the set
        return updated_list_error
    except Exception as e:
        print(f"Error in list_error_code_without_E: {e}")
        return set()

# Cleaning Functions
def clean_text_keywords(text, custom_keywords, list_error_code_without_E):
    words = word_tokenize(text)
    corrected_words = []
    custom_keywords_set = set(custom_keywords)
    for word in words:
        if word in custom_keywords_set:
            if word in list_error_code_without_E:
                corrected_words.append('E' + word)
            else:
                corrected_words.append(word)
        else:
            closest_matches = difflib.get_close_matches(word, custom_keywords, n=3, cutoff=0.99)
            if closest_matches:
                corrected_words.append(closest_matches[0])
    return ' '.join(corrected_words)

def remove_whitespace(text):
    return text.strip()

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_dates(text):
    date_patterns = [
        r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b',  # Matches dates like 10/10/23, 10-10-23, 10.10.2023
        r'\b\d{6,8}\b',                           # Matches continuous digits that might represent dates like 10102023
        r'\b\d{5}\b'                              # Matches 5-digit numbers like 10423
    ]
    for pattern in date_patterns:
        text = re.sub(pattern, '', text)
    return text

def remove_punctuation(text):
    punct_str = string.punctuation.replace("'", "")
    return text.translate(str.maketrans("", "", punct_str))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return " ".join([w for w in text.split() if w.lower() not in stop_words])

def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

def convert_acronyms(text):
    words = [acronyms_dict.get(word, word) for word in text.split()]
    return " ".join(words)

def convert_contractions(text):
    words = [contraction_dict.get(word, word) for word in text.split()]
    return " ".join(words)

# Custom Keywords List Generation
def part_list_keywords(data):
    all_tokens = []
    for desc in data:
        if isinstance(desc, str):
            tokens = [token.lower() for token in desc.split() if len(token) >= 3]
            all_tokens.extend(tokens)
    return set(all_tokens)

def custom_keywords_list(dataset):
    text = dataset.fillna('').astype(str).tolist()
    part_words = part_list_keywords(text)
    combined_text = " ".join(text).lower()
    words = word_tokenize(combined_text)
    word_freq = Counter(words)
    high_freq_words = {word for word, freq in word_freq.items() if freq >= threshold}
    custom_keywords = set(part_words).union(high_freq_words)
    custom_keywords = {word for word in custom_keywords if word not in stopwords.words('english')}
    return custom_keywords

def extract_nouns_and_verbs(text):
    doc = nlp(text)
    actions = [token.text for token in doc if token.pos_ == 'VERB']
    observations = [token.text for token in doc if token.pos_ == 'NOUN']
    return {'actions': actions, 'observations': observations}

# N-gram Extraction
def extract_ngrams_individual(data, n):
    vectorizer = TfidfVectorizer(ngram_range=(n, n), stop_words='english')
    results = []
    for text in data:
        if text.strip():  # ensure the text is not empty
            try:
                X = vectorizer.fit_transform([text])
                features = vectorizer.get_feature_names_out()
                results.append(", ".join(features))
            except ValueError:  # handling empty vocabulary case
                results.append("")
        else:
            results.append("")
    return results

# Updated Part Info and Bi-gram Extraction (n=2)
def extract_keywords_tfidf_bigrams(texts, top_n=10):
    # Check if texts is empty or contains only stop words
    if not texts or all(len(text.strip()) == 0 for text in texts):
        return []  # Return an empty list if there are no valid words

    vectorizer = TfidfVectorizer(ngram_range=(2, 2), stop_words='english')  # Now extracting bi-grams
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        dense = tfidf_matrix.todense()
        denselist = dense.tolist()
        df_tfidf = pd.DataFrame(denselist, columns=feature_names)
        top_keywords = {word: df_tfidf[word].sum() for word in feature_names}
        sorted_keywords = sorted(top_keywords.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_n]
    except ValueError:
        # Handle cases where vocabulary is empty after cleaning
        return []
    # Correlation function for combining bi-gram keywords, parts, and claim status
# def correlate_keywords(row):
#     keywords = [kw[0] for kw in row['significant_keywords']]
#     part_info = row['Part Info']
#     service_line_type = row['Service Line Type']
#     return pd.Series([keywords, part_info, service_line_type])
#
# Main Data Processing
def process_data(df):
    col_summary = "Service Incident Summary"
    col_modify_summary = 'Cleaned_Summary'

    # Apply text cleaning process
    df[col_modify_summary] = df[col_summary].apply(remove_whitespace)
    df[col_modify_summary] = df[col_modify_summary].apply(remove_dates)
    df[col_modify_summary] = df[col_modify_summary].apply(remove_html)
    df[col_modify_summary] = df[col_modify_summary].apply(remove_punctuation)
    df[col_modify_summary] = df[col_modify_summary].apply(convert_acronyms)
    df[col_modify_summary] = df[col_modify_summary].apply(convert_contractions)
    df[col_modify_summary] = df[col_modify_summary].apply(remove_stopwords)
    df[col_modify_summary] = df[col_modify_summary].apply(lemmatize)

    # Load error codes and custom keywords
    list_error_code_without = list_error_code_without_E()
    custom_keywords = custom_keywords_list(df[col_modify_summary])

    # Apply keyword correction using custom keywords and error codes
    df[col_modify_summary] = df[col_modify_summary].apply(
        lambda x: clean_text_keywords(x, custom_keywords, list_error_code_without)
    )

    # Part-of-Speech (POS) Tagging - Extract actions (verbs) and observations (nouns)
    df['Actions and Observations'] = df[col_modify_summary].apply(extract_nouns_and_verbs)
#
    # Extract n-grams for n=1, n=2, and n=3
    df['Keywords for n=1'] = extract_ngrams_individual(df[col_modify_summary], 1)
    df['Keywords for n=2'] = extract_ngrams_individual(df[col_modify_summary], 2)
    df['Keywords for n=3'] = extract_ngrams_individual(df[col_modify_summary], 3)

    # New Addition: Combine Part Number and Description
    # df['Part Info'] = df['Part Number'].astype(str) + " - " + df['Part Description']
    # df['Part Info'] = df['Part Info'].replace(to_replace=r'^\s*$', value='No part needed / no update', regex=True)
    # df = df[~df['Part Info'].str.startswith(('NO CAUSAL PART', 'No part'), na=False)]

    # New Addition: Extract significant bi-gram keywords based on TF-IDF with empty text check
    df['significant_keywords'] = df[col_summary].apply(lambda x: extract_keywords_tfidf_bigrams([x]) if x and len(x.strip()) > 0 else [])

    # New Addition: Correlation of keywords with Part Info, Service Line Type, and Claim Status
    #df[['Keywords', 'Part Info', 'Service Line Type']] = df.apply(correlate_keywords, axis=1)

    return df
def is_english(text):
    try:
        # Detect language of the text
        return detect(text) == 'en'
    except LangDetectException:
        # If detection fails, consider it non-English
        return False
# Load Data and Process
file_path = "R:/K_Archive_/Claim data analysis/DATASET/CPS DATA/CPS_Merged_28_05_2024.csv"
#file_path = 'C:/Users/mn1006/Documents/Service Incident Summary.xlsx'
try:
    df = pd.read_csv(file_path, low_memory=False)
    # Filter to include only 'MERRYCHEF' brand incidents
    df = df[df['Brand'].str.upper() == 'MERRYCHEF']
    df = df[df['Service Line Type'] == 'Parts']
    # Step 2: Eliminate rows where a specific value appears in a column
    # Example: Remove rows where the value 'to_remove' appears in the 'Column_Name' column
# Replace 'Column_Name' and 'to_remove' with your column and value
    try:
        # Filter rows where 'Model' column starts with 'E2S' and handle NaN values
        df = df[df['Model'].notna() & df['Model'].str.startswith('X12', na=False)]
        df = df[~df['Part Number'].str.startswith(('NO CAUSAL PART', 'No part'), na=False)]
        df = df[~df['Part Description'].str.startswith(('NO CAUSAL PART', 'No part'), na=False)]
    except KeyError:
        print("'Model' column not found in the dataset.")
    try:
        df = df[df['Serial Number'].apply(lambda x: str(x).isdigit() and len(str(x)) >= 13)]
    except KeyError:
        print("'Serial Number' column not found in the dataset.")
    try:
        df = df[df['Service Incident Summary'].apply(is_english)]
    except KeyError:
        print("'Service Incident Summary' column not found in the dataset.")
    # Ensure case-insensitive matching
except FileNotFoundError:
    print(f"Error: {file_path} not found.")
else:
    # Process the data
    processed_df = process_data(df)

    # Save Results
    output_file = 'Processed_Service_Incident_Summary_Keywords_for_ConneX_12.csv'
    processed_df[
        ['Service Incident Summary','Keywords for n=1', 'Keywords for n=2',
         'Keywords for n=3', 'Part Number', 'Part Description', 'Service Line Type']
    ].to_csv(output_file, index=False)

    print(f"Processed data saved to {output_file}")


