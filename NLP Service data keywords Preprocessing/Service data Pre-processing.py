# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# import re
#
# # Function to clean the text
# def clean_text(text):
#     text = re.sub('\n', ' ', text)  # replace newlines with spaces
#     text = re.sub('\[.*?\]', '', text)  # remove text within brackets
#     text = re.sub('<.*?>+', '', text)  # remove HTML tags
#     text = re.sub('https?://\S+|www\.\S+', '', text)  # remove URLs
#     text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)  # remove punctuations
#     text = re.sub('\n', '', text)  # remove line breaks
#     text = re.sub('\w*\d\w*', '', text)  # remove words containing numbers
#     return text.lower()
#
# # Function to extract n-grams individually for each row
# def extract_ngrams_individual(data, n):
#     vectorizer = TfidfVectorizer(ngram_range=(n, n), stop_words='english')
#     results = []
#     for text in data:
#         if text.strip():  # ensure the text is not empty
#             try:
#                 X = vectorizer.fit_transform([text])
#                 features = vectorizer.get_feature_names_out()
#                 results.append(", ".join(features))
#             except ValueError:  # handling empty vocabulary case
#                 results.append("")
#         else:
#             results.append("")
#     return results
#
# # Load the Excel file
# service_incident_file_path = '/path/to/your/file.xlsx'
# service_incident_data = pd.read_excel(service_incident_file_path)
#
# # Clean the "Service Incident Summary" column
# service_incident_data['Cleaned Service Incident Summary'] = service_incident_data['Service Incident Summary'].apply(clean_text)
#
# # Apply n-gram extraction individually for each row
# service_incident_data['Keywords for n=1'] = extract_ngrams_individual(service_incident_data['Cleaned Service Incident Summary'], 1)
# service_incident_data['Keywords for n=2'] = extract_ngrams_individual(service_incident_data['Cleaned Service Incident Summary'], 2)
# service_incident_data['Keywords for n=3'] = extract_ngrams_individual(service_incident_data['Cleaned Service Incident Summary'], 3)
#
# # Create a new DataFrame to store the required columns
# result_df = service_incident_data[['Service Incident Summary', 'Keywords for n=1', 'Keywords for n=2', 'Keywords for n=3']]
#
# # Save the results to a new CSV file
# output_file_path = '/path/to/save/Processed_Service_Incident_Keywords_Individual.csv'
# result_df.to_csv(output_file_path, index=False)
#
# print("Processing complete. File saved to:", output_file_path)

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

# Error List Loading
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
    df[col_modify_summary] = df[col_modify_summary].apply(lambda x: clean_text_keywords(x, custom_keywords, list_error_code_without))

    # Extract n-grams
    df['Keywords for n=1'] = extract_ngrams_individual(df[col_modify_summary], 1)
    df['Keywords for n=2'] = extract_ngrams_individual(df[col_modify_summary], 2)
    df['Keywords for n=3'] = extract_ngrams_individual(df[col_modify_summary], 3)

    return df

# Load Data and Process
file_path = 'C:/Users/mn1006/Documents/Service Incident Summary.xlsx'
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Error: {file_path} not found.")
else:
    processed_df = process_data(df)

    # Save Results
    output_file = 'Processed_Service_Incident_Summary_Keywords.csv'
    processed_df[['Service Incident Summary', 'Keywords for n=1', 'Keywords for n=2', 'Keywords for n=3']].to_csv(
        output_file, index=False)

    print(f"Processed data saved to {output_file}")

# output_file = 'Processed_Service_Incident_Summary_Keywords2.csv'
#     processed_df[
#         ['Service Incident Summary', 'Entities', 'Actions and Observations', 'Keywords for n=1', 'Keywords for n=2',
#          'Keywords for n=3']].to_csv(output_file, index=False)
#
#     print(f"Processed data saved to {output_file}")

