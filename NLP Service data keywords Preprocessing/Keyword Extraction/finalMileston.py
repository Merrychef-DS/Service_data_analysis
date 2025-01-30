import difflib
import numpy as np
import pandas as pd
import re
import spacy
import json
import string
import nltk
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
import os
EIKON_ERROR_LIST = "eikon_error_list.csv"
CONNEX_ERROR_LIST = "connex_error_list.csv"
acronyms_json = 'english_acronyms.json'
contraction_json = 'english_contractions.json'
current_path = os.path.dirname(os.path.realpath(__file__))
feature_counts = 50
output_dir = current_path
ERROR_PIVOT_TABLE = current_path
nltk.download('punkt')
threshold = 5
col_error_code = 'Error Code'
def process_asset_data(input_file_path, output_file_path):
    """
    Processes asset data from an Excel file and saves the cleaned data to a CSV file.

    Args:
        input_file_path (str): Path to the input Excel file.
        output_file_path (str): Path to the output CSV file.
    """
    try:
        # Load the Excel workbook and select the sheet
        df = pd.read_excel(input_file_path, sheet_name="Sheet1")

        # Fill NaN values in the 'Quantity' column with 0 and convert to int64
        df["Quantity"] = df["Quantity"].fillna(0).astype(np.int64)

        # Convert specific columns to string type
        df = df.astype({
            "Asset Name": str,
            "Work Description": str,
            "Completion Note": str,
            "Part Description": str,
            "Part Name": str,
            "WO#": str,
            "Billing Account Name": str
        })

        def extract_serial_number(value):
            """
            Extracts and validates serial number from a string value.

            Args:
                value (str): The value to extract the serial number from.

            Returns:
                str or None: Valid serial number if matches the pattern, otherwise None.
            """
            return value if re.fullmatch(r'\d{13}', value) else None

        # Split 'Work Description' into 'Serial_Number_from_WD', 'Category', and 'Description'.
        df[['Serial_Number_from_WD', 'Category', 'Description']] = df["Work Description"].str.split(":", n=2, expand=True)

        # Apply serial number extraction and validation.
        df["Serial_Number_from_WD"] = df["Serial_Number_from_WD"].apply(extract_serial_number)
        df["Asset Name"] = df["Asset Name"].apply(extract_serial_number)

        # Update 'Asset Name' where it's NaN with 'Serial_Number_from_WD'.
        df.loc[df["Asset Name"].isna() & df["Serial_Number_from_WD"].notna(), "Asset Name"] = df["Serial_Number_from_WD"]

        # Fill any remaining NaN values in 'Serial_Number_from_WD' with 'Asset Name'.
        df["Serial_Number_from_WD"] = df["Serial_Number_from_WD"].fillna(df["Asset Name"])

        # Convert 'Asset Name' and 'Serial_Number_from_WD' to string type.
        df["Asset Name"] = df["Asset Name"].astype(str)
        df["Serial_Number_from_WD"] = df["Serial_Number_from_WD"].astype(str)

        # Save the processed DataFrame to a CSV file
        df.to_csv(output_file_path, index=False, quoting=1)  # quoting=1 uses csv.QUOTE_ALL
        print("Data successfully processed and saved to:", output_file_path)

    except Exception as e:
        print(f"Error in process_asset_data: {e}")

def update_category_from_system(main_file_path, secondary_file_path, output_file_path):
    """
    Updates the 'Category' column in the main DataFrame based on 'System' values from a secondary DataFrame.

    Args:
        main_file_path (str): Path to the main CSV file.
        secondary_file_path (str): Path to the secondary CSV file containing system data.
        output_file_path (str): Path to save the updated CSV file.
    """
    try:
        # Load the datasets from CSV files
        main_df = pd.read_csv(main_file_path)
        secondary_df = pd.read_csv(secondary_file_path)

        # Filter rows in the main DataFrame where 'Category' is 'Other' and 'Part Name' is not NaN
        condition = (main_df['Category'] == 'Other') & (main_df['Part Name'].notna())
        filtered_main_df = main_df[condition]

        # Rename column in secondary DataFrame to match the main DataFrame
        secondary_df.rename(columns={"Part No": "Part_No"}, inplace=True)

        # Merge DataFrames to add 'System' values from secondary DataFrame
        merged_df = filtered_main_df.merge(
            secondary_df[['Part_No', 'System']],
            left_on='Part Name',
            right_on='Part_No',
            how='left'
        )

        # Create a mapping from 'Part Name' to 'System'
        system_map = dict(zip(merged_df['Part Name'], merged_df['System']))

        # Update 'Category' in the main DataFrame based on the 'System' values
        main_df['Category'] = main_df.apply(
            lambda row: system_map.get(row['Part Name'], row['Category'])
            if row['Category'] == 'Other' else row['Category'],
            axis=1
        )

        # Drop auxiliary columns
        main_df.drop(columns=['Part_No', 'System'], inplace=True, errors='ignore')

        # Save the updated DataFrame to a new CSV file
        main_df.to_csv(output_file_path, index=False)
        print("Data successfully updated and saved to:", output_file_path)

    except Exception as e:
        print(f"Error in update_category_from_system: {e}")

def process_Data(file_path, output_file_path):
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Fill NaN values in the 'Quantity' column with 0 and convert to int64
        df["Quantity"] = df["Quantity"].fillna(0).astype(np.int64)

        # Convert specific columns to string type
        df = df.astype({
            "Asset Name": str,
            "Work Description": str,
            "Completion Note": str,
            "Part Description": str,
            "WO#": str,
            "Billing Account Name": str
        })

        def validate_serial_number(value):
            return value if re.fullmatch(r'\d{13}', value) else None

        def read_error_file():
            try:
                eikon_error = pd.read_csv(EIKON_ERROR_LIST)
                connex_error = pd.read_csv(CONNEX_ERROR_LIST)
                print("Error file list loaded successfully.")
            except FileNotFoundError as fnf_error:
                print(f"{os.getcwd()} : The file was not found in the specified directory. {fnf_error}")
                eikon_error = pd.DataFrame()  # Returning empty DataFrame in case of error
                connex_error = pd.DataFrame()  # Returning empty DataFrame in case of error
            except Exception as e:
                print(f"{os.getcwd()} : An error occurred: {e}")
                eikon_error = pd.DataFrame()  # Returning empty DataFrame in case of error
                connex_error = pd.DataFrame()  # Returning empty DataFrame in case of error
            return eikon_error, connex_error

        def list_errors():
            try:
                eikon_error, connex_error = read_error_file()
                eikon_error_codes = eikon_error[col_error_code].tolist()
                connex_error_codes = connex_error[col_error_code].tolist()
                error_list = set()
                error_list.update(eikon_error_codes)
                error_list.update(connex_error_codes)
                error_keywords = [error_code.lower() for error_code in error_list]
                return error_keywords
            except Exception as e:
                print(f"{current_path} : Error :", e)
                return set()

        def list_error_code_without_E():
            try:
                list_error = list_errors()
                updated_list_error = set()
                # Iterate over each error code and add
                for code in list_error:
                    # Check if the error code starts with 'E' and is alphanumeric
                    if code.startswith('e') and code[1:].isalnum():
                        # Remove 'E' and add the remaining part to the set
                        updated_list_error.add(code[1:])
                return updated_list_error
            except Exception as e:
                print(f"{current_path} : Error :", e)
                return set()

        def clean_text_keywords(text, custom_keywords, list_error_code_without_E):
            words = word_tokenize(text)
            corrected_words = []
            custom_keywords_set = set(custom_keywords)
            for word in words:
                # Retain words that are in the custom_keywords list
                if word in custom_keywords_set:
                    if word in set(list_error_code_without_E):
                        corrected_words.append('E' + word)
                    else:
                        corrected_words.append(word)
                # For words not in the list, find the closest matches within the custom_keywords
                else:
                    closest_matches = difflib.get_close_matches(word, custom_keywords, n=3, cutoff=0.99)
                    # If a close match is found, add it (ensuring only desired words are included)
                    if closest_matches:
                        corrected_words.append(closest_matches[0])
            return ' '.join(corrected_words)

        dataset = df
        col_summary = "Description"
        col_modify_summary = 'Description_cleaned'
        col_desc = "Description"
        col_range = 'Category'
        col_part_no = 'Part Name'
        col_keyword = 'Keywords'

        # Initialize Spacy NLP model
        nlp = spacy.load('en_core_web_sm')

        # Fill missing values in text columns
        dataset[col_summary] = dataset[col_summary].fillna('')
        dataset[col_desc] = dataset[col_desc].fillna('')

        def part_list_keywords(data):
            try:
                # Initialize a list to store all tokens
                all_tokens = []
                # Iterate over each description in the provided data
                for desc in data:
                    if isinstance(desc, str):  # Ensure the description is a string
                        # Tokenize part description by splitting on whitespace and convert to lowercase
                        tokens = [token.lower() for token in desc.split() if len(token) >= 3]
                        # Extend the list of all tokens with the filtered tokens
                        all_tokens.extend(tokens)
                # Convert list of all tokens to a set to remove duplicates
                all_tokens_set = set(all_tokens)
                # print(all_tokens_set)
                print("Fetched All Part Descriptions as Tokens")
                return all_tokens_set

            except Exception as e:
                print(f"{current_path} :", e)

        def error_related_keywords(eikon_error, connex_error):
            try:
                # Define the set to collect keywords from all columns of both data frames to ensure uniqueness immediately
                all_keywords = set()
                without_E_list_error_code = set()
                # Process each DataFrame in a loop
                for df in [eikon_error, connex_error]:
                    # Iterate over each column in the DataFrame
                    for column in df.columns:
                        # print(column)
                        # Apply processing to each cell in the column to extract clean alphabetical words,
                        # then flatten the list with explode, remove NA, and directly update the set to avoid duplicates
                        all_keywords.update(
                            df[column].apply(
                                lambda x: [word.lower() for word in str(x).split() if word.isalnum()]
                            ).explode().dropna().unique()
                        )
                without_E_list_error_code = list_error_code_without_E()
                all_keywords.update(without_E_list_error_code)
                # Convert the set to a list for the final output
                unique_keywords = list(all_keywords)

                return unique_keywords
            except Exception as e:
                print(f"{current_path} : An error occurred:", e)

        def custom_keywords_list(dataset):
            try:
                text = dataset["Description"].fillna('').astype(str).tolist()
                part_desc_col = dataset["Description"].fillna('').astype(str)
                part_words = part_list_keywords(part_desc_col)
                combined_text = " ".join(text).lower()
                words = word_tokenize(combined_text)
                word_freq = Counter(words)
                high_freq_words = {clean_text(word) for word, freq in word_freq.items() if freq >= threshold}
                custom_keywords = set(clean_text(word) for word in part_words)
                custom_keywords.update(high_freq_words)
                eikon_error, connex_error = read_error_file()
                custom_keywords.update(error_related_keywords(eikon_error, connex_error))
                stop_words = set(stopwords.words('english'))
                custom_keywords = {word for word in custom_keywords if word not in stop_words}
                return custom_keywords
            except Exception as e:
                print(f"Error generating custom keywords list: {e}")
                return set()

        def read_acronyms(acronyms_json):
            try:
                with open(acronyms_json, 'r') as file:
                    acronyms_dict = json.load(file)
                acronyms_list = list(acronyms_dict.keys())
                return acronyms_dict, acronyms_list
            except Exception as e:
                print("Error in reading acronyms:", e)

        # Function to read contractions from JSON file
        def read_contractions(contraction_json):
            try:
                with open(contraction_json, 'r') as file:
                    contraction_dict = json.load(file)
                contraction_list = list(contraction_dict.keys())
                return contraction_list, contraction_dict
            except Exception as e:
                print("Error in reading contractions:", e)

        def clean_text(text):
            try:
                if text is not None:
                    text = re.sub('\n', '', text)  # converting text to one line
                    text = re.sub('\[.*?\]', '', text)  # removing square brackets
                    return text
            except Exception as e:
                print("Error in cleaning text:", e)

        def lemmatize(text, nlp):
            try:
                doc = nlp(text)
                lemmatized_text = [token.lemma_ for token in doc]
                return " ".join(lemmatized_text)
            except Exception as e:
                print("Error in lemmatizing text:", e)

        # Remove mentions and tags
        def remove_mentions_and_tags(text):
            try:
                text = re.sub(r'@\S*', "", text)
                return re.sub(r'#\S*', "", text)
            except Exception as e:
                print("Error in removing mentions and tags:", e)

        def remove_stopwords(text):
            try:
                stop_words = list(set(stopwords.words('english')))
                if text is not None:
                    addstops = ["among", "onto", "shall", "thrice", "thus", "twice", "unto", "us", "would", "ok",
                                "fine", "tested"]
                    stop_words.extend(addstops)
                    no_stopword_text = [w for w in text.split() if w.lower() not in stop_words]
                    return " ".join(no_stopword_text)
                else:
                    return ""
            except Exception as e:
                print("Error in removing stopwords:", e)

        def convert_acronyms(text, acronyms_dict, acronyms_list):
            try:
                words = []
                for word in re.findall(r"\w+|\s+|[^\w\s]", text, re.UNICODE):
                    if word in acronyms_list:
                        words += acronyms_dict[word].split()
                    else:
                        words.append(word)
                text_converted = "".join(words)
                return text_converted
            except Exception as e:
                print("Error in converting acronyms:", e)

        # Function to convert contractions in a text
        def convert_contractions(text, contraction_list, contraction_dict):
            try:
                words = []
                for word in re.findall(r"\w+|\s+|[^\w\s]", text, re.UNICODE):
                    if word in contraction_list:
                        words += contraction_dict[word].split()
                    else:
                        words.append(word)
                text_converted = "".join(words)
                return text_converted
            except Exception as e:
                print("Error in converting contractions:", e)

        def remove_http(text):
            try:
                http = "https?://\S+|www\.\S+"  # matching strings beginning with http (but not just "http")
                pattern = r"({})".format(http)  # creating pattern
                return re.sub(pattern, "", text)
            except Exception as e:
                print("Error in removing http links:", e)

        def remove_whitespace(text):
            try:
                # print("Removed white space")
                return text.strip()
            except Exception as e:
                print("Error in removing white space:", e)

        # Removing punctuations
        def remove_punctuation(text):
            try:
                punct_str = string.punctuation
                punct_str = punct_str.replace("'",
                                              "")  # discarding apostrophe from the string to keep the contractions intact
                return text.translate(str.maketrans("", "", punct_str))
            except Exception as e:
                print("Error in removing punctuation:", e)
                return text  # Returning the original text in case of an error

        # Removing HTML tags
        def remove_html(text):
            try:
                html = re.compile(r'<.*?>')
                return html.sub(r'', text)
            except Exception as e:
                print("Error in removing HTML tags:", e)

        def extract_ngrams_with_freq(texts, n, num_features):
            try:
                # Convert all entries to strings and filter out empty strings
                texts = [str(text).strip() for text in texts if str(text).strip()]
                if not texts:
                    return []
                vectorizer = TfidfVectorizer(max_features=num_features, min_df=1, stop_words='english',
                                             ngram_range=(n, n))
                try:
                    X = vectorizer.fit_transform(texts)
                    feature_names = vectorizer.get_feature_names_out()
                    # Calculate frequencies
                    all_text = ' '.join(texts)
                    word_freq = Counter(word_tokenize(all_text))
                    freq_list = [(word, word_freq[word]) for word in feature_names]
                    # Sort the list in descending order of frequencies
                    freq_list.sort(key=lambda x: x[1], reverse=True)
                    return freq_list
                except ValueError as ev:
                    print(f"Error: {ev}")
                    return []  # Return an empty list if no valid features are found
            except Exception as e:
                print(f"Error: {e}")
                return []
        # Load necessary lists and dictionaries
        custom_keywords = custom_keywords_list(dataset)
        print("custom_keywords", custom_keywords)
        acronyms_dict, acronyms_list = read_acronyms(acronyms_json)
        contraction_list, contraction_dict = read_contractions(contraction_json)
        list_error_code_without_E = list_error_code_without_E()
        # Apply cleaning steps
        dataset[col_modify_summary] = dataset[col_summary].apply(lambda x: remove_whitespace(x))
        dataset[col_modify_summary] = dataset[col_modify_summary].apply(lambda x: clean_text(x))
        dataset[col_modify_summary] = dataset[col_modify_summary].apply(lambda x: remove_mentions_and_tags(x))
        dataset[col_modify_summary] = dataset[col_modify_summary].apply(lambda x: remove_http(x))
        dataset[col_modify_summary] = dataset[col_modify_summary].apply(lambda x: remove_html(x))
        dataset[col_modify_summary] = dataset[col_modify_summary].apply(lambda x: remove_punctuation(x))
        dataset[col_modify_summary] = dataset[col_modify_summary].apply(lambda x: convert_acronyms(x, acronyms_dict, acronyms_list))
        dataset[col_modify_summary] = dataset[col_modify_summary].apply(lambda x: convert_contractions(x, contraction_list, contraction_dict))
        dataset[col_modify_summary] = dataset[col_modify_summary].apply(lambda x: remove_stopwords(x))
        dataset[col_modify_summary] = dataset[col_modify_summary].apply(lambda x: lemmatize(x, nlp))
        dataset[col_modify_summary] = dataset[col_modify_summary].apply(lambda x: clean_text_keywords(x, custom_keywords, list_error_code_without_E))

        # Extract keywords and categorize
        grouped = dataset.groupby(col_range)

        results = []
        def join_texts(series):
            return ' '.join(map(str, series))

        # Prepare vectorizer
        vectorizer = TfidfVectorizer()

        # Process each category, including 'Other'
        for name, group in grouped:
            combined_text = join_texts(group[col_modify_summary])
            col_desc_values = group[col_desc].unique().tolist()

            if combined_text.strip():
                for n in range(1, 4):
                    keywords = extract_ngrams_with_freq([combined_text], n, feature_counts)
                    results.append({
                        col_range: name,
                        col_desc: col_desc_values,
                        col_part_no: "N/A",
                        col_keyword: keywords,
                        'n': n
                    })
            else:
                keywords = ["default_keyword"]
                results.append({
                    col_range: name,
                    col_desc: col_desc_values,
                    col_part_no: "N/A",
                    col_keyword: keywords,
                    'n': 'default'
                })

        # Convert results to DataFrame
        keywords_df = pd.DataFrame(results)

        # Filter out 'Other' category
        category_keywords_df = keywords_df[keywords_df[col_range] != 'Other']

        # Check format of 'Keywords' column
        def process_keywords(keywords):
            if isinstance(keywords, str):
                return keywords
            elif isinstance(keywords, list):
                return ' '.join([kw[0] for kw in keywords])  # Join only the keyword part of the tuple
            return ''

        # Ensure to avoid SettingWithCopyWarning
        category_keywords_df.loc[:, col_keyword] = category_keywords_df[col_keyword].apply(process_keywords)

        category_keywords_list = category_keywords_df[col_keyword].tolist()
        category_names = category_keywords_df[col_range].tolist()

        # Vectorize category keywords
        category_vectors = vectorizer.fit_transform(category_keywords_list)

        # Function to find the closest category
        def find_closest_category(keywords):
            if not keywords:
                return 'Unknown'
            keywords_text = ' '.join([kw[0] for kw in keywords])  # Join only the keyword part of the tuple
            keywords_vector = vectorizer.transform([keywords_text])
            similarities = cosine_similarity(keywords_vector, category_vectors)
            closest_index = similarities.argmax()
            return category_names[closest_index]

        assigned_categories = []
        # Process 'Other' category directly
        other_condition = dataset[col_range] == 'Other'
        for idx, row in dataset[other_condition].iterrows():
            description = row[col_summary]
            cleaned_description = clean_text(description)  # Ensure this matches the cleaning applied earlier
            for n in range(1, 4):
                keywords = extract_ngrams_with_freq([cleaned_description], n, feature_counts)
                assigned_category = find_closest_category(keywords)
                assigned_categories.append({
                    "Index": idx,
                    "Original Category": row[col_range],
                    "keywords": keywords,
                    "Description": description,
                    "Assigned Category": assigned_category,
                    'n': n
                })
            dataset.loc[idx, col_range] = assigned_category

        # Drop intermediate column and export the updated dataset
        dataset.drop(columns=[col_modify_summary], inplace=True)
        dataset.to_csv(output_file_path, index=False, quoting=1)
        print("Third script executed and data saved to:", output_file_path)
        assigned_categories_df = pd.DataFrame(assigned_categories)
        pivoted_df = assigned_categories_df.pivot_table(
            index=['Index', 'Description'],
            columns='n',
            values='keywords',
            aggfunc='first'
        ).reset_index()
        assigned_categories_for_n1 = assigned_categories_df[assigned_categories_df['n'] == 1][
            ['Index', 'Description', 'Assigned Category']]
        assigned_categories_for_n1 = assigned_categories_for_n1.rename(
            columns={'Assigned Category': 'Assigned Category for n=1'})
        pivoted_df = pivoted_df.merge(assigned_categories_for_n1, on=['Index', 'Description'], how='left')
        # Rename columns for clarity
        pivoted_df.columns = ['Index', 'Description', 'Keywords for 1', 'keywords for 2',
                              'keywords for 3','Assigned Category for n=1']
        assigned_categories_output_file_path = r"R:\K_Archive_\Nur_Dir\nur final\Keywords for Different N grams.csv"
        #assigned_categories_df.to_csv(assigned_categories_output_file_path, index=False)
        pivoted_df.to_csv(assigned_categories_output_file_path, index=False)
    except Exception as e:
        print(f"Error in third_script: {e}")


def main():
    try:
        main_file_path = "Data.xlsx"
        intermediate_file_path1 = r"R:\K_Archive_\Nur_Dir\nur final\First_Updated_Data.csv"
        secondary_file_path = r"R:\K_Archive_\Nur_Dir\Nur\Merrychef_eikon_part_list DC classified.csv"
        intermediate_file_path2 = r"R:\K_Archive_\Nur_Dir\nur final\Second_Updated_Data.csv"
        final_output_path = r"R:\K_Archive_\Nur_Dir\nur final\Final1_Updated_Data.csv"

        # Call first script
        process_asset_data(main_file_path, intermediate_file_path1)

        # Call second script
        update_category_from_system(intermediate_file_path1, secondary_file_path, intermediate_file_path2)

        # Clean up intermediate files
        os.remove(intermediate_file_path1)

        # Call third script
        process_Data(intermediate_file_path2, final_output_path)

        # Clean up intermediate files
        os.remove(intermediate_file_path2)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

