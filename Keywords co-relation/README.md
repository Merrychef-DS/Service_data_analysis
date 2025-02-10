#### Author: Kirti Gupta

This is the initial Data analysis mechanism.

**Project Overview**  
This project focuses on processing and analyzing asset data by:  
- Cleaning and preparing data for analysis.  
- Assigning categories to assets using keywords and TF-IDF vectors.  
- Extracting and cleaning keywords to identify patterns in error and part descriptions.  

**Requirements**  
**Libraries Required:**  
- Python packages: pandas, numpy, spacy, nltk, sklearn, json, string, re.  

**Files Used:**  
- **Turbo_tech_.xlsx**: Main asset data file.  
- **Merrychef_eikon_part_list DC classified.csv**: System mapping file for part categories.  
- **eikon_error_list.csv & connex_error_list.csv**: Error code files.  
- **english_acronyms.json & english_contractions.json**: Text processing rules.  

---

**Functions and Processing Steps**  

### 1. Processing Asset Data  
- **process_asset_data(input_file_path)**:  
  - Loads Excel data.  
  - Handles missing values.  
  - Extracts serial numbers.  
  - Splits descriptions into meaningful parts.  
  - Updates asset names where values are missing, using available serial numbers.  

### 2. Updating Categories  
- **update_category_from_system(main_df, secondary_df)**:  
  - Updates the "Category" column where it is labelled as "Other" but a part name exists in the row.  

### 3. Error Code Handling  
- **read_error_file()**: Loads error codes from CSV files.  
- **list_errors() & list_error_code_without_E()**:  
  - Generates error-related keywords.  
  - Combines error codes from multiple sources, normalizing formats for consistency.  
  - Removes the "E" prefix from error codes to standardize data.  

### 4. Keyword Extraction  
- **custom_keywords_list**:  
  - Combines keywords from error codes, high-frequency words, and part descriptions.  
  - Maintains a custom list of keywords for cleaning text fields.  

- **part_list_keywords**: Extracts unique tokens from part descriptions.  
- **error_related_keywords**: Generates keywords from error data.  

### 5. Data Cleaning and Text Preprocessing  
- Utility functions for cleaning text:  
  - **clean_text**: Removes noise (HTML, punctuation, etc.).  
  - **lemmatize**: Converts words to their base forms for consistency.  
  - **remove_stopwords**: Eliminates common words that do not add value.  
  - Expands acronyms and contractions.  

### 6. N-Gram Extraction & Categorization  
- **extract_ngrams & extract_ngrams_with_freq**:  
  - Leverages TF-IDF to extract meaningful n-grams (unigrams, bigrams, trigrams).  
  - Aggregates Description_cleaned text within each group for pattern identification.  

- **Category Assignment**:  
  - Uses **cosine similarity** to find the most relevant category for each keyword.  
  - Updates "Other" records by assigning them to the closest matching category.  

### 7. Data Processing Pipeline  
- **Process_data**:  
  - Converts columns like "Quantity" to integers, replacing missing values with 0.  
  - Identifies key columns like Description and Description_cleaned for text cleaning and analysis.  
  - **Cleaning_Stage()**:  
    - Executes the full data cleaning pipeline, including:  
      - Whitespace removal.  
      - Text normalization.  
      - Keyword matching.  

---

**Enhancing System-Specific Classification Using Advanced Models**  
In addition to keyword extraction and TF-IDF-based classification, **advanced machine learning models** were implemented to enhance classification accuracy. These models help identify issues in ovens by analyzing service solution descriptions, leading to improved problem-solving efficiency.

---

**Models Evaluated**  
The following models were trained and tested for system classification using the processed data and extracted keywords. Each model was chosen for its unique strengths and evaluated based on performance, computational requirements, and limitations.  

### 1. BERT (bert-base-uncased)  
- **Description:** Text classification for system-specific keywords from service solutions.  
- **Advantages:** Effective at capturing complex patterns in data, delivering highly accurate results.  
- **Limitations:** Requires significant computational resources; slower inference times.  

### 2. DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)  
- **Description:** Adaptation for system-specific text classification.  
- **Advantages:** Faster and requires less memory than BERT, retaining much of BERT's accuracy. (Recommended)  
- **Limitations:** Slightly reduced accuracy compared to BERT; some specific capabilities might be lost due to distillation.  

### 3. RoBERTa (SamLowe/roberta-base-go_emotions)  
- **Description:** Emotion classification adapted to service solution descriptions.  
- **Advantages:** Highly effective at nuanced emotion detection across various texts. (Recommended)  
- **Limitations:** Requires large datasets for optimal performance.  

### 4. LLaMA 3.1 70B-instruct  
- **Description:** Top-tier performance in a range of natural language understanding and generation tasks.  
- **Advantages:** Extremely high accuracy and nuanced understanding of language due to its size.  
- **Limitations:** Resource-intensive; focused on text generation rather than direct classification.  

### 5. LLaMA 3.1 13B  
- **Description:** Efficient for diverse text generation tasks.  
- **Advantages:** Good balance of size and performance; suitable for diverse tasks without extreme resource use.  
- **Limitations:** May not perform as well on the most demanding tasks as its larger counterparts.  

### 6. LLaMA 3.1 7B  
- **Description:** Robust understanding of complex instructions for system classification.  
- **Advantages:** More efficient than larger models while delivering strong performance.  
- **Limitations:** Accuracy slightly lower than larger counterparts.  

### 7. Random Forest  
- **Description:** Baseline model for classification tasks using extracted features.  
- **Advantages:** Straightforward implementation, offering insights into dataset limitations.  
- **Limitations:** Lower accuracy (52%), highlighting the need for advanced deep learning methods.
