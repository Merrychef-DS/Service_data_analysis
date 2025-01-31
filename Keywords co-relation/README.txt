---

**Models Evaluated**  
The following models were trained and tested for system classification using the processed data and extracted keywords. Each model was chosen for its unique strengths and evaluated based on performance, computational requirements, and limitations.  

| Model Name | Description | Advantages | Limitations |
|------------|-------------|------------|-------------|
| **BERT (bert-base-uncased)** | Text classification for system-specific keywords from service solutions. | Effective at capturing complex patterns in data, delivering highly accurate results. | Requires significant computational resources; slower inference times. |
| **DistilBERT (distilbert-base-uncased-finetuned-sst-2-english)** | Adaptation for system-specific text classification. | Faster and requires less memory than BERT, retaining much of BERT's accuracy. (Recommended) | Slightly reduced accuracy compared to BERT; some specific capabilities might be lost due to distillation. |
| **RoBERTa (SamLowe/roberta-base-go_emotions)** | Emotion classification adapted to service solution descriptions. | Highly effective at nuanced emotion detection across various texts. (Recommended) | Requires large datasets for optimal performance. |
| **LLaMA 3.1 70B-instruct** | Top-tier performance in a range of natural language understanding and generation tasks. | Extremely high accuracy and nuanced understanding of language due to its size. | Resource-intensive; focused on text generation rather than direct classification. |
| **LLaMA 3.1 13B** | Efficient for diverse text generation tasks. | Good balance of size and performance; suitable for diverse tasks without extreme resource use. | May not perform as well on the most demanding tasks as its larger counterparts. |
| **LLaMA 3.1 7B** | Robust understanding of complex instructions for system classification. | More efficient than larger models while delivering strong performance. | Accuracy slightly lower than larger counterparts. |
| **Random Forest** | Baseline model for classification tasks using extracted features. | Straightforward implementation, offering insights into dataset limitations. | Lower accuracy (52%), highlighting the need for advanced deep learning methods. |

---

**Implementation and Results**  
1. **Preprocessing and Feature Engineering:**  
   - Text fields were cleaned, normalized, and converted into numerical representations using TF-IDF vectors.  
   - Extracted keywords were used as inputs to the models for classification.  
2. **Model Training and Evaluation:**  
   - Each model was trained on labeled data derived from service solution descriptions and part-system mappings.  
   - Evaluation metrics included accuracy, precision, recall, and F1-score to ensure comprehensive performance measurement.  
3. **Insights and Recommendations:**  
   - BERT achieved the highest accuracy but required significant computational resources, making it suitable for smaller, high-priority datasets.  
   - DistilBERT provided a good trade-off between speed and accuracy, making it ideal for real-time or resource-constrained environments.  
   - RoBERTa demonstrated excellent performance in nuanced classification tasks, particularly where descriptions involved implicit system mappings.  
   - Random Forest served as a baseline but struggled to achieve high accuracy due to the complexity of the data and lack of deep contextual understanding.  
4. **Future Enhancements:**  
   - Incorporating fine-tuning strategies for domain-specific text with larger datasets could further improve the performance of deep learning models.  
   - Combining outputs from multiple models (e.g., ensemble methods) may enhance overall accuracy and reliability.

