# Fine-Tuning T5 for UK-to-US Dialect Conversion

## 1. Introduction

The goal of this project is to develop a machine learning model capable of converting UK English sentences into their US English equivalents. While rule-based systems can handle simple transformations like spelling changes, they often lack the ability to comprehend context. To provide a more robust solution, we use the **EnglishVoice/t5-base-uk-to-us-english** model, a pre-trained T5 (Text-to-Text Transfer Transformer) variant fine-tuned specifically for UK-to-US English conversion tasks. This model can handle both lexical and contextual transformations, making it more effective in dialect conversion.

## 2. Methodology

### 2.1 Dataset Preparation

The dataset consists of pairs of UK and US English sentences, demonstrating the various linguistic differences between the two dialects (spelling, word choice, and capitalization). The dataset was split into 80% for training and 20% for testing.

#### Example Data:

| Input Text (UK English)                           | Target Text (US English)                          |
|---------------------------------------------------|---------------------------------------------------|
| I CoLoUr 🎨 the centre of my favourite book.       | I color the center of my favorite book.           |
| He is travelling ✈️ to the THEATRE.               | He is traveling to the theater.                   |
| I have a flat near the lift.                      | I have an apartment near the elevator.            |

### 2.2 Tokenization

To prepare the data for input into the model, we used the `T5Tokenizer`. The text was preprocessed and tokenized, with a special prefix `translate UK to US:` added to the input sentences to indicate the task.

```python
inputs = ["translate UK to US: " + text for text in examples["input_text"]]
```

### 2.3 Model Selection
We selected EnglishVoice/t5-base-uk-to-us-english, a pre-trained T5 model specifically fine-tuned for UK-to-US conversion, from Hugging Face's model hub. This model leverages a sequence-to-sequence architecture that is ideal for translation tasks.

#### Why T5?
- Pre-trained for UK-to-US conversion: This model is specifically fine-tuned to handle UK-to-US English transformations.
- Text-to-Text Framework: T5 is designed to handle text generation tasks by treating all NLP problems as text-to-text transformations, making it suitable for tasks like translation and dialect conversion.
- Pre-trained on Large Corpus: The model has been trained on a large dataset, allowing it to capture linguistic nuances and contextual understanding.

### 2.4 Training Strategy
We used the following training setup:
- Loss Function: Cross-entropy loss, which is commonly used in sequence-to-sequence models for text generation.
- Batch Size: 4 or 5 (chosen based on GPU memory constraints).
- Epochs: 5 (chosen experimentally to ensure convergence without overfitting).
- Evaluation Strategy: We evaluated the model after each epoch to monitor its progress.

#### Training Arguments:
```python
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save model checkpoints
    evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
    per_device_train_batch_size=4,  # Batch size for training
    per_device_eval_batch_size=4,  # Batch size for evaluation
    num_train_epochs=5,  # Number of epochs
    save_steps=100,  # Save checkpoint every 100 steps
    save_total_limit=2,  # Keep only the last 2 checkpoints
    logging_dir="./logs"  # Directory to save logs
)
```

## 3. Architectural Choices
### 3.1 Why EnglishVoice/t5-base-uk-to-us-english?
The EnglishVoice/t5-base-uk-to-us-english model was selected because:

- It is pre-trained specifically for UK-to-US conversion.
- Its text-to-text framework is ideal for tasks like dialect conversion.
- The model’s pre-training on a large corpus allows it to understand linguistic nuances such as spelling, punctuation, and word usage differences between UK and US English.

### 3.2 Alternative Approaches Considered

Approach | Pros| Cons 
--- | --- | --- 
Rule-Based Approach | Simple and fast	Limited to predefined transformations | lacks context awareness
Seq2Seq Models (RNN-based) | Captures context well | Computationally expensive, long training time
Transformer-based Models (BART/T5) | Scalable, handles context effectively | Requires task-specific fine-tuning 

## 4. Results
The fine-tuned model was tested on unseen data to evaluate its performance.

Example Inference:
Input:
```"I CoLoUr 🎨 the centre of my favourite book."```

Predicted Output:
```"I color the center of my favorite book."```

The model correctly converted UK spelling, punctuation, and even handled emojis within the text.

## 5. Conclusion and Future Work
#### Conclusion:
This project demonstrates that fine-tuning EnglishVoice/t5-base-uk-to-us-english can effectively perform UK-to-US English dialect conversion, handling both lexical and contextual transformations. The model performs well with diverse vocabulary, including text with emojis.

#### Future Work:
- Training on a larger dataset: To improve model generalization and performance on diverse text inputs.
- Further Fine-tuning: Extended fine-tuning could improve accuracy.
- API Deployment: The model could be deployed as an API for real-time UK-to-US text conversion.

#### Suggestions for Improvement
- Model Evaluation Metrics: To formally evaluate the model, can use metrics like BLEU, ROUGE, or Accuracy to quantitatively assess performance. For example, comparing the output to ground-truth translations and calculating the BLEU score will give a more precise measure of performance.
- Emoji Handling: Emojis were removed during preprocessing, but in some cases, they might be relevant for context. May want to treat them differently (e.g., keeping them as is or adding special tokens for emojis).
- Logging and Checkpoints: Consider adding more logging to the training process (such as through wandb or TensorBoard). Could also save intermediate checkpoints during training so you can pick up training without starting over.
- Data Augmentation: dataset is relatively small (though sufficient for demonstration). If possible, use data augmentation techniques to artificially increase the size of your dataset, especially for cases where dialect conversion might require more diverse training data.
- Batch Size: The batch size of 5 is quite small. If the hardware supports it, you could try increasing the batch size for faster training and better gradient estimates.

## 6. Instructions
### 6.1 Dependencies
Python version required is 3.11.11

Ensure the following dependencies are installed before running the notebook:

requirements.txt
```txt
pandas==2.2.2
torch==2.5.1+cu124
scikit-learn==1.6.1
transformers==4.47.1
datasets==3.2.0
```

### 6.2 How to Run the Notebook
Clone or open the Colab notebook.
Install the dependencies above using the command:
```bash
!pip install pandas==2.2.2 torch==2.5.1+cu124 scikit-learn==1.6.1 transformers==4.47.1 datasets==3.2.0
```
Run each code block sequentially from top to bottom to prepare the data, train the model, and perform inference.
The final model will be saved as fine_tuned_t5_uktous/, which can be used for inference.

### 6.3 Load the finetuned model which I trained

Download the entire folder of the model with the same folder name name: [Link to model](https://drive.google.com/drive/folders/1obL4HpsToFR1mENsMjIavsWYUZmsQQ0I?usp=sharing). If on colab, store the folder in the content section of the colab runtime directory.

Ensure that torch==2.5.1+cu124 and transformers==4.47.1 are installed. Python version 3.11.11

Run the code below:
```python
# Load the model and tokenizer for future use
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re


# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"

model = T5ForConditionalGeneration.from_pretrained("my_uk_to_us_t5").to(device)
tokenizer = T5Tokenizer.from_pretrained("my_uk_to_us_t5")

# Preprocessing function for text
def preprocess_text(text):
    text = text.strip().lower()  # Remove extra spaces and convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9\s\.,!?]', '', text)  # Remove unwanted characters (emoji, special symbols, etc.)
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'\s+([.,!?;])', r'\1', text)  # Fix spaces before punctuation
    text = text.capitalize()  # Capitalize the first letter
    return text

# Function for Inference
def translate_uk_to_us(text):
    input_text = "UK to US: " + text
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Evaluate the model on the test dataset
# test_sentence = "I CoLoUr 🎨 the centre of my favourite book."
test_sentence = "He is travelling to the theatre."
test_sentence = "UK to US: " + preprocess_text(test_sentence)
print(test_sentence)
print("Translated: ",translate_uk_to_us(test_sentence))
```



### 6.4 Known Limitations and Potential Improvements
Limited Dataset: The model may not perform well on out-of-domain data or sentences with highly unusual vocabulary.

Sometimes the model does not get trained properly and only outputs blanks, in that case train the model again or increase batch size.

#### Time Constraints:
- The fine-tuned model was trained on GPU (T4 GPU on Colab), it took around 2 minutes to train the model. If you train on CPU, it will take 60 minutes plus with the trained parameters.
- For faster training, consider reducing the number of epochs or batch size, though this may affect performance.

## 7. Evaluation

The model is evaluated on these parameters:

1. BLEU (Bilingual Evaluation Understudy): Measures how many n-grams in the predicted sentence match those in the reference (ground truth).
2. ROUGE (Recall-Oriented Understudy for Gisting Evaluation): Measures recall, i.e., how many n-grams in the reference sentence are covered by the prediction.
3. Accuracy: Measures how many predictions match the ground truth exactly.

### 7.1 Evaluation Metrics Breakdown:

The model gave these scores for the metrics.

#### BLEU Score: 0.96

This is an excellent BLEU score, indicating that the model's predictions match the reference translations quite well in terms of n-gram overlap. A score above 0.9 generally suggests high translation quality.
#### ROUGE Scores:

ROUGE-1: 0.98 – The model has a very high recall of unigrams (individual words), meaning it's capturing a lot of the key words in the reference.
ROUGE-2: 0.98 – A high recall of bigrams (two-word sequences), suggesting the model is capturing key two-word combinations well.
ROUGE-L: 0.98 – This is a recall-based metric that focuses on the longest matching subsequence of words, and this score suggests that the model is capturing the flow and structure of the sentences well.
#### Accuracy: 92.71%

This indicates that nearly 93% of the predicted sentences exactly match the reference sentences, which is a strong performance in terms of exact matches.
#### SacreBLEU Score: 96.94

This score is extremely high and reinforces that the model's predictions are very close to the reference translations when considering standardized tokenization.

## 8. References
Pre-trained model: [EnglishVoice/t5-base-uk-to-us-english](https://huggingface.co/EnglishVoice/t5-base-uk-to-us-english) – Hugging Face model for UK-to-US english conversion.


