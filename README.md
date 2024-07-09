Automated Response Generation for Customer Support

## Objective
Build a model to generate automated responses to customer queries.

## Dataset
We used the [Customer Support Responses](https://huggingface.co/datasets/Kaludi/Customer-Support-Responses) dataset from Hugging Face.

## 1. Dataset Exploration and Preprocessing
### Dataset Overview
The dataset contains paired customer queries and support responses. Each entry in the dataset includes:
- query: The customer query text.
- response: The corresponding customer support response text.

### Data Preprocessing
The following preprocessing steps were applied to the dataset:
1. *Tokenization*: Split the text into tokens.
2. *Padding/Truncation*: Padded or truncated the sequences to a fixed length to ensure uniformity in the input data.

## 2. Model Training
### Model Selection
We chose a google's flan-T5 base model for generating responses. Specifically, we utilized that model architecture for its robust language generation capabilities.

### Training Environment
- *Framework*: PyTorch 
- *Hardware*: T4 GPU
- *Software*: Jupyter Notebook, Hugging Face Transformers library

## 5. Evaluation
### Evaluation Metrics
To assess the quality and appropriateness of the generated responses, we used the following metrics:
- *BLEU Score*: Evaluated the overlap between the generated and actual responses.
- *Human Evaluation*: Conducted human evaluations to rate the relevance and coherence of the responses.

### Results
![W B Chart 09_07_2024, 22_48_33](https://github.com/jagadeeshD3/pesto_take-home-assessment/assets/80314569/415c10d4-a232-4ef0-b3dc-a3be2ea3dc89)
![W B Chart 09_07_2024, 22_48_50](https://github.com/jagadeeshD3/pesto_take-home-assessment/assets/80314569/ebbb6b35-2a45-4779-ac40-57539522d337)


## Conclusion
The automated response generation model successfully generates coherent and relevant responses to customer queries. The use of a goole's flan-T5 base and fine-tuning techniques contributed to the high-quality performance of the model.

