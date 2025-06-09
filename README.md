# Amazon Product Review Sentiment Analysis

This repository contains the code for sentiment analysis on Amazon product reviews using a Transformer-based model.

---

## Project Overview

The model predicts whether a product review is **positive** or **negative** with probability scores and visual feedback.

The model was trained using a **BERT** architecture fine-tuned on Amazon review data for sentiment classification.

The actual trained model files are hosted on Hugging Face to avoid large file uploads on GitHub.

---

## How to Use

### Run on Kaggle Notebook

- Open the Kaggle notebook [`amazon-product-review-nlp.ipynb`](https://www.kaggle.com/code/arpitarout01/amazon-product-review-nlp)  
- The notebook loads the model files from Hugging Face and performs inference on input reviews.

---

## Model Files

The following model files are hosted on [Hugging Face](https://huggingface.co/your-username/amazon-review-sentiment-model):

- [`model.safetensors`](https://huggingface.co/arpitarout01/amazon-review-sentiment-model/blob/main/model.safetensors)
- [`config.json`](https://huggingface.co/arpitarout01/amazon-review-sentiment-model/blob/main/config.json)
- [`tokenizer_config.json`](https://huggingface.co/arpitarout01/amazon-review-sentiment-model/blob/main/tokenizer_config.json)
- [`special_tokens_map.json`](https://huggingface.co/arpitarout01/amazon-review-sentiment-model/blob/main/special_tokens_map.json)
- [`vocab.txt`](https://huggingface.co/arpitarout01/amazon-review-sentiment-model/blob/main/vocab.txt)

The notebook downloads these files directly for inference.

---

## Notes

- This repo contains only the code and documentation.  
- The heavy model files are loaded from Hugging Face at runtime.  
- For a live demo or deployment, you can use Kaggle notebooks as the platform.

---

## Contact

Created by [Arpita Rout](https://github.com/arpitarout01)


