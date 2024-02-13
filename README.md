# Kaggle challenge of LLM_Detect_AI_Generated_Text

![chatgpt](https://miro.medium.com/v2/resize:fit:1358/1*WtV5aYkcQV--GXhv1OsTDg.jpeg)


my solution to the challenge of [LLM DETECT AI vs Student GENERATED TEXT](https://www.kaggle.com/c/llm-detect-ai-generated-text/overview) on Kaggle.

## Problem Statement
The task is to detect AI-generated text. The dataset contains 2 classes: AI-generated text and Student-generated text. The original dataset provided by the competition was pretty inbalaced, so I considered using another dataset to balance the classes. I used the [/kaggle/input/daigt-v2-train-dataset](https://www.kaggle.com/abhishek/daigt-v2-train-dataset) dataset to balance the classes.

## Approach
compute Entropy of the text and use it as a feature to train a classifier. I used a simple support vector machine model to classify the text.

## kaggle notebook
you can find my kaggle notebook including detailed EDA on the dataset and the code for the model [here](https://www.kaggle.com/shirinymn/entropy-svm/)