# Spam SMS Detection

## Project Overview
This project focuses on detecting spam SMS messages using machine learning techniques. By analyzing a dataset of SMS messages, we developed and optimized a logistic regression model to identify whether a message is spam or not, achieving high accuracy.

## Dataset
The dataset contains SMS messages labeled as "ham" (not spam) or "spam". It is sourced from Kaggle and includes 5,574 messages.

## Project Steps
1. **Data Analysis and Preprocessing**
    - Loaded and explored the dataset with various encoding techniques to handle different character sets.
    - Converted all text to lowercase to maintain uniformity.
    - Split the data into training and testing sets.
    - Used TF-IDF vectorization to transform the text data into numerical features.
2. **Model Development**
    - Developed and optimized a logistic regression model.
    - Achieved an accuracy of 96.77% in detecting spam messages.
3. **Model Evaluation**
    - Evaluated the model using accuracy score, confusion matrix, and classification report.
    - The model showed high precision and recall for both "ham" and "spam" categories.
4. **Prediction**
    - Tested the model with new messages to predict whether they are spam or ham.

## Requirements
- Python 3.x
- Pandas
- Scikit-learn

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Ayesha-Taranum/Spam-SMS-Detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd spam-sms-detection
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the Jupyter notebook to execute the project:
    ```bash
    jupyter notebook
    ```
2. Open `spam_sms_detection.ipynb` and run the cells to see the analysis and model development process.

## Results
- The logistic regression model achieved an accuracy of 96.77%.
- The confusion matrix and classification report indicate high precision and recall for both spam and ham messages.

## Example Predictions
```python
new_messages = ["need your help now", "Meeting at 3 pm today."]
new_messages_tfidf = tfidf_vectorizer.transform(new_messages)
predictions = classifier.predict(new_messages_tfidf)

for message, prediction in zip(new_messages, predictions):
    print(f"Message: {message}\nPrediction: {'spam' if prediction == 'spam' else 'ham'}\n")
