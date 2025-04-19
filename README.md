# **Email Spam Detection using Machine Learning**

## **Project Overview**
This project aims to classify emails as **spam** or **ham** (non-spam) using machine learning algorithms. By utilizing various machine learning techniques, this system is designed to automatically detect spam emails based on features extracted from the email content. The goal is to build an efficient spam filter that can help prevent unwanted emails from reaching a user's inbox.

## **Technologies Used**
- **Python** for implementing machine learning models and data processing.
- **Scikit-learn** for training and evaluating machine learning models.
- **Pandas** and **NumPy** for data manipulation and preprocessing.
- **Natural Language Processing (NLP)** techniques for text preprocessing (tokenization, stopword removal, etc.).
- **TF-IDF (Term Frequency-Inverse Document Frequency)** for converting text data into numerical format.
- **Jupyter Notebooks** for development and testing.

## **Dataset**
The project uses the **SpamAssassin Public Dataset**, which contains a collection of labeled email messages (spam and non-spam). The dataset includes various features such as the email subject, body text, and metadata, which are used for classification.

## **Approach**
1. **Data Preprocessing:** 
   - The email dataset is cleaned by removing unnecessary characters, stopwords, and special symbols.
   - Tokenization is performed to split text into words, and TF-IDF vectorization is used to convert the email content into numerical features suitable for machine learning.

2. **Feature Engineering:** 
   - TF-IDF vectorization is used to transform the raw email text into a numerical matrix, representing the frequency of words while accounting for the importance of each word in the dataset.

3. **Modeling:**
   - Several machine learning algorithms were trained on the dataset, including **Logistic Regression**, **Naive Bayes**, **Random Forest**, and **Support Vector Machine (SVM)**.
   - Hyperparameter tuning was performed to optimize model performance using techniques like Grid Search.

4. **Evaluation:**
   - The models were evaluated based on accuracy, precision, recall, and F1-score.
   - Cross-validation was applied to ensure that the model generalizes well to unseen data.

5. **Deployment (Optional for demo):**
   - A simple script was developed that takes an input email (in the form of text) and predicts whether it is spam or ham using the trained model.

## **Results**
The models were able to effectively classify spam emails with high accuracy. The **Naive Bayes** classifier achieved the best performance, with a precision and recall score close to 90%. This demonstrates the effectiveness of traditional machine learning techniques in solving text classification problems.

## **How to Run the Project**
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/rohithvarma73/Email-Spam-Detection.git
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook or Python script to start the training and testing process:
   ```bash
   jupyter notebook spam_detection.ipynb
   ```
4. To use the trained model for predictions, you can run the `predict.py` script, passing the email text as input.

## **Contributions**
Feel free to fork this repository and contribute to enhancing the model or adding new features.
