# CodSoft ML Internship Projects

This repository contains three completed machine learning projects from the CodSoft internship:

1. Movie Genre Classification
2. Spam SMS Detection
3. Customer Churn Prediction

## Project Structure

- `task1_movie_genre/` — movie genre classification using text-based logistic regression
- `task4_spam_sms/` — spam SMS detection using TF-IDF + SMOTE + LinearSVC
- `task3_customer_churn/` — customer churn prediction using a Random Forest model
- `Genre Classification Dataset/` — training dataset for the movie project
- `spam.csv` — SMS dataset for the spam detection project
- `requirements.txt` — Python dependencies

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the Notebooks

Open each notebook in Jupyter or VS Code and run the cells in order:

- `task1_movie_genre/movie_genre_classification.ipynb`
- `task4_spam_sms/spam_sms_classification.ipynb`

## Notes

- The notebooks were updated to use repo-relative paths so they work from a cloned GitHub repository without hardcoded Windows paths.
- The movie project expects the dataset in `Genre Classification Dataset/train_data.txt`.
- The spam project expects `spam.csv` at the repository root.

## GitHub Push Workflow

```bash
git init
git add .
git commit -m "Add CodSoft ML internship projects"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Project Summary

### 1) Movie Genre Classification
- Uses title + description text
- Applies text preprocessing and TF-IDF vectorization
- Trains a Logistic Regression model
- Predicts the top movie genres from plot text

### 2) Spam SMS Detection
- Uses SMS text messages
- Performs TF-IDF vectorization
- Balances the data with SMOTE
- Trains a LinearSVC model for spam detection
- Evaluates with confusion matrix and classification report

## License

This project is intended for learning and portfolio purposes.
