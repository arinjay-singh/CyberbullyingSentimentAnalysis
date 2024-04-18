# Cyberbullying Detection using Sentiment Analysis

In recent years, cyberbullying has become a pervasive issue affecting many individuals across various online platforms. Despite increasing awareness, statistics indicate that substantial actions to mitigate cyberbullying are still lacking. To address this, our project develops a sophisticated BiLSTM model for detecting cyberbullying instances in text. We also compare its performance against other models like Logistic Regression (LR) and VADER (Valence Aware Dictionary and sEntiment Reasoner). Our system currently achieves an accuracy rate of 81% on the test set, demonstrating its potential in identifying and classifying cyberbullying behavior effectively.

## Project Structure Overview

### Python Notebooks
- `bi_lstm_model.ipynb`: Implements the BiLSTM model training, and including some preprocessing, model setup, and training loop, as well as evaluation and graphing of the model's performance.
- `vader_sentiment_analysis.ipynb`: Creates and Evaluates the Vader Sentiment Model on our testing data.
- `lr_model.ipynb`: Our implementation of a Logistic Regression Model as well as it's testing.
- `google_emotion_dataset_preprocessing_pt1.ipynb`: Preprocessing for our Google emotions dataset. Converts an entry label into whether it is cyberbullying or not given the emotional labels that it has. It then saves all of the converted entries to a new CSV file after processing.
- `google_emotion_dataset_preprocessing_pt2.ipynb`: The newly processed text is then converted to lowercase, tokenized, and rid of stopwords, and punctuation, and saved to another CSV file.
- `twitter_data_preprocessing.ipynb`: Takes in our Cyberbullying Twitter Dataset and preprocesses it for usage, saving it to a new file after processing.

### Datasets
Our models are trained and evaluated using several datasets, some of these are the final datasets after preprocessing:
- `cyberbullying_tweets.csv`
- `go_emotions_dataset.csv`
- `google_cyberbullying_dataset.csv`
- `processed_concat_data.csv`
- `processed_cyberbullying_tweets.csv`
- `processed_google_data.csv`

### Our Model and Embeddings
This section lists the files related to our neural network model and the embeddings used for training:
- `cyberbullying_detection_model_epoch10.h5` - The saved BiLSTM model after 10 epochs of training.
- `word2vec_model.model` - Word2Vec model used for generating word embeddings.
- `word2vec_embeddings.txt` - Text file containing the word embeddings.

## Deprecated Files
The `deprecated_files` folder contains older versions of notebooks and data analyses that are retained for archival purposes.

### How to Run
For this project simply clone the repo and open up each notebook you'd like to run in the code editor of your choice.

__Note:__ in order to run these files you need some prerequisite libraries, please run pip/pip3 install [library], some include tensorflow, sklearn, and word2vec, among others.
          Python 3.9 or 3.10 is recommended. If on the M chip series of macbooks (M1-M3) please download with `pip install tensorflow-metal`.
