import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from scipy.sparse import vstack
import numpy as np

timestamp = datetime.now().strftime("%Y%m%d%H%M")

def get_phone_column(df, clf, vectorizer):
    for column in df.columns:
        X_test = vectorizer.transform(df[column].astype(str))
        predictions = clf.predict(X_test)
        if np.mean(predictions) > 0.5:
            return column
    return None

def train_phone_model(train_csv_file_path, unlabeled_csv_file_path, epochs=10):
    model_file_path = 'src/saved_models/phone_models/phone_classifier.pkl'

    if os.path.isfile(model_file_path):
        with open(model_file_path, 'rb') as file:
            clf = pickle.load(file)
        with open('src/saved_models/phone_models/vectorizer.pkl', 'rb') as file:
            vectorizer = pickle.load(file)
        with open('src/saved_models/phone_models/phone_column.pkl', 'rb') as file:
            phone_column = pickle.load(file)
        print("Loaded model from file.")
        return clf, vectorizer, phone_column
    else:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
        
            train_data = pd.read_csv(train_csv_file_path)
            print("Loaded training data.")
        
            train_data['is_valid_phone'] = train_data['is_valid_phone'].fillna(0)

            vectorizer = CountVectorizer(lowercase=True, binary=True)
            X_train = vectorizer.fit_transform(train_data['phone_number'].astype(str))
            y_train = train_data['is_valid_phone'].astype(int)
            print("Vectorized training data.")

            clf = RandomForestClassifier()
            clf.fit(X_train, y_train)
            print("Trained classifier on training data.")

            unlabeled_data = pd.read_csv(unlabeled_csv_file_path)
            print("Loaded unlabeled data.")
        
            phone_column = get_phone_column(unlabeled_data, clf, vectorizer)
            if phone_column is None:
                raise ValueError("No phone column found in the unlabeled data.")
            X_unlabeled = vectorizer.transform(unlabeled_data[phone_column].astype(str))
            pseudo_labels = clf.predict(X_unlabeled)
            print("Predicted pseudo labels for unlabeled data.")

            new_unlabeled_data = pd.DataFrame(columns=unlabeled_data.columns)
            new_unlabeled_data = new_unlabeled_data._append(unlabeled_data, ignore_index=True)
            new_row = pd.Series(['phone' if col == phone_column else '' for col in new_unlabeled_data.columns], index=new_unlabeled_data.columns)
            new_unlabeled_data = pd.concat([pd.DataFrame(new_row).T, new_unlabeled_data], ignore_index=True)
            print("Prepared new unlabeled data.")

            with open(model_file_path, 'wb') as file:
                pickle.dump(clf, file)
            with open('src/saved_models/phone_models/vectorizer.pkl', 'wb') as file:
                pickle.dump(vectorizer, file)
            with open('src/saved_models/phone_models/phone_column.pkl', 'wb') as file:
                pickle.dump(phone_column, file)
            print("Saved model to file.")


        return clf, vectorizer, phone_column
    
def generate_unlabeled_phone_predictions(train_csv_file_path, unlabeled_csv_file_path):
    clf, vectorizer, phone_column = train_phone_model(train_csv_file_path, unlabeled_csv_file_path)

    unlabeled_data = pd.read_csv(unlabeled_csv_file_path)

    new_row = pd.Series(['phone' if col == phone_column else '' for col in unlabeled_data.columns], index=unlabeled_data.columns)
    unlabeled_data.columns = new_row

    file_exists = os.path.isfile(f'data/processed/unlabeled_predictions_{timestamp}.csv')

    if file_exists:
        unlabeled_data.to_csv(f'data/processed/unlabeled_predictions_{timestamp}.csv', mode='a', index=False, header=False)
    else:
        unlabeled_data.to_csv(f'data/processed/unlabeled_predictions_{timestamp}.csv', index=False, header=True)

    return f'data/processed/unlabeled_predictions_{timestamp}.csv'