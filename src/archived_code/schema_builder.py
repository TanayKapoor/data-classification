import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def import_csvs(directory_path):
    try:
        files = os.listdir(directory_path)
        csv_files = [file for file in files if file.endswith(".csv")]
        return csv_files
    except FileNotFoundError:
        print(f"Directory not found at {directory_path}")
        return []
    except Exception as e:
        print(f"Error occurred while importing CSV files: {e}")
        return []

def import_and_concat_data(directory_path):
    csv_files = import_csvs(directory_path)
    dfs = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(directory_path, file))
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def explore_and_generate_schema(df):
    try:

        X = df.drop(columns=['target_column'])  # Adjust 'target_column' based on your data
        y = df['target_column']  # Adjust 'target_column' based on your data

        classifier = RandomForestClassifier()
        classifier.fit(X, y)

        # Print feature importances
        feature_importances = pd.Series(classifier.feature_importances_, index=X.columns)
        print("Feature Importances:")
        print(feature_importances.sort_values(ascending=False))

        # Generate a database schema based on feature importances
        # You may customize this part based on your requirements
        schema = {}
        for column, importance in feature_importances.iteritems():
            if importance > 0.02:  # Adjust the threshold as needed
                schema[column] = 'numeric'  # You may adjust the type based on importance thresholds
            else:
                schema[column] = 'categorical'  # You may adjust the type based on importance thresholds

        print("Generated Database Schema:")
        print(schema)

    except Exception as e:
        print(f"Error occurred during schema exploration: {e}")

def main():
    directory_path = "data/processed/mbl_dataset/"
    df = import_and_concat_data(directory_path)

    if not df.empty:
        explore_and_generate_schema(df)
    else:
        print("No CSV files found or an error occurred during import.")

if __name__ == "__main__":
    main()
