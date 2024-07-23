import os
import pandas as pd
from datetime import datetime

def generate_predictions(train_csv_file_path, unlabeled_csv_file_path, output_folder, column_name, model_trainer):
    clf, vectorizer, column = model_trainer(train_csv_file_path, unlabeled_csv_file_path)

    unlabeled_data = pd.read_csv(unlabeled_csv_file_path)

    new_row = pd.Series([column_name if col == column else '' for col in unlabeled_data.columns], index=unlabeled_data.columns)
    unlabeled_data.columns = new_row

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_file_path = os.path.join(output_folder, f'unlabeled_predictions_{timestamp}.csv')

    file_exists = os.path.isfile(output_file_path)

    if file_exists:
        unlabeled_data.to_csv(output_file_path, mode='a', index=False, header=False)
    else:
        unlabeled_data.to_csv(output_file_path, index=False, header=True)

    return output_file_path