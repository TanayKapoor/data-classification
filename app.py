from src.models.phone_learning_model import train_phone_model, generate_unlabeled_phone_predictions
from src.models.email_learing_model import train_email_model, generate_unlabeled_email_predictions
from src.predictions_generator import generate_predictions
import pandas as pd
import os

phone_train_csv_file_path = 'data/raw/phone_dataset.csv'
email_train_csv_file_path = 'data/raw/email-training.csv'

unlabelled_csv_file_path = 'data/raw/user-data-v1.csv'

train_phone_model(phone_train_csv_file_path, unlabelled_csv_file_path)
train_email_model(email_train_csv_file_path, unlabelled_csv_file_path)

phone_predictions_file_path = generate_unlabeled_phone_predictions(phone_train_csv_file_path, unlabelled_csv_file_path)
email_predictions_file_path = generate_unlabeled_email_predictions(email_train_csv_file_path, unlabelled_csv_file_path)

print(f"Phone predictions saved to {phone_predictions_file_path}")
print(f"Email predictions saved to {email_predictions_file_path}")

output_folder = 'data'

phone_predictions_header = pd.read_csv(phone_predictions_file_path, nrows=1)
email_predictions_header = pd.read_csv(email_predictions_file_path, nrows=1)

combined_header = phone_predictions_header.combine_first(email_predictions_header)

combined_predictions_file_path = os.path.join(output_folder, 'combined_predictions.csv')
combined_header.to_csv(combined_predictions_file_path, index=False)

print(f"Combined header saved to {combined_predictions_file_path}")




