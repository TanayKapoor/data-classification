import pandas as pd
import numpy as np
import os
import sys
import logging
import datetime
import shutil

# import csv file
def import_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        logging.error(f"File not found at {file_path}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error occured while importing csv file: {e}")
        sys.exit(1)

def clean_data(df):
    try:
        df = df.dropna()
        df = df.drop_duplicates()
        return df
    except Exception as e:
        logging.error(f"Error occured while cleaning data: {e}")
        sys.exit(1)

# copy cleaned file to archive

def copy_to_archive(file_path, archive_path):
    try:
        if not os.path.exists(archive_path):
            os.makedirs(archive_path)

        file_name = os.path.basename(file_path)
        file_name, file_ext = os.path.splitext(file_name)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        new_file_name = f"{file_name}_{timestamp}{file_ext}"

        new_file_path = os.path.join(archive_path, new_file_name)
        shutil.copy2(file_path, new_file_path)
        
    except Exception as e:
        logging.error(f"Error occured while copying to archive: {e}")
        sys.exit(1)       

def generate_dq_report(df):
    try:
        rows, cols = df.shape
        cols_names = df.columns
        missing_values = df.isnull().sum()
        unique_values = df.nunique()
        data_types = df.dtypes
        duplicate_rows = df.duplicated().sum()

        dq_report = {
            "rows": rows,
            "columns": cols,
            "columns_names": cols_names,
            "missing_values": missing_values,
            "unique_values": unique_values,
            "data_types": data_types,
            "duplicate_rows": duplicate_rows
        }        

        dq_score = generate_dq_score(dq_report)
        dq_report["dq_score"] = dq_score

        return dq_report
    except Exception as e:
        logging.error(f"Error occured while generating dq report: {e}")
        sys.exit(1)

def generate_dq_score(dq_report):
    try:
        missing_values = dq_report["missing_values"]
        unique_values = dq_report["unique_values"]
        duplicate_rows = dq_report["duplicate_rows"]

        dq_score = 100
        for col, val in missing_values.items():
            dq_score -= (val / dq_report["rows"]) * 100

        for val in unique_values:
            if val == 1:
                dq_score -= 10

        dq_score -= (duplicate_rows / dq_report["rows"]) * 100

        dq_score = max(0, dq_score)

        return dq_score
    except Exception as e:
        logging.error(f"Error occured while generating dq score: {e}")
        sys.exit(1)

def export_dq_report(dq_report, file_path):
    try:
        with open(file_path, 'w') as f:
            for key, value in dq_report.items():
                f.write(f"{key}: {value}\n")
    except Exception as e:
        logging.error(f"Error occured while exporting dq report: {e}")
        sys.exit(1)


def export_cleaned_data(df, file_path):
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        logging.error(f"Error occured while exporting cleaned data: {e}")
        sys.exit(1)

def main():
    logging.basicConfig(filename='data_import.log', level=logging.INFO)

    # import data for multiple files in 'data/raw/mbl_dataset'
    raw_data_dir = 'data/raw/mbl_dataset'
    archive_data_dir = 'data/archive/mbl_dataset'
    processed_data_dir = 'data/processed/mbl_dataset'
    dq_report_dir = 'data/dq_reports/mbl_dataset'

    # create directories if not exist
    for dir in [raw_data_dir, archive_data_dir, processed_data_dir, dq_report_dir]:
        if not os.path.exists(dir):
            os.makedirs(dir)


    for file in os.listdir(raw_data_dir):
        file_path = os.path.join(raw_data_dir, file)
        df = import_csv(file_path)
        df = clean_data(df)
        copy_to_archive(file_path, archive_data_dir)
        export_cleaned_data(df, os.path.join(processed_data_dir, file))
        dq_report = generate_dq_report(df)
        export_dq_report(dq_report, os.path.join(dq_report_dir, file + ".txt"))

    logging.info("Data import process completed successfully")





if __name__ == "__main__":
    main()