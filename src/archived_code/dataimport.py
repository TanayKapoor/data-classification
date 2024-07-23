import os
from firebase_admin import firestore, credentials
import firebase_admin
import csv

key_path = 'src/archived_code/serviceAccountKey.json'
dir_path = 'src/archived_code/dataset'

cred = credentials.Certificate(key_path)
firebase_admin.initialize_app(cred)

db = firestore.client()

def import_csv_to_firestore(csv_file, collection_name):
    batch_size = 500  
    batch = db.batch()
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader, start=1):
            doc_ref = db.collection(collection_name).document()
            batch.set(doc_ref, row)
            if i % batch_size == 0:
                batch.commit()
                batch = db.batch()
                print(f'Committed {i} rows to Firestore collection {collection_name}')
        batch.commit() 
        print(f'Committed {i} rows to Firestore collection {collection_name}')

for filename in os.listdir(dir_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(dir_path, filename)
        collection_name = os.path.splitext(filename)[0]
        import_csv_to_firestore(file_path, collection_name)