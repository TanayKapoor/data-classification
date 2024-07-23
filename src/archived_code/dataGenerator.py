from faker import Faker
import pandas as pd
import random
import uuid
from firebase_admin import firestore, credentials
import firebase_admin
import csv
from tqdm import tqdm

fake = Faker()

num_records = 2000
common_domains = [
    'gmail.com', 'yahoo.com', 'hotmail.com', 'aol.com', 'outlook.com',
    'icloud.com', 'mail.com', 'zoho.com'
]

def generate_sample_data(num_records):
    data = []
    for _ in tqdm(range(num_records), desc="Generating sample data"):
        entity_type = random.choice(['Patient', 'Doctor', 'Appointment', 'MedicalRecord', 'Billing'])
        is_patient = entity_type == 'Patient'
        is_doctor = entity_type == 'Doctor'
        is_appointment = entity_type == 'Appointment'
        is_medical = entity_type == 'MedicalRecord'
        is_billing = entity_type == 'Billing'

        first_name = fake.first_name()
        last_name = fake.last_name()
        email_domain = random.choice(common_domains)
        email = f"{first_name.lower()}.{last_name.lower()}@{email_domain}"

        record = {
            "id": str(uuid.uuid4()),
            "entity_type": entity_type,
            "first_name": first_name,
            "middle_name": fake.first_name() if random.random() < 0.3 else None,
            "last_name": last_name,
            "email": email,
            "phone": fake.phone_number(),
            "secondary_phone": fake.phone_number() if random.random() < 0.2 else None,
            "address": fake.address(),
            "city": fake.city(),
            "state": fake.state(),
            "zip": fake.zipcode(),
            "country": 'USA' if random.random() < 0.9 else fake.country(),
            "dob": fake.date_of_birth(minimum_age=0, maximum_age=90) if is_patient else None,
            "gender": random.choices(['Male', 'Female', 'Other'], weights=[0.48, 0.48, 0.04])[0] if is_patient else None,
            "ssn": fake.ssn() if is_patient else None,
            "insurance_provider": fake.company() if is_patient and random.random() < 0.7 else None,
            "insurance_policy_number": fake.bban() if is_patient and random.random() < 0.7 else None,
            "emergency_contact_name": fake.name() if is_patient and random.random() < 0.6 else None,
            "emergency_contact_phone": fake.phone_number() if is_patient and random.random() < 0.6 else None,
            "blood_type": random.choice(['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']) if is_patient else None,
            "specialty": fake.job() if is_doctor else None,
            "department": fake.company_suffix() if is_doctor and random.random() < 0.5 else None,
            "patient_id": fake.random_int(min=1, max=1000) if is_appointment or is_medical or is_billing else None,
            "doctor_id": fake.random_int(min=1, max=1000) if is_appointment or is_medical else None,
            "available_hours": f"{fake.time()} - {fake.time()}" if is_doctor and random.random() < 0.5 else None,
            "appointment_date": fake.date_time_this_year(before_now=True, after_now=False) if is_appointment else None,
            "reason_for_visit": fake.sentence(nb_words=6) if is_appointment else None,
            "appointment_notes": fake.text(max_nb_chars=200) if is_appointment and random.random() < 0.5 else None,
            "appointment_status": random.choice(['Scheduled', 'Completed', 'Cancelled']) if is_appointment else None,
            "appointment_id": fake.random_int(min=1, max=1000) if is_medical or is_billing else None,
            "diagnosis": fake.sentence(nb_words=6) if is_medical else None,
            "treatment": fake.text(max_nb_chars=200) if is_medical else None,
            "medication": fake.word() if is_medical else None,
            "allergy_information": fake.text(max_nb_chars=100) if is_medical and random.random() < 0.3 else None,
            "family_history": fake.paragraph(nb_sentences=3) if is_medical and random.random() < 0.3 else None,
            "created_at": fake.date_time_this_year(before_now=True, after_now=False),
            "updated_at": fake.date_time_this_year(before_now=True, after_now=False) if random.random() < 0.5 else None,
            "amount": fake.random_int(min=100, max=5000) if is_billing else None, 
            "billing_date": fake.date_this_year(before_today=True, after_today=False) if is_billing else None,
            "payment_method": random.choice(['Cash', 'Credit Card', 'Insurance']) if is_billing and random.random() < 0.8 else None,  
            "insurance_claim_number": fake.bban() if is_billing and random.random() < 0.6 else None,
            "billing_status": random.choice(['Pending', 'Paid', 'Overdue']) if is_billing else None,
            "related_id": fake.random_int(min=1, max=1000) if random.random() < 0.3 else None,
            "additional_notes": fake.text(max_nb_chars=300) if random.random() < 0.5 else None  
        }

        data.append(record)

    return pd.DataFrame(data)

def save_to_csv(df, csv_file_path):
    df.to_csv(csv_file_path, index=False)

def initialize_firestore(key_path):
    cred = credentials.Certificate(key_path)
    firebase_admin.initialize_app(cred)
    return firestore.client()

def import_csv_to_firestore(csv_file, collection_name, db):
    batch_size = 1000
    batch = db.batch()

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        total_rows = sum(1 for _ in reader)
        file.seek(0)  
        next(reader) 

        with tqdm(total=total_rows, desc="Uploading to Firestore") as pbar:
            for i, row in enumerate(reader, start=1):
                doc_ref = db.collection(collection_name).document()
                batch.set(doc_ref, row)
                if i % batch_size == 0:
                    batch.commit()
                    batch = db.batch()
                    print(f'Committed {i} rows to Firestore collection {collection_name}')
                pbar.update(1)
            batch.commit()
            pbar.update(total_rows - i)
            print(f'Committed {i} rows to Firestore collection {collection_name}')

if __name__ == "__main__":
    df = generate_sample_data(num_records)
    csv_file_path = 'hospital_combined_dummy_data.csv'
    save_to_csv(df, csv_file_path)

    key_path = 'src/archived_code/serviceAccountKey.json'
    db = initialize_firestore(key_path)

    collection_name = 'hospital_records'
    import_csv_to_firestore(csv_file_path, collection_name, db)
