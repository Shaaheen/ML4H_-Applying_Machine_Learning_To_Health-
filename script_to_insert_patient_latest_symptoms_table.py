from datetime import datetime

import pymysql

from Patient import Patient
#PYTHON SCRIPT THAT GETS THE LAST REPORTED SYMPTOM INFO AND THE SECOND LAST REPORTED SYMPTOM INFO - FOR RESULT SET BUILDING
#Second last symptom info is needed as this is the date where the feature set ends

print("Connecting to ML4H DB..")

conn = pymysql.connect(host='nightmare.cs.uct.ac.za', port=3306, user='ochomo001', passwd='oesaerex', db='ochomo001')

print("Connected")

cur2 = conn.cursor()
print("Executing SQL query..")
#Gets relevant symptom observation data
cur2.execute("select person_id, obs_datetime from Obs_artvisit_sympt WHERE value_coded_name_id IN (110, 4315, 156, 524, 1576, 2325, 3, 888, 17, 10894, 4407, 9345, 838, 4355, 11333) ")
print("Executed")
cur2.close()

patients = {}
patient_ids = []


for row in cur2:
    date = row[1]
    #print(date)
    if (row[0] not in patient_ids):
        patient_ids.append(row[0])
        patients[row[0]] = (Patient(int(row[0])))

    #Set second and latest reported symptom dates
    if (row[0] in patient_ids):
        if (patients[row[0]]).latest_datetime < date:
            (patients[row[0]]).second_latest_datetime = (patients[row[0]]).latest_datetime
            (patients[row[0]]).latest_datetime = date
            (patients[row[0]]).calculate_time_between_symptoms()

#INSERT RESULT SET INFO INTO ONE TABLE FOR QUICK ACCESS
print("Inserting...")
for j in patient_ids:
    cursor = conn.cursor()
    if (patients[j]).time_between_symptom_reports.days < 365:
        cursor.execute("INSERT INTO patient_last_symptom_dates (person_id, last_symptom_date, second_last_symptom_date, days_between_last_symptoms) VALUES (%s, %s, %s,%s)",
                       (j, str( (patients[j]).latest_datetime ), str( (patients[j]).second_latest_datetime ), patients[j].time_between_symptom_reports.days))
    else:
        cursor.execute(
            "INSERT INTO patient_last_symptom_dates (person_id, last_symptom_date, second_last_symptom_date, days_between_last_symptoms) VALUES (%s, %s, %s,%s)",
            (j, str( (patients[j]).latest_datetime ), None, None))

    cursor.close()


conn.commit()
print("Inserted")
conn.close()
