from datetime import datetime

import pymysql

from Patient import Patient

mydate = datetime(1943,3, 13, 15 , 10, 20)  #year, month, day
mydate2 = datetime(1943,3, 17, 15 , 10, 20)  #year, month, day

print(mydate2-mydate)

print("Connecting to ML4H DB..")

conn = pymysql.connect(host='nightmare.cs.uct.ac.za', port=3306, user='ochomo001', passwd='oesaerex', db='ochomo001')

print("Connected")

cur2 = conn.cursor()
print("Executing SQL query..")
cur2.execute("select person_id, obs_datetime from Obs_artvisit_sympt")
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

    if (row[0] in patient_ids):
        if (patients[row[0]]).latest_datetime < date:
            (patients[row[0]]).second_latest_datetime = (patients[row[0]]).latest_datetime
            (patients[row[0]]).latest_datetime = date
            (patients[row[0]]).calculate_time_between_symptoms()

for i in patients:
    print(str(patients[i].nam) + " : " + str(patients[i].latest_datetime) + " " + str(patients[i].second_latest_datetime) + " " + str(patients[i].time_between_symptom_reports) )

count = 0
for i in patient_ids:
    if (patients[i]).time_between_symptom_reports.days < 365:
        count+=1

print(count)

print("Inserting...")
for j in patient_ids:
    cursor = conn.cursor()
    if (patients[j]).time_between_symptom_reports.days < 365:
        #print( j, str( (patients[j]).latest_datetime ), str( (patients[j]).second_latest_datetime ), patients[j].time_between_symptom_reports.days )
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
