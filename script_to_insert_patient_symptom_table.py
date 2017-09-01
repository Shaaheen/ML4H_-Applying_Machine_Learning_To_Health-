import multiprocessing
from datetime import datetime, timedelta
import time

import pymysql
from joblib import Parallel
from joblib import delayed

from Patient import Patient

def main():
    global date
    mydate = datetime(1943, 3, 13, 15, 10, 20)  # year, month, day
    mydate2 = datetime(1943, 3, 17, 15, 10, 20)  # year, month, day

    print(mydate2 - mydate)

    print("Connecting to ML4H DB..")

    conn = pymysql.connect(host='nightmare.cs.uct.ac.za', port=3306, user='ochomo001', passwd='oesaerex', db='ochomo001')

    print("Connected")

    cur2 = conn.cursor()
    print("Executing SQL query..")
    cur2.execute("select person_id, second_last_symptom_date from patient_last_symptom_dates")
    print("Executed")
    cur2.close()

    patients = {}
    patient_ids = []

    print("Querying...")
    for row in cur2:
        date = row[1]
        if date is None:
            continue
        #dt = datetime.strptime(row[1], "%b %d %Y %H:%M")
        # print(date)
        if (row[0] not in patient_ids):
            patient_ids.append(row[0])
            patients[row[0]] = (Patient(int(row[0])))
            patients[row[0]].second_latest_datetime = row[1]

    cur3 = conn.cursor()
    print("Executing SQL query..")
    #cur3.execute("select a.person_id,second_last_symptom_date from Obs_artvisit_sympt a inner join patient_last_symptom_dates b on b.person_id = a.person_id where second_last_symptom_date IS NOT NULL")
    cur3.execute("select person_id,obs_datetime from Obs_artvisit_sympt")
    print("Executed")
    cur3.close()

    count = 0
    for row in cur3:
        if row[0] in patient_ids:
            obs_date = row[1]
            if obs_date > (patients[row[0]].second_latest_datetime - timedelta(days=10)) and obs_date < patients[row[0]].second_latest_datetime :
                patients[row[0]].symptoms_in_prev_month +=1
        count+=1
    print(count)

    print("Done querying.")

    f = open('symptoms_in_last_month.txt', 'w')
    count = 0
    for patient_id in patient_ids:
        f.write(str(patient_id) + "," + str(patients[patient_id].symptoms_in_prev_month) + "\r\n")
        count += 1
    f.close()

    print(count)

    print("Inserting...")
    for j in patient_ids:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO ML4H_prev_10_symptoms (person_id, from_date, number_of_symptoms) VALUES (%s, %s, %s)",
            (j, str( (patients[j]).second_latest_datetime ), str( (patients[j]).symptoms_in_prev_month )))

        cursor.close()

    conn.commit()
    print("Inserted")
    conn.close()


def set_patients_prev_month_symptoms(row,patient_ids,patients, conn):
    date = row[1]
    if row[1] is None:
        return
    if (row[0] in patient_ids):
        cur3 = conn.cursor()
        cur3.execute("select count(*) from Obs_artvisit_sympt where person_id=" + str(
            row[0]) + " AND obs_datetime>(DATE_SUB(DATE('" + str(
            date.strftime('%Y-%m-%d %H:%M:%S')) + "'),INTERVAL 30 DAY))")
        cur3.close()

        for row3 in cur3:
            patients[row[0]].symptoms_in_prev_month = int(row3[0])



main()