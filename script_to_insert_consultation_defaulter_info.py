import multiprocessing
from datetime import datetime, timedelta
import time

import pymysql
from joblib import Parallel
from joblib import delayed

from Consultation_Patient import Consultation_Patient
from Patient import Patient

#PYTHON SCRIPT TO EXCTRACT RESULTSET FOR CONSULTATION DEFAULTER PROBLEM
def main():
    global date

    print("Connecting to ML4H DB..")

    conn = pymysql.connect(host='nightmare.cs.uct.ac.za', port=3306, user='ochomo001', passwd='oesaerex',
                           db='ochomo001')

    print("Connected")

    patient_ids_missed = []
    patients_missed = {}

    cur2 = conn.cursor()
    print("Executing SQL query..")
    #Gets all the patients that have ever missed a consultation
    cur2.execute("select person_id, max(obs_datetime) from obs where concept_id=1805 and value_coded_name_id=1103 group by person_id")
    print("Executed")
    cur2.close()

    for row in cur2:
        if row[0] not in patient_ids_missed:
            patient_ids_missed.append(row[0])
            patients_missed[row[0]] = ( Consultation_Patient( int(row[0]) ) )
            patients_missed[row[0]].latest_obs_datetime = row[1]

    patient_ids_attended = []
    patients_attended = {}
    print(len(patient_ids_missed))
    cur3 = conn.cursor()
    print("Executing SQL query..")
    cur3.execute("select person_id, value_coded_name_id, obs_datetime from obs where concept_id=1805 and value_coded_name_id IN (1102,1103) and obs_datetime>'2008-01-01' order by obs_datetime")
    #Gets all consultation observations
    print("Executed")
    cur3.close()
    for row in cur3:
        if row[0] in patient_ids_missed:
            #If patient prev missed, then add to total consultation sum or total missed consultation sum
            if row[2] <= patients_missed[row[0]].latest_obs_datetime :
                patient = patients_missed[row[0]]
                if int(row[1]) == 1102:
                    patients_missed[row[0]].features[0] += 1
                    patients_missed[row[0]].last_consultations[0] = patients_missed[row[0]].last_consultations[1]
                    patients_missed[row[0]].last_consultations[1] = 0
                elif int(row[1]) == 1103:
                    patients_missed[row[0]].features[1] += 1
                    patients_missed[row[0]].last_consultations[0] = patients_missed[row[0]].last_consultations[1]
                    patients_missed[row[0]].last_consultations[1] = 1

        # If patient never missed, then add to total consultation sum and total missed consultation sum
        else:
            if row[0] not in patient_ids_attended:
                patient_ids_attended.append(row[0])
                patients_attended[row[0]] = (Consultation_Patient(int(row[0])))

            if row[0] in patient_ids_attended:
                if int(row[1]) == 1102:
                    patients_attended[row[0]].features[0] += 1
                    patients_attended[row[0]].last_consultations[0] = patients_attended[row[0]].last_consultations[1]
                    patients_attended[row[0]].last_consultations[1] = 0
                elif int(row[1]) == 1103:
                    patients_attended[row[0]].features[1] += 1
                    patients_attended[row[0]].last_consultations[0] = patients_attended[row[0]].last_consultations[1]
                    patients_attended[row[0]].last_consultations[1] = 1

                if patients_attended[row[0]].latest_obs_datetime is None:
                    patients_attended[row[0]].latest_obs_datetime = row[2]
                elif patients_attended[row[0]].latest_obs_datetime <= row[2]:
                    patients_attended[row[0]].latest_obs_datetime = row[2]

    print(len(patient_ids_attended))

    #Join together patients who have previously missed consultations and patients who have never missed any
    patient_ids = patient_ids_attended + patient_ids_missed

    patients = dict(patients_attended)  # or orig.copy()
    patients.update(patients_missed)

    for id in patient_ids:
        if patients[id].last_consultations[1] == 1:
            patients[id].features[1]-=1
        elif patients[id].last_consultations[1] == 0:
            patients[id].features[0]-=1

    #Insert info into one table
    print("Inserting...")
    for j in patient_ids:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO ML4H_patient_consultation_defaulters "
            "(person_id, tot_consultations_attended, tot_consultations_missed, last_consultation_date, last_consultation_attendance, missed_last_appointment) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (int(j) , int(patients[j].features[0]), int(patients[j].features[1]),
             patients[j].latest_obs_datetime, int(patients[j].last_consultations[0]), int(patients[j].last_consultations[1]) ))

        cursor.close()

    conn.commit()
    print("Inserted")

    conn.close()

main()
