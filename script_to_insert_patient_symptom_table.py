from datetime import datetime

import pymysql

from Patient import Patient

#PYTHON SCRIPT THAT GETS ALL THE TEMPORAL INFO ABOUT A PATIENT AND LOADS IT INTO A TABLE FOR QUICKER ACCESS
#TEMPORAL INFO SUCH AS HOW MANY DAYS AGO WAS A SPECIFIC SYMPTOM REPORTED
def main():
    global date

    print("Connecting to ML4H DB..")

    conn = pymysql.connect(host='nightmare.cs.uct.ac.za', port=3306, user='ochomo001', passwd='oesaerex',
                           db='ochomo001')

    print("Connected")

    cur2 = conn.cursor()
    print("Executing SQL query..")
    #Gets the second last data and last date info about patients
    cur2.execute("select person_id, second_last_symptom_date, last_symptom_date from patient_last_symptom_dates")
    print("Executed")
    cur2.close()

    patients = {}
    patient_ids = []

    print("Querying...")
    for row in cur2:
        date = row[1]
        if date is None:
            continue
        if (row[0] not in patient_ids):
            patient_ids.append(row[0])
            patients[row[0]] = (Patient(int(row[0])))
            patients[row[0]].second_latest_datetime = row[1]
            patients[row[0]].latest_datetime = row[2]


    cur3 = conn.cursor()
    print("Executing SQL query..")
    #Gets all symptom observation data
    cur3.execute("select person_id,obs_datetime,value_coded_name_id from Obs_artvisit_sympt")
    print("Executed")
    cur3.close()
    #The value_coded_name_ids of the specific symptoms accounted for in the symptom prediction problem
    symptom_array = [110, 4315, 156, 524, 1576, 2325, 3, 888, 17, 10894, 4407, 9345, 838, 4355, 11333]
    count = 0
    for row in cur3:
        if row[0] in patient_ids:
            obs_date = row[1]
            days_between_sympts = -((obs_date - patients[row[0]].second_latest_datetime).days)
            if row[2] is None:
                continue
            #If haven't reported symptom before then change to most recently reported symptom
            if int(row[2]) in symptom_array and patients[row[0]].symptoms_recentness_array[
                symptom_array.index(int(row[2]))] == -1 and days_between_sympts >= 0:
                patients[row[0]].symptoms_recentness_array[symptom_array.index(int(row[2]))] = days_between_sympts
            #If have reported before but found a more recent report then change value
            elif int(row[2]) in symptom_array and days_between_sympts < patients[row[0]].symptoms_recentness_array[
                symptom_array.index(int(row[2]))] and days_between_sympts >= 0:
                patients[row[0]].symptoms_recentness_array[symptom_array.index(int(row[2]))] = days_between_sympts
        count += 1

    print("Done querying.")

    #inserts all most recent reported symptoms into one table
    print("Inserting...")
    for j in patient_ids:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO ML4H_most_recent_symptom_days_limit "
            "(person_id, cough, fever, abdominal_pain, skin_rash, lactic_acidosis, lipodystrophy, anemia, "
            "anorexia, diarrhea, leg_pain, night_sweats, other, peripheral_neuropathy, vomiting, weight_loss ) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            # (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)             )
            (int(j) , int(patients[j].symptoms_recentness_array[0]), int(patients[j].symptoms_recentness_array[1]),
             int(patients[j].symptoms_recentness_array[2]), int(patients[j].symptoms_recentness_array[3]),
             int(patients[j].symptoms_recentness_array[4]), int(patients[j].symptoms_recentness_array[5]),
             int(patients[j].symptoms_recentness_array[6]), int(patients[j].symptoms_recentness_array[7]),
             int(patients[j].symptoms_recentness_array[8]), int(patients[j].symptoms_recentness_array[9]),
             int(patients[j].symptoms_recentness_array[10]), int(patients[j].symptoms_recentness_array[11]),
            int(patients[j].symptoms_recentness_array[12]), int(patients[j].symptoms_recentness_array[13]) ,
             int(patients[j].symptoms_recentness_array[14]) ))

        cursor.close()

    conn.commit()
    print("Inserted")
    conn.close()

main()
