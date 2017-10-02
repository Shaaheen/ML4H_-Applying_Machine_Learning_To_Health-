Shaaheen Sacoor - Hons Project

Predicting likely next symptom for a patient
---------Patient.py-------------
= Patient object that is used to store information about a patient(Features)
---------symptom_with_temporal.py ----------
= Main file with the application of machine learning for the problem (scikit-learn)
= The feature and result set is retrieved
= The feature and result set is split into training and result set
= The training set is balanced with the NearMiss sampler technique and RandomOversampler
= Training set is fitted with diff models and evaluated
----------script_to_insert_patient_symptom_table.py-----------
= Script to build the temporal features for the symptom proble
= Since more complex feature, was built in python and then loaded back into database
-----------script_to_insert_patient_latest_symptom_table.py-----------
= Building the result set for the symptom problem
= Get the second to last symptom info and last symptom info for the result set
= Add these to the database for quick access for ml application

Predicting next likely consultation attendance for a patient
------------ConsultationPatient.py---------------
= Consultation Patient object used to store all consultation information about a patient
------------consultation_ML_script.py-------------
= Main file with the application of machine learning for consultation problem
= Retrieve feature and result sets from database
= Get results from AllKNN sampler balanced dataset (Know it is best balanced technique)
= Get results from all other defined sampler techniques
(AllKNN done first as it gives the main results for the project)
-------------script_to_insert_consultation_defaulter_info.py---------------
= This script extracts the customised result set for the consultation problem
= The patients who had missed a consultation at some point in the past where rolled
back to when they just missed their consultation. This formed the new result set