#Script to apply machine learning techniques to medical data to try predict a patient's last reported symptom to a clinic

import matplotlib.pyplot as plt
import numpy
import pandas
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import ClusterCentroids
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import TomekLinks
from numpy import *
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm, datasets
#from numpy import np
import pymysql
from joblib import Parallel, delayed
import multiprocessing

import Patient
import temporal_symptoms_ML
from Patient import Patient
from ml4h_file_utils import classifaction_report_csv
variable_file = None
def main():
    print("Symptom prediction Experiment")
    global variable_file
    print("Connecting to ML4H DB..")

    conn = pymysql.connect(host='nightmare.cs.uct.ac.za', port=3306, user='ochomo001', passwd='oesaerex', db='ochomo001')

    print("Connected")

    cur = conn.cursor()
    print("Executing SQL query..")
    print("SQL Script: select * from ML4H_symptom_totals a inner join ML4H_most_recent_symptom_days_limit b on a.person_id=b.person_id")
    print("Retrieving symptom features in sql query...")
    #Query to get the totals of all the symptoms and amount of days since last reporting of a symptom
    # that a patient has reported to the clinic
    #Used as features to predict the next possible that a patient will report
    cur.execute("select * from ML4H_symptom_totals a inner join ML4H_most_recent_symptom_days_limit b on a.person_id=b.person_id")
    print("Executed")
    cur.close()

    #Loading features
    patient_ids = []
    sum_of_features = []
    for k in range( len(Patient.features) ): #Counter for the occurences of symptoms
        sum_of_features.append(0)

    patients = {}
    print("loading in data into program...")
    for row in cur:
        if (row[0] not in patient_ids): #Create patient if haven't already
            patient_ids.append(row[0])
            patients[row[0]] = (Patient(int(row[0])))

        #Set the patients feature array to the retrieved features
        patients[row[0]].feature_symptom_array = [ int(row[2]) + int(row[10]),int(row[3]),int(row[4]),int(row[5]),int(row[6]),
                                                   int(row[7]), int(row[8]), int(row[9]), int(row[11]),
                                                   int(row[12]), int(row[13]), int(row[14]), int(row[15]),int(row[16]),
                                                   int(row[17]), 0, 0, 0, int(row[21]),int(row[22]),int(row[23])
                                                   ,int(row[24]),int(row[25]),int(row[26]),int(row[27]),int(row[28])
                                                   ,int(row[29]),int(row[30]),int(row[31]),int(row[32]),int(row[33])
                                                   ,int(row[34]),int(row[35]) ]
        #To check the balance of features
        for i in range( len(sum_of_features)- 3 ):
            if i == 8: #Exception for Cough for duration - joining with Cough feature
                sum_of_features[0] += int(row[i + 2])
            elif i==16 or i==17 or i == 18: #Exceptions for the patient demographic features (added later)
                continue
            elif i > 8:
                sum_of_features[i-1]+= int(row[i + 2])
            else:
                sum_of_features[i] += int(row[i + 2])


    cur1 = conn.cursor()
    print("Executing SQL query..")
    print("SQL script: select person_id, value_coded_name_id,name, MAX(obs_datetime) from Obs_artvisit_sympt a inner join concept_name n on a.value_coded_name_id = n.concept_name_id where value_coded_name_id IN (524,110,4315,156,3,888,11335,17,2325,4407,10894,9345,838,4355,11333) Group by person_id")
    print("Getting the last symptom reported for the result set...")
    #Gets the last reported symptom that a patient reported - RESULT CLASS (Trying to predict this)
    cur1.execute("select person_id, value_coded_name_id,name, MAX(obs_datetime) from Obs_artvisit_sympt a inner join concept_name n on a.value_coded_name_id = n.concept_name_id where value_coded_name_id IN (524,110,4315,156,3,888,11335,17,2325,4407,10894,9345,838,4355,11333) Group by person_id")
    print("Executed")
    cur1.close()

    #Loads result set - last symptom reported by patient
    for row in cur1:
        if row[2] == "None" or row[2] =='None' or row[2] is None:
            continue
        if (row[0]  in patient_ids):
            if row[2] == "Cough of any duration":
                patients[row[0]].last_symptom = "Cough"
            else:
                patients[row[0]].last_symptom = row[2]
            patients[row[0]].set_sympt_class() #Indexing - Binaryzing classes for ROC scores later on


    print("Loaded data")
    print()

    # Adding temporal aspect and adding more patient specific  - Sex, Age, Last Drug, num of symptoms in prev month
    # If last reported symptom is longer than a specified number of days(eg. 30) then change result to No symptom
    # Reason: Too long after to be helpful i.e predicting that someone will eventually report a skin rash is not as helpful
    # as predicting a skin rash next month.
    print("Executing temporal query...")
    print("SQL script: select * from ML4H_symptom_resultset")
    print("Retrieving patient demographic info and changing last symptom report to "
          "No symptom if last reported symptom is longer than 40 days..")
    cur4 = temporal_symptoms_ML.query_for_40_day_prev_symptoms_FROM_TABLE(conn)
    for row in cur4:
        if  (row[0] in patient_ids):
            if int(row[3]) < 41 and patients[row[0]].last_symptom != "None"  :
                patients[row[0]].last_symptom = "No symptoms"
                patients[row[0]].set_sympt_class()  # Reindexing - Binaryzing classes for ROC scores later on
            patients[row[0]].feature_symptom_array[ Patient.features.index("Age") ] = int( row[5].year )
            patients[row[0]].feature_symptom_array[ Patient.features.index("Last Drug") ] = int( row[6] )
            patients[row[0]].feature_symptom_array[ Patient.features.index("Tot Prev Month Symptoms") ] = int( row[7] )

    print("Executed.")

    conn.close()

    #Load all data into arrays for ML application
    ml4hX = []
    ml4hY = []
    ml4hY_multiclass = []
    for id in patient_ids:
        if patients[id].check_if_null_features(): #If any features null then Machine learning can't handle
            continue
        if patients[id].last_symptom == "None":
            continue
        if patients[id].feature_symptom_array[ Patient.features.index("Age") ] == 0: #Means age is incorrect
            continue
        ml4hX.append( patients[id].feature_symptom_array )
        ml4hY.append(patients[id].last_symptom)
        ml4hY_multiclass.append(patients[id].last_symptom_class) #Numeric verson of result set

    print()
    print("Lenght of Feature array: " + str(len(ml4hX)))
    print("Lenght of ResultSet array: "  + str(len(ml4hY)))


    # Opening significance file for writing
    variable_file = open("symptom_variable_significance.csv",'w')
    for symptom in Patient.features:
        variable_file.write("," + symptom)
    variable_file.write("\n")


    print("Orig")
    check_symptom_result_distribution(ml4hY)
    print()
    X_train1, X_validation1, Y_train1, Y_validation1 = model_selection.train_test_split(ml4hX, ml4hY_multiclass,
                                                                                        test_size=0.3,
                                                                                        random_state=7)
    print("Y_train")
    check_symptom_result_distribution(Y_train1)
    print("Y_Validation")
    check_symptom_result_distribution(Y_validation1)
    kfold2 = model_selection.KFold(n_splits=10, random_state=7)

    # Trying different samplers
    X_cha, Y_cha = NearMiss(ratio=0.025).fit_sample(X_train1, Y_train1)
    check_symptom_result_distribution(Y_cha)
    X_fitted_higher, Y_fitted_higher = RandomOverSampler().fit_sample(X_cha, Y_cha)
    # X_diff, Y_diff = ADASYN().fit(X_cha,Y_cha)
    # check_symptom_result_distribution(Y_diff)
    apply_machine_learning_techniques(X_fitted_higher, Y_fitted_higher, "Adjusted",X_validation1, Y_validation1)

    variable_file.close()
    samplers = [
        # ["RandomUnderSampler_0.6", RandomUnderSampler()],
        ["NearMiss_0.025", NearMiss(ratio=1.2)],
        #[ "RandomOver_0.3",  RandomOverSampler() ],
        ["CondensedNearestNeighbour0.3", CondensedNearestNeighbour(ratio=0.3)],
        #["RepeatedEditedNearestNeighbours0.2",RepeatedEditedNearestNeighbours(ratio=0.2)],
        #["ALLKNN_0.4",AllKNN(ratio=0.005)],
        ["TomekLinks_0.3", TomekLinks(ratio=0.005)]
    ]

    for sampler in samplers:
        print(sampler[0])
        # X_resamp, Y_resamp = sampler[1].fit_sample(ml4hX, ml4hY_multiclass)
        X_resamp, Y_resamp = sampler[1].fit_sample(X_fitted_higher, Y_fitted_higher)
        #check_symptom_result_distribution(Y_resamp)
        # print()
        apply_machine_learning_techniques(X_resamp, Y_resamp, sampler[0])
        print("............")

    print("Done sampling.")
    nearmiss = NearMiss(ratio=0.03)
    X_resampled_nm, y_resampled_nm = nearmiss.fit_sample(ml4hX, ml4hY_multiclass)
    # check_symptom_result_distribution(y_resampled_nm)
    # print()

    print("RandomOVer")
    randomOVer = RandomOverSampler()
    X_resampled_ranO, y_resampled_ranO = randomOVer.fit_sample(X_resampled_nm, y_resampled_nm)
    check_symptom_result_distribution(y_resampled_ranO)
    print()

    print("NEAR MISS ADJUSTED OVERSAMPLE")
    apply_machine_learning_techniques(X_resampled_ranO, y_resampled_ranO, "NM_ADJ_Over")

    print("feature distribution")
    for j in range(len(sum_of_features)):
        print(Patient.features[j] + " : " + str(sum_of_features[j]))

    print()
    print("result class distribution")

    check_symptom_result_distribution(ml4hY)
    variable_file.close()


#Apply machine learning techniques to the sample set
def apply_machine_learning_techniques( X_train1, Y_train1, balance_name, X_validation1, Y_validation1):
    # Write to file
    f = open(balance_name + '_symptoms.csv', 'w')
    f.write(balance_name + "\r\n")
    f.write( check_symptom_result_distribution(Y_train1) )
    global classifier
    print("Applying ML Techniques...")
    validation_size1 = 0.5
    seed1 = 7

    # X_train1, X_validation1, Y_train1, Y_validation1 = model_selection.train_test_split(X, Y,
    #                                                                                     test_size=validation_size1,
    #                                                                                     random_state=seed1)
    seed = 7
    scoring = 'accuracy'
    # scoring = 'roc_auc'
    # Add models to apply
    models1 = []
    models1.append(('Logistic Regression', LogisticRegression()))
    #models1.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
    models1.append(('K Neighbours Classifier', KNeighborsClassifier()))
    models1.append(('Decision Tree Classifier', DecisionTreeClassifier()))
    models1.append(('Gaussian NB', GaussianNB()))
    models1.append(('Random Forrest', RandomForestClassifier()))
    models1.append(('MLPClassifier', MLPClassifier()))
    models1.append(('AdaBoostClassifier', AdaBoostClassifier()))
    # models1.append(('GaussianProcessClassifier', GaussianProcessClassifier()))
    models1.append(('Support Vector Machine', SVC()))
    # evaluate each model in turn
    results1 = []
    names1 = []
    for name1, model1 in models1:
        if name1 == 'Random Forrest': #Get ROC value
            apply_model_with_ROC(X_train1, Y_train1, model1, f, True,X_validation1, Y_validation1)
        else:
            apply_model_with_ROC(X_train1, Y_train1, model1, f, False,X_validation1, Y_validation1)

        #Do cross-validation for each technique
        kfold1 = model_selection.KFold(n_splits=10, random_state=seed1)
        cv_results1 = model_selection.cross_val_score(model1, X_train1, Y_train1, cv=kfold1, scoring=scoring)
        results1.append(cv_results1)
        names1.append(name1)
        msg = "%s: %f (%f)" % (name1, cv_results1.mean(), cv_results1.std())
        model1.fit(X_train1, Y_train1)
        classifier = OneVsRestClassifier(model1)
        classifier.fit(X_train1, Y_train1)
        print(msg)
        f.write(name1+"," + str( accuracy_score(Y_validation1, model1.predict(X_validation1)) ) + "\n")
        f.write("\r\n")
        print()
        if name1 == 'Random Forrest':  # Get ROC value:
            print(model1.feature_importances_)
    f.close()


#Checks the balance of the resulting classes
def check_symptom_result_distribution(Y):
    string_repr = ""
    sum_of_result_classes1 = []
    for j in range(len(Patient.sql_symptoms)):
        sum_of_result_classes1.append(0)

    for y in Y:
        if isinstance( y ,int ):
            sum_of_result_classes1[y] += 1
        elif type(y) == int:
            sum_of_result_classes1[y] += 1
        else:
            if y not in Patient.sql_symptoms:
                sum_of_result_classes1[y] += 1
            else:
                sum_of_result_classes1[Patient.sql_symptoms.index(y)] += 1

    for result in range(len(Patient.sql_symptoms)):
        string_repr += Patient.sql_symptoms[result] + " , " + str( sum_of_result_classes1[result] ) +"\n"
        print( Patient.sql_symptoms[result] + " : " + str( sum_of_result_classes1[result] ) )

    string_repr += "\r\n"

    return string_repr

#Apply to get ROC
def apply_model_with_ROC( X_train, y_train, model2, file, if_rand_forest, X_test, y_test_orig):
    global classifier
    symptom_result_classes = []
    for n in range( len(Patient.sql_symptoms) ):
        symptom_result_classes.append(n)
    yRoc = label_binarize(y_train, classes=symptom_result_classes)
    y_test = label_binarize(y_test_orig, classes=symptom_result_classes)

    # print(yRoc)
    n_classes = yRoc.shape[1]
    ml4hX_multiclass = numpy.array(X_train, numpy.int64)
    ml4hX_multiclass_test = numpy.array(X_test, numpy.int64)

    # shuffle and split training and test sets
    # X_train, X_test, y_train, y_test = train_test_split(ml4hX_multiclass, yRoc, test_size=0.5,
    #                                                     random_state=0)
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(model2)
    kfold1 = model_selection.KFold(n_splits=10, random_state=8)
    cv_results1 = model_selection.cross_val_score(classifier, X_train, y_train, cv=kfold1, scoring='accuracy')
    print("CVV: " + str(cv_results1.mean()))

    classifier.fit(X_train, yRoc)

    y_score = classifier.predict(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    ave = 0
    ave_accuracy = 0
    if if_rand_forest :
        variable_file.write( str(file.name) + "\n" )
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ave+=roc_auc[i]
        accuracy_meas = accuracy_score( y_test[:, i], y_score[:, i] )
        ave_accuracy += accuracy_meas
        print( Patient.sql_symptoms[i] + " : " +  str(roc_auc[i]) + " , " + str(accuracy_meas) )
        if file is not None:
            file.write(Patient.sql_symptoms[i] + "," + str(roc_auc[i]) + "," + str(accuracy_meas) + "\n")
        if if_rand_forest:
            variable_file.write( Patient.sql_symptoms[i] )
            for significance in classifier.estimators_[i].feature_importances_:
                variable_file.write("," + str(significance))

            variable_file.write("\n")

    if if_rand_forest:
        variable_file.write("\n")

    # Compute micro-average ROC curve and ROC area
    fpr["samples"], tpr["samples"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["samples"] = auc(fpr["samples"], tpr["samples"])
    print("ROC: " + str(roc_auc["samples"]) + " avg : " + str(ave/n_classes))
    if file is not None:
        file.write("ROC," + str(roc_auc["samples"]) + "\n")
        file.write("Avg ROC ," + str(ave/n_classes) + "\n")
        file.write("Avg Pred ," + str(ave_accuracy/n_classes) + "\n")



main()