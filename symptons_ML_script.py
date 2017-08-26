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
from imblearn.under_sampling import NearMiss
from numpy import *
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
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
from Patient import Patient
from ml4h_file_utils import classifaction_report_csv

def main():
    print("Connecting to ML4H DB..")

    conn = pymysql.connect(host='nightmare.cs.uct.ac.za', port=3306, user='ochomo001', passwd='oesaerex', db='ochomo001')

    print("Connected")

    cur = conn.cursor()
    print("Executing SQL query..")
    #Query to get the totals of all the symptoms that a patient has reported to the clinic
    #Used as features to predict the next possible that a patient will report
    cur.execute("select person_id, count(*) AS Tot_Num_Sympt, sum(case when value_coded_name_id =110 then 1 else 0 end) AS Cough, sum(case when value_coded_name_id = 4315 then 1 else 0 end) AS Fever, sum(case when value_coded_name_id = 156 then 1 else 0 end) AS Abdominal_pain, sum(case when value_coded_name_id = 524 then 1 else 0 end) AS skin_rash,sum(case when value_coded_name_id=1576 then 1 else 0 end) AS Lactic_acisdosis, sum(case when value_coded_name_id=2325 then 1 else 0 end) AS Lipodystrophy,sum(case when value_coded_name_id=3 then 1 else 0 end) AS Anemia,sum(case when value_coded_name_id=888 then 1 else 0 end) AS Anorexia,sum(case when value_coded_name_id=11335 then 1 else 0 end) AS Cough_any_duration,sum(case when value_coded_name_id=17 then 1 else 0 end) AS Diarrhea,sum(case when value_coded_name_id=30 then 1 else 0 end) AS Hepatitis,sum(case when value_coded_name_id=226 then 1 else 0 end) AS Jaundice,sum(case when value_coded_name_id=10894 then 1 else 0 end) AS Leg_pain,sum(case when value_coded_name_id=4407 then 1 else 0 end) AS Night_Sweats,sum(case when value_coded_name_id=9345 then 1 else 0 end) AS Other,sum(case when value_coded_name_id=838 then 1 else 0 end) AS Peripheral_neuropathy,sum(case when value_coded_name_id=4355 then 1 else 0 end) AS Vomiting,sum(case when value_coded_name_id=11333 then 1 else 0 end) AS Weight_loss,min(obs_datetime), max(obs_datetime) from Obs_artvisit_sympt a inner join concept_name n on a.value_coded_name_id=n.concept_name_id group by person_id")
    print("Executed")
    cur.close()

    #Loading features
    patient_ids = []
    sum_of_features = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    patients = {}
    print("loading in data into program...")
    for row in cur:
        if (row[0] not in patient_ids):
            patient_ids.append(row[0])
            patients[row[0]] = (Patient(int(row[0])))
        patients[row[0]].feature_symptom_array = [ int(row[2]) + int(row[10]),int(row[3]),int(row[4]),int(row[5]),int(row[6]),
                                                   int(row[7]), int(row[8]), int(row[9]), int(row[11]),
                                                   int(row[12]), int(row[13]), int(row[14]), int(row[15]),int(row[16]),
                                                   int(row[17]), int(row[18]),int(row[19]) ]
        #To check the balance of features
        for i in range(len(sum_of_features)+1):
            if i == 8:
                sum_of_features[0] += int(row[ i + 2])
            elif i > 8:
                sum_of_features[i-1]+= int(row[i + 2])
            else:
                sum_of_features[i] += int(row[i + 2])

    cur1 = conn.cursor()
    print("Executing SQL query..")
    #Gets the last reported symptom that a patient reported - RESULT CLASS (Trying to predict this)
    cur1.execute("select person_id, value_coded_name_id,name, MAX(obs_datetime) from Obs_artvisit_sympt a inner join concept_name n on a.value_coded_name_id = n.concept_name_id where value_coded_name_id IN (524,110,4315,156,3,888,11335,17,2325,4407,10894,9345,838,4355,11333) Group by person_id")
    print("Executed")
    cur1.close()

    #Loads result set - last symptom reported by patient
    for row in cur1:
        if row[2] == "None" or row[2] =='None' or row[2] is None:
            continue
        if (row[0] not in patient_ids):
            patient_ids.append(row[0])
            patients[row[0]] = (Patient(int(row[0])))
        if row[2] == "Cough of any duration":
            patients[row[0]].last_symptom = "Cough"
        else:
            patients[row[0]].last_symptom = row[2]
        patients[row[0]].set_sympt_class() #Indexing - Binaryzing classes for ROC scores later on


    conn.close()

    print("Loaded data")
    print()
    print("Patient Ids:")
    print(patient_ids)
    print("Patient objects")
    print(patients)

    #Load all data into array for ML application
    ml4hX =[]
    ml4hY = []
    ml4hY_multiclass = []
    for id in patient_ids:
        if patients[id].check_if_null_features():
            continue
        if patients[id].last_symptom == "None":
            continue
        ml4hX.append( patients[id].feature_symptom_array )
        ml4hY.append(patients[id].last_symptom)
        ml4hY_multiclass.append(patients[id].last_symptom_class)

    print("Feature set:")
    #print(ml4hX)
    print("Result set:")
    #print(ml4hY)

    print("Orig")
    check_symptom_result_distribution(ml4hY)
    print()
    # print("ADASYN")
    # ada = ADASYN( random_state=40)
    # X_resampled, y_resampled = ada.fit_sample(ml4hX, ml4hY_multiclass)
    # check_symptom_result_distribution(y_resampled)

    # print("ALLKNN")
    # allknn = AllKNN()
    # X_resampled_allknn, y_resampled_allknn = allknn.fit_sample(ml4hX, ml4hY)
    # check_symptom_result_distribution(y_resampled_allknn)

    print("NearMiss")
    nearmiss = NearMiss(ratio=0.025)
    X_resampled_nm, y_resampled_nm = nearmiss.fit_sample(ml4hX, ml4hY_multiclass)
    check_symptom_result_distribution(y_resampled_nm)
    print()

    print("RandomOVer")
    randomOVer = RandomOverSampler()
    X_resampled_ranO, y_resampled_ranO = randomOVer.fit_sample(X_resampled_nm, y_resampled_nm)
    check_symptom_result_distribution(y_resampled_ranO)
    print()


    print("SMOTE")
    sme = SMOTE(ratio=0.5)
    X_res, y_res = sme.fit_sample(ml4hX, ml4hY)
    check_symptom_result_distribution(y_res)
    print()


    # sme = SMOTEENN(random_state=42, k = 3)
    # X_res, y_res = sme.fit_sample(ml4hX, ml4hY)
    # check_symptom_result_distribution(y_res)

    # print("Cluster Centroids")
    # cc = ClusterCentroids(random_state=0)
    # X_resampled_cluster, y_resampled_cluster = cc.fit_sample(ml4hX, ml4hY)
    # check_symptom_result_distribution(y_resampled_cluster)
    # smote = SMOTE(kind='borderline1')
    # X_resampled_smote, y_resampled_smote = smote.fit_sample(ml4hX, ml4hY)
    # check_symptom_result_distribution(y_resampled_smote)

    print()
    print(len(ml4hX))
    print(len(ml4hY))
    c =0
    for t in ml4hY:
        if t == "None":
            c += 1
    print(c)
    print()

    randf = RandomForestClassifier()
    randf.fit(ml4hX,ml4hY)
    print(randf.feature_importances_)
    print("ORIGINAL")
    apply_machine_learning_techniques(ml4hX,ml4hY_multiclass, "Original")
    print("NEAR MISS UNDERSAMPLE 0.025")
    apply_machine_learning_techniques(X_resampled_nm, y_resampled_nm, "NM_Under_0.025" )
    print("NEAR MISS ADJUSTED OVERSAMPLE")
    apply_machine_learning_techniques(X_resampled_ranO, y_resampled_ranO, "NM_ADJ_Over")

    labels_features = ["Cough", "Fever", "Abdominal_pain", "skin_rash", "Lactic_acisdosis", "Lipodystrophy",
                       "Anemia", "Anorexia", "Cough_any_duration", "Diarrhea", "Hepatitis", "Jaundice", "Leg_pain",
                       "Night_Sweats",
                       "None", "Peripheral_neuropathy", "Vomiting", "Weight_loss", "Unknown"]

    print("feature distribution")
    for j in range(len(sum_of_features)):
        print(Patient.symptoms[j] + " : " + str(sum_of_features[j]))

    print()
    print("result class distribution")

    check_symptom_result_distribution(ml4hY)

    #apply_model_with_ROC(ml4hX,ml4hY_multiclass)


def apply_machine_learning_techniques(X,Y,balance_name):
    f = open(balance_name + '_symptoms.csv', 'w')
    f.write(balance_name + "\r\n")
    global classifier
    print("Applying ML Techniques...")
    validation_size1 = 0.2
    seed1 = 7
    ml4hY_multiclass_bin = label_binarize(Y,
                                          classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    # X_train1, X_validation1, Y_train1, Y_validation1 = model_selection.train_test_split(ml4hX, ml4hY, test_size=validation_size1,
    #                                                                                 random_state=seed1)
    X_train1, X_validation1, Y_train1, Y_validation1 = model_selection.train_test_split(X, Y,
                                                                                        test_size=validation_size1,
                                                                                        random_state=seed1)
    seed = 7
    scoring = 'accuracy'
    # scoring = 'roc_auc'
    models1 = []
    models1.append(('Logistic Regression', LogisticRegression()))
    models1.append(('Linear Discriminant Analysis', LinearDiscriminantAnalysis()))
    models1.append(('K Neighbours Classifier', KNeighborsClassifier()))
    models1.append(('Decision Tree Classifier', DecisionTreeClassifier()))
    models1.append(('Gaussian NB', GaussianNB()))
    models1.append(('Random Forrest', RandomForestClassifier()))
    models1.append(('MLPClassifier', MLPClassifier()))
    models1.append(('AdaBoostClassifier', AdaBoostClassifier()))
    # models1.append(('GaussianProcessClassifier', GaussianProcessClassifier()))
    # models1.append(('Support Vector Machine', SVC()))
    # evaluate each model in turn
    results1 = []
    names1 = []
    for name1, model1 in models1:
        apply_model_with_ROC(X,Y,model1,f)
        kfold1 = model_selection.KFold(n_splits=10, random_state=seed1)
        cv_results1 = model_selection.cross_val_score(model1, X_train1, Y_train1, cv=kfold1, scoring=scoring)
        results1.append(cv_results1)
        names1.append(name1)
        msg = "%s: %f (%f)" % (name1, cv_results1.mean(), cv_results1.std())
        model1.fit(X_train1, Y_train1)
        classifier = OneVsRestClassifier(model1)
        classifier.fit(X_train1, Y_train1)
        # roc_auc_score(
        #     Y_validation1, model1.predict(X_validation1)
        # )
        # classifaction_report_csv(classification_report(Y_validation1, model1.predict(X_validation1)),name1)
        print(msg)
        f.write(name1+"," + str( cv_results1.mean() ) + "\n")
        f.write("\r\n")
        print()
    f.close()


#Checks the balance of the resulting classes
def check_symptom_result_distribution(Y):
    # result_set_classes1 = ["Cough", "Fever", "Abdominal pain", "Skin rash",
    #                        "Lactic acidosis", "Lipodystrophy", "Anemia", "Anorexia",
    #                        "Diarrhea", "Hepatitis", "Jaundice", "Leg pain / numbness",
    #                       "Night sweats", "Peripheral neuropathy", "Vomiting", "Weight loss / Failure to thrive / malnutrition",
    #                        "Other symptom"]
    sum_of_result_classes1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
        print( Patient.sql_symptoms[result] + " : " + str( sum_of_result_classes1[result] ) )


def apply_model_with_ROC(X,Y, model2, file):
    global classifier
    yRoc = label_binarize(Y, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    # print(yRoc)
    n_classes = yRoc.shape[1]
    ml4hX_multiclass = numpy.array(X, numpy.int32)
    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(ml4hX_multiclass, yRoc, test_size=.5,
                                                        random_state=0)
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(model2)
    y_score = classifier.fit(X_train, y_train).predict(X_test)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    ave = 0
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        ave+=roc_auc[i]
        print( Patient.sql_symptoms[i] + " : " +  str(roc_auc[i]))
        if file is not None:
            file.write(  Patient.sql_symptoms[i] + "," +  str(roc_auc[i]) + "\n")

    # Compute micro-average ROC curve and ROC area
    fpr["samples"], tpr["samples"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["samples"] = auc(fpr["samples"], tpr["samples"])
    print("ROC: " + str(roc_auc["samples"]) + " avg : " + str(ave/n_classes))
    if file is not None:
        file.write("ROC," + str(roc_auc["samples"]) + "\n")
        file.write("Avg ROC ," + str(ave/n_classes) + "\n")
    #
    # # print(roc_auc_score(ml4hY,knn1.predict(ml4hX)))
    # # print(knn1.predict([230, 40]))
    # # print(X_validation1)
    # # print(predictions1)
    # # print(Y_validation1)
    # # print("h")
    # # print(accuracy_score(Y_validation1, predictions1))
    # # print(confusion_matrix(Y_validation1, predictions1))
    # # print(classification_report(Y_validation1, predictions1))




main()