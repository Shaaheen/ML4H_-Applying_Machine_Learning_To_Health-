# Machine Learning script to predict whether a patient is likely to miss their next scheduled medical consultation
# Application - Can set aside more resources if likely to miss next visit - Send SMSs, reminders, visit home, etc

import matplotlib.pyplot as plt
import numpy
import pandas
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
from imblearn.metrics import sensitivity_specificity_support
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import OneSidedSelection
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import RepeatedEditedNearestNeighbours
from imblearn.under_sampling import TomekLinks
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
# from numpy import np
import pymysql
from joblib import Parallel, delayed
import multiprocessing
from imblearn.over_sampling import ADASYN
from Consultation_Patient import Consultation_Patient
from Patient import Patient
from ml4h_file_utils import classifaction_report_csv

# To binarize the occupation feature
occupations = []


def main():
    print("Connecting to ML4H DB..")

    conn = pymysql.connect(host='nightmare.cs.uct.ac.za', port=3306, user='ochomo001', passwd='oesaerex',
                           db='ochomo001')

    print("Connected")

    cur = conn.cursor()
    print("Executing SQL query..")

    # cur.execute("select person_id, sum(case when value_coded_name_id=1102 then 1 else 0 end) AS consultations_attended,sum(case when value_coded_name_id=1103 then 1 else 0 end) AS consultations_missed from obs where concept_id=1805 group by person_id")
    # Gets patient details - Used as features
    cur.execute(
        "select a.person_id, gender, birthdate ,value,sum(case when value_coded_name_id=1102 then 1 else 0 end) AS consultations_attended,sum(case when value_coded_name_id=1103 then 1 else 0 end) AS consultations_missed,city_village from obs a inner join person b on a.person_id = b.person_id inner join person_attribute c on c.person_id=a.person_id inner join person_address d on a.person_id = d.person_id where concept_id=1805 AND birthdate IS NOT NULL AND city_village IS NOT NULL group by person_id;")
    print("Executed.")

    patient_ids = []
    patients = {}
    consultation_features = []
    consultation_results = []
    occupations = []
    locations = []
    for row in cur:
        if row[2] is None:
            continue
        if row[0] not in patient_ids:
            patient_ids.append(row[0])
            patients[row[0]] = (Consultation_Patient(int(row[0])))
        patients[row[0]].consultations_attended = int(row[4])
        patients[row[0]].consultations_missed = int(row[5])

        patients[row[0]].features[0] = int(row[4])
        patients[row[0]].features[1] = int(row[5])
        if row[1] == "M":
            patients[row[0]].features[2] = 0
        elif row[1] == "F":
            patients[row[0]].features[2] = 1

        patients[row[0]].features[3] = row[2].year

        patients[row[0]].features[4] = get_feature_index(row[3],occupations)

        patients[row[0]].features[5] = get_feature_index(row[6], locations)


    cur.close()

    cur1 = conn.cursor()
    print("Executing SQL query..")
    # Gets if patient missed last consultation - RESULT CLASS
    cur1.execute(
        "select person_id, value_coded_name_id, max(obs_datetime) from obs where concept_id=1805 group by person_id")
    print("Executed.")
    print()
    for row in cur1:
        if row[0] in patient_ids:
            if int(row[1]) == 1102:
                patients[row[0]].attended_last_consultation = True
            elif int(row[1]) == 1103:
                patients[row[0]].attended_last_consultation = False

            consultation_results.append(patients[row[0]].attended_last_consultation)

    cur1.close()
    check_result_distr(consultation_results)
    # consultation_features_resa= numpy.array(consultation_features, numpy.int32)

    conn.close()

    for id in patient_ids:
        if patients[id].attended_last_consultation == True:
            patients[id].consultations_attended -= 1
            patients[id].features[0] -= 1
        elif patients[id].attended_last_consultation == False:
            patients[id].features[1] -= 1
            patients[id].consultations_missed -= 1
        else:
            print("WTF")
        consultation_features.append(patients[id].features)

    print( len( consultation_results ) )
    print( len( consultation_features ) )

    # DATA IS IMBALANCED
    # Trying to balance data appropriately - Using multiple sampler tools to see which is best
    samplers = [['ADASYN', ADASYN(ratio=0.06)],
                ['SMOTE', SMOTE(ratio=0.06)],
                ['SMOTEENN', SMOTEENN(random_state=22)],
                ['SMOTETomek', SMOTETomek(ratio=0.06, random_state=22)],
                ['ALLKNN', AllKNN()],
                ['NearMiss', NearMiss(ratio=0.2)],
                ['NearMiss', NearMiss(ratio=0.01)],
                ['CondensedNearestNeighbour', CondensedNearestNeighbour()],
                ['TomekLinks', TomekLinks()],
                ['NeighbourhoodCleaningRule', NeighbourhoodCleaningRule()],
                ['InstanceHardnessThreshold', InstanceHardnessThreshold(ratio=0.2)],
                ['RandomUnderSampler', RandomUnderSampler(ratio=0.2)]
                ]

    # for sampler in samplers:
    #     print(sampler[0])
    #     X_resamp, Y_resamp = sampler[1].fit_sample(consultation_features, consultation_results)
    #     check_result_distr(Y_resamp)
    #
    #     apply_machine_learning_techniques(X_resamp, Y_resamp)

    X_resamp1, Y_resamp1 = NearMiss(ratio=0.01).fit_sample(consultation_features, consultation_results)
    check_result_distr(Y_resamp1)
    X_resamp2, Y_resamp2 = ADASYN(ratio=0.02).fit_sample(X_resamp1, Y_resamp1)
    check_result_distr(Y_resamp2)
    apply_machine_learning_techniques(X_resamp2, Y_resamp2)
    model_random_forest = RandomForestClassifier()
    model_random_forest.fit(X_resamp2, Y_resamp2)
    print(model_random_forest.feature_importances_)

    print()
    X_resamp3, Y_resamp3 = ADASYN(ratio=0.05).fit_sample(X_resamp1, Y_resamp1)
    check_result_distr(Y_resamp3)
    apply_machine_learning_techniques(X_resamp3, Y_resamp3)

    X_resamp4, Y_resamp4 = NearMiss(ratio=0.03).fit_sample(consultation_features, consultation_results)
    check_result_distr(Y_resamp4)
    X_resamp5, Y_resamp5 = ADASYN(ratio=0.065).fit_sample(X_resamp4, Y_resamp4)
    check_result_distr(Y_resamp5)
    apply_machine_learning_techniques(X_resamp5, Y_resamp5)

    X_resamp4, Y_resamp4 = NearMiss(ratio=0.02).fit_sample(consultation_features, consultation_results)
    check_result_distr(Y_resamp4)
    X_resamp5, Y_resamp5 = ADASYN().fit_sample(X_resamp4, Y_resamp4)
    check_result_distr(Y_resamp5)
    apply_machine_learning_techniques(X_resamp5, Y_resamp5)

    X_resamp4, Y_resamp4 = NearMiss(ratio=0.3).fit_sample(consultation_features, consultation_results)
    check_result_distr(Y_resamp4)
    X_resamp5, Y_resamp5 = ADASYN().fit_sample(X_resamp4, Y_resamp4)
    check_result_distr(Y_resamp5)
    apply_machine_learning_techniques(X_resamp5, Y_resamp5)


def get_feature_index(feature, feature_categories):
    if feature not in feature_categories:
        feature_categories.append(feature)
    return feature_categories.index(feature)


# Checks distribution of result class
def check_result_distr(Y):
    count_y = 0
    count_n = 0
    for b in Y:
        if b:
            count_y += 1
        elif not b:
            count_n += 1

    print("Attended Last Consultation : " + str(count_y))
    print("Missed Last Consultation : " + str(count_n))


def apply_machine_learning_techniques(X_resamp, Y_resamp):
    print("Applying ML Techniques...")
    validation_size1 = 0.50
    seed1 = 7
    X_train1, X_validation1, Y_train1, Y_validation1 = model_selection.train_test_split(X_resamp, Y_resamp,
                                                                                        test_size=validation_size1,
                                                                                        random_state=seed1)
    seed = 7
    scoring = 'accuracy'
    scoring = 'roc_auc'
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
        kfold1 = model_selection.KFold(n_splits=10, random_state=seed1)
        cv_results1 = model_selection.cross_val_score(model1, X_train1, Y_train1, cv=kfold1, scoring=scoring)

        results1.append(cv_results1)
        names1.append(name1)
        msg = "%s: %f (%f)" % (name1, cv_results1.mean(), cv_results1.std())
        model1.fit(X_train1, Y_train1)
        # classifaction_report_csv(classification_report(Y_validation1, model1.predict(X_validation1)),name1)
        print(msg)
        print("Roc : " + str(roc_auc_score(Y_validation1, model1.predict(X_validation1))))
        print(classification_report(Y_validation1, model1.predict(X_validation1)))
        confusion_matr = confusion_matrix(Y_validation1, model1.predict(X_validation1), labels=[True, False])
        print("      True   | False")
        print("True  " + str(confusion_matr[0][0]) + "  " + str(confusion_matr[0][1]))
        print("False  " + str(confusion_matr[1][0]) + "  " + str(confusion_matr[1][1]))
        # print(sensitivity_specificity_support(Y_validation1, model1.predict(X_validation1), average='macro'))
    print()


main()
