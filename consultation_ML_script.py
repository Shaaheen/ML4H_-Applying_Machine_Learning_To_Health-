# Machine Learning script to predict whether a patient is likely to miss their next scheduled medical consultation
# Application - Can set aside more resources if likely to miss next visit - Send SMSs, reminders, visit home, etc

import numpy
import pymysql
from imblearn.metrics import sensitivity_specificity_support
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.under_sampling import InstanceHardnessThreshold
from imblearn.under_sampling import NearMiss
from imblearn.under_sampling import NeighbourhoodCleaningRule
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from numpy import *
from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Consultation_Patient import Consultation_Patient

# To binarize the occupation feature
occupations = []

def main():
    print("Connecting to ML4H DB..")

    conn = pymysql.connect(host='nightmare.cs.uct.ac.za', port=3306, user='ochomo001', passwd='oesaerex',
                           db='ochomo001')

    print("Connected")

    cur = conn.cursor()
    print("Executing SQL query..")
    print("SQL script: select person_id,tot_consultations_attended,tot_consultations_missed, gender, age, city_village, profession, last_consultation_date, last_consultation_attendance,missed_last_appointment from ML4H_consultation_defaulter_sets")
    print("Retrieving all consultation features...")
    cur.execute(
        "select person_id,tot_consultations_attended,tot_consultations_missed, gender, age, city_village, profession, last_consultation_date, last_consultation_attendance,missed_last_appointment from ML4H_consultation_defaulter_sets")
    print("Executed.")

    patient_ids = []
    patients = {}
    consultation_features = []
    consultation_results = []
    occupations = []
    locations = []
    #["Consultations_Attended", "Consultations_Missed", "Sex", "Age", "Occupation", "Location","Day_of_week","Last_appointment" ]

    #load in all the consultation features into each patient and the result class
    for row in cur:
        if row[0] not in patient_ids:
            patient_ids.append(row[0])
            patients[row[0]] = (Consultation_Patient(int(row[0])))

            patients[row[0]].features[0] = int(row[1])
            patients[row[0]].features[1] = int(row[2])
            if row[3] == "M":
                patients[row[0]].features[2] = 0
            elif row[3] == "F":
                patients[row[0]].features[2] = 1

            patients[row[0]].features[3] = int(row[4])

            patients[row[0]].features[5] = get_feature_index(row[5], locations)

            patients[row[0]].features[4] = get_feature_index(row[6],occupations)

            patients[row[0]].features[6] = row[7].weekday()

            patients[row[0]].features[7] = int(row[8])

            consultation_features.append(patients[row[0]].features)
            consultation_results.append(row[9])

    cur.close()

    conn.close()

    print( len( consultation_results ) )
    print( len( consultation_features ) )

    #Splot sets up, keep a hold out set
    X_train1, X_validation1, Y_train1, Y_validation1 = model_selection.train_test_split(consultation_features, consultation_results,
                                                                                        test_size=0.3,
                                                                                        random_state=7)

    print("Y_train")
    check_result_distr(Y_train1)
    print("Y_val")
    check_result_distr(Y_validation1)
    # DATA IS IMBALANCED
    # Trying to balance data appropriately - Using multiple sampler tools to see which is best
    samplers = [
                ['ALLKNN', AllKNN()],
                ['NearMiss', NearMiss()],
                ['CondensedNearestNeighbour', CondensedNearestNeighbour()],
                ['TomekLinks', TomekLinks()],
                ['NeighbourhoodCleaningRule', NeighbourhoodCleaningRule()],
                ['InstanceHardnessThreshold', InstanceHardnessThreshold()],
                ['RandomUnderSampler', RandomUnderSampler()]
                ]


    #Write results of AllKNN results(best sampler) to file
    f1 = open('consultation_technique_comparison.csv','w')

    X_resamp, Y_resamp = samplers[0][1].fit_sample(X_train1, Y_train1)
    X_resamp_orig, Y_resamp_orig = samplers[0][1].fit_sample(consultation_features, consultation_results)
    results_final = apply_machine_learning_techniques(X_resamp_orig, Y_resamp_orig, X_resamp, Y_resamp, X_validation1, Y_validation1)
    f1.write(',Logistic Regression' +", " 'K Neighbours Classifier' + "," + 'Decision Tree Classifier' + "," +  'Gaussian NB'+ "," +  'Random Forrest' + "," + 'MLPClassifier' + "," + 'AdaBoostClassifier'+ "," +  'Support Vector Machine')
    f1.write("\n")
    f1.write("Roc " + results_final[0] + "\n")
    f1.write("Sensitivity " + results_final[1] + "\n")
    f1.write("Specificity " + results_final[2] +"\n")
    f1.write("Unseen Roc " + results_final[3] )

    f1.close()


    f = open('consultation_balance_comparison.csv', 'w')
    f.write("Sampler,Attended ,Missed ," + 'Logistic Regression' +", " 'K Neighbours Classifier' + "," + 'Decision Tree Classifier' + "," +  'Gaussian NB'+ "," +  'Random Forrest' + "," + 'MLPClassifier' + "," + 'AdaBoostClassifier'+ "," +  'Support Vector Machine')
    f.write("\n")
    orig_distribution = check_result_distr(Y_train1)
    orig_results = apply_machine_learning_techniques(X_train1, Y_train1, X_validation1, Y_validation1)[0]
    f.write("Orig" + "," + orig_distribution + orig_results + "\n")

    for sampler in samplers:
        print(sampler[0])
        X_resamp, Y_resamp = sampler[1].fit_sample(X_train1, Y_train1)
        distribution = check_result_distr(Y_resamp)
        results = apply_machine_learning_techniques(X_resamp, Y_resamp, X_validation1, Y_validation1)[0]

        f.write(sampler[0] + "," +  distribution + results + "\n")

    f.close()


#For binaryzing categorical location and occupation features
def get_feature_index(feature, feature_categories):
    if feature not in feature_categories:
        feature_categories.append(feature)
    return feature_categories.index(feature)


# Checks distribution of result class
def check_result_distr(Y):
    count_y = 0
    count_n = 0
    for b in Y:
        if b=='0' or b==0:
            count_y += 1
        elif b=='1' or b==1:
            count_n += 1

    print("Attended Last Consultation : " + str(count_y))
    print("Missed Last Consultation : " + str(count_n))
    return (str(count_y) + ',' + str(count_n))


def apply_machine_learning_techniques(X_orig, Y_orig, X_train1, Y_train1, X_validation1, Y_validation1):
    print("Applying ML Techniques...")
    validation_size1 = 0.50
    seed1 = 7

    seed = 7
    scoring = 'accuracy'
    scoring = 'roc_auc'
    string_roc_results = ""
    string_sensitivity_results = ""
    string_specifity_results = ""
    string_unseen_roc = ""
    models1 = []

    X_train1, X_validation1, Y_train1, Y_validation1 = model_selection.train_test_split(X_orig, Y_orig,
                                                                                        test_size=0.2,
                                                                                        random_state=7)

    models1.append(('Logistic Regression', LogisticRegression()))
    models1.append(('K Neighbours Classifier', KNeighborsClassifier()))
    models1.append(('Decision Tree Classifier', DecisionTreeClassifier()))
    models1.append(('Gaussian NB', GaussianNB()))
    models1.append(('Random Forrest', RandomForestClassifier()))
    models1.append(('MLPClassifier', MLPClassifier()))
    models1.append(('AdaBoostClassifier', AdaBoostClassifier()))
    models1.append(('Support Vector Machine', SVC() ))
    # evaluate each model in turn
    results1 = []
    names1 = []
    for name1, model1 in models1:
        kfold1 = model_selection.KFold(n_splits=10, random_state=seed1)
        check_result_distr(Y_train1)
        print(len(numpy.unique(Y_train1)))
        #Cross validation evaluation
        cv_results1 = model_selection.cross_val_score(model1, X_train1, Y_train1, cv=kfold1, scoring='roc_auc')
        cv_results2 = model_selection.cross_val_score(model1, X_train1, Y_train1, cv=kfold1, scoring='accuracy')

        results1.append(cv_results1)

        names1.append(name1)
        msg = "%s: %f (%f)" % (name1, cv_results2.mean(), cv_results2.std())
        model1.fit(X_train1, Y_train1)
        if name1 == 'Random Forrest':  # Get variable significance
            print(model1.feature_importances_)
        elif name1 == 'AdaBoostClassifier':  # Get variable significance
            print(model1.feature_importances_)
        elif name1 == 'Support Vector Machine':  # Get variable significance
            # print(model1.coef_)
            print()
        elif name1 == 'Decision Tree Classifier':  # Get variable significance
            print(model1.feature_importances_)
            sum_float = 0
            for float_val in model1.feature_importances_:
                sum_float+=float_val

            for importance in model1.feature_importances_:
                print( str( (importance/sum_float) )  )
        elif name1 == 'Logistic Regression':  # Get variable significance
            coefficients = (numpy.std(X_train1, 0) * model1.coef_)[0]
            print(coefficients)
            sum_float = 0
            for float_val in coefficients:
                sum_float += abs(float_val)

            for importance in coefficients:
                print(str((abs(importance) / sum_float)))

        #Print out all metrics
        print(msg)
        print("Roc : " + str(cv_results1.mean() ))
        print("Unseen Roc : " + str(roc_auc_score(Y_validation1, model1.predict(X_validation1))))
        string_roc_results+="," + str(cv_results1.mean() )
        print(classification_report(Y_validation1, model1.predict(X_validation1)))
        confusion_matr = confusion_matrix(Y_validation1, model1.predict(X_validation1), labels=[True, False])
        print("      True   | False")
        print("True  " + str(confusion_matr[0][0]) + "  " + str(confusion_matr[0][1]))
        print("False  " + str(confusion_matr[1][0]) + "  " + str(confusion_matr[1][1]))
        print(sensitivity_specificity_support(Y_validation1, model1.predict(X_validation1), average='macro'))

        #Calculate all relevant metrics
        tn, fp, fn, tp = confusion_matrix(Y_validation1, model1.predict(X_validation1)).ravel()
        specificity = int(tn.T) / (int(tn.T) + int(fp.T))
        sensitivity = int(tp.T) / (int(tp.T) + int(fn.T))
        print(str(specificity) + " , " + str(sensitivity) )
        string_sensitivity_results += "," + str(sensitivity)
        string_specifity_results += "," + str(specificity)
        string_unseen_roc += "," + str(roc_auc_score(Y_validation1, model1.predict(X_validation1) ))

    print()
    return [string_roc_results,string_specifity_results,string_sensitivity_results,string_unseen_roc]


main()
