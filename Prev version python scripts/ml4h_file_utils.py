import pandas as pd

def classifaction_report_csv(report, classifier_name):
    print("exporting classification report..")
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-1]:
        if (len(line) <2): continue
        row = {}
        row_data = line.split('      ')
        start_index = 0
        for i in range(len(row_data)):
            if len(row_data[i]) >1:
                start_index = i
                break
        row['class'] = row_data[start_index]
        row['precision'] = float(row_data[start_index+1])
        row['recall'] = float(row_data[start_index+2])
        row['f1_score'] = float(row_data[start_index+3])
        row['support'] = float(row_data[start_index+4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(classifier_name + 'classification_report.csv', index = False)
    print("Done exporting.")


