import datetime


class Patient:
    features = ["Cough", "Fever", "Abdominal_pain", "skin_rash",
                "Lactic_acisdosis", "Lipodystrophy", "Anemia", "Anorexia",
                "Diarrhea", "Leg_pain", "Night_Sweats", "Other",
                "Peripheral_neuropathy", "Vomiting", "Weight_loss", "Sex",
                "Age", "Last Drug", "Tot Prev Month Symptoms"]

    temporal_symptoms = ["Cough", "Fever", "Abdominal_pain", "skin_rash",
                "Lactic_acisdosis", "Lipodystrophy", "Anemia", "Anorexia",
                "Diarrhea", "Hepatitis", "Jaundice", "Leg_pain",
                "Night_Sweats", "Other", "Peripheral_neuropathy", "Vomiting",
                "Weight_loss", "Sex", "Birthdate", "Last Drug", "Symptoms_prev_month"]

    sql_symptoms = ["Cough", "Fever", "Abdominal pain", "Skin rash",
                     "Lipodystrophy", "Anemia", "Anorexia", "Diarrhea",
                    "Leg pain / numbness", "Night sweats", "Peripheral neuropathy", "Vomiting",
                    "Weight loss / Failure to thrive / malnutrition", "Other symptom", "No symptoms"]

    def __init__(self, nam):
        self.nam = nam
        self.num_encounters = 0
        self.num_adverse_effects = 0
        self.adverse = "Not Adverse"
        self.secondary = 0
        self.feature_symptom_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.feature_temporal_sympt_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cough = 0
        self.fever = 0
        self.abdom = 0
        self.rash = 0
        self.lactic = 0
        self.lipodystrophy = 0
        self.anemia = 0
        self.anorexia = 0
        self.cough_any = 0
        self.diarrhea = 0
        self.hepatitis = 0
        self.jaundice = 0
        self.leg_pain = 0
        self.night_sweats = 0
        self.other = 0
        self.peripheral_neuropath = 0
        self.vomiting = 0
        self.weight_loss = 0
        self.unknown = 0
        self.last_symptom = "None"
        self.last_symptom_class = 15
        self.latest_datetime = datetime.datetime(1900, 3, 13, 15, 10, 20)  # placeholder
        self.second_latest_datetime = datetime.datetime(1899, 3, 13, 15, 10, 20)  # placeholder
        self.time_between_symptom_reports = None
        self.continued_symptoms = None
        self.symptoms_in_prev_month = 0

    def __str__(self):
        return str("cough : " + str(self.cough) + " skin rash : " + str(self.rash))

    def __repr__(self):
        # return str("enc: " + str(self.num_encounters)+" adverse: " + str(self.num_adverse_effects) + " SE: " + str(self.adverse))
        stri = ""
        return str("cough : " + str(self.cough) + " skin rash : " + str(self.rash))

    def set_sympt_class(self):

        self.last_symptom_class = Patient.sql_symptoms.index(self.last_symptom)
        # print(self.last_symptom + " : " + str(self.last_symptom_class))

    def check_if_null_features(self):
        for i in range( len(self.features) ):
            if self.feature_symptom_array[i] > 0:
                return False
        return True

    def calculate_time_between_symptoms(self):
        self.time_between_symptom_reports = self.latest_datetime - self.second_latest_datetime
