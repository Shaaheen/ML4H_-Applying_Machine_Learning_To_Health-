class Consultation_Patient:
    def __init__(self, person):
        self.person_id = person
        self.sex = None
        self.age = None
        self.consultations_attended = 0
        self.consultations_missed = 0
        self.features_list = ["Consultations_Attended", "Consultations_Missed", "Sex", "Birthdate", "Occupation" ]
        self.features = [0,0,0,0,0]

        self.attended_last_consultation = None