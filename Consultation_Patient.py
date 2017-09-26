
#Representation of a consultation patient object
class Consultation_Patient:
    def __init__(self, person):
        self.person_id = person
        self.sex = None
        self.age = None
        self.consultations_attended = 0
        self.consultations_missed = 0
        self.features_list = ["Consultations_Attended", "Consultations_Missed", "Sex", "Age", "Occupation", "Location","Day_of_week","Last_appointment" ]
        self.features = [0,0,0,0,0,0,0,0]
        self.latest_obs_datetime = None
        self.attended_last_consultation = None
        self.last_consultations = [-1,-1]