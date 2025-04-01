import sys 
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path='artifact\model.pkl'
            preprocessor_path='artifact\preprocessor.pkl'
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
        StudentID:int,
        Age:int,
        Gender:int,
        Ethnicity:int,
        ParentalEducation:int,
        StudyTimeWeekly:float,
        Absences:int,
        Tutoring:int,
        ParentalSupport:int,
        Extracurricular:int,
        Sports:int,
        Music:int,
        Volunteering:int,
        GPA:float,
        GradeClass:float,):
        
        self.StudentID = StudentID
        self.Age = Age
        self.Gender = Gender
        self.Ethnicity = Ethnicity
        self.ParentalEducation = ParentalEducation
        self.StudyTimeWeekly = StudyTimeWeekly
        self.Absences = Absences
        self.Tutoring = Tutoring
        self.ParentalSupport = ParentalSupport
        self.Extracurricular = Extracurricular
        self.Sports = Sports
        self.Music = Music
        self.Volunteering = Volunteering
        self.GPA = GPA
        self.GradeClass = GradeClass

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.Age],
                "Gender": [self.Gender],
                "Ethnicity": [self.Ethnicity],
                "StudentID": [self.StudentID],
                "ParentalEducation": [self.ParentalEducation],
                "StudyTimeWeekly": [self.StudyTimeWeekly],
                "Absences": [self.Absences],
                "Tutoring": [self.Tutoring],
                "ParentalSupport": [self.ParentalSupport],
                "Extracurricular": [self.Extracurricular],
                "Sports": [self.Sports],
                "Music": [self.Music],
                "Volunteering": [self.Volunteering],
                "GPA": [self.GPA],
                "GradeClass": [self.GradeClass],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)