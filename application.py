from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            StudentID = request.form.get('StudentID'),
            Age = request.form.get('Age'),
            Gender = request.form.get('Gender'),
            Ethnicity = request.form.get('Ethnicity'),
            ParentalEducation = request.form.get('ParentalEducation'),
            StudyTimeWeekly = request.form.get('StudyTimeWeekly'),
            Absences = request.form.get('Absences'),
            Tutoring = request.form.get('Tutoring'),
            ParentalSupport = request.form.get('ParentalSupport'),
            Extracurricular = request.form.get('Extracurricular'),
            Sports = request.form.get('Sports'),
            Music = request.form.get('Music'),
            Volunteering = request.form.get('Volunteering'),
            GPA = request.form.get('GPA'),
            GradeClass = request.form.get('GradeClass'),
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")