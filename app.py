from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
import sys

application = Flask(__name__)

app = application

## Routing for home page

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    
    else:
        try: 
            class_labels = {
                0: "Sadness",
                1: "Joy",
                2: "Love",
                3: "Anger",
                4: "Fear",
                5: "Surprise"
            }

            predict_pipeline = PredictPipeline(class_labels=class_labels)

            text = request.form.get('text')

            if not text:
                return render_template('home.html', results="Input text is missing.")           

            
            results = predict_pipeline.predict([text])

            return render_template('home.html', results = results[0])
        
        except Exception as e:
            return CustomException(e, sys)
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug= True)
