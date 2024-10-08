from django.shortcuts import render

# Create your views here.

from django.contrib import messages
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from .forms import PredictionForm

# Load the pre-trained model from the file
# Define the path to your model.pkl file
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(MODEL_PATH, 'rb') as file:
    model = pickle.load(file)

def predict(request):
    result = None

    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
             # Get form data and convert to numpy array
            data = np.array([
                form.cleaned_data['age'],
                form.cleaned_data['sex'],
                form.cleaned_data['cp'],
                form.cleaned_data['trestbps'],
                form.cleaned_data['chol'],
                form.cleaned_data['fbs'],
                form.cleaned_data['restecg'],
                form.cleaned_data['thalach'],
                form.cleaned_data['exang'],
                form.cleaned_data['oldpeak'],
                form.cleaned_data['slope'],
            ]).reshape(1, -1)

            # Predict using the loaded model
            result = model.predict(data)[0]
            result = 'Heart Disease Detected !! \n Get your appointment today.' if result == 1 else 'No Heart Disease !'
            # prediction_input = input_data    #np.array(input_data).reshape(1, -1)
            
            # result = "Don't have" # Example logic (just sum the inputs)
    else:
        form = PredictionForm()

    return render(request, 'cvd/index.html', {'form': form, 'result': result})

# import joblib

# # Load the model at the start of the Django server
# model = joblib.load('path/to/your_model.pkl')

# # Inside your view:
# result = model.predict(prediction_input)
