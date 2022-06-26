# Heart Attack Analysis & Prediction Dataset
## About this dataset
- Age : Age of the patient
- Sex : Sex of the patient
- exang: exercise induced angina (1 = yes; 0 = no)
- ca: number of major vessels (0-3)
- cp : Chest Pain type chest pain type
    - Value 1: typical angina
    - Value 2: atypical angina
    - Value 3: non-anginal pain
    - Value 4: asymptomatic
- trtbps : resting blood pressure (in mmHg)
- chol : cholestoral in mg/dl fetched via BMI sensor
- fbs : (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
- rest_ecg : resting electrocardiographic results
    - Value 0: normal
    - Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    - Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria
- thalach : maximum heart rate achieved
- target : 0= less chance of heart attack 1= more chance of heart attack

## Run web server
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Send request to API for predict
Sample data for list
```
json={"age":[63, 37], "sex":[1, 1], "cp":[3, 2],"trtbps":[145, 130],"chol":[233, 250],"fbs":[1, 0],"restecg":[0,1],"thalachh":[150, 187],"exng":[0,0],"oldpeak":[2.3, 3.5],"slp":[0,0],"caa":[0,0],"thall":[1,2]}
```
```
python request.py
```