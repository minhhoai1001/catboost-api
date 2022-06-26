import requests

URL = "http://127.0.0.1:8000/predict"
response = requests.post(url=URL, json={"age":[63, 37], "sex":[1, 1], "cp":[3, 2],"trtbps":[145, 130],"chol":[233, 250],"fbs":[1, 0],
                            "restecg":[0,1],"thalachh":[150, 187],"exng":[0,0],"oldpeak":[2.3, 3.5],"slp":[0,0],"caa":[0,0],"thall":[1,2]})
if response.status_code == 200:
    print(response.text)