from catboost import CatBoostClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def main():
    scaler = StandardScaler()

    # df = pd.read_csv("heart.csv")
    # print(df.head())
    # X = df.drop(["output"], axis=1)
    # y = df.output
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 0)
    data = {"age":63, "sex":1, "cp":3,"trtbps":145,"chol":233,"fbs":1,"restecg":0,"thalachh":150,"exng":0,"oldpeak":2.3,"slp":0,"caa":0,"thall":1}
    df = pd.DataFrame.from_dict(data, orient='index')
    x_predict = scaler.fit_transform(df)
    # x_test = scaler.transform(X_test)
    model = CatBoostClassifier()
    model.load_model("model")
    result = model.predict(x_predict.T)
    print(result)

if __name__ == "__main__":
    main()