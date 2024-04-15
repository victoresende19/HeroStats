from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error
import sklearn.metrics as metrics
import numpy as np
import pandas as pd

def features(df: pd.DataFrame):
    """
    Esta função faz a escolha das features atraves do feature_importances_.

    Parâmetros:
    - df: conjunto de dados.

    Retorna:
    - Uma lista com as features recomendadas.
    """

    df = pd.get_dummies(df, drop_first=True)

    X = df.drop(['Weight'], axis=1)
    y = df['Weight']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    feature_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    return feature_importances

def regressor(df: pd.DataFrame):
    """
    Esta função faz a previsão do peso.

    Parâmetros:
    - df: conjunto de dados.

    Retorna:
    - métricas de r2, mape, mae e o regressor treinado.
    """

    X = np.array(df[['IMC']].copy())
    y = np.array(df[['Weight']].copy())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)


    regressor = RandomForestRegressor()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    # metricas
    r2 = regressor.score(X_test, y_test) 
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return r2, mape, mae, regressor

