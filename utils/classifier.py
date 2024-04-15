import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report

def features(df):
    """
    Esta função faz a escolha das features atraves do REF e DecisionTree.

    Parâmetros:
    - df: conjunto de dados.

    Retorna:
    - Uma lista com as features recomendadas.
    """

    df = pd.get_dummies(df, drop_first=True)

    alignment_cols = [col for col in df.columns if 'Alignment_' in col]
    X = df.drop(alignment_cols, axis=1) 
    y = df[alignment_cols] 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=10)
    X_train_rfe = rfe.fit_transform(X_train, y_train)
    X_test_rfe = rfe.transform(X_test)

    selected_features = pd.Series(rfe.support_, index=X.columns)
    return selected_features[selected_features == True]

@st.cache_data(show_spinner=False, ttl=24*3600, max_entries=2)
def classifier(df):
    """
    Esta função faz a previsão do alinhamento.

    Parâmetros:
    - df: conjunto de dados.

    Retorna:
    - métricas de precisao, recall, f1score, acurácia e o classificador treinado.
    """

    X = df.loc[:, ['Durability', 'Stealth', 'Marksmanship', 'Cryokinesis', 'Insanity', 'IMC']].values
    y = df.loc[:, 'Alignment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 49)

    classifier =  BernoulliNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=['bad', 'good'], output_dict=True, zero_division=0)
    precision = report['weighted avg']['precision']
    recall = report['weighted avg']['recall']
    f1_score = report['weighted avg']['f1-score']
    acc = report['accuracy']

    return precision, recall, f1_score, acc, classifier
