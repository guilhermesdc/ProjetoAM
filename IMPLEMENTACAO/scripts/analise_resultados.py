# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Aluno: Guilherme Silva de Camargo
# RA: 792183
# ################################################################

# Arquivo com todas as funcoes e codigos referentes a analise dos resultados


import numpy as np
import pandas as pd 
from sklearn.model_selection import *
from sklearn.metrics import *
import matplotlib.pyplot as plt

def metricas(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    conf = confusion_matrix(y_true, y_pred)
    print('Métricas:')
    display(pd.DataFrame([[acc, prec, rec, f1, auc]], columns=['Acurácia', 'Precisão', 'Recall', 'F1-score', 'AUC'], index=[model_name]))
    print('Matriz de Confusao:')
    display(pd.DataFrame(conf, columns=['Predito 0', 'Predito 1'], index=['Verdadeiro 0', 'Verdadeiro 1']))

# grafico de curva de aprendizado com k fold cross validation
def grafico_curva_aprendizado(model, train_x, train_y, model_name, fig):
    train_sizes, train_scores, test_scores = learning_curve(model, train_x, train_y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fig.add_subplot(1, 2, 1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score de treinamento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score de validacao cruzada")
    plt.legend(loc="best")
    plt.title('Curva de Aprendizado - ' + model_name)
    
def grafico_curva_validacao(model, train_x, train_y, model_name, fig, param):
    train_scores, test_scores = validation_curve(model, train_x, train_y, param_name=param, param_range=np.arange(1, 20), cv=5, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fig.add_subplot(1, 2, 2)
    plt.grid()
    plt.fill_between(np.arange(1, 20), train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(np.arange(1, 20), test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(np.arange(1, 20), train_scores_mean, 'o-', color="r", label="Score de treinamento")
    plt.plot(np.arange(1, 20), test_scores_mean, 'o-', color="g", label="Score de validacao cruzada")
    plt.legend(loc="best")
    plt.title('Curva de Validacao - ' + model_name)
    
def grafico_roc(y_true, y_pred, model_name, fig):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    fig.add_subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=model_name)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Porcentagem de Falsos Positivos')
    plt.ylabel('Porcentagem de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc='center right')
    
def grafico_precisao_recall(y_true, y_pred, model_name, fig):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    fig.add_subplot(1, 2, 2)
    plt.plot(recall, precision, label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precisao')
    plt.title('Curva Precisao x Recall')
    plt.legend(loc='center right')
    
def plotaResultados(model_name, model, train_x, train_y, param):
    trainX, testX, trainY, testY = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
    model.fit(trainX, trainY)
    y_pred = model.predict(testX)
    metricas(testY, y_pred, model_name)
    fig = plt.figure(figsize=(20, 5))
    grafico_curva_aprendizado(model, train_x, train_y, model_name, fig)
    grafico_curva_validacao(model, train_x, train_y, model_name, fig, param)
    plt.show()
    fig = plt.figure(figsize=(20, 5))
    grafico_roc(testY, y_pred, model_name, fig)
    grafico_precisao_recall(testY, y_pred, model_name, fig)
    plt.show()