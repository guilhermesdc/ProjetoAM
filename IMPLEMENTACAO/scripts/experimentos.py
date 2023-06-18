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

# Arquivo com todas as funcoes e codigos referentes aos experimentos

import numpy as np
import pandas as pd
from sklearn.model_selection import *
from sklearn.metrics import *
from sklearn.neighbors import *
from sklearn.naive_bayes import *
from sklearn.linear_model import *
from sklearn.neural_network import *
from sklearn.svm import *
from sklearn.ensemble import *
from sklearn.feature_selection import *

def treinarModeloFold(model, trainX, trainY, nSplit):
    kf = KFold(n_splits=nSplit)
    kf.get_n_splits(trainX)
    acc = []
    roc = []
    f1 = []
    recall = []
    precisao = []
    for train_index, test_index in kf.split(trainX):
        X_train, X_test = trainX[train_index], trainX[test_index]
        y_train, y_test = trainY[train_index], trainY[test_index]
        model.fit(X_train, y_train)
        y = model.predict(X_test)
        acc += [accuracy_score(y_test, y)]
        roc += [roc_auc_score(y_test, y)]
        f1 += [f1_score(y_test, y)]
        recall += [recall_score(y_test, y)]
        precisao += [precision_score(y_test, y)]
    return np.mean(acc), np.std(acc), np.mean(roc), np.mean(recall), np.mean(precisao), np.mean(f1)

def gridSearch(model, trainX, trainY, nSplit, param_grid):
    grid = GridSearchCV(model, param_grid, cv=nSplit, n_jobs=-1, scoring='roc_auc')
    grid.fit(trainX, trainY)
    return grid.best_params_

def knn(train_x, train_y, nfolds, params):
    best_param = gridSearch(KNeighborsClassifier(), train_x, train_y, nfolds, params)
    model_knn = KNeighborsClassifier(**best_param)
    acc, std, roc, recall, precisao, f1 = treinarModeloFold(model_knn, train_x, train_y, nfolds)
    return acc, std, roc, recall, precisao, f1, best_param, model_knn

def naiveBayes(train_x, train_y, nfolds, params):
    best_param = gridSearch(GaussianNB(), train_x, train_y, nfolds, params)
    model_nb = GaussianNB(**best_param)
    acc, std, roc, recall, precisao, f1 = treinarModeloFold(model_nb, train_x, train_y, nfolds)
    return acc, std, roc, recall, precisao, f1, best_param, model_nb

def regressaoLogistica(train_x, train_y, nfolds, params):
    best_param = gridSearch(LogisticRegression(), train_x, train_y, nfolds, params)
    model_lr = LogisticRegression(**best_param)
    acc, std, roc, recall, precisao, f1 = treinarModeloFold(model_lr, train_x, train_y, nfolds)
    return acc, std, roc, recall, precisao, f1, best_param, model_lr

def redesNeurais(train_x, train_y, nfolds, params):
    best_param = gridSearch(MLPClassifier(), train_x, train_y, nfolds, params)
    model_mlp = MLPClassifier(**best_param)
    acc, std, roc, recall, precisao, f1 = treinarModeloFold(model_mlp, train_x, train_y, nfolds)
    return acc, std, roc, recall, precisao, f1, best_param, model_mlp

def maquinaVetoresSuporte(train_x, train_y, nfolds, params):
    best_param = gridSearch(SVC(), train_x, train_y, nfolds, params)
    model_svm = SVC(**best_param, probability=True)
    acc, std, roc, recall, precisao, f1 = treinarModeloFold(model_svm, train_x, train_y, nfolds)
    return acc, std, roc, recall, precisao, f1, best_param, model_svm

def florestaAleatoria(train_x, train_y, nfolds, params):
    best_param = gridSearch(RandomForestClassifier(), train_x, train_y, nfolds, params)
    model_rf = RandomForestClassifier(**best_param)
    acc, std, roc, recall, precisao, f1 = treinarModeloFold(model_rf, train_x, train_y, nfolds)
    return acc, std, roc, recall, precisao, f1, best_param, model_rf

def selecionarMelhoresFeatures(train_x, train_y, nfold):
    df = pd.DataFrame(columns=["Modelo", "Acuracia", "Desvio Padrao", "Curva ROC", "Recall", "Precisao", "F1-score"])
    models = [LogisticRegression(max_iter=3000)]
    for i in range(len(models)):
        acc, std, roc, recall, precisao, f1 = treinarModeloFold(models[i], train_x, train_y, nfold)
        df.loc[i] = [models[i], acc, std, roc, recall, precisao, f1]
    
    return df

def converteSelecionaFeatures(df_dataset_train, df_dataset_test, func_select, k):
    train_x, train_y = df_dataset_train.values[:, :-1], df_dataset_train.values[:, -1]
    test_x = df_dataset_test.values

    features_exp = selecionaFeaturesKBest(train_x, train_y, k, func_select)
    train_x = train_x[:, features_exp]
    test_x = test_x[:, features_exp]
    
    return train_x, train_y, test_x

def selecionaFeaturesFpr(X, Y, func):
    selector = SelectFpr(score_func=func)
    selector.fit(X, Y)
    features = selector.get_support(indices=True)
    return features

def selecionaFeaturesKBest(X, Y, k, func):
    selector = SelectKBest(score_func=func, k=k)
    selector.fit(X, Y)
    features = selector.get_support(indices=True)
    return features