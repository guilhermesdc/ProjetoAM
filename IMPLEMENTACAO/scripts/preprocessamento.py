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

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento

import pandas as pd 
from sklearn.preprocessing import normalize
from sklearn.impute import KNNImputer

def converteMatrix(data_set):
    X = data_set.iloc[:, :-1].values
    Y = data_set.iloc[:, -1].values
    return X, Y

def substituiNaN(data_set):
    data_set = data_set.fillna(data_set.mean())
    return data_set

def substituiNaNKNN(data_set):
    inputer = KNNImputer(n_neighbors=3)
    inputer.set_output(transform="pandas")
    df = inputer.fit_transform(data_set)
    data_set = pd.DataFrame(df, columns=data_set.columns, index=data_set.index)
    return data_set

def normalizaDados(X):
    X = normalize(X)
    return X

def converteCategorias(data_set):
    data_set = pd.get_dummies(data_set)
    return data_set

def removeCategorias(data_set, name):
    data_set = data_set.drop([name], axis=1)
    return data_set

def removeColunaZero(data_set, data_set2):
    leg = [''] * len(data_set.columns)
    cont = 0
    for i in range(len(data_set.columns)):
        if data_set.iloc[:, i].sum() == 0:
            leg[cont] += data_set.columns[i]
            cont += 1
            leg[cont] += data_set.columns[i].replace('_y', '_x')
            cont += 1
    for i in range(len(leg)):
        if leg[i] in data_set.columns:
            data_set = data_set.drop(leg[i], axis=1)
        if leg[i] in data_set2.columns:
            data_set2 = data_set2.drop(leg[i], axis=1)
    return data_set, data_set2

def agrupaRace(data_set):
    data_set = data_set.replace('White', 'White')
    data_set = data_set.replace('WHITE', 'White')
    data_set = data_set.replace('Non Hispanic', 'White')
    data_set = data_set.replace('Hispanic', 'Hispanic')
    data_set = data_set.replace('HISPANIC', 'Hispanic')
    data_set = data_set.replace('Asian', 'Asian')
    data_set = data_set.replace('ASIAN', 'Asian')
    data_set = data_set.replace('Black', 'Black')
    data_set = data_set.replace('BLACK', 'Black')
    data_set = data_set.replace('black', 'Black')
    data_set = data_set.replace('Native American', 'Native American')
    data_set = data_set.replace('Unknown', 'White')
    return data_set

def droparDuplicadas(df_dataset_clinical, df_dataset_expression, df_dataset_mutation):
    df_dataset_clinical.drop_duplicates(inplace=True)
    df_dataset_expression.drop_duplicates(inplace=True)
    df_dataset_mutation.drop_duplicates(inplace=True)
    print("Dados duplicados removidos com sucesso!")