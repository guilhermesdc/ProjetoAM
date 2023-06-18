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

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os 

def converteStrToFloat(data_set, coluna):
    data_set[coluna] = data_set[coluna].str.replace(',', '.')
    data_set[coluna] = data_set[coluna].astype(float)

def carregaDados(path):
    df_clinical_data = pd.read_csv(os.path.join(path, "clinical_data.csv"), sep=',', index_col='Sample ID')
    df_expression_data = pd.read_csv(os.path.join(path, "genetic_expression_data.csv"), sep=',', index_col='Sample ID')
    df_mutation_data = pd.read_csv(os.path.join(path, "genetic_mutation_data.csv"), sep=',', index_col='Sample ID')
    df_train = pd.read_csv(os.path.join(path, "train.csv"), sep=',', index_col='Sample ID')
    df_test = pd.read_csv(os.path.join(path, "test.csv"), sep=',', index_col='Sample ID')
    
    converteStrToFloat(df_clinical_data, "Bone Marrow Blast Percentage")
    
    df_dataset = pd.merge(df_clinical_data, df_expression_data, on='Sample ID', how='left')
    df_dataset = pd.merge(df_dataset, df_mutation_data, on='Sample ID', how='left')
    df_dataset.drop_duplicates(inplace=True)
    
    print("Dados carregados com sucesso!")
    
    return df_dataset, df_clinical_data, df_expression_data, df_mutation_data, df_train, df_test

def medidasDescritivas(df_dataset):
    return df_dataset.describe()

def matrizCorrelacao(df_dataset):
    return df_dataset.corr()

def matrizCovariancia(df_dataset):
    return df_dataset.cov()

def boxplot(df_dataset, df_dataset_classe, atributos, classe):
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle("Boxplots utilizando os atributos númericos", fontsize=16) 
    for i in range(0, 6):
        sns.boxplot(x=df_dataset_classe[classe], y=df_dataset[atributos[i]], ax=axes[i // 3, i % 3])
    
def graficoBarra(df_dataset, df_dataset_classe, atributos, classe):
    fig, axes = plt.subplots(2, 4, figsize=(30, 10))
    fig.suptitle("Graficos de barras para os atributos númericos e categoricos", fontsize=16)
    idx = 0; idx2 = 0
    for i in range(0, 8):
        if idx2 == 4:
            idx += 1
            idx2 = 0
        if(idx == 1) and (idx2 > 0):
            sns.barplot(x=df_dataset[atributos[i]], y=df_dataset_classe[classe], ax=axes[idx, idx2])
        else:
            sns.barplot(x=df_dataset_classe[classe], y=df_dataset[atributos[i]], ax=axes[idx, idx2])
        idx2 += 1
    
def histogramaConj(df_dataset, df_dataset_classe, atributos, classe):
    fig, axes = plt.subplots(2, 4, figsize=(30, 10))
    fig.suptitle("Histogramas para os atributos númericos e categoricos", fontsize=16)
    idx = 0; idx2 = 0
    for i in range(0, 8):
        if idx2 == 4:
            idx += 1
            idx2 = 0
        sns.histplot(x=df_dataset[atributos[i]], hue=df_dataset_classe[classe], ax=axes[idx, idx2], multiple='dodge')
        idx2 += 1
    
def graficoDispersao(df_dataset, df_dataset_classe, atributos, classe):
    fig, axes = plt.subplots(2, 5, figsize=(30, 10))
    fig.suptitle("Graficos de dispersão utilizando os atributos númericos", fontsize=16)
    idx = 0; idx2 = 0
    for i in range(0, 5):
        for j in range(i + 1, 5):
            if idx2 == 5:
                idx += 1
                idx2 = 0
            sns.scatterplot(x=df_dataset[atributos[j]], y=df_dataset[atributos[i]], hue=df_dataset_classe[classe], ax=axes[idx, idx2], palette="Set2")
            idx2 += 1

def histogramaAtr(df_dataset, df_dataset_classe, atr, classe):
    plt.figure(figsize=(30, 10))
    plt.title("Histograma para a raça")
    sns.histplot(x=df_dataset[atr], hue=df_dataset_classe[classe], multiple='dodge')
    