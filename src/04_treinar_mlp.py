# ============================================================
# PROJETO: Classificação de Vinhos utilizando Rede Neural MLP
# ETAPA 04: Construção e treinamento inicial da MLP
# ============================================================

from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# 1. Identificar o diretório do projeto
PASTA_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_DADOS = PASTA_PROJETO / "data" / "wine.csv"


# 2. Carregar a base de dados
if not ARQUIVO_DADOS.exists():
    raise FileNotFoundError(
        "Arquivo data/wine.csv não encontrado. "
        "Execute primeiro 01_explorar_dados.py."
    )

dados = pd.read_csv(ARQUIVO_DADOS)

print("Base de dados carregada!")
print("Total de amostras:", len(dados))


# 3. Separar características e classes
if "class" not in dados.columns:
    raise ValueError("A coluna 'class' não foi encontrada na base.")

X = dados.drop(columns=["class"])
y = dados["class"]


# 4. Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# 5. Construir o Pipeline
modelo = Pipeline([
    ("normalizacao", StandardScaler()),
    ("rede_neural", MLPClassifier(
        hidden_layer_sizes=(16,),
        activation="relu",
        solver="lbfgs",
        alpha=0.001,
        max_iter=3000,
        random_state=42
    ))
])


# 6. Treinar a Rede Neural
print("\nIniciando o treinamento...")
modelo.fit(X_train, y_train)
print("Treinamento concluído!")


# 7. Realizar as previsões
previsoes = modelo.predict(X_test)


# 8. Calcular a acurácia
acuracia = accuracy_score(y_test, previsoes)

print("\n===== RESULTADO DO TESTE =====")
print(f"Acurácia: {acuracia * 100:.2f}%")


# 9. Exibir relatório de classificação
print("\n===== RELATÓRIO DE CLASSIFICAÇÃO =====")
print(classification_report(
    y_test,
    previsoes,
    labels=[1, 2, 3],
    target_names=["Classe 1", "Classe 2", "Classe 3"],
    zero_division=0
))


# 10. Exibir matriz de confusão
matriz = confusion_matrix(
    y_test,
    previsoes,
    labels=[1, 2, 3]
)

print("\n===== MATRIZ DE CONFUSÃO =====")
print(matriz)


# 11. Verificar a arquitetura efetivamente criada
rede = modelo.named_steps["rede_neural"]

print("\n===== ARQUITETURA DA REDE =====")
print("Quantidade de camadas:", rede.n_layers_)
print("Neurônios por camada: 13 -> 16 -> 3")

print("Dimensão dos pesos entre entrada e camada oculta:")
print(rede.coefs_[0].shape)

print("Dimensão dos pesos entre camada oculta e saída:")
print(rede.coefs_[1].shape)

print("Iterações realizadas:", rede.n_iter_)

print("\nExperimento inicial concluído!")
