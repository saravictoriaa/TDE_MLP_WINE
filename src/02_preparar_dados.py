# ============================================================
# PROJETO: Classificação de Vinhos utilizando Rede Neural MLP
# ETAPA 02: Preparação e divisão dos dados
# ============================================================

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# 1. Identificar os diretórios do projeto
PASTA_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_DADOS = PASTA_PROJETO / "data" / "wine.csv"


# 2. Carregar a base salva na etapa anterior
if not ARQUIVO_DADOS.exists():
    raise FileNotFoundError(
        "Arquivo data/wine.csv não encontrado. "
        "Execute primeiro 01_explorar_dados.py."
    )

dados = pd.read_csv(ARQUIVO_DADOS)

print("Base de dados carregada!")
print("Total de amostras:", len(dados))


# 3. Separar as características da classe
if "class" not in dados.columns:
    raise ValueError("A coluna 'class' não foi encontrada na base.")

X = dados.drop(columns=["class"])
y = dados["class"]


# 4. Verificar as dimensões iniciais
print("\n===== DIMENSÕES ORIGINAIS =====")
print("X:", X.shape)
print("y:", y.shape)


# 5. Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# 6. Exibir as dimensões dos conjuntos
print("\n===== DIMENSÕES APÓS A DIVISÃO =====")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# 7. Verificar a distribuição das classes
print("\n===== CLASSES NO TREINAMENTO =====")
print(y_train.value_counts().sort_index())

print("\n===== CLASSES NO TESTE =====")
print(y_test.value_counts().sort_index())


# 8. Conferir os percentuais
print("\n===== PERCENTUAIS =====")
print(f"Treinamento: {len(X_train) / len(X) * 100:.2f}%")
print(f"Teste: {len(X_test) / len(X) * 100:.2f}%")

print("\nPreparação dos dados concluída!")
