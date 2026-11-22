# ============================================================
# PROJETO: Classificação de Vinhos utilizando Rede Neural MLP
# ETAPA 03: Normalização dos dados
# ============================================================

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1. Identificar os diretórios do projeto
PASTA_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_DADOS = PASTA_PROJETO / "data" / "wine.csv"


# 2. Carregar a base de dados
if not ARQUIVO_DADOS.exists():
    raise FileNotFoundError(
        "Arquivo data/wine.csv não encontrado. "
        "Execute primeiro 01_explorar_dados.py."
    )

dados = pd.read_csv(ARQUIVO_DADOS)


# 3. Separar características e classes
if "class" not in dados.columns:
    raise ValueError("A coluna 'class' não foi encontrada na base.")

X = dados.drop(columns=["class"])
y = dados["class"]


# 4. Separar treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)


# 5. Criar o normalizador
normalizador = StandardScaler()


# 6. Ajustar e normalizar somente os dados de treinamento
X_train_normalizado = normalizador.fit_transform(X_train)


# 7. Transformar os dados de teste usando os parâmetros do treino
X_test_normalizado = normalizador.transform(X_test)


# 8. Converter arrays para DataFrames para facilitar a visualização
df_treino = pd.DataFrame(
    X_train_normalizado,
    columns=X.columns,
    index=X_train.index
)

df_teste = pd.DataFrame(
    X_test_normalizado,
    columns=X.columns,
    index=X_test.index
)


# 9. Exibir as dimensões
print("===== DIMENSÕES =====")
print("Treinamento:", df_treino.shape)
print("Teste:", df_teste.shape)


# 10. Verificar as médias do treinamento normalizado
print("\n===== MÉDIAS DO TREINAMENTO =====")
print(df_treino.mean().round(6))


# 11. Verificar os desvios padrões do treinamento
# ddof=0 utiliza a mesma convenção do StandardScaler.
print("\n===== DESVIOS PADRÕES DO TREINAMENTO =====")
print(df_treino.std(ddof=0).round(6))


# 12. Visualizar as primeiras amostras normalizadas
print("\n===== AMOSTRAS NORMALIZADAS =====")
print(df_treino.head().round(3))


# 13. Conferir as classes originais
print("\n===== CLASSES =====")
print("Treinamento:", y_train.shape)
print("Teste:", y_test.shape)

print("\nNormalização concluída!")
