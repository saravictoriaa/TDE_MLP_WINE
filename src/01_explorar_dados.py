# ============================================================
# PROJETO: Classificação de Vinhos utilizando Rede Neural MLP
# ETAPA 01: Obtenção e exploração da base de dados
# Fonte: UCI Machine Learning Repository - Wine (ID 109)
# ============================================================

from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo


# 1. Definir os diretórios do projeto
PASTA_PROJETO = Path(__file__).resolve().parents[1]
PASTA_DADOS = PASTA_PROJETO / "data"
ARQUIVO_DADOS = PASTA_DADOS / "wine.csv"

PASTA_DADOS.mkdir(parents=True, exist_ok=True)


# 2. Obter a base Wine diretamente da UCI
print("\n===== OBTENÇÃO DA BASE DE DADOS =====")
print("Consultando a base Wine na UCI...")

wine = fetch_ucirepo(id=109)


# 3. Separar características e variável de saída
X = wine.data.features.copy()
y = wine.data.targets.copy()


# 4. Validar a estrutura da base
if X.shape != (178, 13):
    raise ValueError(
        f"Estrutura inesperada para X: {X.shape}. "
        "Esperado: 178 amostras e 13 características."
    )

if y.shape[0] != 178 or y.shape[1] != 1:
    raise ValueError(
        f"Estrutura inesperada para y: {y.shape}. "
        "Esperado: uma coluna de classes com 178 registros."
    )

if X.isnull().any().any() or y.isnull().any().any():
    raise ValueError("A base apresenta valores ausentes.")


# 5. Exibir informações gerais
print("\n===== INFORMAÇÕES GERAIS =====")
print("Nome da base:", wine.metadata.name)
print("Quantidade de amostras:", X.shape[0])
print("Quantidade de características:", X.shape[1])
print("Quantidade de classes:", y.iloc[:, 0].nunique())


# 6. Visualizar as primeiras amostras
print("\n===== PRIMEIRAS AMOSTRAS =====")
print(X.head())


# 7. Verificar as características disponíveis
print("\n===== CARACTERÍSTICAS =====")
print(X.columns.tolist())


# 8. Verificar os tipos de dados
print("\n===== TIPOS DE DADOS =====")
print(X.dtypes)


# 9. Verificar valores ausentes
print("\n===== VALORES AUSENTES =====")
print("Nas características:")
print(X.isnull().sum())
print("\nNa variável de saída:")
print(y.isnull().sum())


# 10. Verificar a distribuição das classes
print("\n===== DISTRIBUIÇÃO DAS CLASSES =====")
print(y.iloc[:, 0].value_counts().sort_index())


# 11. Reunir características e classes em um único DataFrame
dados = pd.concat([X, y], axis=1)

# Padronizar o nome da coluna de saída para uso nos demais scripts.
coluna_alvo = y.columns[0]
if coluna_alvo != "class":
    dados = dados.rename(columns={coluna_alvo: "class"})


# 12. Salvar a base localmente
dados.to_csv(
    ARQUIVO_DADOS,
    index=False,
    encoding="utf-8"
)

print("\nBase salva com sucesso em:")
print(ARQUIVO_DADOS)
print("\nExploração dos dados concluída!")
