# Classificação de Vinhos com Rede Neural MLP

Projeto acadêmico desenvolvido para o Trabalho Discente Efetivo (TDE) da unidade curricular **Fundamentos de Inteligência Artificial**, do curso de Análise e Desenvolvimento de Sistemas.

O projeto utiliza uma Rede Neural Artificial do tipo **Multilayer Perceptron (MLP)** para classificar amostras de vinho em três classes, a partir de 13 características químicas.

## Objetivo

Aplicar uma rede neural MLP em um problema de classificação supervisionada, compreender as etapas de preparação dos dados, treinamento e avaliação do modelo e analisar seu desempenho em 20 ciclos independentes de treinamento e teste.

## Base de dados

Foi utilizada a base **Wine**, disponibilizada pelo UCI Machine Learning Repository:

- Fonte: [UCI Machine Learning Repository — Wine (ID 109)](https://archive.ics.uci.edu/dataset/109/wine)
- Quantidade de amostras: 178
- Características de entrada: 13
- Classes de saída: 3
- Valores ausentes identificados na exploração: nenhum

A distribuição observada na base é:

| Classe | Amostras | Percentual aproximado |
|---:|---:|---:|
| 1 | 59 | 33,15% |
| 2 | 71 | 39,89% |
| 3 | 48 | 26,97% |
| **Total** | **178** | **100%** |

O script `01_explorar_dados.py` consulta a base na UCI e salva uma cópia local em `data/wine.csv`. Essa primeira obtenção exige conexão com a internet. O arquivo CSV é mantido no repositório para facilitar a reprodução das demais etapas.

## Arquitetura da rede neural

A arquitetura implementada possui uma camada de entrada, uma camada oculta e uma camada de saída:

| Camada | Neurônios | Função |
|---|---:|---|
| Entrada | 13 | Receber as características químicas |
| Oculta | 16 | Aprender padrões e relações entre as características |
| Saída | 3 | Representar as três classes de vinho |

**Topologia:** **13 → 16 → 3**

Configurações principais do `MLPClassifier`:

- Função de ativação da camada oculta: ReLU (`activation="relu"`).
- Otimizador: L-BFGS (`solver="lbfgs"`).
- Regularização: `alpha=0.001`.
- Limite máximo de iterações: `max_iter=3000`.

A saída multiclasse do classificador representa as três classes possíveis. A previsão corresponde à classe com maior pontuação/probabilidade estimada pelo modelo.

## Normalização das características

As 13 características químicas possuem escalas diferentes. Por isso, é utilizada a padronização (`StandardScaler`), que transforma cada valor de entrada pela expressão:

$$
z = \frac{x - \mu}{\sigma}
$$

Em que:

- $x$ é o valor original da característica;
- $\mu$ é a média calculada no conjunto de treinamento;
- $\sigma$ é o desvio padrão calculado no conjunto de treinamento;
- $z$ é o valor padronizado.

O normalizador é ajustado exclusivamente com os dados de treinamento (`fit`) e depois aplicado ao conjunto de teste (`transform`). Esse procedimento evita o vazamento de informações do teste para o treinamento (*data leakage*).

## Metodologia

O experimento oficial é executado pelo script `05_experimento_20_ciclos.py` e segue estas etapas:

1. Carregamento das 178 amostras da base local `data/wine.csv`.
2. Separação das 13 características (`X`) e da classe-alvo (`y`).
3. Divisão aleatória e estratificada em treinamento e teste, com proporção de 80% e 20%, respectivamente.
4. Ajuste do `StandardScaler` exclusivamente com os dados de treinamento e aplicação dos mesmos parâmetros ao conjunto de teste.
5. Criação e treinamento de uma nova MLP em cada ciclo.
6. Avaliação das previsões e armazenamento das métricas.
7. Consolidação dos resultados e geração das tabelas e gráficos.

São realizados **20 ciclos independentes de holdout aleatório estratificado**. As sementes utilizadas são 43 a 62. Cada ciclo utiliza uma nova divisão dos dados e uma nova instância do modelo. Portanto, a normalização também é ajustada novamente em cada ciclo, sem utilizar informações do conjunto de teste no ajuste do normalizador.

> **Nota metodológica:** esse procedimento é uma repetição de divisões aleatórias de treinamento e teste (*Repeated Random Holdout*), não uma validação cruzada de 20 folds. Como as divisões são repetidas, uma mesma amostra original pode aparecer no conjunto de teste em mais de um ciclo.

## Métricas de avaliação

Em cada ciclo são calculadas:

- **Acurácia:** proporção de previsões corretas.
- **Precisão macro:** média aritmética da precisão das três classes.
- **Recall macro:** média aritmética do recall das três classes.
- **F1-score macro:** média aritmética do F1-score das três classes.

Ao final, são calculados média, desvio padrão amostral, mínimo e máximo das métricas entre os 20 ciclos.

Também é gerado um relatório de classificação e uma matriz de confusão acumulada com as previsões dos 20 conjuntos de teste. Essa matriz representa 720 avaliações (20 ciclos × 36 amostras de teste), e não 720 vinhos únicos.

## Resultados obtidos

Na execução validada durante o desenvolvimento, foram obtidos os seguintes resultados médios entre os 20 ciclos:

| Métrica | Média | Desvio padrão |
|---|---:|---:|
| Acurácia | 97,92% | 1,53 pontos percentuais |
| Precisão macro | 97,95% | 1,49 pontos percentuais |
| Recall macro | 98,02% | 1,53 pontos percentuais |
| F1-score macro | 97,91% | 1,53 pontos percentuais |

A acurácia mínima observada foi de 94,44%, e a máxima foi de 100,00%.

### Matriz de confusão acumulada

Linhas representam as classes reais e colunas representam as classes previstas:

| Classe real | Previsto: classe 1 | Previsto: classe 2 | Previsto: classe 3 |
|---|---:|---:|---:|
| Classe 1 | 240 | 0 | 0 |
| Classe 2 | 3 | 269 | 8 |
| Classe 3 | 0 | 4 | 196 |

Foram registradas 705 classificações corretas e 15 incorretas nas 720 avaliações acumuladas. A acurácia calculada sobre esse conjunto de previsões acumuladas é 97,92%. Esse valor é distinto, conceitualmente, da média aritmética das 20 acurácias individuais, embora neste experimento os valores arredondados coincidam.

Os resultados são referentes às divisões e configurações utilizadas neste experimento. Eles não garantem o mesmo desempenho em outros conjuntos de dados ou em novas condições de avaliação.

## Estrutura do projeto

```text
TDE_MLP_WINE/
│
├── data/
│   └── wine.csv
│
├── resultados/
│   ├── resultados_20_ciclos.csv
│   ├── resumo_metricas_20_ciclos.csv
│   ├── grafico_acuracia_20_ciclos.png
│   ├── matriz_confusao_20_ciclos.png
│   └── matriz_confusao_normalizada_20_ciclos.png
│
├── src/
│   ├── 01_explorar_dados.py
│   ├── 02_preparar_dados.py
│   ├── 03_normalizar_dados.py
│   ├── 04_treinar_mlp.py
│   └── 05_experimento_20_ciclos.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

### Responsabilidade dos scripts

| Script | Responsabilidade |
|---|---|
| `01_explorar_dados.py` | Obtém a base Wine na UCI, explora os dados e salva `data/wine.csv`. |
| `02_preparar_dados.py` | Demonstra a separação entre treinamento e teste. |
| `03_normalizar_dados.py` | Demonstra a padronização das características com `StandardScaler`. |
| `04_treinar_mlp.py` | Executa um treinamento e teste experimental da MLP. |
| `05_experimento_20_ciclos.py` | Executa o experimento oficial de 20 ciclos, calcula métricas e gera os resultados. |

Os scripts 02, 03 e 04 são etapas didáticas independentes. O script 05 refaz a divisão e a normalização dentro de cada ciclo; não é necessário executar os scripts anteriores para iniciar o experimento oficial, desde que `data/wine.csv` já exista.

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- ucimlrepo
- Rich

## Como executar o projeto

### 1. Pré-requisitos

- Python instalado.
- Git instalado, caso deseje clonar ou versionar o projeto.
- Conexão com a internet para obter a base diretamente da UCI na primeira execução do script 01.

Confira a instalação do Python:

```powershell
python --version
```

### 2. Clonar o repositório

```powershell
git clone https://github.com/saravictoriaa/TDE_MLP_WINE.git
cd TDE_MLP_WINE
```


### 3. Criar e ativar o ambiente virtual (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativação de scripts, para a sessão atual pode ser necessário executar:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Depois, tente ativar o ambiente novamente.

### 4. Instalar as dependências

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Obter ou atualizar a base de dados

Caso `data/wine.csv` ainda não exista, execute:

```powershell
python .\src\01_explorar_dados.py
```

O script consulta a UCI e salva o arquivo CSV localmente. Se o CSV já estiver presente e você quiser preservar exatamente a cópia versionada no repositório, não é necessário executar novamente essa etapa.

### 6. Executar as etapas didáticas (opcional)

```powershell
python .\src\02_preparar_dados.py
python .\src\03_normalizar_dados.py
python .\src\04_treinar_mlp.py
```

### 7. Executar o experimento oficial

Com `data/wine.csv` disponível, execute:

```powershell
python .\src\05_experimento_20_ciclos.py
```

Os arquivos de tabelas e gráficos serão salvos automaticamente na pasta `resultados/`.

## Arquivos gerados

- `resultados_20_ciclos.csv`: métricas e quantidade de iterações de cada ciclo.
- `resumo_metricas_20_ciclos.csv`: média, desvio padrão, mínimo e máximo das métricas.
- `grafico_acuracia_20_ciclos.png`: acurácia por ciclo e média dos experimentos.
- `matriz_confusao_20_ciclos.png`: matriz de confusão em quantidades absolutas.
- `matriz_confusao_normalizada_20_ciclos.png`: matriz de confusão normalizada por classe real, em percentual.

## Observações

- O tempo de execução depende do equipamento e do ambiente Python; o tempo exibido no terminal é apenas uma medição daquela execução.
- O arquivo `.venv/` não deve ser enviado ao GitHub. As dependências são instaladas a partir de `requirements.txt`.
- Os arquivos de dados e resultados são mantidos no repositório para documentar e facilitar a reprodução do trabalho.

