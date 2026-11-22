# ============================================================
# PROJETO: CLASSIFICACAO DE VINHOS UTILIZANDO REDE NEURAL MLP
# Trabalho Discente Efetivo - Fundamentos de Inteligencia Artificial
#
# Fonte: UCI Machine Learning Repository
# Dataset: Wine (ID 109)
#
# Arquitetura: 13 -> 16 -> 3
# Metodologia: 20 ciclos de treinamento e teste
# ============================================================


# ============================================================
# 1. IMPORTACAO DAS BIBLIOTECAS
# ============================================================

from pathlib import Path
import time

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from rich.console import Console
from rich.theme import Theme
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    MofNCompleteColumn,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


# ============================================================
# 2. IDENTIDADE VISUAL DO PROJETO
# ============================================================

TEMA = Theme(
    {
        "wine.title": "bold #F3E9D2",
        "wine.border": "#722F37",
        "wine.accent": "bold #C9828C",
        "wine.value": "bold #D7B56D",
        "wine.muted": "#C7B8B0",
        "wine.success": "bold #8FAF83",
        "wine.error": "bold #E07A7A",
    }
)

console = Console(theme=TEMA)

# Paleta utilizada nos graficos
COR_VINHO = "#722F37"
COR_PROFUNDA = "#4A1725"
COR_ROSE = "#C9828C"
COR_DOURADO = "#D7B56D"
COR_CHAMPAGNE = "#F3E9D2"

MAPA_VINHO = LinearSegmentedColormap.from_list(
    "wine_palette",
    [
        COR_CHAMPAGNE,
        COR_ROSE,
        COR_VINHO,
        COR_PROFUNDA,
    ],
)


# ============================================================
# 3. CONFIGURACOES DO PROJETO
# ============================================================

PASTA_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_DADOS = PASTA_PROJETO / "data" / "wine.csv"
PASTA_RESULTADOS = PASTA_PROJETO / "resultados"

NUM_CICLOS = 20
TAMANHO_TESTE = 0.20
SEMENTE_BASE = 42

CLASSES = [1, 2, 3]
NOME_CLASSES = ["Classe 1", "Classe 2", "Classe 3"]


# ============================================================
# 4. CABECALHO DO PROGRAMA
# ============================================================

def exibir_cabecalho():
    """Apresenta a identificacao do projeto no terminal."""

    console.print()
    console.print(
        Panel(
            "[wine.title]CLASSIFICACAO DE VINHOS[/]\n"
            "[wine.accent]Rede Neural Artificial - MLP[/]\n"
            "[wine.muted]Trabalho Discente Efetivo | Fundamentos de IA[/]",
            title="[wine.value]UCI WINE | DATASET 109[/]",
            border_style="wine.border",
            padding=(1, 4),
            expand=False,
        )
    )


# ============================================================
# 5. CARREGAMENTO E PREPARACAO DOS DADOS
# ============================================================

def carregar_dados():
    """Carrega a base Wine e separa caracteristicas e classes."""

    console.print("\n[wine.accent]ETAPA 01 - CARREGAMENTO DOS DADOS[/]")

    if not ARQUIVO_DADOS.exists():
        raise FileNotFoundError(
            f"Arquivo de dados nao encontrado: {ARQUIVO_DADOS}"
        )

    dados = pd.read_csv(ARQUIVO_DADOS)

    if dados.empty:
        raise ValueError("O arquivo de dados esta vazio.")

    if dados.isnull().any().any():
        raise ValueError("A base de dados possui valores ausentes.")

    if "class" not in dados.columns:
        raise ValueError("A coluna 'class' nao foi encontrada.")

    X = dados.drop(columns=["class"])
    y = dados["class"]

    if X.shape[1] != 13:
        raise ValueError("A base deve possuir 13 caracteristicas.")

    if sorted(y.unique().tolist()) != CLASSES:
        raise ValueError("A base deve conter as classes 1, 2 e 3.")

    if not all(pd.api.types.is_numeric_dtype(tipo) for tipo in X.dtypes):
        raise TypeError("Todas as caracteristicas devem ser numericas.")

    # Tabela com informacoes gerais da base
    tabela = Table(
        title="Informacoes da Base Wine",
        box=box.ROUNDED,
        border_style="wine.border",
        title_style="wine.value",
        header_style="wine.accent",
    )
    tabela.add_column("Propriedade", style="wine.title")
    tabela.add_column("Valor", justify="right", style="wine.value")
    tabela.add_row("Amostras", str(len(dados)))
    tabela.add_row("Caracteristicas", str(X.shape[1]))
    tabela.add_row("Classes", str(y.nunique()))
    tabela.add_row("Valores ausentes", "0")
    console.print(tabela)

    # Tabela com a distribuicao das classes
    distribuicao = Table(
        title="Distribuicao das Classes",
        box=box.ROUNDED,
        border_style="wine.border",
        title_style="wine.value",
        header_style="wine.accent",
    )
    distribuicao.add_column("Classe")
    distribuicao.add_column("Quantidade", justify="right")

    for classe, quantidade in y.value_counts().sort_index().items():
        distribuicao.add_row(str(classe), str(quantidade))

    console.print(distribuicao)
    console.print("[wine.success]Base carregada e validada com sucesso![/]")

    return X, y


# ============================================================
# 6. CONSTRUCAO DA REDE NEURAL MLP
# ============================================================

def criar_modelo(semente):
    """Cria o pipeline de normalizacao e a rede neural MLP."""

    return Pipeline(
        [
            # A normalizacao e ajustada somente com os dados de treino.
            ("normalizacao", StandardScaler()),
            (
                "rede_neural",
                MLPClassifier(
                    hidden_layer_sizes=(16,),
                    activation="relu",
                    solver="lbfgs",
                    alpha=0.001,
                    max_iter=3000,
                    random_state=semente,
                ),
            ),
        ]
    )


# ============================================================
# 7. EXECUCAO DOS 20 CICLOS DE TREINAMENTO E TESTE
# ============================================================

def executar_experimentos(X, y):
    """Executa os ciclos independentes e armazena suas metricas."""

    resultados = []
    classes_reais = []
    classes_previstas = []

    console.print("\n[wine.accent]ETAPA 02 - EXECUCAO DOS EXPERIMENTOS[/]")

    # perf_counter mede o tempo decorrido com alta resolucao.
    # A medicao comeca imediatamente antes dos 20 ciclos.
    inicio = time.perf_counter()

    with Progress(
        SpinnerColumn(style="wine.accent"),
        TextColumn("[wine.accent]{task.description}"),
        BarColumn(
            style="wine.muted",
            complete_style="wine.border",
            finished_style="wine.success",
            pulse_style="wine.accent",
        ),
        MofNCompleteColumn(separator="/"),
        TextColumn("[wine.value]{task.percentage:>3.0f}%"),
        console=console,
        refresh_per_second=10,
    ) as progress:
        tarefa = progress.add_task(
            "Preparando os experimentos...",
            total=NUM_CICLOS,
        )

        for ciclo in range(1, NUM_CICLOS + 1):
            # Semente diferente em cada ciclo para controlar a aleatoriedade.
            semente = SEMENTE_BASE + ciclo

            progress.update(
                tarefa,
                description=f"Ciclo {ciclo:02d}/{NUM_CICLOS} - Treinando MLP",
            )

            # Divisao estratificada: preserva a proporcao das classes.
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TAMANHO_TESTE,
                stratify=y,
                random_state=semente,
            )

            # Uma nova instancia do pipeline e criada em cada ciclo.
            modelo = criar_modelo(semente)

            # O Pipeline ajusta o StandardScaler apenas em X_train
            # e, em seguida, treina a MLP.
            modelo.fit(X_train, y_train)

            # Previsoes para amostras que nao participaram do treino.
            previsoes = modelo.predict(X_test)

            # Calculo das metricas do ciclo.
            acuracia = accuracy_score(y_test, previsoes)
            precisao = precision_score(
                y_test,
                previsoes,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            )
            recall = recall_score(
                y_test,
                previsoes,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            )
            f1 = f1_score(
                y_test,
                previsoes,
                labels=CLASSES,
                average="macro",
                zero_division=0,
            )

            rede = modelo.named_steps["rede_neural"]

            resultados.append(
                {
                    "ciclo": ciclo,
                    "acuracia": acuracia,
                    "precisao_macro": precisao,
                    "recall_macro": recall,
                    "f1_macro": f1,
                    "iteracoes": rede.n_iter_,
                }
            )

            classes_reais.extend(y_test.tolist())
            classes_previstas.extend(previsoes.tolist())

            progress.update(
                tarefa,
                advance=1,
                description=f"Ciclo {ciclo:02d}/{NUM_CICLOS} concluido",
            )

    # O tempo e calculado logo apos o fim do bloco dos 20 ciclos.
    # Portanto, nao inclui geracao dos graficos nem salvamento dos CSVs.
    tempo_total = time.perf_counter() - inicio

    console.print(
        f"\n[wine.value]Tempo dos {NUM_CICLOS} ciclos: "
        f"{tempo_total:.3f} segundos[/]"
    )
    console.print("[wine.success]Os 20 ciclos foram concluidos![/]")

    df_resultados = pd.DataFrame(resultados)

    return df_resultados, classes_reais, classes_previstas, tempo_total


# ============================================================
# 8. EXIBICAO DOS RESULTADOS INDIVIDUAIS
# ============================================================

def exibir_resultados_ciclos(df_resultados):
    """Apresenta as metricas individuais dos 20 ciclos."""

    tabela = Table(
        title="Desempenho Individual dos 20 Ciclos",
        box=box.ROUNDED,
        border_style="wine.border",
        title_style="wine.value",
        header_style="wine.accent",
    )

    tabela.add_column("Ciclo", justify="center")
    tabela.add_column("Acuracia", justify="right")
    tabela.add_column("Precisao", justify="right")
    tabela.add_column("Recall", justify="right")
    tabela.add_column("F1-score", justify="right")
    tabela.add_column("Iteracoes", justify="right")

    for _, linha in df_resultados.iterrows():
        tabela.add_row(
            f"{int(linha['ciclo']):02d}",
            f"{linha['acuracia'] * 100:.2f}%",
            f"{linha['precisao_macro'] * 100:.2f}%",
            f"{linha['recall_macro'] * 100:.2f}%",
            f"{linha['f1_macro'] * 100:.2f}%",
            str(int(linha["iteracoes"])),
        )

    console.print()
    console.print(tabela)


# ============================================================
# 9. ANALISE ESTATISTICA DOS RESULTADOS
# ============================================================

def analisar_resultados(df_resultados):
    """Calcula estatisticas das quatro metricas nos 20 ciclos."""

    colunas_metricas = [
        "acuracia",
        "precisao_macro",
        "recall_macro",
        "f1_macro",
    ]

    # Media, desvio padrao amostral, minimo e maximo em percentual.
    resumo = (
        df_resultados[colunas_metricas]
        .agg(["mean", "std", "min", "max"])
        .T
        * 100
    )

    tabela = Table(
        title="Resumo Estatistico dos 20 Ciclos (%)",
        box=box.ROUNDED,
        border_style="wine.border",
        title_style="wine.value",
        header_style="wine.accent",
    )

    tabela.add_column("Metrica", style="wine.title")
    tabela.add_column("Media", justify="right")
    tabela.add_column("Desvio padrao", justify="right")
    tabela.add_column("Minimo", justify="right")
    tabela.add_column("Maximo", justify="right")

    for metrica, valores in resumo.iterrows():
        tabela.add_row(
            metrica.replace("_", " ").title(),
            f"{valores['mean']:.2f}%",
            f"{valores['std']:.2f}%",
            f"{valores['min']:.2f}%",
            f"{valores['max']:.2f}%",
        )

    console.print()
    console.print(tabela)

    media = df_resultados["acuracia"].mean()
    desvio = df_resultados["acuracia"].std()
    menor = df_resultados["acuracia"].min()
    maior = df_resultados["acuracia"].max()

    painel = Panel(
        "[wine.muted]Acuracia media[/]\n"
        f"[wine.value]{media * 100:.2f}%[/]\n\n"
        f"[wine.muted]Desvio padrao: {desvio * 100:.2f} p.p.[/]\n"
        f"[wine.muted]Menor resultado: {menor * 100:.2f}%[/]\n"
        f"[wine.muted]Maior resultado: {maior * 100:.2f}%[/]",
        title="[wine.accent]DESEMPENHO GERAL[/]",
        border_style="wine.border",
        padding=(1, 3),
        expand=False,
    )

    console.print(painel)

    return resumo, media


# ============================================================
# 10. RELATORIO DE CLASSIFICACAO E MATRIZES DE CONFUSAO
# ============================================================

def avaliar_classificacao(classes_reais, classes_previstas):
    """Gera relatorio de classificacao e matrizes acumuladas."""

    relatorio = classification_report(
        classes_reais,
        classes_previstas,
        labels=CLASSES,
        target_names=NOME_CLASSES,
        zero_division=0,
        output_dict=True,
    )

    tabela = Table(
        title="Relatorio de Classificacao - Previsoes Acumuladas",
        box=box.ROUNDED,
        border_style="wine.border",
        title_style="wine.value",
        header_style="wine.accent",
    )

    tabela.add_column("Classe", style="wine.title")
    tabela.add_column("Precisao", justify="right")
    tabela.add_column("Recall", justify="right")
    tabela.add_column("F1-score", justify="right")
    tabela.add_column("Suporte", justify="right")

    for nome in NOME_CLASSES + ["macro avg", "weighted avg"]:
        valores = relatorio[nome]
        tabela.add_row(
            nome,
            f"{valores['precision'] * 100:.2f}%",
            f"{valores['recall'] * 100:.2f}%",
            f"{valores['f1-score'] * 100:.2f}%",
            f"{valores['support']:.0f}",
        )

    console.print()
    console.print(tabela)

    acuracia_acumulada = accuracy_score(
        classes_reais,
        classes_previstas,
    )

    console.print(
        Panel(
            f"[wine.value]{acuracia_acumulada * 100:.2f}%[/]\n"
            "[wine.muted]Acuracia sobre as previsoes acumuladas[/]",
            title="[wine.accent]ACURACIA GLOBAL[/]",
            border_style="wine.border",
            expand=False,
        )
    )

    # Matriz de confusao absoluta
    matriz = confusion_matrix(
        classes_reais,
        classes_previstas,
        labels=CLASSES,
    )

    tabela_matriz = Table(
        title="Matriz de Confusao - Quantidades",
        box=box.ROUNDED,
        border_style="wine.border",
        title_style="wine.value",
        header_style="wine.accent",
    )
    tabela_matriz.add_column("Classe real")

    for nome in NOME_CLASSES:
        tabela_matriz.add_column(nome, justify="right")

    for indice, nome in enumerate(NOME_CLASSES):
        tabela_matriz.add_row(
            nome,
            *[str(int(valor)) for valor in matriz[indice]],
        )

    console.print()
    console.print(tabela_matriz)

    # Matriz normalizada por classe real, apresentada em percentual.
    matriz_normalizada = (
        confusion_matrix(
            classes_reais,
            classes_previstas,
            labels=CLASSES,
            normalize="true",
        )
        * 100
    )

    tabela_normalizada = Table(
        title="Matriz de Confusao - Percentuais por Classe Real",
        box=box.ROUNDED,
        border_style="wine.border",
        title_style="wine.value",
        header_style="wine.accent",
    )
    tabela_normalizada.add_column("Classe real")

    for nome in NOME_CLASSES:
        tabela_normalizada.add_column(nome, justify="right")

    for indice, nome in enumerate(NOME_CLASSES):
        tabela_normalizada.add_row(
            nome,
            *[f"{valor:.2f}%" for valor in matriz_normalizada[indice]],
        )

    console.print(tabela_normalizada)

    return matriz, matriz_normalizada


# ============================================================
# 11. GERACAO DOS GRAFICOS
# ============================================================

def gerar_graficos(df_resultados, matriz, matriz_normalizada):
    """Gera os tres graficos da analise experimental."""

    console.print("\n[wine.accent]ETAPA 03 - GERACAO DOS GRAFICOS[/]")

    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # 11.1. Acuracia nos 20 ciclos
    # --------------------------------------------------------

    media = df_resultados["acuracia"].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#FBF8F4")
    ax.set_facecolor("#FBF8F4")

    ax.plot(
        df_resultados["ciclo"],
        df_resultados["acuracia"] * 100,
        color=COR_VINHO,
        marker="o",
        markerfacecolor=COR_ROSE,
        markeredgecolor=COR_PROFUNDA,
        linewidth=2,
        label="Acuracia por ciclo",
    )

    ax.axhline(
        y=media,
        color=COR_DOURADO,
        linestyle="--",
        linewidth=2,
        label=f"Media: {media:.2f}%",
    )

    ax.fill_between(
        df_resultados["ciclo"],
        df_resultados["acuracia"] * 100,
        90,
        color=COR_ROSE,
        alpha=0.10,
    )

    ax.set_xlabel("Ciclo")
    ax.set_ylabel("Acuracia (%)")
    ax.set_title(
        "Acuracia da MLP nos 20 Ciclos",
        color=COR_PROFUNDA,
        fontweight="bold",
    )
    ax.set_ylim(90, 101)
    ax.set_yticks(range(90, 101, 1))
    ax.set_xticks(range(1, NUM_CICLOS + 1))
    ax.grid(True, alpha=0.25, color=COR_ROSE)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        PASTA_RESULTADOS / "grafico_acuracia_20_ciclos.png",
        dpi=300,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 11.2. Matriz de confusao absoluta
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#FBF8F4")

    display = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=NOME_CLASSES,
    )
    display.plot(
        ax=ax,
        values_format="d",
        cmap=MAPA_VINHO,
        colorbar=False,
    )

    ax.set_title(
        "Matriz de Confusao - 20 Ciclos",
        color=COR_PROFUNDA,
        fontweight="bold",
    )
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")

    fig.tight_layout()
    fig.savefig(
        PASTA_RESULTADOS / "matriz_confusao_20_ciclos.png",
        dpi=300,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    # --------------------------------------------------------
    # 11.3. Matriz de confusao normalizada
    # --------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#FBF8F4")

    display_normalizado = ConfusionMatrixDisplay(
        confusion_matrix=matriz_normalizada,
        display_labels=NOME_CLASSES,
    )
    display_normalizado.plot(
        ax=ax,
        values_format=".2f",
        cmap=MAPA_VINHO,
        colorbar=False,
    )

    ax.set_title(
        "Matriz de Confusao Normalizada - 20 Ciclos (%)",
        color=COR_PROFUNDA,
        fontweight="bold",
    )
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")

    fig.tight_layout()
    fig.savefig(
        PASTA_RESULTADOS / "matriz_confusao_normalizada_20_ciclos.png",
        dpi=300,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    console.print(
        "[wine.success]Os tres graficos foram gerados com sucesso![/]"
    )


# ============================================================
# 12. SALVAMENTO DOS RESULTADOS
# ============================================================

def salvar_resultados(df_resultados, resumo):
    """Salva as metricas individuais e o resumo estatistico em CSV."""

    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    arquivo_resultados = (
        PASTA_RESULTADOS / "resultados_20_ciclos.csv"
    )
    df_resultados.to_csv(arquivo_resultados, index=False)

    arquivo_resumo = (
        PASTA_RESULTADOS / "resumo_metricas_20_ciclos.csv"
    )
    resumo.rename_axis("metrica").to_csv(arquivo_resumo)

    tabela = Table(
        title="Arquivos Gerados",
        box=box.ROUNDED,
        border_style="wine.border",
        title_style="wine.value",
        header_style="wine.accent",
    )
    tabela.add_column("Arquivo", style="wine.title")
    tabela.add_column("Status", justify="center")

    arquivos = [
        arquivo_resultados,
        arquivo_resumo,
        PASTA_RESULTADOS / "grafico_acuracia_20_ciclos.png",
        PASTA_RESULTADOS / "matriz_confusao_20_ciclos.png",
        PASTA_RESULTADOS / "matriz_confusao_normalizada_20_ciclos.png",
    ]

    for arquivo in arquivos:
        status = (
            "[wine.success]Gerado[/]"
            if arquivo.exists()
            else "[wine.error]Nao encontrado[/]"
        )
        tabela.add_row(arquivo.name, status)

    console.print()
    console.print(tabela)


# ============================================================
# 13. FUNCAO PRINCIPAL
# ============================================================

def main():
    """Coordena todas as etapas do experimento."""

    exibir_cabecalho()
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    # 1. Carregamento e validacao da base
    X, y = carregar_dados()

    # 2. Execucao dos 20 ciclos e medicao do tempo
    (
        df_resultados,
        classes_reais,
        classes_previstas,
        tempo_total,
    ) = executar_experimentos(X, y)

    # 3. Resultados individuais
    exibir_resultados_ciclos(df_resultados)

    # 4. Estatisticas das metricas
    resumo, media = analisar_resultados(df_resultados)

    # 5. Relatorio e matrizes de confusao
    matriz, matriz_normalizada = avaliar_classificacao(
        classes_reais,
        classes_previstas,
    )

    # 6. Geracao dos graficos
    gerar_graficos(
        df_resultados,
        matriz,
        matriz_normalizada,
    )

    # 7. Salvamento das tabelas
    salvar_resultados(df_resultados, resumo)

    # 8. Painel de finalizacao
    console.print()
    console.print(
        Panel(
            "[wine.success]EXPERIMENTO CONCLUIDO COM SUCESSO![/]\n\n"
            f"[wine.muted]Acuracia media: [/][wine.value]"
            f"{media * 100:.2f}%[/]\n"
            f"[wine.muted]Tempo dos 20 ciclos: [/][wine.value]"
            f"{tempo_total:.3f} segundos[/]\n"
            "[wine.muted]20 ciclos executados e avaliados.[/]\n"
            "[wine.muted]Consulte a pasta resultados/.[/]",
            title="[wine.value]FINALIZACAO[/]",
            border_style="wine.border",
            padding=(1, 3),
            expand=False,
        )
    )


# ============================================================
# 14. PONTO DE ENTRADA DO PROGRAMA
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as erro:
        console.print()
        console.print(
            Panel(
                f"[wine.error]{type(erro).__name__}: {erro}[/]",
                title="[wine.error]ERRO NA EXECUCAO[/]",
                border_style="wine.error",
                padding=(1, 2),
            )
        )
        console.print_exception(show_locals=False)
        raise
