import os
import sys
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import moeabench as mb
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# =======================================================
# CORREÇÃO DE ROTA (Path Fix)
# =======================================================
DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))
DIR_RAIZ = os.path.dirname(DIR_ATUAL)
if DIR_RAIZ not in sys.path:
    sys.path.insert(0, DIR_RAIZ)

from src.meamt import MEAMT
from src.meamt_enxuto import MEAMT_ENXUTO

PASTA_RESULTADOS = "/mnt/steam_ssd/resultados_finais"
PASTA_GRAFICOS = "./graficos"
os.makedirs(PASTA_GRAFICOS, exist_ok=True)

def calcular_spacing(front):
    """
    Calcula a métrica de Spacing (SP). Quanto menor, mais uniforme é a distribuição.
    Utiliza a distância de Manhattan (cityblock) conforme a definição clássica de Schott.
    """
    if len(front) < 2: 
        return 0.0
    
    # Calcula a distância entre todos os pares de pontos
    dist_matrix = cdist(front, front, metric='cityblock')
    
    # Ignora a distância do ponto para ele mesmo
    np.fill_diagonal(dist_matrix, np.inf)
    
    # Distância para o vizinho mais próximo de cada ponto
    d_i = dist_matrix.min(axis=1)
    
    # Média das distâncias
    d_mean = np.mean(d_i)
    
    # Cálculo da variância das distâncias (Fórmula clássica do Spacing)
    sp = np.sqrt(np.sum((d_i - d_mean)**2) / (len(front) - 1))
    
    return sp

def extrair_metadados(nome_arquivo):
    nome_sem_ext = os.path.basename(nome_arquivo).replace('.zip', '')
    partes = nome_sem_ext.split('_')
    return partes[0], partes[1], int(partes[2].replace('M', '')), int(partes[3].replace('N', ''))

def obter_fronteira_global(exps):
    """Calcula a fronteira empírica global Z* unindo os resultados de todos os algoritmos"""
    pool = []
    for exp in exps:
        if hasattr(exp, 'runs') and len(exp.runs) > 0:
            front = exp.front()
            # CORREÇÃO 1: front já é o SmartArray com os objetivos. 
            # Verificamos apenas se ele não está vazio.
            if front is not None and len(front) > 0:
                pool.append(np.array(front)) 
            
    if not pool: 
        return np.array([])
    
    pool = np.vstack(pool)
    pool = np.unique(pool, axis=0)
    
    nds = NonDominatedSorting()
    indices_nd = nds.do(pool, only_non_dominated_front=True)
    
    return pool[indices_nd]

def compilar_resultados():
    arquivos_zip = glob.glob(os.path.join(PASTA_RESULTADOS, "*.zip"))
    print(f"Encontrados {len(arquivos_zip)} ficheiros para analisar.")
    
    grupos = {}
    for arq in arquivos_zip:
        prob, alg, m, n = extrair_metadados(arq)
        chave = (prob, m, n)
        if chave not in grupos:
            grupos[chave] = []
        grupos[chave].append((alg, arq))

    dados_finais = []

    for (prob, m, n), lista_arqs in tqdm(grupos.items(), desc="A processar Grupos"):
        exps = {}
        for alg, arq in lista_arqs:
            if not os.path.exists(arq):
                continue
                
            caminho_base = arq.replace('.zip', '')
            exp = mb.experiment()
            exp.load(caminho_base)
            exps[alg] = exp

        if not exps: 
            continue

        # --- ACHAR A FRONTEIRA DE REFERÊNCIA (Z*) ---
        Z_ref = None
        
        if "Mochila" not in prob and hasattr(mb.mops, prob):
            mop_class = getattr(mb.mops, prob)
            if "DTLZ" in prob:
                k_derivado = n - m + 1
                Z_ref = mop_class(M=m, K=k_derivado).pf(1500)
            else:
                Z_ref = mop_class(M=m, N=n).pf(1500)
        else:
            Z_ref = obter_fronteira_global(list(exps.values()))

        if Z_ref is None or len(Z_ref) == 0: 
            continue

        # --- CALCULAR MÉTRICAS ---
        for alg, exp in exps.items():
            
            hv_matrix = mb.metrics.hv(exp, ref=list(exps.values()), scale='raw', gens=-1)
            igd_matrix = mb.metrics.igdplus(exp, ref=Z_ref, gens=-1)
            
            hv_vals = hv_matrix.gen(-1)
            igd_vals = igd_matrix.gen(-1)
            
            for i, run in enumerate(exp.runs):
               
                F_atual = np.array(run.front())
                pareto_subset, taxa_erro, spacing_val = None, None, None
                spacing_val = calcular_spacing(F_atual)
                
                if "Mochila" in prob and len(F_atual) > 0:
                    distancias = cdist(F_atual, Z_ref)
                    menores = distancias.min(axis=1)
                    pontos_na_fronteira = np.sum(menores < 1e-6)
                    pareto_subset = pontos_na_fronteira / len(F_atual)
                    taxa_erro = 1.0 - pareto_subset
                
                dados_finais.append({
                    "Problema": prob, "Algoritmo": alg,
                    "M (Objetivos)": m, "N (Variáveis)": n,
                    "Semente": i + 1,
                    "Hipervolume": hv_vals[i],
                    "IGDPlus": igd_vals[i],
                    "Spacing": spacing_val,
                    "Pareto Subset": pareto_subset,
                    "Taxa de Erro": taxa_erro
                })

    df = pd.DataFrame(dados_finais)
    df.to_csv("resultados_compilados.csv", index=False)
    return df

# ----------------------------------------------------
# GERAÇÃO DE TABELAS (Padrão Acadêmico - Por Instância)
# ----------------------------------------------------
def gerar_tabelas(df):
    if df.empty: 
        return
    
    print("\nA gerar Tabelas de Resultados Isoladas por Problema...")
    
    metricas = ["Hipervolume", "IGDPlus", "Spacing", "Pareto Subset", "Taxa de Erro"]
    metricas_existentes = [m for m in metricas if df[m].notnull().any()]
    
    # 1. Agrupa e calcula as estatísticas gerais
    colunas_agrupamento = ["Problema", "M (Objetivos)", "N (Variáveis)", "Algoritmo"]
    resumo = df.groupby(colunas_agrupamento)[metricas_existentes].agg(['mean', 'std'])
    
    # 2. Formatar os valores para o padrão LaTeX "Média \pm Std"
    df_tabela = pd.DataFrame(index=resumo.index)
    for metrica in metricas_existentes:
        df_tabela[metrica] = resumo[metrica].apply(
            lambda x: f"${x['mean']:.4f} \\pm {x['std']:.4f}$" if pd.notnull(x['mean']) else "-"
            , axis=1
        )
    
    df_tabela = df_tabela.reset_index()
    
    # Encontra todas as combinações únicas de Problema + M + N
    instancias = df_tabela[['Problema', 'M (Objetivos)', 'N (Variáveis)']].drop_duplicates()
    
    arquivo_tex = "tabelas_artigo.tex"
    
    # 3. Abre um único arquivo LaTeX e escreve as tabelas separadamente
    with open(arquivo_tex, "w", encoding="utf-8") as f_tex:
        f_tex.write("% =========================================================\n")
        f_tex.write("% COLE ESTE CÓDIGO DIRETAMENTE NO SEU DOCUMENTO LATEX\n")
        f_tex.write("% Certifique-se de ter \\usepackage{booktabs} no preâmbulo\n")
        f_tex.write("% =========================================================\n\n")
        
        for _, row in instancias.iterrows():
            prob = row['Problema']
            m = row['M (Objetivos)']
            n = row['N (Variáveis)']
            
            # Filtra apenas os dados dessa instância específica
            filtro = (df_tabela['Problema'] == prob) & (df_tabela['M (Objetivos)'] == m) & (df_tabela['N (Variáveis)'] == n)
            df_instancia = df_tabela[filtro].copy()
            
            # Seleciona apenas a coluna do Algoritmo e as Métricas (oculta Prob, M e N da tabela final)
            df_print = df_instancia[['Algoritmo'] + metricas_existentes]
            
            # Salva um CSV individual caso você queira abrir no Excel depois
            nome_base = f"{prob}_M{m}_N{n}"
            df_print.to_csv(f"{PASTA_GRAFICOS}/Tabela_{nome_base}.csv", index=False)
            
            # 4. Geração do Código LaTeX
            # escape=False garante que o $\pm$ seja interpretado como código matemático e não como texto
            col_format = "l" + "c" * len(metricas_existentes)
            latex_str = df_print.to_latex(index=False, column_format=col_format, escape=False)
            
            # Adiciona o cabeçalho descritivo da tabela
            f_tex.write(f"% --- Tabela para {prob} (M={m}, N={n}) ---\n")
            f_tex.write("\\begin{table}[htpb]\n")
            f_tex.write("\\centering\n")
            f_tex.write(f"\\caption{{Resultados de desempenho para o problema {prob} ($M={m}$, $N={n}$). Valores expressos em média $\\pm$ desvio padrão sobre 30 execuções independentes.}}\n")
            f_tex.write(f"\\label{{tab:{nome_base}}}\n")
            
            # Tratamento cosmético para deixar a tabela com visual de artigo "Premium" (estilo booktabs)
            latex_str = latex_str.replace("\\toprule", "\\hline\\hline").replace("\\midrule", "\\hline").replace("\\bottomrule", "\\hline\\hline")
            
            f_tex.write(latex_str)
            f_tex.write("\\end{table}\n\n")
            
    print(f"✅ Geração concluída! Verifique o arquivo '{arquivo_tex}' na raiz do projeto.")


if __name__ == "__main__":
    if os.path.exists("resultados_compilados.csv"):
        print("CSV bruto encontrado. A carregar dados...")
        df_resultados = pd.read_csv("resultados_compilados.csv")
    else:
        df_resultados = compilar_resultados()
        
    # Substitui a chamada dos gráficos pela geração de tabelas
    gerar_tabelas(df_resultados)
    print("Processo totalmente concluído!")