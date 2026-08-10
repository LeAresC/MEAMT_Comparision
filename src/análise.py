import os
import sys
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import moeabench as mb

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

def sanitizar_outliers(exps_dict, Z_ref):
    nadir_real = np.max(Z_ref, axis=0)
    nadir_seguro = np.where(nadir_real <= 0, 1.0, nadir_real)
    limite_tolerancia = nadir_seguro * 1.5 
    
    for nome_alg, exp in exps_dict.items():
        if exp is None or not hasattr(exp, 'runs'): continue
        for run in exp.runs:
            if hasattr(run, '_F_nd_history') and len(run._F_nd_history) > 0:
                F_atual = np.array(run._F_nd_history[-1])
                if F_atual.size == 0: continue 
                mascara = np.all(F_atual <= limite_tolerancia, axis=1)
                run._F_nd_history[-1] = F_atual[mascara]

def calcular_spacing(front):
    if len(front) < 2: return 0.0
    dist_matrix = cdist(front, front, metric='cityblock')
    np.fill_diagonal(dist_matrix, np.inf)
    d_i = dist_matrix.min(axis=1)
    d_mean = np.mean(d_i)
    return np.sqrt(np.sum((d_i - d_mean)**2) / (len(front) - 1))

def calcular_igd_plus_fast(front, Z_ref):
    """
    Versão ultrarrápida vetorizada em Numpy para calcular IGD+.
    Cálculo integral, sem limites de amostragem.
    """
    if len(front) == 0 or len(Z_ref) == 0:
        return np.nan
        
    # Broadcasting para distância unilateral (max(a_i - z_i, 0))
    diff = front[np.newaxis, :, :] - Z_ref[:, np.newaxis, :]
    diff = np.maximum(diff, 0)
    dists = np.sqrt(np.sum(diff**2, axis=2))
    min_dists = np.min(dists, axis=1)
    
    return np.mean(min_dists)

def extrair_metadados(nome_arquivo):
    nome_sem_ext = os.path.basename(nome_arquivo).replace('.zip', '')
    partes = nome_sem_ext.split('_')
    return partes[0], partes[1], int(partes[2].replace('M', '')), int(partes[3].replace('N', ''))

def obter_fronteira_global(exps):
    """
    Calcula a fronteira empírica global Z* unindo os resultados.
    Sem cortes de tamanho, preservando 100% dos pontos não-dominados.
    """
    pool = []
    for exp in exps:
        if hasattr(exp, 'runs') and len(exp.runs) > 0:
            front = exp.front() 
            if front is not None and len(front) > 0:
                pool.append(np.array(front)) 
            
    if not pool: return np.array([])
    
    pool = np.vstack(pool)
    pool = np.unique(pool, axis=0)
    
    is_efficient = np.ones(pool.shape[0], dtype=bool)
    for i, c in enumerate(pool):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(pool[is_efficient] < c, axis=1)
            is_efficient[i] = True
            
    return pool[is_efficient]

def compilar_resultados():
    arquivos_zip = glob.glob(os.path.join(PASTA_RESULTADOS, "*.zip"))
    print(f"Encontrados {len(arquivos_zip)} ficheiros para analisar.")
    
    grupos = {}
    for arq in arquivos_zip:
        prob, alg, m, n = extrair_metadados(arq)
        chave = (prob, m, n)
        if chave not in grupos: grupos[chave] = []
        grupos[chave].append((alg, arq))

    dados_finais = []

    for (prob, m, n), lista_arqs in tqdm(grupos.items(), desc="A processar Grupos"):
        exps = {}
        for alg, arq in lista_arqs:
            if not os.path.exists(arq): continue
            caminho_base = arq.replace('.zip', '')
            exp = mb.experiment()
            exp.load(caminho_base)
            exps[alg] = exp

        if not exps: continue

        # --- ACHAR A FRONTEIRA DE REFERÊNCIA (Z*) ---
        Z_ref = None
        if "Mochila" not in prob and hasattr(mb.mops, prob):
            mop_class = getattr(mb.mops, prob)
            if "DTLZ" in prob:
                k_derivado = n - m + 1
                # Mantém 1500 apenas para problemas contínuos (onde a fronteira é uma fórmula teórica)
                Z_ref = mop_class(M=m, K=k_derivado).pf(1500)
            else:
                Z_ref = mop_class(M=m, N=n).pf(1500)
        else:
            # Usa a fronteira global empírica INTEGRAL
            Z_ref = obter_fronteira_global(list(exps.values()))

        if Z_ref is None or len(Z_ref) == 0: continue

        # =========================================================
        # 1. CÁLCULO DE MÉTRICAS ROBUSTAS (DADOS BRUTOS)
        # =========================================================
        igd_vals_dict = {}
        spacing_vals_dict = {}
        pareto_vals_dict = {}
        erro_vals_dict = {}

        for alg, exp in exps.items():
            igd_vals_dict[alg] = []
            spacing_vals_dict[alg] = []
            pareto_vals_dict[alg] = []
            erro_vals_dict[alg] = []
            
            for run in exp.runs:
                F_atual = np.array(run.front())
                
                if len(F_atual) == 0:
                    igd_vals_dict[alg].append(np.nan)
                    spacing_vals_dict[alg].append(np.nan)
                    pareto_vals_dict[alg].append(np.nan)
                    erro_vals_dict[alg].append(np.nan)
                    continue
                
                # IGD+ integral
                igd_vals_dict[alg].append(calcular_igd_plus_fast(F_atual, Z_ref))
                
                # Spacing integral
                spacing_vals_dict[alg].append(calcular_spacing(F_atual))
                
                if "Mochila" in prob:
                    # Pareto Subset calculado contra 100% da fronteira Z_ref
                    distancias = cdist(F_atual, Z_ref)
                    menores = distancias.min(axis=1)
                    pontos_na_fronteira = np.sum(menores < 1e-6)
                    p_sub = pontos_na_fronteira / len(F_atual)
                    pareto_vals_dict[alg].append(p_sub)
                    erro_vals_dict[alg].append(1.0 - p_sub)
                else:
                    pareto_vals_dict[alg].append(None)
                    erro_vals_dict[alg].append(None)

        # =========================================================
        # 2. SANITIZAÇÃO DE OUTLIERS
        # =========================================================
        sanitizar_outliers(exps, Z_ref)

        # =========================================================
        # 3. CÁLCULO DO HIPERVOLUME
        # =========================================================
        for alg, exp in exps.items():
            hv_matrix = mb.metrics.hv(exp, ref=list(exps.values()), scale='raw', gens=-1)
            hv_vals = hv_matrix.gen(-1)
            
            for i in range(len(exp.runs)):
                hv_final = hv_vals[i]
                if np.isnan(hv_final): hv_final = 0.0
                
                dados_finais.append({
                    "Problema": prob, "Algoritmo": alg,
                    "M (Objetivos)": m, "N (Variáveis)": n,
                    "Semente": i + 1,
                    "Hipervolume": hv_final,
                    "IGDPlus": igd_vals_dict[alg][i],
                    "Spacing": spacing_vals_dict[alg][i],
                    "Pareto Subset": pareto_vals_dict[alg][i],
                    "Taxa de Erro": erro_vals_dict[alg][i]
                })

    df = pd.DataFrame(dados_finais)
    df.to_csv("resultados_compilados.csv", index=False)
    return df

def gerar_tabelas(df):
    if df.empty: return
    print("\nA gerar Tabelas de Resultados Isoladas por Problema...")
    
    metricas = ["Hipervolume", "IGDPlus", "Spacing", "Pareto Subset", "Taxa de Erro"]
    metricas_existentes = [m for m in metricas if df[m].notnull().any()]
    
    colunas_agrupamento = ["Problema", "M (Objetivos)", "N (Variáveis)", "Algoritmo"]
    resumo = df.groupby(colunas_agrupamento)[metricas_existentes].agg(['mean', 'std'])
    
    df_tabela = pd.DataFrame(index=resumo.index)
    for metrica in metricas_existentes:
        df_tabela[metrica] = resumo[metrica].apply(
            lambda x: f"${x['mean']:.4f} \\pm {x['std']:.4f}$" if pd.notnull(x['mean']) else "-"
            , axis=1
        )
    
    df_tabela = df_tabela.reset_index()
    instancias = df_tabela[['Problema', 'M (Objetivos)', 'N (Variáveis)']].drop_duplicates()
    arquivo_tex = "tabelas_artigo.tex"
    
    with open(arquivo_tex, "w", encoding="utf-8") as f_tex:
        f_tex.write("% =========================================================\n")
        f_tex.write("% COLE ESTE CÓDIGO DIRETAMENTE NO SEU DOCUMENTO LATEX\n")
        f_tex.write("% Certifique-se de ter \\usepackage{booktabs} no preâmbulo\n")
        f_tex.write("% =========================================================\n\n")
        
        for _, row in instancias.iterrows():
            prob = row['Problema']
            m = row['M (Objetivos)']
            n = row['N (Variáveis)']
            
            filtro = (df_tabela['Problema'] == prob) & (df_tabela['M (Objetivos)'] == m) & (df_tabela['N (Variáveis)'] == n)
            df_instancia = df_tabela[filtro].copy()
            df_print = df_instancia[['Algoritmo'] + metricas_existentes]
            
            nome_base = f"{prob}_M{m}_N{n}"
            df_print.to_csv(f"{PASTA_GRAFICOS}/Tabela_{nome_base}.csv", index=False)
            
            col_format = "l" + "c" * len(metricas_existentes)
            latex_str = df_print.to_latex(index=False, column_format=col_format, escape=False)
            
            f_tex.write(f"% --- Tabela para {prob} (M={m}, N={n}) ---\n")
            f_tex.write("\\begin{table}[htpb]\n")
            f_tex.write("\\centering\n")
            f_tex.write(f"\\caption{{Resultados de desempenho para o problema {prob} ($M={m}$, $N={n}$). Valores expressos em média $\\pm$ desvio padrão sobre 30 execuções independentes.}}\n")
            f_tex.write(f"\\label{{tab:{nome_base}}}\n")
            
            latex_str = latex_str.replace("\\toprule", "\\hline\\hline").replace("\\midrule", "\\hline").replace("\\bottomrule", "\\hline\\hline")
            f_tex.write(latex_str)
            f_tex.write("\\end{table}\n\n")
            
    print(f"✅ Geração concluída! Verifique o arquivo '{arquivo_tex}' na raiz do projeto.")

if __name__ == "__main__":
    # Remove o CSV antigo se quiser forçar o recálculo bruto
    if os.path.exists("resultados_compilados.csv"):
        print("CSV bruto encontrado. A carregar dados...")
        df_resultados = pd.read_csv("resultados_compilados.csv")
    else:
        df_resultados = compilar_resultados()
        
    gerar_tabelas(df_resultados)
    print("Processo totalmente concluído!")