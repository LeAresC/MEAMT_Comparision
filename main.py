import os
import gc
import sys
import numpy as np
import moeabench as mb
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ==========================================
# 1. IMPORTAÇÕES LOCAIS (DA PASTA SRC)
# ==========================================
sys.path.append(os.path.abspath("."))
from src.meamt import MEAMT
from src.meamt_enxuto import MEAMT_ENXUTO
from src.mochila import MochilaMultiobjetivo

# ==========================================
# 2. CONFIGURAÇÕES GLOBAIS
# ==========================================
MAX_FES = 300000
SEMENTES = 30

# Tamanhos bases definidos
POP_PADRAO = 300            # População para NSGA3, NSGA2, MOEAD
MIN_TABLES_MEAMT = 30       # Tamanho da tabela do MEAMT normal
MIN_TABLES_ENXUTO = 100    # Tamanho da tabela do MEAMT enxuto

CAMINHO_BASE = "/mnt/steam_ssd/resultados_finais"
PASTA_RESULTADOS = os.path.expanduser(CAMINHO_BASE)
os.makedirs(PASTA_RESULTADOS, exist_ok=True)

MAX_WORKERS = 15

# A função agora recebe 'mop' para descobrir o M e gerar as populações exatas
def get_algoritmos(mop):
    m = mop.M
    
    # 1. MEAMT Normal: 20 indivíduos por tabela, 2^M tabelas
    pop_meamt = MIN_TABLES_MEAMT * (2 ** m)
    gen_meamt = MAX_FES // pop_meamt
    
    # 2. MEAMT Enxuto: 100 indivíduos por tabela, M+1 tabelas
    pop_enxuto = MIN_TABLES_ENXUTO * (m + 1)
    gen_enxuto = MAX_FES // pop_enxuto
    
    # 3. Algoritmos Padrão: População fixa em 300
    gen_padrao = MAX_FES // POP_PADRAO
    
    return [
        MEAMT(population=pop_meamt, generations=gen_meamt),
        MEAMT_ENXUTO(population=pop_enxuto, generations=gen_enxuto),
        mb.moeas.NSGA3(population=POP_PADRAO, generations=gen_padrao),
        mb.moeas.NSGA2(population=POP_PADRAO, generations=gen_padrao),
        mb.moeas.MOEAD(population=POP_PADRAO, generations=gen_padrao)
    ]

def get_problemas():
    problemas = []
    
    #--- A. Suíte DTLZ (3, 5, 10 Objetivos) ---
    for n_obj in [7]:
       problemas.append(mb.mops.DTLZ1(M=n_obj, N=n_obj + 5 - 1))
       problemas.append(mb.mops.DTLZ2(M=n_obj, N=n_obj + 10 - 1))
       problemas.append(mb.mops.DTLZ3(M=n_obj, N=n_obj + 10 - 1))
       problemas.append(mb.mops.DTLZ4(M=n_obj, N=n_obj + 10 - 1))
        
    # --- B. Suíte Mochila (30, 50, 100 itens; 2 a 6 Objetivos) ---
    for n_itens in [30, 50, 100]:
        for n_obj in [3, 5, 7]:
            problemas.append(MochilaMultiobjetivo(n_itens=n_itens, n_obj=n_obj))
            
    # --- C. DPF (3, 5, 10) ---
    for n_obj in [7]:
         n_var_dpf = n_obj + 20 - 1
         problemas.append(mb.mops.DPF1(M=n_obj, N=n_var_dpf))
         problemas.append(mb.mops.DPF2(M=n_obj, N=n_var_dpf))
         problemas.append(mb.mops.DPF3(M=n_obj, N=n_var_dpf))
         problemas.append(mb.mops.DPF4(M=n_obj, N=n_var_dpf))
         problemas.append(mb.mops.DPF5(M=n_obj, N=n_var_dpf))

    return problemas


# ==========================================
# 3. A FUNÇÃO DO TRABALHADOR (WORKER)
# ==========================================
def rodar_experimento_isolado(mop_instance, moea_instance, idx, total_exps):
    nome_mop = mop_instance.__class__.__name__
    nome_moea = getattr(moea_instance, 'name', moea_instance.__class__.__name__)
    
    nome_experimento = f"{nome_mop}_{nome_moea}_M{mop_instance.M}_N{mop_instance.N}"
    
    exp = mb.experiment()
    exp.mop = mop_instance
    exp.moea = moea_instance
    exp.name = nome_experimento
    
    caminho_base = os.path.join(PASTA_RESULTADOS, exp.name)
    caminho_zip = f"{caminho_base}.zip"
    
    if os.path.exists(caminho_zip):
        return f"⏩ [PULADO] {exp.name} ({idx:03d}/{total_exps:03d}) - Já executado."

    for semente_idx in range(SEMENTES):
        # Roda estritamente 1 semente de cada vez
        exp.run(repeat=1, silent=True)
        
        # Pega na semente que ACABOU de rodar e decepa o histórico dela imediatamente
        semente_atual = exp.runs[-1]
        
        if hasattr(semente_atual, '_F_history') and semente_atual._F_history:
            semente_atual._F_history = [semente_atual._F_history[-1]]
            
        if hasattr(semente_atual, '_F_nd_history') and semente_atual._F_nd_history:
            semente_atual._F_nd_history = [semente_atual._F_nd_history[-1]]
            
        # Deleta as variáveis para sempre
        if hasattr(semente_atual, '_X_history'):
            semente_atual._X_history = []
            
        if hasattr(semente_atual, '_X_nd_history'):
            semente_atual._X_nd_history = []
            
        gc.collect()

    exp.save(caminho_base, mode="all")
    
    nome_exp_salvo = exp.name
    
    del exp
    del mop_instance
    del moea_instance
    gc.collect() 
    
    return f"✅ [CONCLUÍDO] {nome_exp_salvo} ({idx:03d}/{total_exps:03d})"


# ==========================================
# 4. ORQUESTRAÇÃO COM BARRA DE PROGRESSO
# ==========================================
if __name__ == "__main__":
    mb.system.version(show=True)
    print(f"\n🚀 Iniciando Otimização Dinâmica em .ZIP (max_tasks_per_child=1)...")
    
    # 1. Pré-calcular as tarefas para sabermos o total exato
    tarefas_iniciais = []
    problemas = get_problemas()
    
    for mop in problemas:
        # Aqui, os algoritmos são gerados especificamente para o 'M' deste problema
        algoritmos_adaptados = get_algoritmos(mop)
        for moea in algoritmos_adaptados:
            tarefas_iniciais.append((mop, moea))
            
    total_exps = len(tarefas_iniciais)
    
    # 2. Formatar as tarefas para o formato final com o idx e total
    tarefas = []
    for idx, (mop, moea) in enumerate(tarefas_iniciais, start=1):
        tarefas.append((mop, moea, idx, total_exps))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS, max_tasks_per_child=1) as executor:
        futuros = [
            executor.submit(rodar_experimento_isolado, mop, moea, i, tot) 
            for (mop, moea, i, tot) in tarefas
        ]
        
        for futuro in tqdm(as_completed(futuros), total=total_exps, desc="Progresso Geral", unit="exp", dynamic_ncols=True):
            try:
                msg = futuro.result()
                if msg is not None:
                    tqdm.write(str(msg))
            except Exception as e:
                tqdm.write(f"❌ [ERRO CRÍTICO] Falha em um experimento: {e}")

    print("\n🎉 Todas as simulações dinâmicas foram concluídas com sucesso!")