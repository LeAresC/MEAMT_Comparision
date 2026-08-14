import numpy as np
import random
import sys
import os
import moeabench as mb
from moeabench.progress import get_active_pbar
from deap import creator

# Importa as suas funções base do core
sys.path.append(os.path.abspath("."))
from src.meamt_core import build_toolbox, gen_inicial_tables, run

class MEAMT(mb.moeas.BaseMoea):
    def __init__(self, problem=None, population=None, generations=None, seed=None):
        super().__init__(problem, population, generations, seed)
        self.name = "MEAMT"
        
    def evaluation(self):
        # ==========================================
        # 1. CONTRATO DO MOEABENCH: Acesso ao Problema
        # ==========================================
        mop = self.get_problem()
        n_obj = mop.M
        n_var = mop.N
        
        max_fes = self.population * self.generations
        
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
            
        def avaliacao_identidade(x):
            return x
        
        if hasattr(creator, "FitnessMin"):
            del creator.FitnessMin
            del creator.Individual
            del creator.SubPopulation

        # Variáveis de Estado para o Histórico
        self.F_gens = []
        self.X_gens = []
        self.tabelas_ref = None       
        self.last_gen_saved = 0       
        self.fes_gasto = 0            

        # ==========================================
        # 2. CONTRATO DO MOEABENCH: Avaliação Oficial
        # ==========================================
        def avaliacao_em_lote(func_dummy, individuos_invalidos):
            X_eval = np.array([list(ind) for ind in individuos_invalidos])
            
            # A chamada à evaluation_benchmark é OBRIGATÓRIA.
            # Ela processa penalidades, restrições e conta os FES para o framework.
            resultado = self.evaluation_benchmark(X_eval)['F']
            
            self.fes_gasto += len(individuos_invalidos)
            gen_atual = self.fes_gasto // self.population
            
            pbar = get_active_pbar()
            if pbar:
                pbar.update_to(gen_atual)
            
            # Lógica de Snapshot (Foto da Geração)
            if self.tabelas_ref is not None and gen_atual > self.last_gen_saved:
                self.last_gen_saved = gen_atual
                unicos = {id(ind): ind for t in self.tabelas_ref.values() for ind in t}
                pop_atual = list(unicos.values())
                
                self.X_gens.append(np.array([list(ind) for ind in pop_atual]))
                self.F_gens.append(np.array([ind.fitness.values for ind in pop_atual]))

            return [tuple(fit) for fit in resultado]
            
        toolbox = build_toolbox(avaliacao_identidade, n_var, self.population, n_obj)
        toolbox.register("map", avaliacao_em_lote)

        # Inicializa a População baseada nos limites reais do problema (mop.xl, mop.xu)
        pop_inicial = toolbox.population()
        for ind in pop_inicial:
            for i in range(n_var):
                xl = mop.xl[i] if isinstance(mop.xl, (list, np.ndarray)) else mop.xl
                xu = mop.xu[i] if isinstance(mop.xu, (list, np.ndarray)) else mop.xu
                ind[i] = random.uniform(xl, xu)

        # Primeira avaliação
        fitnesses = toolbox.map(toolbox.evaluate, pop_inicial)
        for ind, fit in zip(pop_inicial, fitnesses):
            ind.fitness.values = fit
            
        num_tables = 1 << n_obj 
        max_table_size = max(1, self.population // num_tables)
        
        tabelas = gen_inicial_tables(pop_inicial, num_tables, max_table_size, n_obj)
        self.tabelas_ref = tabelas 
        
        # Salva o Snapshot da Geração 0
        unicos_0 = {id(ind): ind for t in self.tabelas_ref.values() for ind in t}
        pop_0 = list(unicos_0.values())
        self.X_gens.append(np.array([list(ind) for ind in pop_0]))
        self.F_gens.append(np.array([ind.fitness.values for ind in pop_0]))

        avaliacoes_iniciais = len(pop_inicial)
        
        # Roda o motor Steady-State do MEAMT
        run(
            tables=tabelas, 
            num_tables=num_tables, 
            max_table_size=max_table_size, 
            max_fes=max_fes,             
            avaliacoes_iniciais=avaliacoes_iniciais,    
            toolbox=toolbox, 
            cxpb=0.9, 
            mutpb=1.0, 
            n_obj=n_obj
        )
        
        # Extrai o Resultado Final da Tabela ND
        fronteira_nd = tabelas[0]
        X_final = np.array([list(ind) for ind in fronteira_nd])
        F_final = np.array([ind.fitness.values for ind in fronteira_nd])
        
        # ==========================================
        # 3. CONTRATO DO MOEABENCH: Retorno Exato (7 Elementos)
        # ==========================================
        return (
            self.F_gens,      # 1. Histórico Completo Objetivos (Lista de Arrays)
            self.X_gens,      # 2. Histórico Completo Variáveis (Lista de Arrays)
            F_final,          # 3. Fronteira Final ND (Matriz Numpy)
            self.F_gens,      # 4. Histórico ND F (Espelhamos o histórico para evitar crash)
            self.X_gens,      # 5. Histórico ND X
            [np.array([])],   # 6. Histórico Dominados F (Lista com array vazio, exigido pela API)
            [np.array([])]    # 7. Histórico Dominados X (Lista com array vazio)
        )