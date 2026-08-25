import numpy as np
import random
import sys
import os
import moeabench as mb
from moeabench.progress import get_active_pbar
from deap import creator, tools

# Importa as suas funções base do core
sys.path.append(os.path.abspath("."))
from src.meamt_core_ndom import build_toolbox, gen_inicial_tables, run

class MEAMT_NDOM(mb.moeas.BaseMoea):
    def __init__(self, problem=None, population=None, generations=None, seed=None):
        super().__init__(problem, population, generations, seed)
        self.name = "MEAMTNDOM"
        
    def evaluation(self):
        # ==========================================
        # 1. CONTRATO DO MOEABENCH: Acesso ao Problema
        # ==========================================
        mop = self.get_problem()
        n_obj = mop.M
        n_var = mop.N
        
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
            
            
            # Na arquitetura geracional, os FES avançam em lotes de ~pop_size
            gen_atual = self.fes_gasto // self.population
            
            pbar = get_active_pbar()
            if pbar:
                pbar.update_to(gen_atual)
                 
            # Lógica de Snapshot (Foto da Geração para o Histórico)
            if self.tabelas_ref is not None and gen_atual > self.last_gen_saved:
                self.last_gen_saved = gen_atual
                
                # Coleta todos os indivíduos únicos distribuídos nas tabelas direcionais
                unicos = {id(ind): ind for t in self.tabelas_ref.values() for ind in t}
                pop_atual = list(unicos.values())
                
                if pop_atual:
                    self.X_gens.append(np.array([list(ind) for ind in pop_atual]))
                    self.F_gens.append(np.array([ind.fitness.values for ind in pop_atual]))

            return [tuple(fit) for fit in resultado]
            
        toolbox = build_toolbox(avaliacao_identidade, n_var, self.population, n_obj)
        toolbox.register("map", avaliacao_em_lote)

        # Inicializa a População baseada nos limites reais do problema
        pop_inicial = toolbox.population()
        for ind in pop_inicial:
            for i in range(n_var):
                xl = mop.xl[i] if isinstance(mop.xl, (list, np.ndarray)) else mop.xl
                xu = mop.xu[i] if isinstance(mop.xu, (list, np.ndarray)) else mop.xu
                ind[i] = random.uniform(xl, xu)

        # Primeira avaliação (Geração 0)
        fitnesses = toolbox.map(toolbox.evaluate, pop_inicial)
        for ind, fit in zip(pop_inicial, fitnesses):
            ind.fitness.values = fit
            
        # ==========================================
        # 3. ALOCAÇÃO E INICIALIZAÇÃO DAS TABELAS (Igualitária)
        # ==========================================
        num_tables = 1 << n_obj 
        max_table_size = [0] * num_tables
        
        # Divisão igualitária entre todas as tabelas ativas
        tabelas_ativas = num_tables - 1
        vagas_por_tabela = self.population // tabelas_ativas
        
        for i in range(1, num_tables):
            max_table_size[i] = vagas_por_tabela
            
        # O resto da divisão inteira vai para a tabela global (Tabela 7)
        max_table_size[-1] += self.population % tabelas_ativas 
        
        tabelas = gen_inicial_tables(pop_inicial, num_tables, max_table_size, n_obj)
        
        # Inicializa a Tabela 0 (Cofre) no Wrapper para o Snapshot 0
        tabelas[0] = creator.SubPopulation()
        todas_iniciais = [ind for i in range(1, num_tables) for ind in tabelas[i]]
        if todas_iniciais:
            fronts_ini = tools.sortNondominated(todas_iniciais, len(todas_iniciais), first_front_only=True)
            tabelas[0].extend(fronts_ini[0])

        self.tabelas_ref = tabelas
        
        # ==========================================
        # 4. MOTOR EVOLUTIVO (Nova Assinatura Geracional)
        # ==========================================
        tables = run(
            tables=tabelas, 
            num_tables=num_tables, 
            pop_size=self.population,
            ngen=self.generations,
            max_table_size=max_table_size, 
            toolbox=toolbox, 
            cxpb=0.9, 
            mutpb=1.0, 
            n_obj=n_obj
        )
        
        # ==========================================
        # 5. EXTRAÇÃO DO ARQUIVO GLOBAL (Fronteira 0)
        # ==========================================
        # MUDANÇA ABSOLUTA: Não precisa mais buscar em todas as tabelas nem dar sort.
        # A Tabela 0 já é o nosso Arquivo Externo Elitista perfeito!
        arquivo_externo = tables[0]
        
        X_final = np.array([list(ind) for ind in arquivo_externo])
        F_final = np.array([ind.fitness.values for ind in arquivo_externo])
        
        # ==========================================
        # 6. CONTRATO DO MOEABENCH: Retorno Exato
        # ==========================================
        return (
            self.F_gens,      
            self.X_gens,      
            F_final,          
            self.F_gens,      
            self.X_gens,      
            [np.array([])],   
            [np.array([])]    
        )