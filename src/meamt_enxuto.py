import numpy as np
import random
import os
import sys
import moeabench as mb
from deap import creator, tools

# Importa do core ENXUTO
sys.path.append(os.path.abspath("."))
from src.meamt_core_enxuto import build_toolbox, gen_inicial_tables, run

class MEAMT_ENXUTO(mb.moeas.BaseMoea):
    def __init__(self, problem=None, population=None, generations=None, seed=None):
        super().__init__(problem, population, generations, seed)
        self.name = "MEAMT-ENXUTO"
        
    def evaluation(self):
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

        # Variáveis de Estado para o Histórico Rigoroso
        self.F_gens = []
        self.X_gens = []
        self.F_nd_gens = []  
        self.X_nd_gens = []
        self.tabelas_ref = None       
        self.last_gen_saved = 0       
        self.fes_gasto = 0            

        # Função auxiliar para extrair e salvar a foto da geração
        def tirar_foto_geracao(pop_atual):
            self.X_gens.append(np.array([list(ind) for ind in pop_atual]))
            self.F_gens.append(np.array([ind.fitness.values for ind in pop_atual]))
            
            fronts = tools.sortNondominated(pop_atual, len(pop_atual))
            nd_front = fronts[0] # Extrai apenas a Frente 0 (Não-Dominados)
            
            self.X_nd_gens.append(np.array([list(ind) for ind in nd_front]))
            self.F_nd_gens.append(np.array([ind.fitness.values for ind in nd_front]))

        def avaliacao_em_lote(func_dummy, individuos_invalidos):
            X_eval = np.array([list(ind) for ind in individuos_invalidos])
            resultado = self.evaluation_benchmark(X_eval)['F']
            
            if resultado.shape[1] != n_obj:
                print(f"!!! ERRO DE DENSIDADE !!!")
                print(f"Problema: {self.problem.name}")
                print(f"Esperado n_obj: {n_obj}")
                print(f"Recebido do MoeaBench: {resultado.shape[1]}")
            
            self.fes_gasto += len(individuos_invalidos)
            gen_atual = self.fes_gasto // self.population
            
            # Gatilho para a foto da geração
            if self.tabelas_ref is not None and gen_atual > self.last_gen_saved:
                self.last_gen_saved = gen_atual
                unicos = {id(ind): ind for t in self.tabelas_ref.values() for ind in t}
                tirar_foto_geracao(list(unicos.values()))

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
            
        # AQUI É A VERSÃO ENXUTA: M+1 tabelas
        num_tables = 1 + n_obj 
        max_table_size = max(1, self.population // num_tables)
        
        # AQUI É A VERSÃO ENXUTA: Sem o argumento n_obj
        tabelas = gen_inicial_tables(pop_inicial, num_tables, max_table_size)
        self.tabelas_ref = tabelas 
        
        # Foto da Geração 0
        unicos_0 = {id(ind): ind for t in self.tabelas_ref.values() for ind in t}
        tirar_foto_geracao(list(unicos_0.values()))

        avaliacoes_iniciais = len(pop_inicial)
        
        # AQUI É A VERSÃO ENXUTA: Sem o argumento n_obj
        run(
            tables=tabelas, 
            num_tables=num_tables, 
            max_table_size=max_table_size, 
            max_fes=max_fes,             
            avaliacoes_iniciais=avaliacoes_iniciais,    
            toolbox=toolbox, 
            cxpb=0.9, 
            mutpb=1.0, 
            reset=100
        )
        
        # Extrai o Resultado Final da Tabela ND
        fronteira_nd = tabelas[0]
        X_final = np.array([list(ind) for ind in fronteira_nd])
        F_final = np.array([ind.fitness.values for ind in fronteira_nd])
        
        # Retorna a tupla exata com o histórico
        return (
            self.F_gens,      # 1. Histórico Geral F
            self.X_gens,      # 2. Histórico Geral X
            F_final,          # 3. Fronteira Final ND
            self.F_nd_gens,   # 4. Histórico ND F 
            self.X_nd_gens,   # 5. Histórico ND X
            [np.array([])],   # 6. Histórico Dominados F
            [np.array([])]    # 7. Histórico Dominados X
        )