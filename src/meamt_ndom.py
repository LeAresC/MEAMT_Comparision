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
        # HEURÍSTICA DE ALOCAÇÃO DE POPULAÇÃO
        # ==========================================
        num_tables = 1 << n_obj 
        max_table_size = [0] * num_tables
        
        # 1. Calcula o "Peso Geométrico" de cada tabela
        pesos = [0] * num_tables
        for i in range(1, num_tables):
            # Conta quantos objetivos estão ativos nesta máscara (k)
            k = bin(i).count('1') 
            
            # Heurística Exponencial: A necessidade de pontos cresce com a dimensão.
            # Usar base 3 garante que o miolo (k=3) tenha muito mais vagas que as arestas (k=2).
            pesos[i] = 3 ** k 
            
        soma_pesos = sum(pesos)
        
        # 2. Distribui as vagas proporcionalmente ao peso
        pop_restante = self.population
        for i in range(1, num_tables):
            # Garante no mínimo 2 vagas (elitismo) para tabelas de vértice (k=1)
            tamanho = max(2, int((pesos[i] / soma_pesos) * self.population))
            max_table_size[i] = tamanho
            pop_restante -= tamanho
            
        # 3. Correção de Arredondamento
        # Qualquer sobra populacional vai para a tabela mais exigente (a última: [1,1,1...])
        max_table_size[num_tables - 1] += pop_restante 
        
        # A nova assinatura retorna apenas as tabelas (sem z_ideal)
        tabelas = gen_inicial_tables(pop_inicial, num_tables, max_table_size, n_obj)
        self.tabelas_ref = tabelas 
        
        # Salva o Snapshot da Geração 0
        unicos_0 = {id(ind): ind for t in self.tabelas_ref.values() for ind in t}
        pop_0 = list(unicos_0.values())
        self.X_gens.append(np.array([list(ind) for ind in pop_0]))
        self.F_gens.append(np.array([ind.fitness.values for ind in pop_0]))
        
        # ==========================================
        # 3. MOTOR EVOLUTIVO (Nova Assinatura Geracional)
        # ==========================================
        run(
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
        # 4. EXTRAÇÃO DO ARQUIVO GLOBAL (Fronteira 0)
        # ==========================================
        # Agrupa todos os sobreviventes das tabelas direcionais em uma única piscina
        populacao_final = []
        for i in range(1, num_tables):
            populacao_final.extend(tabelas[i])
            
        # first_front_only=True garante que pegaremos apenas a fronteira perfeita (Front 0)
        fronteiras = tools.sortNondominated(populacao_final, len(populacao_final), first_front_only=True)
        fronteira_nd = fronteiras[0] if fronteiras else []
        
        X_final = np.array([list(ind) for ind in fronteira_nd])
        F_final = np.array([ind.fitness.values for ind in fronteira_nd])
        
        # ==========================================
        # 5. CONTRATO DO MOEABENCH: Retorno Exato
        # ==========================================
        return (
            self.F_gens,      # 1. Histórico Completo Objetivos (Lista de Arrays)
            self.X_gens,      # 2. Histórico Completo Variáveis (Lista de Arrays)
            F_final,          # 3. Fronteira Final ND (Matriz Numpy)
            self.F_gens,      # 4. Histórico ND F (Espelhado para evitar crash)
            self.X_gens,      # 5. Histórico ND X
            [np.array([])],   # 6. Histórico Dominados F (Exigido pela API)
            [np.array([])]    # 7. Histórico Dominados X 
        )