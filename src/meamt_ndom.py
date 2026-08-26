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


def _normalize_constraints(constraints, n_individuals):
    G = np.asarray(constraints, dtype=float)

    if G.size == 0 and n_individuals > 0:
        raise ValueError("G vazio para um lote com indivíduos")
    if G.ndim == 0:
        if n_individuals != 1:
            raise ValueError("G escalar só é válido para um indivíduo")
        G = G.reshape(1, 1)
    elif G.ndim == 1:
        if n_individuals == 1:
            G = G.reshape(1, -1)
        elif G.shape[0] == n_individuals:
            G = G.reshape(n_individuals, 1)
        else:
            raise ValueError(
                f"shape de G incompatível: {G.shape} para {n_individuals} indivíduos"
            )
    elif G.ndim == 2:
        if G.shape[0] == n_individuals:
            pass
        elif n_individuals > 1 and G.shape == (1, n_individuals):
            G = G.T
        else:
            raise ValueError(
                f"shape de G incompatível: {G.shape} para {n_individuals} indivíduos"
            )
    else:
        raise ValueError(f"G deve ter no máximo 2 dimensões; recebido ndim={G.ndim}")

    if not np.all(np.isfinite(G)):
        raise ValueError("G contém valores de restrição não finitos")
    return G


def _normalize_objectives(objectives, n_individuals, n_obj):
    F = np.asarray(objectives, dtype=float)
    if F.ndim == 1 and n_individuals == 1:
        F = F.reshape(1, -1)
    if F.ndim != 2 or F.shape != (n_individuals, n_obj):
        raise ValueError(
            f"shape de F incompatível: {F.shape}; esperado {(n_individuals, n_obj)}"
        )
    if not np.all(np.isfinite(F)):
        raise ValueError("F contém valores de objetivo não finitos")
    return F


def _number_of_constraints(problem):
    for name in ("get_n_ieq_constr", "n_ieq_constr"):
        value = getattr(problem, name, None)
        if value is not None:
            value = value() if callable(value) else value
            return int(value)
    return 0


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
        n_constraints = _number_of_constraints(mop)
        
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
            
        def avaliacao_identidade(x):
            return x
        
        for class_name in ("FitnessMin", "Individual", "SubPopulation"):
            if hasattr(creator, class_name):
                delattr(creator, class_name)

        # Variáveis de Estado para o Histórico
        self.F_gens = []
        self.X_gens = []
        self.F_nd_gens = []
        self.X_nd_gens = []
        self.F_dom_gens = []
        self.X_dom_gens = []
        self.fes_gasto = 0

        # ==========================================
        # 2. CONTRATO DO MOEABENCH: Avaliação Oficial
        # ==========================================
        def avaliacao_em_lote(func_dummy, individuos_invalidos):
            X_eval = np.array([list(ind) for ind in individuos_invalidos])
            
            # A chamada à evaluation_benchmark é OBRIGATÓRIA.
            # Ela processa penalidades, restrições e conta os FES para o framework.
            resultado = self.evaluation_benchmark(X_eval)
            F = _normalize_objectives(
                resultado["F"], len(individuos_invalidos), n_obj
            )

            if "G" in resultado:
                G = _normalize_constraints(
                    resultado["G"], len(individuos_invalidos)
                )
                CV = np.maximum(G, 0.0).sum(axis=1)
            elif n_constraints > 0:
                raise ValueError(
                    "problema restrito não retornou G em evaluation_benchmark"
                )
            else:
                CV = np.zeros(len(individuos_invalidos), dtype=float)

            for ind, cv in zip(individuos_invalidos, CV):
                ind.fitness.constraint_violation = float(cv)
            
            self.fes_gasto += len(individuos_invalidos)

            return [tuple(fit) for fit in F]
            
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

        def snapshot_callback(current_tables):
            unique = {
                id(ind): ind
                for table_id in range(1, num_tables)
                for ind in current_tables[table_id]
            }
            population = list(unique.values())

            if population:
                nd = tools.sortNondominated(
                    population,
                    len(population),
                    first_front_only=True,
                )[0]
                nd_ids = {id(ind) for ind in nd}
                dominated = [ind for ind in population if id(ind) not in nd_ids]
            else:
                nd = []
                dominated = []

            def objective_array(individuals):
                if not individuals:
                    return np.zeros((0, n_obj))
                return np.asarray([ind.fitness.values for ind in individuals])

            def decision_array(individuals):
                if not individuals:
                    return np.zeros((0, n_var))
                return np.asarray([list(ind) for ind in individuals])

            self.F_gens.append(objective_array(population))
            self.X_gens.append(decision_array(population))
            self.F_nd_gens.append(objective_array(nd))
            self.X_nd_gens.append(decision_array(nd))
            self.F_dom_gens.append(objective_array(dominated))
            self.X_dom_gens.append(decision_array(dominated))

            pbar = get_active_pbar()
            if pbar:
                generation = min(len(self.F_gens) - 1, self.generations)
                pbar.update_to(generation)
        
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
            n_obj=n_obj,
            snapshot_callback=snapshot_callback,
        )
        
        # ==========================================
        # 5. EXTRAÇÃO DO ARQUIVO GLOBAL (Fronteira 0)
        # ==========================================
        # MUDANÇA ABSOLUTA: Não precisa mais buscar em todas as tabelas nem dar sort.
        # A Tabela 0 já é o nosso Arquivo Externo Elitista perfeito!
        arquivo_externo = tables[0]
        
        F_final = (
            np.asarray([ind.fitness.values for ind in arquivo_externo])
            if arquivo_externo
            else np.zeros((0, n_obj))
        )
        
        # ==========================================
        # 6. CONTRATO DO MOEABENCH: Retorno Exato
        # ==========================================
        return (
            self.F_gens,      
            self.X_gens,      
            F_final,          
            self.F_nd_gens,
            self.X_nd_gens,
            self.F_dom_gens,
            self.X_dom_gens,
        )
