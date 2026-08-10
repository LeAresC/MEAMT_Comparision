import random
import bisect
import numpy as np
from deap import creator, base, tools

# ==========================================
# 1. SETUP DE CLASSES E TOOLBOX
# ==========================================
def setup_deap_classes(n_obj):
    """
    Inicializa as classes do DEAP.
    Como o MoeaBench lida com problemas genéricos, 
    nós padronizamos para minimização (FitnessMin) e ajustamos a 
    função de avaliação no Wrapper caso o problema original seja de maximização.
    """
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * n_obj)
        creator.create("Individual", list, fitness=creator.FitnessMin, Parent_Table=None)
        creator.create("SubPopulation", list, score=0.0)

def build_toolbox(funcao_avaliacao, ind_size, n_pop, n_obj):
    """
    Constrói o toolbox do DEAP.
    O tipo de gene (float ou bool) e os operadores genéticos são 
    passados dinamicamente para não engessar o algoritmo.
    """
    setup_deap_classes(n_obj)
    
    toolbox = base.Toolbox()
    
    # O default é float, mas pode ser sobrescrito pelo Wrapper da Mochila
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=n_pop)
    
    toolbox.register("evaluate", funcao_avaliacao) 
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=0.0, up=1.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=10.0, low=0.0, up=1.0, indpb=1.0/ind_size)
    
    return toolbox

# ==========================================
# 2. OPERADORES PRINCIPAIS DO MEAMT
# ==========================================
def calc_combined_fitness(ind, table_idx, n_obj, z_ideal):
    """Calcula a distância de Tchebycheff combinada para a tabela."""
    fit = 0
    for b in range(n_obj):
        if (table_idx >> b) & 1:
            fit = max(fit,abs(ind.fitness.values[n_obj - 1 - b] - z_ideal[n_obj - 1 - b]))
    return fit

def gen_inicial_tables(pop_ini, num_tables, table_size, n_obj):
    """Distribui a população inicial nas tabelas do algoritmo."""
    tables = dict()
    
    # Tabela 0: ND (Non Dominated)
    fronteira = tools.sortNondominated(pop_ini, len(pop_ini), first_front_only=True)[0]
    tables[0] = creator.SubPopulation(fronteira[:table_size[0]]) 
    tables[0].score = 0.0
    z_ideal = np.full(n_obj, np.inf)
    for ind in pop_ini:
        z_ideal = np.minimum(z_ideal, ind.fitness.values)
    # Tabelas Direcionais
    for i in range(1, num_tables):
        pop_ordenada = sorted(pop_ini, key=lambda ind: calc_combined_fitness(ind, i, n_obj, z_ideal))
        tables[i] = creator.SubPopulation(pop_ordenada[:table_size[i]])
        tables[i].score = 0.0
        
    return tables, z_ideal

def select_parents(tables, num_tables):
    """Seleciona pais baseado no 'score' de sucesso das tabelas."""
    selected = []
    for _ in range(2):
        random1 = random.randint(0, num_tables - 1)
        random2 = random.randint(0, num_tables - 1)

        if len(tables[random1]) == 0: winner = random2
        elif len(tables[random2]) == 0: winner = random1
        elif tables[random1].score >= tables[random2].score:
            winner = random1
        else:
            winner = random2

        ind = random.choice(tables[winner])
        ind.Parent_Table = winner
        selected.append(ind)
        
    return selected

def dominates(ind1, ind2):
    """Verifica a dominância de Pareto absoluta."""
    fit1 = ind1.fitness.wvalues
    fit2 = ind2.fitness.wvalues
    
    nao_e_pior = all(f1 >= f2 for f1, f2 in zip(fit1, fit2))
    e_melhor = any(f1 > f2 for f1, f2 in zip(fit1, fit2))
    
    return nao_e_pior and e_melhor

def update_nd_table(tabela_nd, offspring, max_table_size):
    """Mantém a tabela 0 rigorosamente não-dominada e limitada."""
    is_dominated = False
    individuos_para_remover = []
    
    for i, individuo_atual in enumerate(tabela_nd):
        if dominates(individuo_atual, offspring):
            is_dominated = True
            break
        elif dominates(offspring, individuo_atual):
            individuos_para_remover.append(i)
            
    if not is_dominated:
        for i in reversed(individuos_para_remover):
            del tabela_nd[i]
            
        tabela_nd.append(offspring)
        
        if len(tabela_nd) > 300:
           tabela_nd.pop(random.randrange(len(tabela_nd))) 
             
            
    return tabela_nd

def insert_in_tables(tables, num_tables, off, max_table_size, n_obj, z_ideal):
    """Insere o indivíduo gerado nas tabelas apropriadas."""
    # 1. Tenta atualizar a Tabela ND
    tabela_nd = tables[0]
    nova_selecao = update_nd_table(tabela_nd, off, max_table_size)
    tabela_nd[:] = nova_selecao

    if any(ind is off for ind in tabela_nd):
        if off.Parent_Table is not None:
            tables[off.Parent_Table].score += 1

    # 2. Tenta inserir nas Tabelas Direcionais
    for i in range(1, num_tables):
        bisect.insort(tables[i], off, key=lambda ind: calc_combined_fitness(ind, i, n_obj, z_ideal))

        removed_ind = None
        
        if len(tables[i]) > max_table_size[i]:
            removed_ind = tables[i].pop() 
        
        # Se sobreviveu ao truncamento, recompensa a tabela do pai
        if off is not removed_ind and off.Parent_Table is not None:
            tables[off.Parent_Table].score += 1

# ==========================================
# 3. LOOP EVOLUTIVO (O CORAÇÃO DO ALGORITMO)
# ==========================================
def run(tables, num_tables, max_table_size, max_fes, avaliacoes_iniciais, toolbox, cxpb, mutpb, n_obj, z_ideal):
    """
    A rotina Steady-State do MEA-MT.
    Removidos todos os loggers, pois o MoeaBench rastreia os dados.
    """
    fes_count = avaliacoes_iniciais
    while fes_count < max_fes:
        
        # 1. Seleção
        offspring = []
        parents = select_parents(tables, num_tables)

        off1, off2 = toolbox.clone(parents[0]), toolbox.clone(parents[1])
        off1.Parent_Table = parents[0].Parent_Table
        off2.Parent_Table = parents[1].Parent_Table

        # 2. Recombinação
        if random.random() < cxpb:
            toolbox.mate(off1, off2)
            del off1.fitness.values, off2.fitness.values

        # 3. Mutação
        if random.random() < mutpb:
            toolbox.mutate(off1)
            del off1.fitness.values
        if random.random() < mutpb:
            toolbox.mutate(off2)
            del off2.fitness.values
        
        offspring.extend([off1, off2])
        
        # 4. Avaliação (Onde o limite FES é controlado)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        if invalid_ind:
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                 ind.fitness.values = fit
                 z_ideal = np.minimum(z_ideal, ind.fitness.values)
                 fes_count += 1 

        # 5. Inserção Steady-State
        for off in offspring:
             insert_in_tables(tables, num_tables, off, max_table_size, n_obj, z_ideal)
        