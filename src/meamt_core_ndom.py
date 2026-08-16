import random
import numpy as np
from deap import creator, base, tools

# ==========================================
# 1. SETUP DE CLASSES E TOOLBOX
# ==========================================
def setup_deap_classes(n_obj):
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * n_obj)
        creator.create("Individual", list, fitness=creator.FitnessMin, Parent_Table=None)
        creator.create("SubPopulation", list, score=0.0)

def build_toolbox(funcao_avaliacao, ind_size, n_pop, n_obj):
    setup_deap_classes(n_obj)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=ind_size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=n_pop)
    toolbox.register("evaluate", funcao_avaliacao) 
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=0.0, up=1.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=0.0, up=1.0, indpb=1.0/ind_size)
    return toolbox

# ==========================================
# 1.5 CACHE DE PERFORMANCE (OTIMIZAÇÃO EXTREMA)
# ==========================================
_INACTIVE_OBJS_CACHE = {}

def get_inactive_objs(mask, n_obj):
    """Retorna estritamente os índices que devem ser zerados. Ignora os ativos."""
    key = (mask, n_obj)
    if key not in _INACTIVE_OBJS_CACHE:
        _INACTIVE_OBJS_CACHE[key] = [i for i in range(n_obj) if not ((mask >> i) & 1)]
    return _INACTIVE_OBJS_CACHE[key]

# ==========================================
# 2. OPERADORES PRINCIPAIS DO MEAMT
# ==========================================
def gen_inicial_tables(pop_ini, num_tables, table_size, n_obj):
    tables = dict()
    for i in range(1, num_tables):
        tables[i] = sel_nsga2(pop_ini, i, table_size[i], n_obj) 
    return tables

def select_parents(tables, num_tables):
    selected = []
    for _ in range(2):
        # randrange economiza microssegundos por chamada em relação ao randint
        random1 = random.randrange(1, num_tables)
        random2 = random.randrange(1, num_tables)

        if len(tables[random1]) == 0: winner = random2
        elif len(tables[random2]) == 0: winner = random1
        elif tables[random1].score >= tables[random2].score:
            winner = random1
        else:
            winner = random2

        ind = random.choice(tables[winner])
        selected.append((ind, winner))
        
    return selected

def sel_nsga2(offspring, mask, max_table_size, nobj):
    if len(offspring) < max_table_size:
        return creator.SubPopulation(offspring)

    # OTIMIZAÇÃO: Backup sequencial O(N) puro usando List Comprehension (C-level)
    backups = [ind.fitness.values for ind in offspring]
    
    # OTIMIZAÇÃO: Alteração direta via índice inativo (Elimina a Generator Expression e zip)
    inactive_indices = get_inactive_objs(mask, nobj)
    
    if inactive_indices: # Processa apenas se existirem objetivos inativos
        for ind in offspring:
            vals = list(ind.fitness.values)
            for idx in inactive_indices:
                vals[idx] = 0.0
            ind.fitness.values = tuple(vals)

    survivors = tools.selNSGA2(offspring, max_table_size)

    # OTIMIZAÇÃO: Restauração sequencial limpa usando zip
    for ind, backup in zip(offspring, backups):
        ind.fitness.values = backup

    return creator.SubPopulation(survivors)

def insert_in_tables(tables, num_tables, offspring, max_table_size, n_obj):
    # Inserção direta sem rastreamento prévio
    for i in range(1, num_tables):
        current_score = tables[i].score
        tables[i].extend(offspring)
        tables[i] = sel_nsga2(tables[i], i, max_table_size[i], n_obj)
        tables[i].score = current_score

    # OTIMIZAÇÃO: Fim do loop triplo. Coleta todos os sobreviventes em um Set (C-level)
    surviving_ids = {id(ind) for i in range(1, num_tables) for ind in tables[i]}

    # OTIMIZAÇÃO: Varre estritamente os filhos apenas uma vez, impossibilitando duplicidade de pontos
    for off in offspring:
        if id(off) in surviving_ids:
            parent_table = getattr(off, 'Parent_Table', None)
            if parent_table is not None:
                tables[parent_table].score += 1
    
    for off in offspring:
        off.Parent_Table = None

# ==========================================
# 3. LOOP EVOLUTIVO (O CORAÇÃO DO ALGORITMO)
# ==========================================
def run(tables, num_tables, pop_size, ngen, max_table_size, toolbox, cxpb, mutpb, n_obj):
    max_fes = pop_size * ngen
    fes_count = 0
    while fes_count < max_fes:
        offspring = []
        for i in range(1, num_tables):
            tables[i].score *= 0.95
            
            # NOTA DE PERFORMANCE: A função print() executa operações custosas de I/O de disco/terminal. 
            # Em execuções com milhares de avaliações, manter isso habilitado será o maior gargalo remanescente do código.
            n_bits = bin(i).count("1")
            print(f"tabela {i:03b} (bits={n_bits}) -> score={tables[i].score:.2f}, size={len(tables[i])}")
            
        while len(offspring) < pop_size:
            parents = select_parents(tables, num_tables)

            off1 = toolbox.clone(parents[0][0])
            off2 = toolbox.clone(parents[1][0])
            
            off1.Parent_Table = parents[0][1]
            off2.Parent_Table = parents[1][1]

            if random.random() < cxpb:
              toolbox.mate(off1, off2)
              del off1.fitness.values, off2.fitness.values

            if random.random() < mutpb:
              toolbox.mutate(off1)
              del off1.fitness.values
            if random.random() < mutpb:
              toolbox.mutate(off2)
              del off2.fitness.values
        
            offspring.extend([off1, off2])
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        
        if invalid_ind:
            # Proteção: Garante que não vai estourar o limite máximo
            if fes_count + len(invalid_ind) > max_fes:
                invalid_ind = invalid_ind[:(max_fes - fes_count)]
                offspring = [ind for ind in offspring if ind.fitness.valid] + invalid_ind
                
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                 ind.fitness.values = fit
            
            fes_count += len(invalid_ind)

        insert_in_tables(tables, num_tables, offspring, max_table_size, n_obj)