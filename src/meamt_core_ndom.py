import random
from itertools import chain

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
    toolbox.register("clone", fast_clone)
    return toolbox

# ==========================================
# 1.5 CACHE DE PERFORMANCE (OTIMIZAÇÃO EXTREMA)
# ==========================================
_INACTIVE_OBJS_CACHE = {}
_ACTIVE_OBJS_CACHE = {}

def fast_clone(ind):
    new_ind = creator.Individual(ind)              # copia os genes (floats, imutáveis)
    if ind.fitness.valid:
        new_ind.fitness.values = ind.fitness.values
    new_ind.Parent_Table = ind.Parent_Table
    return new_ind

def get_inactive_objs(mask, n_obj):
    """Retorna estritamente os índices que devem ser zerados. Ignora os ativos."""
    key = (mask, n_obj)
    if key not in _INACTIVE_OBJS_CACHE:
        _INACTIVE_OBJS_CACHE[key] = [i for i in range(n_obj) if not ((mask >> i) & 1)]
    return _INACTIVE_OBJS_CACHE[key]

def get_active_objs(mask, n_obj):
    """Complemento de get_inactive_objs: índices dos objetivos ATIVOS para esta máscara."""
    key = (mask, n_obj)
    if key not in _ACTIVE_OBJS_CACHE:
        inactive = set(get_inactive_objs(mask, n_obj))
        _ACTIVE_OBJS_CACHE[key] = [i for i in range(n_obj) if i not in inactive]
    return _ACTIVE_OBJS_CACHE[key]

# ==========================================
# CORREÇÃO #1: crowding distance restrita aos objetivos ativos
# ==========================================
#
# O DEAP calcula a crowding distance iterando sobre TODOS os objetivos do
# indivíduo (`nobj = len(individuals[0].fitness.values)`). Quando zeramos os
# objetivos inativos para simular uma tabela "parcial", esses objetivos
# zerados viram uma dimensão constante (0.0 para todo mundo) — mas o DEAP,
# mesmo para uma dimensão constante, atribui `crowding_dist = inf` para os
# dois indivíduos que caem nas pontas do sort (efeito colateral do algoritmo
# de Deb et al., não um bug do DEAP em si: ele assume que "extremos" sempre
# merecem inf, o que só é verdade quando o objetivo é informativo).
#
# Resultado prático: quanto mais objetivos INATIVOS uma tabela tem, mais
# chances de indivíduos ganharem `crowding_dist == inf` "de graça" — e isso
# infla o bônus de diversidade (+1.0) pago em insert_in_tables, mesmo sem
# diversidade real nos objetivos que importam pra aquela tabela. Foi esse
# viés que apareceu nos logs: tabelas parciais (mask com bits desligados)
# consistentemente com score muito mais alto que a tabela de todos os
# objetivos ativos.
#
# A correção: implementar nossa própria assignCrowdingDist que itera só
# sobre os índices ATIVOS da máscara, em vez de sobre todos os `nobj`.
# ==========================================

def _assign_crowding_dist_active(individuals, active_indices):
    """Igual a tools.emo.assignCrowdingDist, mas considera apenas os
    objetivos em `active_indices` no cálculo da distância."""
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]
    n_active = len(active_indices)

    if n_active == 0:
        # não deveria ocorrer (toda máscara tem ao menos 1 bit ligado),
        # mas mantemos um fallback seguro
        for i in range(len(individuals)):
            individuals[i].fitness.crowding_dist = 0.0
        return

    for i in active_indices:
        crowd.sort(key=lambda element: element[0][i])
        distances[crowd[0][1]] = float("inf")
        distances[crowd[-1][1]] = float("inf")
        if crowd[-1][0][i] == crowd[0][0][i]:
            continue
        norm = n_active * float(crowd[-1][0][i] - crowd[0][0][i])
        for prev, cur, nxt in zip(crowd[:-2], crowd[1:-1], crowd[2:]):
            distances[cur[1]] += (nxt[0][i] - prev[0][i]) / norm

    for i, dist in enumerate(distances):
        individuals[i].fitness.crowding_dist = dist

def _sel_nsga2_active(individuals, k, active_indices):
    """Igual a tools.selNSGA2, mas usa _assign_crowding_dist_active em vez
    da assignCrowdingDist padrão do DEAP (que olha todos os objetivos)."""
    fronts = tools.sortNondominated(individuals, k)
    for front in fronts:
        _assign_crowding_dist_active(front, active_indices)

    chosen = list(chain(*fronts[:-1]))
    k = k - len(chosen)
    if k > 0:
        sorted_front = sorted(fronts[-1], key=lambda ind: ind.fitness.crowding_dist, reverse=True)
        chosen.extend(sorted_front[:k])

    return chosen

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
    active_indices = get_active_objs(mask, nobj)
    
    if inactive_indices: # Processa apenas se existirem objetivos inativos
        for ind in offspring:
            vals = list(ind.fitness.values)
            for idx in inactive_indices:
                vals[idx] = 0.0
            ind.fitness.values = tuple(vals)

    # CORREÇÃO #1: usamos nossa seleção NSGA2 com crowding distance restrita
    # aos objetivos ativos, em vez de tools.selNSGA2 (que consideraria também
    # os objetivos zerados e geraria inf artificial nas pontas do sort)
    survivors = _sel_nsga2_active(offspring, max_table_size, active_indices)

    # OTIMIZAÇÃO: Restauração sequencial limpa usando zip
    for ind, backup in zip(offspring, backups):
        ind.fitness.values = backup

    return creator.SubPopulation(survivors)

def insert_in_tables(tables, num_tables, offspring, max_table_size, n_obj):
    # 1. Inserção normal e truncamento (Preservando o histórico de score)
    for i in range(1, num_tables):
        current_score = tables[i].score  
        tables[i].extend(offspring)
        tables[i] = sel_nsga2(tables[i], i, max_table_size[i], n_obj)
        tables[i].score = current_score  

    # 2. Rastreamento dos Filhos Sobreviventes
    offspring_ids = {id(off) for off in offspring}

    # 3. Pagamento de Royalties Guiado por DIVERSIDADE
    for i in range(1, num_tables):
        for ind in tables[i]:
            if id(ind) in offspring_ids:
                parent_table = getattr(ind, 'Parent_Table', None)
                if parent_table is not None:
                    cd = getattr(ind.fitness, 'crowding_dist', 0.0)
                    
                    if cd == float('inf'):
                        bonus_diversidade = 1.0  
                    else:
                        bonus_diversidade = min(4.0, cd * 5.0) 
                        
                    tables[parent_table].score += (1.0 + bonus_diversidade)
    
    # 4. Limpeza de memória
    for off in offspring:
        off.Parent_Table = None

    # ==========================================
    # 5. ALOCAÇÃO DINÂMICA — CORRIGIDA (#2 e #3)
    # ==========================================
    # CORREÇÃO #2: em vez de truncar (int()) e jogar todo o resto do
    # arredondamento pra última tabela (viés estrutural a favor dela todo
    # geração), usamos o método dos maiores restos (largest remainder /
    # Hamilton): cada tabela recebe o piso da sua cota proporcional, e as
    # vagas que sobraram do arredondamento vão, uma a uma, para as tabelas
    # com maior parte fracionária perdida — sem favorecer nenhuma tabela
    # em particular.
    #
    # CORREÇÃO #3: se pop_total for pequeno demais para garantir min_vagas
    # em todas as tabelas, o código antigo podia gerar max_table_size
    # negativo pra última tabela. Agora calculamos um piso efetivo seguro
    # (nunca negativo, nunca estoura pop_total) e avisamos o usuário.
    pop_total = sum(max_table_size[1:])
    num_subtables = num_tables - 1
    min_vagas = 3  # piso desejado (garante que nenhuma tabela feche as portas)

    effective_min_vagas = min_vagas
    if pop_total < min_vagas * num_subtables:
        effective_min_vagas = max(1, pop_total // num_subtables)
        print(f"  [AVISO] pop_total={pop_total} é pequeno demais para manter "
              f"min_vagas={min_vagas} em {num_subtables} tabelas. "
              f"Usando piso efetivo de {effective_min_vagas} nesta geração "
              f"(considere aumentar pop_size ou reduzir num_tables/min_vagas).")

    # Coleta os scores (garantindo um piso mínimo de 0.01 para evitar divisões por zero)
    scores = [max(0.01, tables[i].score) for i in range(1, num_tables)]
    total_score = sum(scores)

    vagas_restantes = pop_total - (effective_min_vagas * num_subtables)

    if vagas_restantes > 0 and total_score > 0:
        proporcoes = [s / total_score for s in scores]
        extras_exatos = [p * vagas_restantes for p in proporcoes]
        extras = [int(e) for e in extras_exatos]
        falta = vagas_restantes - sum(extras)

        # distribui as `falta` vagas restantes para quem perdeu mais no
        # arredondamento (maiores partes fracionárias primeiro)
        ordem_por_resto = sorted(
            range(num_subtables),
            key=lambda idx: extras_exatos[idx] - extras[idx],
            reverse=True,
        )
        for idx in ordem_por_resto[:falta]:
            extras[idx] += 1
    else:
        extras = [0] * num_subtables

    for idx, i in enumerate(range(1, num_tables)):
        max_table_size[i] = effective_min_vagas + extras[idx]
# ==========================================
# 3. LOOP EVOLUTIVO (O CORAÇÃO DO ALGORITMO)
# ==========================================
def run(tables, num_tables, pop_size, ngen, max_table_size, toolbox, cxpb, mutpb, n_obj):
    max_fes = pop_size * ngen
    fes_count = 0
    
    # ==========================================
    # PREPARAÇÃO DO ARQUIVO EXTERNO (TABELA 0)
    # ==========================================
    if 0 not in tables:
        tables[0] = creator.SubPopulation()
        
    # Salva a população inicial na Tabela 0 logo de cara
    todas_iniciais = []
    for i in range(1, num_tables):
        todas_iniciais.extend(tables[i])
    if todas_iniciais:
        fronts_ini = tools.sortNondominated(todas_iniciais, len(todas_iniciais), first_front_only=True)
        tables[0].extend(fronts_ini[0])

    # Loop principal
    while fes_count < max_fes:
        offspring = []
        for i in range(1, num_tables):
            tables[i].score *= 0.95
            print(f"  Tabela {i} -> Score: {tables[i].score:6.2f} | Tamanho Alocado: {max_table_size[i]}")
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
            if fes_count + len(invalid_ind) > max_fes:
                invalid_ind = invalid_ind[:(max_fes - fes_count)]
                offspring = [ind for ind in offspring if ind.fitness.valid] + invalid_ind
                
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                 ind.fitness.values = fit
            
            fes_count += len(invalid_ind)

        # ==========================================
        # ATUALIZAÇÃO DO ARQUIVO EXTERNO (TABELA 0) COM TAMANHO FIXO
        # ==========================================
        tables[0].extend([toolbox.clone(ind) for ind in offspring])
        
        if len(tables[0]) > 0:
            # 1. Filtra apenas a primeira fronteira não-dominada global
            fronts = tools.sortNondominated(tables[0], len(tables[0]), first_front_only=True)
            non_dominated = fronts[0]
            
            # 2. Se o cofre ultrapassou o tamanho da população, trunca mantendo a diversidade
            # O selNSGA2 usa a Crowding Distance para escolher os 'pop_size' mais bem distribuídos
            if len(non_dominated) > pop_size:
                truncated_archive = tools.selNSGA2(non_dominated, pop_size)
                tables[0] = creator.SubPopulation(truncated_archive)
            else:
                tables[0] = creator.SubPopulation(non_dominated)
                
            
        # A evolução continua nas tabelas dinâmicas (Tabelas 1 até num_tables-1)
        insert_in_tables(tables, num_tables, offspring, max_table_size, n_obj)
        
    return tables