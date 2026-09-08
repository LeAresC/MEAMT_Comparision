import random
from itertools import chain
from collections import deque

import numpy as np
from deap import creator, base, tools

# ==========================================
# 1. SETUP DE CLASSES E TOOLBOX
# ==========================================

class ConstraintFitness(base.Fitness):
    """Fitness que aplica constraint-domination sem penalizar objetivos."""

    def __init__(self, values=()):
        super().__init__(values)
        self.constraint_violation = 0.0

    def dominates(self, other, obj=slice(None)):
        self_feasible = self.constraint_violation == 0.0
        other_feasible = other.constraint_violation == 0.0

        if self_feasible and not other_feasible:
            return True
        if not self_feasible and other_feasible:
            return False
        if not self_feasible:
            return self.constraint_violation < other.constraint_violation
        return super().dominates(other, obj)

    def __eq__(self, other):
        if not isinstance(other, base.Fitness):
            return NotImplemented
        return (
            self.wvalues == other.wvalues
            and self.constraint_violation
            == getattr(other, "constraint_violation", 0.0)
        )

    def __hash__(self):
        return hash((self.wvalues, self.constraint_violation))


def setup_deap_classes(n_obj):
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", ConstraintFitness, weights=(-1.0,) * n_obj)
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
    new_ind.fitness.constraint_violation = getattr(
        ind.fitness, "constraint_violation", 0.0
    )
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
def _assign_crowding_dist_active(individuals, active_indices):
    """Igual a tools.emo.assignCrowdingDist, mas considera apenas os
    objetivos em `active_indices` no cálculo da distância."""
    if len(individuals) == 0:
        return

    distances = [0.0] * len(individuals)
    crowd = [(ind.fitness.values, i) for i, ind in enumerate(individuals)]
    n_active = len(active_indices)

    if n_active == 0:
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

# ==========================================
# MUDANÇA ARQUITETURAL: OTIMIZAÇÃO EM ESTÁGIOS POR NÍVEL
# ==========================================
# Em vez de rodar todas as 2^n_obj-1 tabelas simultaneamente do início ao
# fim, agrupamos as tabelas por NÍVEL = número de objetivos ativos na
# máscara (popcount). Nível 1 = tabelas de 1 objetivo (n_obj delas),
# nível 2 = tabelas de 2 objetivos (C(n_obj,2) delas), ..., nível n_obj =
# só a tabela com todos os objetivos ativos.
#
# Cada nível roda por um número FIXO de gerações (orçamento total dividido
# igualmente entre os n_obj níveis). Ao final de um nível, a população de
# cada tabela do PRÓXIMO nível é semeada pela união (sem duplicar por id)
# das populações finais de todas as suas tabelas "subconjunto" no nível
# anterior (ex.: tabela {1,2,3} é semeada por {1,2}, {1,3} e {2,3}).
#
# O arquivo externo (equivalente à antiga Tabela 0) É COMPARTILHADO e
# persiste por TODOS os níveis -- nunca é reiniciado entre eles.
#
# Isso NÃO reduz o número total de tabelas (a soma de C(n,k) para todo k
# ainda é 2^n_obj-1) -- só muda quando cada uma é processada, e em qual
# momento o orçamento é gasto nela. O ganho esperado é de eficiência
# (warm start), não de escala combinatória.
# ==========================================

def popcount(m):
    return bin(m).count("1")

def masks_by_level(n_obj):
    """Retorna {nivel: [lista de máscaras com esse popcount]}, nivel de 1 a n_obj."""
    niveis = {}
    for m in range(1, 2 ** n_obj):
        k = popcount(m)
        niveis.setdefault(k, []).append(m)
    return niveis

def select_parents_niveis(tables, masks_ativos):
    """Igual a select_parents, mas sorteia entre as máscaras ATIVAS neste
    nível (não precisa ser um intervalo contíguo 1..num_tables-1)."""
    selected = []
    for _ in range(2):
        m1 = random.choice(masks_ativos)
        m2 = random.choice(masks_ativos)

        if len(tables[m1]) == 0: winner = m2
        elif len(tables[m2]) == 0: winner = m1
        elif tables[m1].score >= tables[m2].score:
            winner = m1
        else:
            winner = m2

        ind = random.choice(tables[winner])
        selected.append((ind, winner))

    return selected

# ==========================================
# PRESSÃO SELETIVA DENTRO DA TABELA: torneio binário em vez de random.choice
# ==========================================
def _dominates_active_constrained(fit_a, fit_b, active_indices):
    """Dominância restrita aos objetivos ativos da tabela, respeitando
    factibilidade primeiro (constraint-domination) -- importante em
    problemas restritos como o DTLZ9."""
    cv_a = getattr(fit_a, 'constraint_violation', 0.0)
    cv_b = getattr(fit_b, 'constraint_violation', 0.0)
    a_feasible = cv_a == 0.0
    b_feasible = cv_b == 0.0
    if a_feasible and not b_feasible:
        return True
    if not a_feasible and b_feasible:
        return False
    if not a_feasible:
        return cv_a < cv_b

    not_worse = True
    strictly_better = False
    for i in active_indices:
        if fit_a.values[i] > fit_b.values[i]:
            not_worse = False
            break
        elif fit_a.values[i] < fit_b.values[i]:
            strictly_better = True
    return not_worse and strictly_better

def _tournament_pick(table, mask, n_obj):
    """Sorteia 2 indivíduos da tabela vencedora e fica com o que domina o
    outro (factibilidade primeiro, depois objetivos ativos da máscara);
    empate desempatado por crowding distance. Em vez de random.choice
    (zero pressão seletiva dentro da tabela)."""
    if len(table) == 1:
        return table[0]
    a, b = random.sample(table, 2)
    active = get_active_objs(mask, n_obj)
    if _dominates_active_constrained(a.fitness, b.fitness, active):
        return a
    if _dominates_active_constrained(b.fitness, a.fitness, active):
        return b
    cd_a = getattr(a.fitness, 'crowding_dist', 0.0)
    cd_b = getattr(b.fitness, 'crowding_dist', 0.0)
    return a if cd_a >= cd_b else b

def select_parents_niveis_torneio(tables, masks_ativos, n_obj):
    """Igual a select_parents_niveis, mas usa _tournament_pick em vez de
    random.choice pra escolher o indivíduo dentro da tabela vencedora."""
    selected = []
    for _ in range(2):
        m1 = random.choice(masks_ativos)
        m2 = random.choice(masks_ativos)

        if len(tables[m1]) == 0: winner = m2
        elif len(tables[m2]) == 0: winner = m1
        elif tables[m1].score >= tables[m2].score:
            winner = m1
        else:
            winner = m2

        ind = _tournament_pick(tables[winner], winner, n_obj)
        selected.append((ind, winner))

    return selected

def insert_in_tables_niveis(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela=10, lineage_log=None):
    """Igual a insert_in_tables, mas iterando só sobre as máscaras ativas
    deste nível em vez de range(1, num_tables). Se `lineage_log` (lista)
    for passado, registra por geração {"mesma_tabela": X, "tabela_diferente": Y}
    contando, entre os sobreviventes desta geração, quantos têm
    Parent_Table igual à tabela onde sobreviveram (auto-reprodução) vs
    diferente (colaboração real entre tabelas)."""
    for mask in masks_ativos:
        current_score = tables[mask].score
        tables[mask].extend(offspring)
        tables[mask] = sel_nsga2(tables[mask], mask, max_table_size[mask], n_obj)
        tables[mask].score = current_score

    offspring_ids = {id(off) for off in offspring}

    if lineage_log is not None:
        mesma, diferente = 0, 0
        for mask in masks_ativos:
            for ind in tables[mask]:
                if id(ind) in offspring_ids:
                    parent_mask = getattr(ind, 'Parent_Table', None)
                    if parent_mask is not None:
                        if parent_mask == mask:
                            mesma += 1
                        else:
                            diferente += 1
        lineage_log.append({"mesma_tabela": mesma, "tabela_diferente": diferente})

    for mask in masks_ativos:
        for ind in tables[mask]:
            if id(ind) in offspring_ids:
                parent_mask = getattr(ind, 'Parent_Table', None)
                if parent_mask is not None and parent_mask in tables:
                    tables[parent_mask].score += 1.0

    for off in offspring:
        off.Parent_Table = None

    pop_total = sum(max_table_size[m] for m in masks_ativos)
    num_subtables = len(masks_ativos)
    min_vagas = 3

    effective_min_vagas = min_vagas
    if pop_total < min_vagas * num_subtables:
        effective_min_vagas = max(1, pop_total // num_subtables)
        print(f"  [AVISO] pop_total={pop_total} pequeno demais pra manter "
              f"min_vagas={min_vagas} nas {num_subtables} tabelas deste nível. "
              f"Usando piso efetivo {effective_min_vagas}.")

    deltas = []
    for mask in masks_ativos:
        score_atual = tables[mask].score
        janela_tabela = historico_scores.setdefault(mask, deque(maxlen=janela))
        if len(janela_tabela) > 0:
            media_anterior = sum(janela_tabela) / len(janela_tabela)
            derivada = score_atual - media_anterior
        else:
            derivada = 0.0
        janela_tabela.append(score_atual)
        deltas.append(max(0.01, derivada))

    total_delta = sum(deltas)
    vagas_restantes = pop_total - (effective_min_vagas * num_subtables)

    if vagas_restantes > 0 and total_delta > 0:
        proporcoes = [d / total_delta for d in deltas]
        extras_exatos = [p * vagas_restantes for p in proporcoes]
        extras = [int(e) for e in extras_exatos]
        falta = vagas_restantes - sum(extras)
        ordem_por_resto = sorted(
            range(num_subtables),
            key=lambda idx: extras_exatos[idx] - extras[idx],
            reverse=True,
        )
        for idx in ordem_por_resto[:falta]:
            extras[idx] += 1
    else:
        extras = [0] * num_subtables

    for idx, mask in enumerate(masks_ativos):
        max_table_size[mask] = effective_min_vagas + extras[idx]

def diversidade_tabela(tbl, mask, n_obj):
    """Desvio padrão médio dos valores nos objetivos ATIVOS da tabela.
    Cai perto de zero quando os membros colapsaram em pontos quase
    idênticos (o problema que identificamos na Tabela 7 do DTLZ5)."""
    if len(tbl) <= 1:
        return 0.0
    active = get_active_objs(mask, n_obj)
    if not active:
        return 0.0
    stds = []
    for i in active:
        vals = [ind.fitness.values[i] for ind in tbl]
        m = sum(vals) / len(vals)
        var = sum((v - m) ** 2 for v in vals) / len(vals)
        stds.append(var ** 0.5)
    return sum(stds) / len(stds)

# ==========================================
# VARIANTE B: peso de alocação multiplicado pela diversidade normalizada
# ==========================================
def insert_in_tables_niveis_diversidade(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela=10, lineage_log=None):
    """Igual a insert_in_tables_niveis, mas o peso de alocação (derivada)
    de cada tabela é multiplicado por sua diversidade normalizada (0 a 1,
    relativa à tabela mais diversa da fase). Uma tabela colapsada some do
    numerador da alocação mesmo com score/derivada alta -- o crescimento
    dela trava perto do piso."""
    for mask in masks_ativos:
        current_score = tables[mask].score
        tables[mask].extend(offspring)
        tables[mask] = sel_nsga2(tables[mask], mask, max_table_size[mask], n_obj)
        tables[mask].score = current_score

    offspring_ids = {id(off) for off in offspring}

    if lineage_log is not None:
        mesma, diferente = 0, 0
        for mask in masks_ativos:
            for ind in tables[mask]:
                if id(ind) in offspring_ids:
                    parent_mask = getattr(ind, 'Parent_Table', None)
                    if parent_mask is not None:
                        if parent_mask == mask:
                            mesma += 1
                        else:
                            diferente += 1
        lineage_log.append({"mesma_tabela": mesma, "tabela_diferente": diferente})

    for mask in masks_ativos:
        for ind in tables[mask]:
            if id(ind) in offspring_ids:
                parent_mask = getattr(ind, 'Parent_Table', None)
                if parent_mask is not None and parent_mask in tables:
                    tables[parent_mask].score += 1.0

    for off in offspring:
        off.Parent_Table = None

    pop_total = sum(max_table_size[m] for m in masks_ativos)
    num_subtables = len(masks_ativos)
    min_vagas = 3

    effective_min_vagas = min_vagas
    if pop_total < min_vagas * num_subtables:
        effective_min_vagas = max(1, pop_total // num_subtables)
        print(f"  [AVISO] pop_total={pop_total} pequeno demais pra manter "
              f"min_vagas={min_vagas} nas {num_subtables} tabelas deste nível. "
              f"Usando piso efetivo {effective_min_vagas}.")

    deltas = []
    for mask in masks_ativos:
        score_atual = tables[mask].score
        janela_tabela = historico_scores.setdefault(mask, deque(maxlen=janela))
        if len(janela_tabela) > 0:
            media_anterior = sum(janela_tabela) / len(janela_tabela)
            derivada = score_atual - media_anterior
        else:
            derivada = 0.0
        janela_tabela.append(score_atual)
        deltas.append(max(0.01, derivada))

    diversidades = [diversidade_tabela(tables[m], m, n_obj) for m in masks_ativos]
    max_div = max(diversidades) if diversidades else 0.0
    if max_div > 0:
        fatores_div = [max(0.01, d / max_div) for d in diversidades]
    else:
        fatores_div = [1.0] * len(masks_ativos)
    deltas = [max(0.01, deltas[idx] * fatores_div[idx]) for idx in range(len(masks_ativos))]

    total_delta = sum(deltas)
    vagas_restantes = pop_total - (effective_min_vagas * num_subtables)

    if vagas_restantes > 0 and total_delta > 0:
        proporcoes = [d / total_delta for d in deltas]
        extras_exatos = [p * vagas_restantes for p in proporcoes]
        extras = [int(e) for e in extras_exatos]
        falta = vagas_restantes - sum(extras)
        ordem_por_resto = sorted(
            range(num_subtables),
            key=lambda idx: extras_exatos[idx] - extras[idx],
            reverse=True,
        )
        for idx in ordem_por_resto[:falta]:
            extras[idx] += 1
    else:
        extras = [0] * num_subtables

    for idx, mask in enumerate(masks_ativos):
        max_table_size[mask] = effective_min_vagas + extras[idx]

# ==========================================
# VARIANTE C: corte direto pro piso quando diversidade < limiar
# ==========================================
def insert_in_tables_niveis_corte(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela=10, lineage_log=None, limiar_diversidade=1e-6):
    """Igual a insert_in_tables_niveis, mas qualquer tabela com diversidade
    abaixo de `limiar_diversidade` (essencialmente pontos idênticos, ruído
    de ponto flutuante) tem peso de alocação ZERADO -- trava exatamente no
    piso, não importa o score/derivada dela."""
    for mask in masks_ativos:
        current_score = tables[mask].score
        tables[mask].extend(offspring)
        tables[mask] = sel_nsga2(tables[mask], mask, max_table_size[mask], n_obj)
        tables[mask].score = current_score

    offspring_ids = {id(off) for off in offspring}

    if lineage_log is not None:
        mesma, diferente = 0, 0
        for mask in masks_ativos:
            for ind in tables[mask]:
                if id(ind) in offspring_ids:
                    parent_mask = getattr(ind, 'Parent_Table', None)
                    if parent_mask is not None:
                        if parent_mask == mask:
                            mesma += 1
                        else:
                            diferente += 1
        lineage_log.append({"mesma_tabela": mesma, "tabela_diferente": diferente})

    for mask in masks_ativos:
        for ind in tables[mask]:
            if id(ind) in offspring_ids:
                parent_mask = getattr(ind, 'Parent_Table', None)
                if parent_mask is not None and parent_mask in tables:
                    tables[parent_mask].score += 1.0

    for off in offspring:
        off.Parent_Table = None

    pop_total = sum(max_table_size[m] for m in masks_ativos)
    num_subtables = len(masks_ativos)
    min_vagas = 3

    effective_min_vagas = min_vagas
    if pop_total < min_vagas * num_subtables:
        effective_min_vagas = max(1, pop_total // num_subtables)
        print(f"  [AVISO] pop_total={pop_total} pequeno demais pra manter "
              f"min_vagas={min_vagas} nas {num_subtables} tabelas deste nível. "
              f"Usando piso efetivo {effective_min_vagas}.")

    deltas = []
    for mask in masks_ativos:
        score_atual = tables[mask].score
        janela_tabela = historico_scores.setdefault(mask, deque(maxlen=janela))
        if len(janela_tabela) > 0:
            media_anterior = sum(janela_tabela) / len(janela_tabela)
            derivada = score_atual - media_anterior
        else:
            derivada = 0.0
        janela_tabela.append(score_atual)
        deltas.append(max(0.01, derivada))

    diversidades = [diversidade_tabela(tables[m], m, n_obj) for m in masks_ativos]
    deltas = [deltas[idx] if diversidades[idx] >= limiar_diversidade else 0.0 for idx in range(len(masks_ativos))]

    total_delta = sum(deltas)
    vagas_restantes = pop_total - (effective_min_vagas * num_subtables)

    if vagas_restantes > 0 and total_delta > 0:
        proporcoes = [d / total_delta for d in deltas]
        extras_exatos = [p * vagas_restantes for p in proporcoes]
        extras = [int(e) for e in extras_exatos]
        falta = vagas_restantes - sum(extras)
        ordem_por_resto = sorted(
            range(num_subtables),
            key=lambda idx: extras_exatos[idx] - extras[idx],
            reverse=True,
        )
        for idx in ordem_por_resto[:falta]:
            extras[idx] += 1
    else:
        extras = [0] * num_subtables

    for idx, mask in enumerate(masks_ativos):
        max_table_size[mask] = effective_min_vagas + extras[idx]

def seed_next_level(tables_nivel_anterior, masks_novo_nivel, table_size, n_obj):
    """Semeia cada tabela do novo nível com a união (sem duplicar por id)
    das populações finais de todas as suas tabelas subconjunto no nível
    anterior, depois roda sel_nsga2 pra reduzir ao tamanho alvo."""
    novas = {}
    for mask in masks_novo_nivel:
        subset_masks = []
        for bit in range(n_obj):
            if (mask >> bit) & 1:
                sub = mask & ~(1 << bit)
                if sub != 0 and sub in tables_nivel_anterior:
                    subset_masks.append(sub)

        candidatos = {}
        for sub in subset_masks:
            for ind in tables_nivel_anterior[sub]:
                candidatos[id(ind)] = ind
        candidatos_list = list(candidatos.values())

        novas[mask] = sel_nsga2(candidatos_list, mask, table_size, n_obj)
    return novas

def run_nivel(tables, archive_holder, masks_ativos, pop_size, ngen_nivel, max_table_size,
              toolbox, cxpb, mutpb, n_obj, historico_scores, janela=10, snapshot_callback=None, lineage_log=None):
    """Roda um único nível (fase) por ngen_nivel gerações. archive_holder é
    uma lista de 1 elemento contendo o SubPopulation do arquivo global --
    usado como referência mutável pra persistir entre chamadas (níveis)."""
    max_fes_nivel = pop_size * ngen_nivel
    fes_count = 0

    clone = toolbox.clone
    mate = toolbox.mate
    mutate = toolbox.mutate

    while fes_count < max_fes_nivel:
        for mask in masks_ativos:
            print(f"  [Nível {popcount(mask)}] Tabela {mask} -> Score: {tables[mask].score:6.2f} | Tamanho Alocado: {max_table_size[mask]}")

        offspring = []
        while len(offspring) < pop_size:
            parents = select_parents_niveis(tables, masks_ativos)

            off1 = clone(parents[0][0])
            off2 = clone(parents[1][0])
            off1.Parent_Table = parents[0][1]
            off2.Parent_Table = parents[1][1]

            if random.random() < cxpb:
                mate(off1, off2)
                del off1.fitness.values, off2.fitness.values
            if random.random() < mutpb:
                mutate(off1)
                del off1.fitness.values
            if random.random() < mutpb:
                mutate(off2)
                del off2.fitness.values

            offspring.append(off1)
            offspring.append(off2)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            if fes_count + len(invalid_ind) > max_fes_nivel:
                invalid_ind = invalid_ind[:(max_fes_nivel - fes_count)]
                offspring = [ind for ind in offspring if ind.fitness.valid] + invalid_ind
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            fes_count += len(invalid_ind)

        archive = archive_holder[0]
        archive.extend([clone(ind) for ind in offspring])
        if len(archive) > 0:
            fronts = tools.sortNondominated(archive, len(archive), first_front_only=True)
            non_dominated = fronts[0]
            if len(non_dominated) > pop_size:
                truncated = tools.selNSGA2(non_dominated, pop_size)
                archive = creator.SubPopulation(truncated)
            else:
                archive = creator.SubPopulation(non_dominated)
        archive_holder[0] = archive

        insert_in_tables_niveis(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela, lineage_log)

        if snapshot_callback is not None:
            tables[0] = archive_holder[0]
            snapshot_callback(tables)

    return tables

def run_staged(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None):
    """Orquestrador: roda os n_obj níveis em sequência, com orçamento fixo
    (total_gens // n_obj gerações por nível) e semeadura entre níveis.
    Retorna um dict `tables` com a tabela 0 = arquivo global e as tabelas
    do ÚLTIMO nível processado (a de todos os objetivos ativos, sozinha)."""
    niveis = masks_by_level(n_obj)
    gens_por_nivel = max(1, total_gens // n_obj)

    archive_holder = [creator.SubPopulation()]
    historico_scores = {}

    # --- Nível 1: semeado da população inicial aleatória ---
    masks_nivel = niveis[1]
    base_size = max(3, pop_size // len(masks_nivel))
    tables = {}
    max_table_size = {}
    for mask in masks_nivel:
        tables[mask] = sel_nsga2(pop_ini, mask, base_size, n_obj)
        max_table_size[mask] = base_size

    tables = run_nivel(tables, archive_holder, masks_nivel, pop_size, gens_por_nivel, max_table_size,
                        toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback)

    # --- Níveis 2..n_obj: semeados pelo nível anterior ---
    for nivel_atual in range(2, n_obj + 1):
        masks_nivel = niveis[nivel_atual]
        base_size = max(3, pop_size // len(masks_nivel))
        max_table_size = {mask: base_size for mask in masks_nivel}
        historico_scores = {}  # nova janela pro novo conjunto de tabelas

        novas_tables = seed_next_level(tables, masks_nivel, base_size, n_obj)
        tables = run_nivel(novas_tables, archive_holder, masks_nivel, pop_size, gens_por_nivel, max_table_size,
                            toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback)

    tables[0] = archive_holder[0]
    return tables

def run_staged_pairs(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None, lineage_log=None):
    """Variante com JANELA DE 2 NÍVEIS ADJACENTES por fase: (1,2) -> (2,3)
    -> (3,4) -> ... -> (n_obj-1, n_obj). O nível mais "velho" da fase
    anterior sai, o nível do meio continua evoluindo sem reseed, e o nível
    novo entra semeado pelo nível anterior (igual ao run_staged, só que
    agora sempre com 2 níveis competindo entre si ao mesmo tempo em vez
    de 1). Se `lineage_log` (lista) for passado, é preenchido por
    insert_in_tables_niveis com estatísticas de auto-reprodução vs
    colaboração cruzada entre tabelas, geração a geração."""
    niveis = masks_by_level(n_obj)
    num_fases = max(1, n_obj - 1)
    gens_por_fase = max(1, total_gens // num_fases)

    archive_holder = [creator.SubPopulation()]

    # --- Fase 1: níveis 1 e 2 juntos, ambos semeados da população inicial ---
    active_masks = list(niveis.get(1, [])) + list(niveis.get(2, []))
    base_size = max(3, pop_size // len(active_masks))
    tables = {}
    max_table_size = {}
    for mask in active_masks:
        tables[mask] = sel_nsga2(pop_ini, mask, base_size, n_obj)
        max_table_size[mask] = base_size
    historico_scores = {}

    tables = run_nivel(tables, archive_holder, active_masks, pop_size, gens_por_fase, max_table_size,
                        toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback, lineage_log)

    # --- Fases seguintes: (2,3), (3,4), ..., (n_obj-1, n_obj) ---
    for nivel_velho in range(2, n_obj):
        nivel_novo = nivel_velho + 1
        masks_velho = niveis[nivel_velho]
        masks_novo = niveis[nivel_novo]
        active_masks = masks_velho + masks_novo

        base_size = max(3, pop_size // len(active_masks))

        # o nível "velho" desta fase (já ativo na fase anterior) CONTINUA
        # evoluindo sem reseed -- só o nível novo é semeado
        tables_continuadas = {m: tables[m] for m in masks_velho if m in tables}
        tables_novas = seed_next_level(tables, masks_novo, base_size, n_obj)

        tables = {**tables_continuadas, **tables_novas}
        max_table_size = {mask: base_size for mask in active_masks}
        historico_scores = {}

        tables = run_nivel(tables, archive_holder, active_masks, pop_size, gens_por_fase, max_table_size,
                            toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback, lineage_log)

    tables[0] = archive_holder[0]
    return tables

def insert_in_tables_niveis_arquivo(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela=10, lineage_log=None, ids_creditados=None):
    """Igual a insert_in_tables_niveis, mas a premiação de score NÃO é
    "sobreviveu em alguma tabela" -- é "o clone deste filho sobreviveu no
    ARQUIVO GLOBAL desta geração" (ids_creditados, calculado em
    run_nivel_arquivo). Alinha o incentivo de score com contribuição real
    pro resultado final, em vez de sobrevivência local (que pode ser só
    ruído numérico entre pontos quase idênticos)."""
    for mask in masks_ativos:
        current_score = tables[mask].score
        tables[mask].extend(offspring)
        tables[mask] = sel_nsga2(tables[mask], mask, max_table_size[mask], n_obj)
        tables[mask].score = current_score

    offspring_ids = {id(off) for off in offspring}

    if lineage_log is not None:
        mesma, diferente = 0, 0
        for mask in masks_ativos:
            for ind in tables[mask]:
                if id(ind) in offspring_ids:
                    parent_mask = getattr(ind, 'Parent_Table', None)
                    if parent_mask is not None:
                        if parent_mask == mask:
                            mesma += 1
                        else:
                            diferente += 1
        lineage_log.append({"mesma_tabela": mesma, "tabela_diferente": diferente})

    if ids_creditados is None:
        ids_creditados = set()
    for off in offspring:
        if id(off) in ids_creditados:
            parent_mask = getattr(off, 'Parent_Table', None)
            if parent_mask is not None and parent_mask in tables:
                tables[parent_mask].score += 1.0

    for off in offspring:
        off.Parent_Table = None

    pop_total = sum(max_table_size[m] for m in masks_ativos)
    num_subtables = len(masks_ativos)
    min_vagas = 3

    effective_min_vagas = min_vagas
    if pop_total < min_vagas * num_subtables:
        effective_min_vagas = max(1, pop_total // num_subtables)
        print(f"  [AVISO] pop_total={pop_total} pequeno demais pra manter "
              f"min_vagas={min_vagas} nas {num_subtables} tabelas deste nível. "
              f"Usando piso efetivo {effective_min_vagas}.")

    deltas = []
    for mask in masks_ativos:
        score_atual = tables[mask].score
        janela_tabela = historico_scores.setdefault(mask, deque(maxlen=janela))
        if len(janela_tabela) > 0:
            media_anterior = sum(janela_tabela) / len(janela_tabela)
            derivada = score_atual - media_anterior
        else:
            derivada = 0.0
        janela_tabela.append(score_atual)
        deltas.append(max(0.01, derivada))

    total_delta = sum(deltas)
    vagas_restantes = pop_total - (effective_min_vagas * num_subtables)

    if vagas_restantes > 0 and total_delta > 0:
        proporcoes = [d / total_delta for d in deltas]
        extras_exatos = [p * vagas_restantes for p in proporcoes]
        extras = [int(e) for e in extras_exatos]
        falta = vagas_restantes - sum(extras)
        ordem_por_resto = sorted(
            range(num_subtables),
            key=lambda idx: extras_exatos[idx] - extras[idx],
            reverse=True,
        )
        for idx in ordem_por_resto[:falta]:
            extras[idx] += 1
    else:
        extras = [0] * num_subtables

    for idx, mask in enumerate(masks_ativos):
        max_table_size[mask] = effective_min_vagas + extras[idx]

def insert_in_tables_niveis_arquivo_absoluto(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela=10, lineage_log=None, ids_creditados=None):
    """Igual a insert_in_tables_niveis_arquivo (mesma premiação por
    contribuição ao arquivo), mas a ALOCAÇÃO usa a regra de três original:
    peso proporcional ao SCORE ABSOLUTO acumulado, sem derivada/sliding
    window. Isola o efeito da derivada mantendo tudo o resto igual."""
    for mask in masks_ativos:
        current_score = tables[mask].score
        tables[mask].extend(offspring)
        tables[mask] = sel_nsga2(tables[mask], mask, max_table_size[mask], n_obj)
        tables[mask].score = current_score

    offspring_ids = {id(off) for off in offspring}

    if lineage_log is not None:
        mesma, diferente = 0, 0
        for mask in masks_ativos:
            for ind in tables[mask]:
                if id(ind) in offspring_ids:
                    parent_mask = getattr(ind, 'Parent_Table', None)
                    if parent_mask is not None:
                        if parent_mask == mask:
                            mesma += 1
                        else:
                            diferente += 1
        lineage_log.append({"mesma_tabela": mesma, "tabela_diferente": diferente})

    if ids_creditados is None:
        ids_creditados = set()
    for off in offspring:
        if id(off) in ids_creditados:
            parent_mask = getattr(off, 'Parent_Table', None)
            if parent_mask is not None and parent_mask in tables:
                tables[parent_mask].score += 1.0

    for off in offspring:
        off.Parent_Table = None

    pop_total = sum(max_table_size[m] for m in masks_ativos)
    num_subtables = len(masks_ativos)
    min_vagas = 3

    effective_min_vagas = min_vagas
    if pop_total < min_vagas * num_subtables:
        effective_min_vagas = max(1, pop_total // num_subtables)
        print(f"  [AVISO] pop_total={pop_total} pequeno demais pra manter "
              f"min_vagas={min_vagas} nas {num_subtables} tabelas deste nível. "
              f"Usando piso efetivo {effective_min_vagas}.")

    # REGRA DE TRÊS: peso = score absoluto (com piso 0.01), sem derivada
    scores = [max(0.01, tables[m].score) for m in masks_ativos]
    total_score = sum(scores)
    vagas_restantes = pop_total - (effective_min_vagas * num_subtables)

    if vagas_restantes > 0 and total_score > 0:
        proporcoes = [s / total_score for s in scores]
        extras_exatos = [p * vagas_restantes for p in proporcoes]
        extras = [int(e) for e in extras_exatos]
        falta = vagas_restantes - sum(extras)
        ordem_por_resto = sorted(
            range(num_subtables),
            key=lambda idx: extras_exatos[idx] - extras[idx],
            reverse=True,
        )
        for idx in ordem_por_resto[:falta]:
            extras[idx] += 1
    else:
        extras = [0] * num_subtables

    for idx, mask in enumerate(masks_ativos):
        max_table_size[mask] = effective_min_vagas + extras[idx]

def run_nivel_arquivo_absoluto_torneio(tables, archive_holder, masks_ativos, pop_size, ngen_nivel, max_table_size,
                                        toolbox, cxpb, mutpb, n_obj, historico_scores, janela=10, snapshot_callback=None, lineage_log=None):
    """Igual a run_nivel_arquivo_torneio, mas usando alocação por score
    absoluto (insert_in_tables_niveis_arquivo_absoluto) em vez de derivada."""
    max_fes_nivel = pop_size * ngen_nivel
    fes_count = 0
    clone, mate, mutate = toolbox.clone, toolbox.mate, toolbox.mutate

    while fes_count < max_fes_nivel:
        for mask in masks_ativos:
            print(f"  [Nível {popcount(mask)}] Tabela {mask} -> Score: {tables[mask].score:6.2f} | Tamanho Alocado: {max_table_size[mask]}")

        offspring = []
        while len(offspring) < pop_size:
            parents = select_parents_niveis_torneio(tables, masks_ativos, n_obj)
            off1 = clone(parents[0][0]); off2 = clone(parents[1][0])
            off1.Parent_Table = parents[0][1]; off2.Parent_Table = parents[1][1]
            if random.random() < cxpb:
                mate(off1, off2)
                del off1.fitness.values, off2.fitness.values
            if random.random() < mutpb:
                mutate(off1); del off1.fitness.values
            if random.random() < mutpb:
                mutate(off2); del off2.fitness.values
            offspring.append(off1); offspring.append(off2)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            if fes_count + len(invalid_ind) > max_fes_nivel:
                invalid_ind = invalid_ind[:(max_fes_nivel - fes_count)]
                offspring = [ind for ind in offspring if ind.fitness.valid] + invalid_ind
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            fes_count += len(invalid_ind)

        archive = archive_holder[0]
        clones_desta_geracao = [clone(ind) for ind in offspring]
        offspring_to_clone_id = {id(off): id(cl) for off, cl in zip(offspring, clones_desta_geracao)}
        ids_clones_desta_geracao = set(offspring_to_clone_id.values())
        archive.extend(clones_desta_geracao)
        if len(archive) > 0:
            fronts = tools.sortNondominated(archive, len(archive), first_front_only=True)
            non_dominated = fronts[0]
            if len(non_dominated) > pop_size:
                archive = creator.SubPopulation(tools.selNSGA2(non_dominated, pop_size))
            else:
                archive = creator.SubPopulation(non_dominated)
        archive_holder[0] = archive

        ids_sobreviventes_arquivo = {id(ind) for ind in archive if id(ind) in ids_clones_desta_geracao}
        ids_creditados = {oid for oid, cid in offspring_to_clone_id.items() if cid in ids_sobreviventes_arquivo}

        insert_in_tables_niveis_arquivo_absoluto(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela, lineage_log, ids_creditados)

        if snapshot_callback is not None:
            tables[0] = archive_holder[0]
            snapshot_callback(tables)

    return tables

def run_staged_pairs_arquivo_absoluto_torneio(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None, lineage_log=None):
    """COM ESTÁGIOS + Variante A (arquivo) + torneio + ALOCAÇÃO POR SCORE
    ABSOLUTO (regra de três, sem derivada) -- isola o efeito da derivada
    contra run_staged_pairs_arquivo_torneio (que usa derivada)."""
    niveis = masks_by_level(n_obj)
    num_fases = max(1, n_obj - 1)
    gens_por_fase = max(1, total_gens // num_fases)

    archive_holder = [creator.SubPopulation()]
    active_masks = list(niveis.get(1, [])) + list(niveis.get(2, []))
    base_size = max(3, pop_size // len(active_masks))
    tables = {m: sel_nsga2(pop_ini, m, base_size, n_obj) for m in active_masks}
    max_table_size = {m: base_size for m in active_masks}
    historico_scores = {}

    tables = run_nivel_arquivo_absoluto_torneio(tables, archive_holder, active_masks, pop_size, gens_por_fase, max_table_size,
                                                 toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback, lineage_log)

    for nivel_velho in range(2, n_obj):
        nivel_novo = nivel_velho + 1
        masks_velho = niveis[nivel_velho]; masks_novo = niveis[nivel_novo]
        active_masks = masks_velho + masks_novo
        base_size = max(3, pop_size // len(active_masks))
        tables_cont = {m: tables[m] for m in masks_velho if m in tables}
        tables_novas = seed_next_level(tables, masks_novo, base_size, n_obj)
        tables = {**tables_cont, **tables_novas}
        max_table_size = {m: base_size for m in active_masks}
        historico_scores = {}
        tables = run_nivel_arquivo_absoluto_torneio(tables, archive_holder, active_masks, pop_size, gens_por_fase, max_table_size,
                                                     toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback, lineage_log)

    tables[0] = archive_holder[0]
    return tables

def run_nivel_arquivo(tables, archive_holder, masks_ativos, pop_size, ngen_nivel, max_table_size,
                       toolbox, cxpb, mutpb, n_obj, historico_scores, janela=10, snapshot_callback=None, lineage_log=None):
    """Igual a run_nivel, mas calcula quais filhos desta geração tiveram um
    clone sobrevivendo no arquivo global (ANTES do truncamento), e credita
    o score só pra esses -- ver insert_in_tables_niveis_arquivo."""
    max_fes_nivel = pop_size * ngen_nivel
    fes_count = 0

    clone = toolbox.clone
    mate = toolbox.mate
    mutate = toolbox.mutate

    while fes_count < max_fes_nivel:
        for mask in masks_ativos:
            print(f"  [Nível {popcount(mask)}] Tabela {mask} -> Score: {tables[mask].score:6.2f} | Tamanho Alocado: {max_table_size[mask]}")

        offspring = []
        while len(offspring) < pop_size:
            parents = select_parents_niveis(tables, masks_ativos)

            off1 = clone(parents[0][0])
            off2 = clone(parents[1][0])
            off1.Parent_Table = parents[0][1]
            off2.Parent_Table = parents[1][1]

            if random.random() < cxpb:
                mate(off1, off2)
                del off1.fitness.values, off2.fitness.values
            if random.random() < mutpb:
                mutate(off1)
                del off1.fitness.values
            if random.random() < mutpb:
                mutate(off2)
                del off2.fitness.values

            offspring.append(off1)
            offspring.append(off2)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            if fes_count + len(invalid_ind) > max_fes_nivel:
                invalid_ind = invalid_ind[:(max_fes_nivel - fes_count)]
                offspring = [ind for ind in offspring if ind.fitness.valid] + invalid_ind
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            fes_count += len(invalid_ind)

        archive = archive_holder[0]
        clones_desta_geracao = [clone(ind) for ind in offspring]
        offspring_to_clone_id = {id(off): id(cl) for off, cl in zip(offspring, clones_desta_geracao)}
        ids_clones_desta_geracao = set(offspring_to_clone_id.values())

        archive.extend(clones_desta_geracao)
        if len(archive) > 0:
            fronts = tools.sortNondominated(archive, len(archive), first_front_only=True)
            non_dominated = fronts[0]
            if len(non_dominated) > pop_size:
                truncated = tools.selNSGA2(non_dominated, pop_size)
                archive = creator.SubPopulation(truncated)
            else:
                archive = creator.SubPopulation(non_dominated)
        archive_holder[0] = archive

        ids_sobreviventes_arquivo = {id(ind) for ind in archive if id(ind) in ids_clones_desta_geracao}
        ids_creditados = {off_id for off_id, clone_id in offspring_to_clone_id.items() if clone_id in ids_sobreviventes_arquivo}

        insert_in_tables_niveis_arquivo(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela, lineage_log, ids_creditados)

        if snapshot_callback is not None:
            tables[0] = archive_holder[0]
            snapshot_callback(tables)

    return tables

def insert_tabelas_sem_recompensa(tables, masks_ativos, offspring, max_table_size, n_obj):
    """Só a etapa de inserção/truncamento por tabela (sem calcular nem
    aplicar nenhuma recompensa de score) -- usada pela versão de crédito
    duplo, que calcula a recompensa separadamente antes de chamar isso."""
    for mask in masks_ativos:
        current_score = tables[mask].score
        tables[mask].extend(offspring)
        tables[mask] = sel_nsga2(tables[mask], mask, max_table_size[mask], n_obj)
        tables[mask].score = current_score
    for off in offspring:
        off.Parent_Table = None

def alocar_por_derivada(tables, masks_ativos, max_table_size, n_obj, historico_scores, janela=10, pisos=None):
    """Só a etapa de alocação por derivada (igual ao final de
    insert_in_tables_niveis_arquivo), separada pra reuso. `pisos` opcional
    (dict mask->piso) permite piso por tabela em vez de um valor único."""
    pop_total = sum(max_table_size[m] for m in masks_ativos)
    num_subtables = len(masks_ativos)

    if pisos is None:
        pisos_efetivos = {m: 3 for m in masks_ativos}
    else:
        pisos_efetivos = dict(pisos)
    piso_total = sum(pisos_efetivos[m] for m in masks_ativos)
    if pop_total < piso_total:
        fator = pop_total / piso_total if piso_total > 0 else 0
        pisos_efetivos = {m: max(1, int(pisos_efetivos[m] * fator)) for m in masks_ativos}
        piso_total = sum(pisos_efetivos.values())

    deltas = []
    for mask in masks_ativos:
        score_atual = tables[mask].score
        janela_tabela = historico_scores.setdefault(mask, deque(maxlen=janela))
        if len(janela_tabela) > 0:
            media_anterior = sum(janela_tabela) / len(janela_tabela)
            derivada = score_atual - media_anterior
        else:
            derivada = 0.0
        janela_tabela.append(score_atual)
        deltas.append(max(0.01, derivada))

    total_delta = sum(deltas)
    vagas_restantes = pop_total - piso_total
    if vagas_restantes > 0 and total_delta > 0:
        proporcoes = [d / total_delta for d in deltas]
        extras_exatos = [p * vagas_restantes for p in proporcoes]
        extras = [int(e) for e in extras_exatos]
        falta = vagas_restantes - sum(extras)
        ordem = sorted(range(num_subtables), key=lambda i: extras_exatos[i]-extras[i], reverse=True)
        for i in ordem[:falta]:
            extras[i] += 1
    else:
        extras = [0]*num_subtables
    for idx, mask in enumerate(masks_ativos):
        max_table_size[mask] = pisos_efetivos[mask] + extras[idx]

def run_nivel_arquivo_credito_duplo_torneio(tables, archive_holder, masks_ativos, pop_size, ngen_nivel, max_table_size,
                                             toolbox, cxpb, mutpb, n_obj, historico_scores, janela=10, snapshot_callback=None, lineage_log=None, pisos=None):
    """Igual a run_nivel_arquivo_torneio, mas credita AS DUAS tabelas-mãe
    de um cruzamento sempre que QUALQUER UM dos dois filhos sobrevive no
    arquivo -- não só a tabela marcada no filho específico que sobreviveu.
    Corrige o viés contra tabelas que são boas PARCEIRAS de cruzamento mas
    raramente "vencem" o sorteio de qual filho fica marcado com qual pai."""
    max_fes_nivel = pop_size * ngen_nivel
    fes_count = 0
    clone, mate, mutate = toolbox.clone, toolbox.mate, toolbox.mutate

    while fes_count < max_fes_nivel:
        for mask in masks_ativos:
            print(f"  [Nível {popcount(mask)}] Tabela {mask} -> Score: {tables[mask].score:6.2f} | Tamanho Alocado: {max_table_size[mask]}")

        offspring = []
        pares_masks = []  # (mask_pai1, mask_pai2) por par de filhos (off1,off2)
        while len(offspring) < pop_size:
            parents = select_parents_niveis_torneio(tables, masks_ativos, n_obj)
            off1 = clone(parents[0][0]); off2 = clone(parents[1][0])
            off1.Parent_Table = parents[0][1]; off2.Parent_Table = parents[1][1]
            if random.random() < cxpb:
                mate(off1, off2)
                del off1.fitness.values, off2.fitness.values
            if random.random() < mutpb:
                mutate(off1); del off1.fitness.values
            if random.random() < mutpb:
                mutate(off2); del off2.fitness.values
            offspring.append(off1); offspring.append(off2)
            pares_masks.append((parents[0][1], parents[1][1]))

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            if fes_count + len(invalid_ind) > max_fes_nivel:
                invalid_ind = invalid_ind[:(max_fes_nivel - fes_count)]
                offspring = [ind for ind in offspring if ind.fitness.valid] + invalid_ind
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            fes_count += len(invalid_ind)

        archive = archive_holder[0]
        clones_desta_geracao = [clone(ind) for ind in offspring]
        offspring_to_clone_id = {id(off): id(cl) for off, cl in zip(offspring, clones_desta_geracao)}
        ids_clones_desta_geracao = set(offspring_to_clone_id.values())
        archive.extend(clones_desta_geracao)
        if len(archive) > 0:
            fronts = tools.sortNondominated(archive, len(archive), first_front_only=True)
            non_dominated = fronts[0]
            if len(non_dominated) > pop_size:
                archive = creator.SubPopulation(tools.selNSGA2(non_dominated, pop_size))
            else:
                archive = creator.SubPopulation(non_dominated)
        archive_holder[0] = archive
        ids_sobreviventes_arquivo = {id(ind) for ind in archive if id(ind) in ids_clones_desta_geracao}

        # credito duplo: para cada par (off1,off2) gerado do mesmo cruzamento,
        # se QUALQUER UM dos dois clones sobreviveu no arquivo, credita as
        # DUAS mascaras-pai (parents[0][1] e parents[1][1]) daquele par
        if lineage_log is not None:
            mesma, diferente = 0, 0

        for idx in range(0, len(offspring), 2):
            off1, off2 = offspring[idx], offspring[idx + 1]
            m1, m2 = pares_masks[idx // 2]
            sobreviveu = (offspring_to_clone_id[id(off1)] in ids_sobreviventes_arquivo or
                          offspring_to_clone_id[id(off2)] in ids_sobreviventes_arquivo)
            if sobreviveu:
                if m1 in tables:
                    tables[m1].score += 1.0
                if m2 in tables:
                    tables[m2].score += 1.0

        for off in offspring:
            off.Parent_Table = None

        insert_tabelas_sem_recompensa(tables, masks_ativos, offspring, max_table_size, n_obj)
        alocar_por_derivada(tables, masks_ativos, max_table_size, n_obj, historico_scores, janela, pisos)

        if snapshot_callback is not None:
            tables[0] = archive_holder[0]
            snapshot_callback(tables)

    return tables

def run_flat_arquivo_credito_duplo_piso_torneio(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None, lineage_log=None, piso_nivel1=10, piso_outros=3):
    """SEM ESTÁGIOS + CRÉDITO DUPLO + piso maior pro nível 1 -- combina as
    duas correções testadas separadamente."""
    num_tables = 2 ** n_obj
    active_masks = list(range(1, num_tables))
    pisos = {m: (piso_nivel1 if popcount(m) == 1 else piso_outros) for m in active_masks}

    base_size = max(3, pop_size // len(active_masks))
    tables = {m: sel_nsga2(pop_ini, m, base_size, n_obj) for m in active_masks}
    max_table_size = {m: base_size for m in active_masks}
    historico_scores = {}
    archive_holder = [creator.SubPopulation()]

    tables = run_nivel_arquivo_credito_duplo_torneio(tables, archive_holder, active_masks, pop_size, total_gens, max_table_size,
                                                       toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback, lineage_log, pisos)
    tables[0] = archive_holder[0]
    return tables

def run_flat_arquivo_credito_duplo_torneio(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None, lineage_log=None):
    """SEM ESTÁGIOS + crédito duplo (às duas tabelas-mãe) + torneio, com
    piso UNIFORME (3) -- testa se o crédito duplo, sozinho, já resolve o
    viés contra o nível 1 sem precisar de piso artificial maior."""
    num_tables = 2 ** n_obj
    active_masks = list(range(1, num_tables))
    base_size = max(3, pop_size // len(active_masks))
    tables = {m: sel_nsga2(pop_ini, m, base_size, n_obj) for m in active_masks}
    max_table_size = {m: base_size for m in active_masks}
    historico_scores = {}
    archive_holder = [creator.SubPopulation()]

    tables = run_nivel_arquivo_credito_duplo_torneio(tables, archive_holder, active_masks, pop_size, total_gens, max_table_size,
                                                       toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback, lineage_log)
    tables[0] = archive_holder[0]
    return tables

def run_nivel_arquivo_torneio(tables, archive_holder, masks_ativos, pop_size, ngen_nivel, max_table_size,
                               toolbox, cxpb, mutpb, n_obj, historico_scores, janela=10, snapshot_callback=None, lineage_log=None):
    """Igual a run_nivel_arquivo, mas usa select_parents_niveis_torneio em
    vez de select_parents_niveis (Variante A + torneio combinados)."""
    max_fes_nivel = pop_size * ngen_nivel
    fes_count = 0
    clone, mate, mutate = toolbox.clone, toolbox.mate, toolbox.mutate

    while fes_count < max_fes_nivel:
        for mask in masks_ativos:
            print(f"  [Nível {popcount(mask)}] Tabela {mask} -> Score: {tables[mask].score:6.2f} | Tamanho Alocado: {max_table_size[mask]}")

        offspring = []
        while len(offspring) < pop_size:
            parents = select_parents_niveis_torneio(tables, masks_ativos, n_obj)
            off1 = clone(parents[0][0]); off2 = clone(parents[1][0])
            off1.Parent_Table = parents[0][1]; off2.Parent_Table = parents[1][1]
            if random.random() < cxpb:
                mate(off1, off2)
                del off1.fitness.values, off2.fitness.values
            if random.random() < mutpb:
                mutate(off1); del off1.fitness.values
            if random.random() < mutpb:
                mutate(off2); del off2.fitness.values
            offspring.append(off1); offspring.append(off2)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            if fes_count + len(invalid_ind) > max_fes_nivel:
                invalid_ind = invalid_ind[:(max_fes_nivel - fes_count)]
                offspring = [ind for ind in offspring if ind.fitness.valid] + invalid_ind
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            fes_count += len(invalid_ind)

        archive = archive_holder[0]
        clones_desta_geracao = [clone(ind) for ind in offspring]
        offspring_to_clone_id = {id(off): id(cl) for off, cl in zip(offspring, clones_desta_geracao)}
        ids_clones_desta_geracao = set(offspring_to_clone_id.values())
        archive.extend(clones_desta_geracao)
        if len(archive) > 0:
            fronts = tools.sortNondominated(archive, len(archive), first_front_only=True)
            non_dominated = fronts[0]
            if len(non_dominated) > pop_size:
                archive = creator.SubPopulation(tools.selNSGA2(non_dominated, pop_size))
            else:
                archive = creator.SubPopulation(non_dominated)
        archive_holder[0] = archive

        ids_sobreviventes_arquivo = {id(ind) for ind in archive if id(ind) in ids_clones_desta_geracao}
        ids_creditados = {oid for oid, cid in offspring_to_clone_id.items() if cid in ids_sobreviventes_arquivo}

        insert_in_tables_niveis_arquivo(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela, lineage_log, ids_creditados)

        if snapshot_callback is not None:
            tables[0] = archive_holder[0]
            snapshot_callback(tables)

    return tables

def run_staged_pairs_arquivo_torneio(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None, lineage_log=None):
    """COM ESTÁGIOS (janela de 2 níveis) + Variante A (recompensa por
    arquivo) + torneio dentro da tabela."""
    niveis = masks_by_level(n_obj)
    num_fases = max(1, n_obj - 1)
    gens_por_fase = max(1, total_gens // num_fases)

    archive_holder = [creator.SubPopulation()]
    active_masks = list(niveis.get(1, [])) + list(niveis.get(2, []))
    base_size = max(3, pop_size // len(active_masks))
    tables = {m: sel_nsga2(pop_ini, m, base_size, n_obj) for m in active_masks}
    max_table_size = {m: base_size for m in active_masks}
    historico_scores = {}

    tables = run_nivel_arquivo_torneio(tables, archive_holder, active_masks, pop_size, gens_por_fase, max_table_size,
                                        toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback, lineage_log)

    for nivel_velho in range(2, n_obj):
        nivel_novo = nivel_velho + 1
        masks_velho = niveis[nivel_velho]; masks_novo = niveis[nivel_novo]
        active_masks = masks_velho + masks_novo
        base_size = max(3, pop_size // len(active_masks))
        tables_cont = {m: tables[m] for m in masks_velho if m in tables}
        tables_novas = seed_next_level(tables, masks_novo, base_size, n_obj)
        tables = {**tables_cont, **tables_novas}
        max_table_size = {m: base_size for m in active_masks}
        historico_scores = {}
        tables = run_nivel_arquivo_torneio(tables, archive_holder, active_masks, pop_size, gens_por_fase, max_table_size,
                                            toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback, lineage_log)

    tables[0] = archive_holder[0]
    return tables

def run_flat_arquivo_torneio(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None, lineage_log=None):
    """SEM ESTÁGIOS: todas as 2^n_obj-1 tabelas ativas desde o início e o
    tempo todo (nenhuma fase, nenhuma janela de níveis) + Variante A
    (recompensa por arquivo) + torneio. Permite colaboração entre
    quaisquer níveis (ex.: nível 1 com nível 3) o tempo todo, ao custo de
    manter todas as tabelas ativas simultaneamente."""
    num_tables = 2 ** n_obj
    active_masks = list(range(1, num_tables))
    base_size = max(3, pop_size // len(active_masks))
    tables = {m: sel_nsga2(pop_ini, m, base_size, n_obj) for m in active_masks}
    max_table_size = {m: base_size for m in active_masks}
    historico_scores = {}
    archive_holder = [creator.SubPopulation()]

    tables = run_nivel_arquivo_torneio(tables, archive_holder, active_masks, pop_size, total_gens, max_table_size,
                                        toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback, lineage_log)
    tables[0] = archive_holder[0]
    return tables

# ==========================================
# SEM ESTÁGIOS + PISO POR NÍVEL (piso maior pras tabelas de 1 objetivo)
# ==========================================
def insert_in_tables_niveis_arquivo_piso_variavel(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela=10, lineage_log=None, ids_creditados=None, pisos=None):
    """Igual a insert_in_tables_niveis_arquivo, mas o piso (min_vagas) pode
    variar por tabela via o dict `pisos` (mask -> piso), em vez de um valor
    único pra todo mundo. Default (pisos=None) usa 3 pra todas, igual antes."""
    for mask in masks_ativos:
        current_score = tables[mask].score
        tables[mask].extend(offspring)
        tables[mask] = sel_nsga2(tables[mask], mask, max_table_size[mask], n_obj)
        tables[mask].score = current_score

    offspring_ids = {id(off) for off in offspring}

    if lineage_log is not None:
        mesma, diferente = 0, 0
        for mask in masks_ativos:
            for ind in tables[mask]:
                if id(ind) in offspring_ids:
                    parent_mask = getattr(ind, 'Parent_Table', None)
                    if parent_mask is not None:
                        if parent_mask == mask:
                            mesma += 1
                        else:
                            diferente += 1
        lineage_log.append({"mesma_tabela": mesma, "tabela_diferente": diferente})

    if ids_creditados is None:
        ids_creditados = set()
    for off in offspring:
        if id(off) in ids_creditados:
            parent_mask = getattr(off, 'Parent_Table', None)
            if parent_mask is not None and parent_mask in tables:
                tables[parent_mask].score += 1.0

    for off in offspring:
        off.Parent_Table = None

    if pisos is None:
        pisos = {m: 3 for m in masks_ativos}

    pop_total = sum(max_table_size[m] for m in masks_ativos)
    num_subtables = len(masks_ativos)
    piso_total = sum(pisos[m] for m in masks_ativos)

    pisos_efetivos = dict(pisos)
    if pop_total < piso_total:
        # reduz todos os pisos proporcionalmente ate caber no orcamento
        fator = pop_total / piso_total if piso_total > 0 else 0
        pisos_efetivos = {m: max(1, int(pisos[m] * fator)) for m in masks_ativos}
        print(f"  [AVISO] pop_total={pop_total} menor que soma dos pisos ({piso_total}). "
              f"Reduzindo pisos proporcionalmente (fator={fator:.2f}).")
        piso_total = sum(pisos_efetivos.values())

    scores = [max(0.01, tables[m].score) for m in masks_ativos]
    deltas = []
    for mask in masks_ativos:
        score_atual = tables[mask].score
        janela_tabela = historico_scores.setdefault(mask, deque(maxlen=janela))
        if len(janela_tabela) > 0:
            media_anterior = sum(janela_tabela) / len(janela_tabela)
            derivada = score_atual - media_anterior
        else:
            derivada = 0.0
        janela_tabela.append(score_atual)
        deltas.append(max(0.01, derivada))

    total_delta = sum(deltas)
    vagas_restantes = pop_total - piso_total

    if vagas_restantes > 0 and total_delta > 0:
        proporcoes = [d / total_delta for d in deltas]
        extras_exatos = [p * vagas_restantes for p in proporcoes]
        extras = [int(e) for e in extras_exatos]
        falta = vagas_restantes - sum(extras)
        ordem_por_resto = sorted(
            range(num_subtables),
            key=lambda idx: extras_exatos[idx] - extras[idx],
            reverse=True,
        )
        for idx in ordem_por_resto[:falta]:
            extras[idx] += 1
    else:
        extras = [0] * num_subtables

    for idx, mask in enumerate(masks_ativos):
        max_table_size[mask] = pisos_efetivos[mask] + extras[idx]

def run_nivel_arquivo_piso_torneio(tables, archive_holder, masks_ativos, pop_size, ngen_nivel, max_table_size,
                                    toolbox, cxpb, mutpb, n_obj, historico_scores, janela=10, snapshot_callback=None, lineage_log=None, pisos=None):
    max_fes_nivel = pop_size * ngen_nivel
    fes_count = 0
    clone, mate, mutate = toolbox.clone, toolbox.mate, toolbox.mutate

    while fes_count < max_fes_nivel:
        for mask in masks_ativos:
            print(f"  [Nível {popcount(mask)}] Tabela {mask} -> Score: {tables[mask].score:6.2f} | Tamanho Alocado: {max_table_size[mask]}")

        offspring = []
        while len(offspring) < pop_size:
            parents = select_parents_niveis_torneio(tables, masks_ativos, n_obj)
            off1 = clone(parents[0][0]); off2 = clone(parents[1][0])
            off1.Parent_Table = parents[0][1]; off2.Parent_Table = parents[1][1]
            if random.random() < cxpb:
                mate(off1, off2)
                del off1.fitness.values, off2.fitness.values
            if random.random() < mutpb:
                mutate(off1); del off1.fitness.values
            if random.random() < mutpb:
                mutate(off2); del off2.fitness.values
            offspring.append(off1); offspring.append(off2)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            if fes_count + len(invalid_ind) > max_fes_nivel:
                invalid_ind = invalid_ind[:(max_fes_nivel - fes_count)]
                offspring = [ind for ind in offspring if ind.fitness.valid] + invalid_ind
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            fes_count += len(invalid_ind)

        archive = archive_holder[0]
        clones_desta_geracao = [clone(ind) for ind in offspring]
        offspring_to_clone_id = {id(off): id(cl) for off, cl in zip(offspring, clones_desta_geracao)}
        ids_clones_desta_geracao = set(offspring_to_clone_id.values())
        archive.extend(clones_desta_geracao)
        if len(archive) > 0:
            fronts = tools.sortNondominated(archive, len(archive), first_front_only=True)
            non_dominated = fronts[0]
            if len(non_dominated) > pop_size:
                archive = creator.SubPopulation(tools.selNSGA2(non_dominated, pop_size))
            else:
                archive = creator.SubPopulation(non_dominated)
        archive_holder[0] = archive

        ids_sobreviventes_arquivo = {id(ind) for ind in archive if id(ind) in ids_clones_desta_geracao}
        ids_creditados = {oid for oid, cid in offspring_to_clone_id.items() if cid in ids_sobreviventes_arquivo}

        insert_in_tables_niveis_arquivo_piso_variavel(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela, lineage_log, ids_creditados, pisos)

        if snapshot_callback is not None:
            tables[0] = archive_holder[0]
            snapshot_callback(tables)

    return tables

def run_flat_arquivo_piso_torneio(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None, lineage_log=None, piso_nivel1=20, piso_outros=3):
    """SEM ESTÁGIOS + piso maior especificamente pras tabelas de 1
    objetivo (nível 1), que antes ficavam grudadas no piso uniforme."""
    num_tables = 2 ** n_obj
    active_masks = list(range(1, num_tables))
    niveis = masks_by_level(n_obj)
    pisos = {}
    for mask in active_masks:
        pisos[mask] = piso_nivel1 if popcount(mask) == 1 else piso_outros

    base_size = max(3, pop_size // len(active_masks))
    tables = {m: sel_nsga2(pop_ini, m, base_size, n_obj) for m in active_masks}
    max_table_size = {m: base_size for m in active_masks}
    historico_scores = {}
    archive_holder = [creator.SubPopulation()]

    tables = run_nivel_arquivo_piso_torneio(tables, archive_holder, active_masks, pop_size, total_gens, max_table_size,
                                             toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback, lineage_log, pisos)
    tables[0] = archive_holder[0]
    return tables

def run_staged_pairs_arquivo(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None, lineage_log=None):
    """Como run_staged_pairs, mas usando run_nivel_arquivo (VARIANTE A:
    score creditado por contribuição ao arquivo, não por sobrevivência
    local)."""
    niveis = masks_by_level(n_obj)
    num_fases = max(1, n_obj - 1)
    gens_por_fase = max(1, total_gens // num_fases)

    archive_holder = [creator.SubPopulation()]

    active_masks = list(niveis.get(1, [])) + list(niveis.get(2, []))
    base_size = max(3, pop_size // len(active_masks))
    tables = {}
    max_table_size = {}
    for mask in active_masks:
        tables[mask] = sel_nsga2(pop_ini, mask, base_size, n_obj)
        max_table_size[mask] = base_size
    historico_scores = {}

    tables = run_nivel_arquivo(tables, archive_holder, active_masks, pop_size, gens_por_fase, max_table_size,
                                toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback, lineage_log)

    for nivel_velho in range(2, n_obj):
        nivel_novo = nivel_velho + 1
        masks_velho = niveis[nivel_velho]
        masks_novo = niveis[nivel_novo]
        active_masks = masks_velho + masks_novo

        base_size = max(3, pop_size // len(active_masks))
        tables_continuadas = {m: tables[m] for m in masks_velho if m in tables}
        tables_novas = seed_next_level(tables, masks_novo, base_size, n_obj)

        tables = {**tables_continuadas, **tables_novas}
        max_table_size = {mask: base_size for mask in active_masks}
        historico_scores = {}

        tables = run_nivel_arquivo(tables, archive_holder, active_masks, pop_size, gens_por_fase, max_table_size,
                                    toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback, lineage_log)

    tables[0] = archive_holder[0]
    return tables

# ==========================================
# VARIANTES B e C: mesma orquestração de run_staged_pairs, trocando só a
# função de insert_in_tables_niveis usada dentro de run_nivel
# ==========================================
def run_staged_pairs_diversidade(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None, lineage_log=None):
    """VARIANTE B: peso de alocação multiplicado pela diversidade normalizada."""
    return _run_staged_pairs_generic(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb,
                                      insert_in_tables_niveis_diversidade, janela, snapshot_callback, lineage_log)

def run_staged_pairs_corte(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None, lineage_log=None, limiar_diversidade=1e-6):
    """VARIANTE C: corte direto pro piso quando diversidade < limiar."""
    def insert_fn(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela, lineage_log):
        return insert_in_tables_niveis_corte(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela, lineage_log, limiar_diversidade)
    return _run_staged_pairs_generic(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb,
                                      insert_fn, janela, snapshot_callback, lineage_log)

def _run_nivel_generic(tables, archive_holder, masks_ativos, pop_size, ngen_nivel, max_table_size,
                        toolbox, cxpb, mutpb, n_obj, historico_scores, insert_fn, janela=10, snapshot_callback=None, lineage_log=None, select_fn=None):
    max_fes_nivel = pop_size * ngen_nivel
    fes_count = 0
    clone = toolbox.clone
    mate = toolbox.mate
    mutate = toolbox.mutate
    if select_fn is None:
        select_fn = lambda tbls, masks: select_parents_niveis(tbls, masks)

    while fes_count < max_fes_nivel:
        for mask in masks_ativos:
            print(f"  [Nível {popcount(mask)}] Tabela {mask} -> Score: {tables[mask].score:6.2f} | Tamanho Alocado: {max_table_size[mask]}")

        offspring = []
        while len(offspring) < pop_size:
            parents = select_fn(tables, masks_ativos)
            off1 = clone(parents[0][0])
            off2 = clone(parents[1][0])
            off1.Parent_Table = parents[0][1]
            off2.Parent_Table = parents[1][1]
            if random.random() < cxpb:
                mate(off1, off2)
                del off1.fitness.values, off2.fitness.values
            if random.random() < mutpb:
                mutate(off1)
                del off1.fitness.values
            if random.random() < mutpb:
                mutate(off2)
                del off2.fitness.values
            offspring.append(off1)
            offspring.append(off2)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            if fes_count + len(invalid_ind) > max_fes_nivel:
                invalid_ind = invalid_ind[:(max_fes_nivel - fes_count)]
                offspring = [ind for ind in offspring if ind.fitness.valid] + invalid_ind
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            fes_count += len(invalid_ind)

        archive = archive_holder[0]
        archive.extend([clone(ind) for ind in offspring])
        if len(archive) > 0:
            fronts = tools.sortNondominated(archive, len(archive), first_front_only=True)
            non_dominated = fronts[0]
            if len(non_dominated) > pop_size:
                truncated = tools.selNSGA2(non_dominated, pop_size)
                archive = creator.SubPopulation(truncated)
            else:
                archive = creator.SubPopulation(non_dominated)
        archive_holder[0] = archive

        insert_fn(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela, lineage_log)

        if snapshot_callback is not None:
            tables[0] = archive_holder[0]
            snapshot_callback(tables)

    return tables

def _run_staged_pairs_generic(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, insert_fn, janela=10, snapshot_callback=None, lineage_log=None, select_fn=None):
    niveis = masks_by_level(n_obj)
    num_fases = max(1, n_obj - 1)
    gens_por_fase = max(1, total_gens // num_fases)

    archive_holder = [creator.SubPopulation()]

    active_masks = list(niveis.get(1, [])) + list(niveis.get(2, []))
    base_size = max(3, pop_size // len(active_masks))
    tables = {}
    max_table_size = {}
    for mask in active_masks:
        tables[mask] = sel_nsga2(pop_ini, mask, base_size, n_obj)
        max_table_size[mask] = base_size
    historico_scores = {}

    tables = _run_nivel_generic(tables, archive_holder, active_masks, pop_size, gens_por_fase, max_table_size,
                                 toolbox, cxpb, mutpb, n_obj, historico_scores, insert_fn, janela, snapshot_callback, lineage_log, select_fn)

    for nivel_velho in range(2, n_obj):
        nivel_novo = nivel_velho + 1
        masks_velho = niveis[nivel_velho]
        masks_novo = niveis[nivel_novo]
        active_masks = masks_velho + masks_novo

        base_size = max(3, pop_size // len(active_masks))
        tables_continuadas = {m: tables[m] for m in masks_velho if m in tables}
        tables_novas = seed_next_level(tables, masks_novo, base_size, n_obj)

        tables = {**tables_continuadas, **tables_novas}
        max_table_size = {mask: base_size for mask in active_masks}
        historico_scores = {}

        tables = _run_nivel_generic(tables, archive_holder, active_masks, pop_size, gens_por_fase, max_table_size,
                                     toolbox, cxpb, mutpb, n_obj, historico_scores, insert_fn, janela, snapshot_callback, lineage_log, select_fn)

    tables[0] = archive_holder[0]
    return tables

def run_staged_pairs_torneio(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None, lineage_log=None):
    """SÓ TORNEIO, sem corte por diversidade: usa a alocação padrão
    (insert_in_tables_niveis, baseada só em score) + seleção por torneio
    binário dentro da tabela vencedora (select_parents_niveis_torneio).
    Isola o efeito do torneio isoladamente, pra comparar com a combinação
    torneio+corte."""
    def select_fn(tables, masks_ativos):
        return select_parents_niveis_torneio(tables, masks_ativos, n_obj)

    return _run_staged_pairs_generic(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb,
                                      insert_in_tables_niveis, janela, snapshot_callback, lineage_log, select_fn)

def run_staged_pairs_torneio_corte(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None, lineage_log=None, limiar_diversidade=1e-6):
    """VARIANTE C + TORNEIO: corte no piso por diversidade colapsada
    (Variante C, já escolhida) combinado com seleção por torneio binário
    dentro da tabela vencedora (mais pressão seletiva local, em vez de
    random.choice)."""
    def insert_fn(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela, lineage_log):
        return insert_in_tables_niveis_corte(tables, masks_ativos, offspring, max_table_size, n_obj, historico_scores, janela, lineage_log, limiar_diversidade)

    def select_fn(tables, masks_ativos):
        return select_parents_niveis_torneio(tables, masks_ativos, n_obj)

    return _run_staged_pairs_generic(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb,
                                      insert_fn, janela, snapshot_callback, lineage_log, select_fn)

def run_staged_pairs_reset_score(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None):
    """Igual a run_staged_pairs, mas ZERA o .score de TODAS as tabelas
    ativas (inclusive as que continuam de uma fase pra outra) a cada
    transição de fase. Motivo: select_parents_niveis decide o torneio de
    seleção de pais comparando SCORE ABSOLUTO (não a derivada) -- uma
    tabela "veterana" com score acumulado de muitas gerações vence o
    torneio contra uma recém-semeada (score=0) quase sempre, mesmo que a
    derivada já indique que a nova é mais produtiva agora. Resetar o
    histórico_scores (janela) sozinho não corrige isso, porque ele só
    afeta a ALOCAÇÃO de população, não o torneio de pais."""
    niveis = masks_by_level(n_obj)
    num_fases = max(1, n_obj - 1)
    gens_por_fase = max(1, total_gens // num_fases)

    archive_holder = [creator.SubPopulation()]

    active_masks = list(niveis.get(1, [])) + list(niveis.get(2, []))
    base_size = max(3, pop_size // len(active_masks))
    tables = {}
    max_table_size = {}
    for mask in active_masks:
        tables[mask] = sel_nsga2(pop_ini, mask, base_size, n_obj)
        tables[mask].score = 0.0
        max_table_size[mask] = base_size
    historico_scores = {}

    tables = run_nivel(tables, archive_holder, active_masks, pop_size, gens_por_fase, max_table_size,
                        toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback)

    for nivel_velho in range(2, n_obj):
        nivel_novo = nivel_velho + 1
        masks_velho = niveis[nivel_velho]
        masks_novo = niveis[nivel_novo]
        active_masks = masks_velho + masks_novo

        base_size = max(3, pop_size // len(active_masks))

        tables_continuadas = {m: tables[m] for m in masks_velho if m in tables}
        tables_novas = seed_next_level(tables, masks_novo, base_size, n_obj)

        tables = {**tables_continuadas, **tables_novas}
        for mask in active_masks:
            tables[mask].score = 0.0  # <<< RESET -- a única diferença real vs run_staged_pairs
        max_table_size = {mask: base_size for mask in active_masks}
        historico_scores = {}

        tables = run_nivel(tables, archive_holder, active_masks, pop_size, gens_por_fase, max_table_size,
                            toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback)

    tables[0] = archive_holder[0]
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

    backups = [ind.fitness.values for ind in offspring]
    
    inactive_indices = get_inactive_objs(mask, nobj)
    active_indices = get_active_objs(mask, nobj)
    
    if inactive_indices:
        for ind in offspring:
            vals = list(ind.fitness.values)
            for idx in inactive_indices:
                vals[idx] = 0.0
            ind.fitness.values = tuple(vals)

    survivors = _sel_nsga2_active(offspring, max_table_size, active_indices)

    for ind, backup in zip(offspring, backups):
        ind.fitness.values = backup

    return creator.SubPopulation(survivors)

def insert_in_tables(tables, num_tables, offspring, max_table_size, n_obj, historico_scores, janela=10):
    # 1. Inserção normal e truncamento (Preservando o histórico de score)
    for i in range(1, num_tables):
        current_score = tables[i].score  
        tables[i].extend(offspring)
        tables[i] = sel_nsga2(tables[i], i, max_table_size[i], n_obj)
        tables[i].score = current_score  

    # 2. Rastreamento dos Filhos Sobreviventes + Premiação
    # Cada indivíduo do offspring que sobreviveu ao truncamento NSGA2 de uma
    # tabela dá exatamente +1 de score para a tabela de ORIGEM dele
    # (Parent_Table) -- sem peso de diversidade, só a contagem bruta de
    # quantos filhos aquela tabela conseguiu "colocar" nas tabelas.
    offspring_ids = {id(off) for off in offspring}
    for i in range(1, num_tables):
        for ind in tables[i]:
            if id(ind) in offspring_ids:
                parent_table = getattr(ind, 'Parent_Table', None)
                if parent_table is not None:
                    tables[parent_table].score += 1.0

    # 3. Limpeza de memória
    for off in offspring:
        off.Parent_Table = None

    # ==========================================
    # 4. ALOCAÇÃO DINÂMICA POR DERIVADA VIA SLIDING WINDOW DE SCORE
    # ==========================================
    # Em vez de EMA do delta bruto, mantemos uma janela deslizante das
    # últimas `janela` gerações de score de cada tabela. A derivada é
    # score_atual - média(janela anterior) -- compara o score de agora
    # contra a média recente, em vez de só o valor da geração passada.
    # Isso dá uma baseline mais estável (menos sensível a um pico isolado
    # numa única geração) com um corte de memória rígido de `janela`
    # gerações, em vez do decaimento suave/indefinido do EMA.
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

    deltas = []
    for i in range(1, num_tables):
        score_atual = tables[i].score
        janela_tabela = historico_scores.setdefault(i, deque(maxlen=janela))

        if len(janela_tabela) > 0:
            media_anterior = sum(janela_tabela) / len(janela_tabela)
            derivada = score_atual - media_anterior
        else:
            derivada = 0.0  # primeira geração: sem histórico ainda, neutro

        janela_tabela.append(score_atual)
        deltas.append(max(0.01, derivada))

    total_delta = sum(deltas)
    vagas_restantes = pop_total - (effective_min_vagas * num_subtables)

    if vagas_restantes > 0 and total_delta > 0:
        proporcoes = [d / total_delta for d in deltas]
        extras_exatos = [p * vagas_restantes for p in proporcoes]
        extras = [int(e) for e in extras_exatos]
        falta = vagas_restantes - sum(extras)

        # método dos maiores restos: sem viés estrutural pra nenhuma tabela
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
def run(
    tables,
    num_tables,
    pop_size,
    ngen,
    max_table_size,
    toolbox,
    cxpb,
    mutpb,
    n_obj,
    snapshot_callback=None,
    janela=10,
):
    max_fes = pop_size * ngen
    fes_count = 0

    # historico_scores guarda, por tabela, a janela deslizante dos últimos
    # `janela` valores de score -- serve de baseline pro cálculo da derivada
    # em insert_in_tables. Persiste durante todo o run() (passado por
    # referência, atualizado dentro de insert_in_tables).
    historico_scores = {}
    
    # ==========================================
    # PREPARAÇÃO DO ARQUIVO EXTERNO (TABELA 0)
    # ==========================================
    tables[0] = creator.SubPopulation()
        
    # Salva a população inicial na Tabela 0 logo de cara
    iniciais_unicos = {
        id(ind): ind
        for i in range(1, num_tables)
        for ind in tables[i]
    }
    todas_iniciais = list(iniciais_unicos.values())
    if todas_iniciais:
        fronts_ini = tools.sortNondominated(todas_iniciais, len(todas_iniciais), first_front_only=True)
        tables[0].extend(fronts_ini[0])

    if snapshot_callback is not None:
        snapshot_callback(tables)

    # Loop principal
    while fes_count < max_fes:
        offspring = []
        for i in range(1, num_tables):
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
            fronts = tools.sortNondominated(tables[0], len(tables[0]), first_front_only=True)
            non_dominated = fronts[0]
            
            if len(non_dominated) > pop_size:
                truncated_archive = tools.selNSGA2(non_dominated, pop_size)
                tables[0] = creator.SubPopulation(truncated_archive)
            else:
                tables[0] = creator.SubPopulation(non_dominated)
                
            
        # A evolução continua nas tabelas dinâmicas (Tabelas 1 até num_tables-1)
        insert_in_tables(tables, num_tables, offspring, max_table_size, n_obj, historico_scores, janela)

        if snapshot_callback is not None:
            snapshot_callback(tables)
        
    return tables