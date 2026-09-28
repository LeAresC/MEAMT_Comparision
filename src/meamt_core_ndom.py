"""MEAMT -- duas versoes hierarquicas num so arquivo.

1) run_hierarquico(...)  -- versao das anotacoes do professor.
   Hierarquia CUMULATIVA a partir do nivel 1: comeca so com as tabelas de
   1 objetivo, depois acrescenta as de 2 (ficando 1+2 ativos), e assim por
   diante; nada congela. Variantes com congelamento tambem estao aqui
   (run_staged_congelado_torneio, run_staged_congelado_hibrido).
   Selecao por dominancia mascarada, score por contribuicao ao arquivo,
   alocacao de populacao por derivada do score.

2) run_tchebycheff(...)  -- versao construida nesta investigacao.
   Fases em ordem DECRESCENTE de objetivos ativos: nivel n-1 (vertices,
   0-D) -> n-2 -> ... -> nivel 1 -> nivel n (frente completa). Uma mascara
   com k objetivos ativos zera k objetivos e deixa uma regiao de dimensao
   n-k-1, entao a dimensao da regiao CRESCE conforme k diminui.

   Fase 1 por Tchebycheff (ordem total; ~4x mais rapido que dominancia
   para convergir). Fases seguintes por crossover SEM mutacao -- cruzar
   pontos ja convergidos preserva a convergencia, e a mutacao a destruia.
   Insercao por sel_aresta, em tres etapas:
     a) filtro de regiao: f_ativo entra como FILTRO binario, nao como
        criterio de ordenacao. Na aresta o objetivo ativo nao mede posicao
        (mediu-se f0 = 6.123e-17 * f1), entao ordenar por ele colapsa a
        tabela numa ponta.
     b) filtro de Pareto COM restricoes (constraint-domination).
     c) truncamento iterativo do SPEA2 nas direcoes NORMALIZADAS -- a
        normalizacao separa "direcao rara" de "longe da frente"; o
        truncamento iterativo evita o colapso da crowding distance.

   run_tchebycheff recebe apenas pop_size e total_gens e faz a divisao
   entre as etapas sozinho.
"""
import copy
import math
import random
import itertools
from itertools import chain
from collections import deque

import numpy as np
from deap import creator, base, tools


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


_INACTIVE_OBJS_CACHE = {}
_ACTIVE_OBJS_CACHE = {}
_REF_POINTS_CACHE = {}
_REF_FULL_CACHE = {}
_FIT_CLS_CACHE = {}


def setup_deap_classes(n_obj):
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", ConstraintFitness, weights=(-1.0,) * n_obj)
        creator.create("Individual", list, fitness=creator.FitnessMin, Parent_Table=None)
        creator.create("SubPopulation", list, score=0.0)


def fast_clone(ind):
    new_ind = creator.Individual(ind)              # copia os genes (floats, imutáveis)
    if ind.fitness.valid:
        new_ind.fitness.values = ind.fitness.values
    new_ind.fitness.constraint_violation = getattr(
        ind.fitness, "constraint_violation", 0.0
    )
    new_ind.Parent_Table = ind.Parent_Table
    return new_ind


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


def popcount(m):
    return bin(m).count("1")


def get_active_objs(mask, n_obj):
    """Complemento de get_inactive_objs: índices dos objetivos ATIVOS para esta máscara."""
    key = (mask, n_obj)
    if key not in _ACTIVE_OBJS_CACHE:
        inactive = set(get_inactive_objs(mask, n_obj))
        _ACTIVE_OBJS_CACHE[key] = [i for i in range(n_obj) if i not in inactive]
    return _ACTIVE_OBJS_CACHE[key]


def get_inactive_objs(mask, n_obj):
    """Retorna estritamente os índices que devem ser zerados. Ignora os ativos."""
    key = (mask, n_obj)
    if key not in _INACTIVE_OBJS_CACHE:
        _INACTIVE_OBJS_CACHE[key] = [i for i in range(n_obj) if not ((mask >> i) & 1)]
    return _INACTIVE_OBJS_CACHE[key]


def masks_by_level(n_obj):
    """Retorna {nivel: [lista de máscaras com esse popcount]}, nivel de 1 a n_obj."""
    niveis = {}
    for m in range(1, 2 ** n_obj):
        k = popcount(m)
        niveis.setdefault(k, []).append(m)
    return niveis


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


def run_cumulativo_desde_nivel1(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10,
                                 snapshot_callback=None, lineage_log=None):
    """ESTRATÉGIA 2 (sugestão do professor), na forma exata da anotação:
    começa com SÓ as tabelas de 1 objetivo, depois adiciona as de 2 (ficando
    1+2 ativos), depois as de 3 (ficando 1+2+3), e assim por diante --
    hierarquia cumulativa a partir do nível 1.

    Diferente de run_staged_cumulativo_arquivo_torneio, que começava já com
    níveis 1 E 2 juntos na primeira fase. Diferente também do híbrido
    congelado: aqui NADA congela -- todos os níveis já introduzidos
    continuam evoluindo e recebendo população até o fim.

    São n_obj fases, cada uma com total_gens/n_obj gerações."""
    niveis = masks_by_level(n_obj)
    gens_por_fase = max(1, total_gens // n_obj)

    archive_holder = [creator.SubPopulation()]

    # Fase 1: SÓ o nível 1
    active_masks = list(niveis[1])
    base_size = max(3, pop_size // len(active_masks))
    tables = {m: sel_nsga2(pop_ini, m, base_size, n_obj) for m in active_masks}
    max_table_size = {m: base_size for m in active_masks}
    historico_scores = {}

    tables = run_nivel_arquivo_torneio(tables, archive_holder, active_masks, pop_size, gens_por_fase,
                                        max_table_size, toolbox, cxpb, mutpb, n_obj, historico_scores,
                                        janela, snapshot_callback, lineage_log)

    # Fases seguintes: adiciona um nível por vez, mantendo TODOS os anteriores ativos
    for nivel_novo in range(2, n_obj + 1):
        masks_anteriores = list(active_masks)
        masks_novo = niveis[nivel_novo]
        active_masks = masks_anteriores + list(masks_novo)

        base_size = max(3, pop_size // len(active_masks))
        tables_continuadas = {m: tables[m] for m in masks_anteriores if m in tables}
        tables_novas = seed_next_level(tables, masks_novo, base_size, n_obj)
        tables = {**tables_continuadas, **tables_novas}
        max_table_size = {m: base_size for m in active_masks}
        historico_scores = {}

        tables = run_nivel_arquivo_torneio(tables, archive_holder, active_masks, pop_size, gens_por_fase,
                                            max_table_size, toolbox, cxpb, mutpb, n_obj, historico_scores,
                                            janela, snapshot_callback, lineage_log)

    tables[0] = archive_holder[0]
    return tables


def run_nivel_congelado_torneio(tables, archive_holder, masks_evoluindo, masks_congeladas, pop_size, ngen_nivel, max_table_size,
                                 toolbox, cxpb, mutpb, n_obj, historico_scores, janela=10, snapshot_callback=None, lineage_log=None):
    """Roda uma fase onde SÓ `masks_evoluindo` recebe filhos e disputa
    população, mas `masks_congeladas` (níveis já otimizados em fases
    anteriores) continuam disponíveis como PAIS no torneio -- fornecendo
    material genético sem consumir orçamento de população.

    Diferente de run_nivel_arquivo_torneio, onde toda máscara ativa tanto
    gera quanto recebe. Aqui a separação é explícita:
      - seleção de pais: masks_evoluindo + masks_congeladas
      - inserção/alocação: só masks_evoluindo
    O score das congeladas ainda é creditado (elas participam de
    cruzamentos que dão certo), mas não é usado pra alocar população --
    elas mantêm o tamanho que tinham ao congelar."""
    max_fes_nivel = pop_size * ngen_nivel
    fes_count = 0
    clone, mate, mutate = toolbox.clone, toolbox.mate, toolbox.mutate

    masks_para_pais = list(masks_evoluindo) + list(masks_congeladas)

    while fes_count < max_fes_nivel:
        for mask in masks_evoluindo:
            print(f"  [Nível {popcount(mask)} ATIVO] Tabela {mask} -> Score: {tables[mask].score:6.2f} | Tamanho: {max_table_size[mask]}")
        for mask in masks_congeladas:
            print(f"  [Nível {popcount(mask)} congelado] Tabela {mask} -> n={len(tables[mask])}")

        offspring = []
        while len(offspring) < pop_size:
            parents = select_parents_niveis_torneio(tables, masks_para_pais, n_obj)
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

        # inserção/alocação SÓ nas máscaras que estão evoluindo nesta fase
        insert_in_tables_niveis_arquivo(tables, masks_evoluindo, offspring, max_table_size, n_obj,
                                         historico_scores, janela, lineage_log, ids_creditados)

        if snapshot_callback is not None:
            tables[0] = archive_holder[0]
            snapshot_callback(tables)

    return tables


def run_staged_congelado_torneio(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10, snapshot_callback=None, lineage_log=None):
    """Estágios nível a nível COM CONGELAMENTO (ideia do professor):
      Fase 1: SÓ o nível 1 evolui, sozinho, com todo o orçamento da fase.
      Fase 2: nível 1 congela (vira só fonte de pais); nível 2 evolui.
      Fase k: níveis 1..k-1 congelados como pais; nível k evolui.
    Cada nível recebe total_gens/n_obj gerações. Isso replica dentro do
    algoritmo o experimento manual que deu o melhor resultado de
    colaboração: otimizar bem uma tabela especialista primeiro e depois
    usá-la como fonte de material genético."""
    niveis = masks_by_level(n_obj)
    gens_por_fase = max(1, total_gens // n_obj)

    archive_holder = [creator.SubPopulation()]
    tables = {}
    max_table_size = {}
    masks_congeladas = []

    for nivel in range(1, n_obj + 1):
        masks_nivel = niveis[nivel]

        if nivel == 1:
            base_size = max(3, pop_size // len(masks_nivel))
            for mask in masks_nivel:
                tables[mask] = sel_nsga2(pop_ini, mask, base_size, n_obj)
                max_table_size[mask] = base_size
        else:
            base_size = max(3, pop_size // len(masks_nivel))
            tables_novas = seed_next_level(tables, masks_nivel, base_size, n_obj)
            tables.update(tables_novas)
            for mask in masks_nivel:
                max_table_size[mask] = base_size

        historico_scores = {}
        tables = run_nivel_congelado_torneio(
            tables, archive_holder, masks_nivel, masks_congeladas, pop_size, gens_por_fase,
            max_table_size, toolbox, cxpb, mutpb, n_obj, historico_scores, janela,
            snapshot_callback, lineage_log
        )

        # ao terminar a fase, este nível vira congelado pras próximas
        masks_congeladas = masks_congeladas + list(masks_nivel)

    tables[0] = archive_holder[0]
    return tables


def run_staged_congelado_hibrido(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=10,
                                  snapshot_callback=None, lineage_log=None, frac_descongelada=0.4):
    """HÍBRIDO: fases congeladas nível a nível (constrói especialistas
    bons, como run_staged_congelado_torneio) seguidas de uma FASE FINAL
    onde TUDO descongela e evolui junto (como o flat), pra refinar a
    cobertura geral sem perder os especialistas já construídos.

    `frac_descongelada` é a fração do orçamento total reservada pra fase
    final (0.4 = 40% das gerações no final com tudo evoluindo junto; os
    60% restantes divididos igualmente entre as n_obj fases congeladas)."""
    niveis = masks_by_level(n_obj)
    gens_final = max(1, int(total_gens * frac_descongelada))
    gens_congeladas = total_gens - gens_final
    gens_por_fase = max(1, gens_congeladas // n_obj)

    archive_holder = [creator.SubPopulation()]
    tables = {}
    max_table_size = {}
    masks_congeladas = []

    # --- Etapa 1: fases congeladas, nível a nível ---
    for nivel in range(1, n_obj + 1):
        masks_nivel = niveis[nivel]
        base_size = max(3, pop_size // len(masks_nivel))
        if nivel == 1:
            for mask in masks_nivel:
                tables[mask] = sel_nsga2(pop_ini, mask, base_size, n_obj)
                max_table_size[mask] = base_size
        else:
            tables_novas = seed_next_level(tables, masks_nivel, base_size, n_obj)
            tables.update(tables_novas)
            for mask in masks_nivel:
                max_table_size[mask] = base_size

        historico_scores = {}
        tables = run_nivel_congelado_torneio(
            tables, archive_holder, masks_nivel, masks_congeladas, pop_size, gens_por_fase,
            max_table_size, toolbox, cxpb, mutpb, n_obj, historico_scores, janela,
            snapshot_callback, lineage_log
        )
        masks_congeladas = masks_congeladas + list(masks_nivel)

    # --- Etapa 2: fase final, TUDO descongelado evoluindo junto ---
    todas_masks = list(range(1, 2 ** n_obj))
    base_size = max(3, pop_size // len(todas_masks))
    for mask in todas_masks:
        if mask not in tables:
            tables[mask] = sel_nsga2(pop_ini, mask, base_size, n_obj)
        max_table_size[mask] = base_size
    historico_scores = {}

    tables = run_nivel_arquivo_torneio(
        tables, archive_holder, todas_masks, pop_size, gens_final, max_table_size,
        toolbox, cxpb, mutpb, n_obj, historico_scores, janela, snapshot_callback, lineage_log
    )

    tables[0] = archive_holder[0]
    return tables


def _cv(ind):
    return getattr(ind.fitness, "constraint_violation", 0.0) or 0.0


def sort_nondominated_restrito(pop, k):
    """sortNondominated que respeita RESTRICOES (constraint-domination de
    Deb): factivel domina infactivel; entre infactiveis vence o de menor
    violacao; entre factiveis, dominancia normal.

    O tools.sortNondominated do DEAP compara SO os valores dos objetivos.
    Em problemas restritos (DTLZ9) isso faz o algoritmo otimizar como se
    fosse irrestrito -- medido: todas as solucoes encontradas violavam as
    restricoes (violacao de 0.33 a 1.26) e ficavam fora da frente real."""
    viol = [ind for ind in pop if _cv(ind) > 0]
    fact = [ind for ind in pop if _cv(ind) <= 0]
    fronts = []
    if fact:
        fronts.extend(tools.sortNondominated(fact, min(k, len(fact))))
    if viol:
        # infactiveis entram DEPOIS de todos os factiveis, ordenados por
        # violacao crescente (cada nivel de violacao vira uma "frente")
        viol.sort(key=_cv)
        fronts.extend([[ind] for ind in viol])
    return fronts if fronts else [list(pop)]


def _grid_coords(P, div):
    """Coordenadas de grid do GrEA (Yang, Li, Liu, Zheng, IEEE TEVC 2013).
    Cada objetivo e dividido em `div` intervalos entre o min e o max da
    populacao; a largura leva um pequeno alargamento para que os extremos
    caiam dentro do grid."""
    mn = P.min(axis=0)
    mx = P.max(axis=0)
    d = (mx - mn) / div
    d[d <= 0] = 1e-12
    lo = mn - d / (2 * div)
    d = (mx + d / (2 * div) - lo) / div
    d[d <= 0] = 1e-12
    return np.floor((P - lo) / d).astype(int)


def sel_aresta(offspring, mask, max_table_size, nobj, tol_rel=0.05):
    """Regra de armazenamento da tabela, na forma exata:
       "contanto que f_ativo seja ~0 e o ponto esteja no Pareto,
        quero os objetivos INATIVOS o mais diversos possivel".

    f_ativo e FILTRO BINARIO, nao criterio de ordenacao. Essa e a
    diferenca central em relacao a sel_nsga2: ordenar por f_ativo na
    aresta e inutil, porque ali f_ativo nao mede posicao no arco. Medido
    no DTLZ3: na aresta f0=0 vale f0 = 6.123e-17 * f1 EXATAMENTE (o erro
    de cos(pi/2) escalado por f1), entao ordenar por f0 equivale a
    ordenar por f1 e a tabela colapsa numa ponta do arco. Zerando f0 o
    criterio some e a ordem vira arbitraria -- colapsa na outra ponta.

    Etapas:
      1. FILTRO regiao: f_ativo/||f|| <= tol_rel (esta na aresta/face)
      2. FILTRO Pareto: nao-dominado no espaco COMPLETO (convergiu; um
         ponto com g alto e dominado por um de mesma direcao com g=0)
      3. DIVERSIDADE: entre os aprovados, truncamento iterativo nas
         direcoes normalizadas dos objetivos INATIVOS -- remove um por vez
         o mais amontoado e recalcula (truncamento do SPEA2), que espalha
         de verdade, ao contrario de crowding distance em lote.
    Se o filtro deixar menos que k, completa com os melhores por
    f_ativo/||f||.
    """
    n = len(offspring)
    if n <= max_table_size:
        return creator.SubPopulation(offspring)
    ativos = get_active_objs(mask, nobj)
    if not ativos:
        return creator.SubPopulation(offspring[:max_table_size])
    inativos = [i for i in range(nobj) if i not in set(ativos)]
    if not inativos:
        # MASCARA COMPLETA: sem objetivos inativos. Delegar para sel_nsga2
        # (crowding em n objetivos) DEGRADA a cobertura -- medido: a fase
        # final recebia 68.6% da fase anterior e terminava com 41.4%.
        # Acumular frentes inteiras e so truncar a ultima tambem nao serve:
        # a maior parte entra por dominancia pura e a diversidade mal atua
        # (275 de 300 vinham assim). Aqui o truncamento SPEA2 e aplicado ao
        # conjunto TODO, usando o rank de dominancia so como desempate.
        fronts = sort_nondominated_restrito(offspring, len(offspring))
        rank = {}
        for r, f in enumerate(fronts):
            for ind in f:
                rank[id(ind)] = r
        P = np.array([list(i.fitness.values) for i in offspring], dtype=float)
        U = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
        D = np.sqrt(((U[:, None, :] - U[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(D, np.inf)
        vivos = np.ones(len(offspring), dtype=bool)
        ranks = np.array([rank[id(i)] for i in offspring])
        while vivos.sum() > max_table_size:
            act = np.flatnonzero(vivos)
            pior_rank = ranks[act].max()
            # remove primeiro entre os de PIOR rank; entre eles, o mais amontoado
            cand = act[ranks[act] == pior_rank]
            if len(cand) == 1:
                vivos[cand[0]] = False
                continue
            ordenadas = np.sort(D[np.ix_(cand, vivos)], axis=1)
            vivos[cand[int(min(range(len(cand)), key=lambda i: tuple(ordenadas[i])))]] = False
        return creator.SubPopulation([offspring[i] for i in np.flatnonzero(vivos)])

    nor = [math.sqrt(sum(v * v for v in ind.fitness.values)) for ind in offspring]
    ativo_rel = [sum(ind.fitness.values[i] for i in ativos) / max(nv, 1e-12)
                 for ind, nv in zip(offspring, nor)]

    # CASO 0-D (mascara de nivel n-1 -> VERTICE): a regiao e um ponto, todos
    # os candidatos tem a MESMA direcao normalizada e a etapa de diversidade
    # fica cega -- ela acaba escolhendo entre pontos que so diferem em g
    # (medido: 120 pontos no mesmo vertice, sel_aresta devolvia norma media
    # 81.9 em vez de 1.0). Aqui a decisao e por norma, que vale exatamente
    # 1+g: e dominancia pura no espaco completo, sem precisar de Tchebycheff.
    if len(ativos) == nobj - 1:
        # decide por DOMINANCIA no espaco completo (generico: nao assume
        # nada sobre a forma da frente). A versao anterior ordenava pela
        # NORMA, que so mede convergencia em DTLZ2/3/4, onde a frente e a
        # esfera unitaria e norma = 1+g -- num problema de frente linear
        # (DTLZ1) ou escalas arbitrarias isso nao vale.
        na_regiao = [i for i in range(n) if ativo_rel[i] <= tol_rel]
        if len(na_regiao) < max_table_size:
            na_regiao = sorted(range(n), key=lambda i: ativo_rel[i])[:max(max_table_size, len(na_regiao))]
        sub = [offspring[i] for i in na_regiao]
        if len(sub) <= max_table_size:
            return creator.SubPopulation(sub)
        fronts = sort_nondominated_restrito(sub, max_table_size)
        escolhidos = []
        for f in fronts:
            if len(escolhidos) + len(f) > max_table_size:
                escolhidos.extend(f[:max_table_size - len(escolhidos)])
                break
            escolhidos.extend(f)
            if len(escolhidos) >= max_table_size:
                break
        return creator.SubPopulation(escolhidos[:max_table_size])

    # 1) filtro de regiao
    idx = [i for i in range(n) if ativo_rel[i] <= tol_rel]
    if len(idx) < max_table_size:
        idx = sorted(range(n), key=lambda i: ativo_rel[i])[:max(max_table_size, len(idx))]

    # 2) filtro de Pareto no espaco completo
    sub = [offspring[i] for i in idx]
    if len(sub) > max_table_size:
        fronts = sort_nondominated_restrito(sub, len(sub))
        acc = []
        for f in fronts:
            if acc and len(acc) + len(f) > max(max_table_size, len(sub) // 2):
                break
            acc.extend(f)
            if len(acc) >= max_table_size:
                break
        if len(acc) >= max_table_size:
            sub = acc
    if len(sub) <= max_table_size:
        return creator.SubPopulation(sub)

    # 3) diversidade maxima nos INATIVOS (direcoes normalizadas)
    P = np.array([list(ind.fitness.values) for ind in sub], dtype=float)
    U = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    X = U[:, inativos]
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    vivos = np.ones(len(sub), dtype=bool)
    while vivos.sum() > max_table_size:
        act = np.flatnonzero(vivos)
        sub_d = D[np.ix_(vivos, vivos)]
        # TRUNCAMENTO DO SPEA2: remover pelo vizinho mais proximo apenas
        # nao basta -- num conjunto uniforme TODOS empatam (mesma distancia
        # ao vizinho) e o argmin varre sempre pela mesma ponta, colapsando.
        # Testado: 10 pontos equiespacados, escolher 5 -> saia 5,6,7,8,9.
        # O SPEA2 desempata comparando o 2o vizinho, o 3o, etc. (ordem
        # lexicografica das distancias ordenadas) -> saia 0,3,5,8,9.
        ordenadas = np.sort(sub_d, axis=1)
        idx_local = int(min(range(len(act)), key=lambda i: tuple(ordenadas[i])))
        vivos[act[idx_local]] = False
    return creator.SubPopulation([sub[i] for i in np.flatnonzero(vivos)])


def sel_grea(offspring, mask, max_table_size, nobj, div=None, tol_rel=0.05):
    """Densidade por GRID (GrEA) no lugar do truncamento SPEA2.

    Motivacao: o truncamento do SPEA2 e O(n^2) por remocao -- com 600
    candidatos e 300 remocoes custa ~0.44 s por chamada, o que inviabiliza
    muitas geracoes. O GrEA foi desenhado para muitos objetivos e a
    densidade sai de contagem de celulas, custo linear.

    Metricas do GrEA usadas:
      GR  (grid ranking)  = soma das coordenadas de grid -> convergencia
      GCD (grid crowding) = ocupacao da celula do individuo -> densidade
    Mantem o mesmo esqueleto de sel_aresta: filtro de regiao pela mascara,
    filtro de Pareto no espaco completo, e so entao a densidade decide.
    """
    n = len(offspring)
    if n <= max_table_size:
        return creator.SubPopulation(offspring)
    ativos = get_active_objs(mask, nobj)
    if not ativos:
        return creator.SubPopulation(offspring[:max_table_size])
    inativos = [i for i in range(nobj) if i not in set(ativos)]
    if div is None:
        div = max(4, int(round(max_table_size ** (1.0 / max(1, nobj - 1)))) * 2)

    P_all = np.array([list(i.fitness.values) for i in offspring], dtype=float)
    nor = np.maximum(np.linalg.norm(P_all, axis=1), 1e-12)

    # 1) filtro de regiao (mascara) -- pulado quando nao ha inativos
    if inativos:
        ativo_rel = P_all[:, ativos].sum(axis=1) / nor
        idx = np.flatnonzero(ativo_rel <= tol_rel)
        if len(idx) < max_table_size:
            idx = np.argsort(ativo_rel)[:max(max_table_size, len(idx))]
    else:
        idx = np.arange(n)
    sub = [offspring[i] for i in idx]
    if len(sub) <= max_table_size:
        return creator.SubPopulation(sub)

    # 2) filtro de Pareto no espaco completo (elimina g alto)
    fronts = sort_nondominated_restrito(sub, len(sub))
    rank = {}
    for r, f in enumerate(fronts):
        for ind in f:
            rank[id(ind)] = r

    # 3) densidade por grid nas DIRECOES normalizadas
    eixos = inativos if inativos else list(range(nobj))
    P = np.array([list(i.fitness.values) for i in sub], dtype=float)
    U = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    G = _grid_coords(U[:, eixos], div)
    GR = G.sum(axis=1)
    ranks = np.array([rank[id(i)] for i in sub])

    from collections import Counter
    chaves = [tuple(g) for g in G]
    ocup = Counter(chaves)
    vivos = np.ones(len(sub), dtype=bool)
    while vivos.sum() > max_table_size:
        act = np.flatnonzero(vivos)
        # remove primeiro do pior rank de Pareto; entre eles, da celula
        # mais cheia; desempate por maior GR (pior convergencia no grid)
        pr = ranks[act].max()
        cand = act[ranks[act] == pr]
        dens = np.array([ocup[chaves[i]] for i in cand])
        melhores = cand[dens == dens.max()]
        alvo = melhores[int(np.argmax(GR[melhores]))]
        ocup[chaves[alvo]] -= 1
        vivos[alvo] = False
    return creator.SubPopulation([sub[i] for i in np.flatnonzero(vivos)])


# =====================================================================
#  ENTRADAS DE ALTO NIVEL
# =====================================================================

def _tcheb(vals, lam):
    return max(l * abs(v) for l, v in zip(lam, vals))


def _chave_tcheb(ind, lam):
    """Tchebycheff respeitando restricoes: factivel antes de infactivel."""
    return (_cv(ind), _tcheb(ind.fitness.values, lam))


def _fase_vertices(n_obj, masks, tam, ngen, toolbox, cxpb, mutpb, evaluate):
    """Nivel n-1 -> vertices. Cada mascara roda ISOLADA (nao ha colaboracao
    possivel entre vertices), por Tchebycheff com peso residual no objetivo
    livre -- o residual e o que impede esse objetivo de ficar indeterminado."""
    tables, tam_d = {}, {}
    for s, mask in enumerate(masks):
        random.seed(11 + s * 7)
        livre = [k for k in range(n_obj) if k not in get_active_objs(mask, n_obj)][0]
        lam = [1.0] * n_obj
        lam[livre] = 0.01
        pop = []
        while len(pop) < tam:
            pop.extend(toolbox.population())
        pop = pop[:tam]
        for ind in pop:
            ind.fitness.values = evaluate(ind)
        pop = sorted(pop, key=lambda i: _chave_tcheb(i, lam))[:tam]
        for _ in range(ngen):
            off = []
            while len(off) < tam:
                c = random.sample(pop, 4)
                a = min(c[:2], key=lambda i: _chave_tcheb(i, lam))
                b = min(c[2:], key=lambda i: _chave_tcheb(i, lam))
                o1, o2 = toolbox.clone(a), toolbox.clone(b)
                if random.random() < cxpb:
                    toolbox.mate(o1, o2)
                    del o1.fitness.values, o2.fitness.values
                if random.random() < mutpb:
                    toolbox.mutate(o1); del o1.fitness.values
                if random.random() < mutpb:
                    toolbox.mutate(o2); del o2.fitness.values
                off += [o1, o2]
            for ind in off:
                if not ind.fitness.valid:
                    ind.fitness.values = evaluate(ind)
            pop = sorted(pop + off, key=lambda i: _chave_tcheb(i, lam))[:tam]
        sp = creator.SubPopulation(pop)
        sp.score = 0.0
        tables[mask] = sp
        tam_d[mask] = tam
    return tables, tam_d


def _fase_dimensional(tables, tam, masks_ant, masks, n_obj, pop_size, pop_ger,
                      ngen, toolbox, cxpb, evaluate, sel_fn, eta_cx, janela,
                      snapshot_callback=None, arquivo=None):
    """Semeia o nivel novo com a populacao do anterior e deixa as duas
    interagirem. Os PAIS saem apenas das tabelas do nivel ANTERIOR: a
    tabela do nivel atual e so arquivo (recebe, nao fornece), senao ela
    retroalimenta o proprio material e colapsa.
    Sem mutacao -- todo o espalhamento vem do crossover."""
    semente = [x for mm in masks_ant for x in tables[mm]]
    base_tam = max(5, pop_size // len(masks))
    for mask in masks:
        sp = sel_fn(list(semente), mask, base_tam, n_obj)
        sp.score = 0.0
        tables[mask] = sp
        tam[mask] = base_tam

    hist = {}
    for _ in range(ngen):
        validas = [mm for mm in masks_ant if len(tables.get(mm, [])) > 0]
        if not validas:
            validas = [mm for mm in masks if len(tables.get(mm, [])) > 0]
        offspring, origem = [], []
        while len(offspring) < pop_ger:
            pais = []
            for _ in range(2):
                m1, m2 = random.choice(validas), random.choice(validas)
                v = m1 if tables[m1].score >= tables[m2].score else m2
                pais.append((_tournament_pick(tables[v], v, n_obj), v))
            o1, o2 = toolbox.clone(pais[0][0]), toolbox.clone(pais[1][0])
            tools.cxSimulatedBinaryBounded(o1, o2, eta=eta_cx, low=0.0, up=1.0)
            del o1.fitness.values, o2.fitness.values
            offspring += [o1, o2]
            origem += [pais[0][1], pais[1][1]]
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind)

        sobrev = {}
        for mask in masks:
            sc = tables[mask].score
            novo = sel_fn(list(tables[mask]) + offspring, mask, tam[mask], n_obj)
            novo.score = sc
            tables[mask] = novo
            sobrev[mask] = {id(x) for x in novo}
        for off, mo in zip(offspring, origem):
            if any(id(off) in sobrev[mask] for mask in masks):
                tables[mo].score += 1.0
        alocar_por_derivada(tables, masks, tam, n_obj, hist, janela)

        if arquivo is not None:
            # arquivo global elitista: acumula os nao-dominados de tudo que
            # ja passou (com restricoes), truncado em pop_size
            arquivo[0] = list(arquivo[0]) + [toolbox.clone(x) for x in offspring]
            fr = sort_nondominated_restrito(arquivo[0], pop_size)
            nd = fr[0] if fr else []
            arquivo[0] = (sel_aresta(nd, (1 << n_obj) - 1, pop_size, n_obj)
                          if len(nd) > pop_size else creator.SubPopulation(nd))
        if snapshot_callback is not None:
            snap = dict(tables)
            snap[0] = arquivo[0] if arquivo is not None else []
            snapshot_callback(snap)
    return tables, tam


def run_tchebycheff(n_obj, pop_size, total_gens, toolbox, evaluate,
                    cxpb=0.9, mutpb=1.0, eta_cx=2.0, janela=10,
                    frac_vertices=0.5, usar_grid=False, retorna_fases=False,
                    gens_vertices=None, snapshot_callback=None):
    """Entrada principal da versao Tchebycheff.

    So precisa de pop_size e total_gens -- a divisao entre as etapas e
    feita aqui dentro:
      * a fase dos vertices leva `frac_vertices` do orcamento de geracoes
        (default 50%): ela e a que mais precisa, por ser a unica que tem de
        convergir do zero, e roda com pop_size/n_tabelas individuos.
      * o restante e dividido igualmente entre as n_obj-1 fases seguintes.
      * a populacao TOTAL de cada fase respeita pop_size; dentro da fase os
        tamanhos por tabela sao dinamicos (alocacao por derivada do score).

    Retorna o dict de tabelas, com a chave 0 = ARQUIVO GLOBAL (elitista,
    nao-dominado com restricoes, truncado em pop_size) -- mesma convencao
    das versoes anteriores, para o wrapper do moeabench. Com
    retorna_fases=True devolve (lista_de_fases, tables).

    snapshot_callback(tables) e chamado a cada geracao das fases
    dimensionais, recebendo as tabelas ativas mais a chave 0.
    """
    setup_deap_classes(n_obj)
    niveis = masks_by_level(n_obj)
    ordem = list(range(n_obj - 1, 0, -1)) + [n_obj]
    sel_fn = sel_grea if usar_grid else sel_aresta

    n_fases_dim = len(ordem) - 1
    # A fase dos vertices e a unica que converge do zero e precisa de bem
    # mais geracoes que as demais -- medido no DTLZ3 n=5: ~1200 geracoes
    # para norma 1.001, contra ~80-150 que bastam nas fases seguintes.
    # `gens_vertices` permite fixa-la diretamente; senao usa a fracao.
    gens_vert = int(gens_vertices) if gens_vertices else max(1, int(total_gens * frac_vertices))
    gens_dim = max(1, (total_gens - gens_vert) // max(1, n_fases_dim))

    masks_v = niveis[ordem[0]]
    tam_v = max(5, pop_size // len(masks_v))
    tables, tam = _fase_vertices(n_obj, masks_v, tam_v, gens_vert,
                                 toolbox, cxpb, mutpb, evaluate)
    fases = [[x for mm in masks_v for x in tables[mm]]]

    # arquivo global (chave 0), como na versao original -- elitista e
    # truncado em pop_size. E o que o wrapper do moeabench le.
    arquivo = [creator.SubPopulation([toolbox.clone(x) for x in fases[0]])]
    if snapshot_callback is not None:
        snap = dict(tables); snap[0] = arquivo[0]
        snapshot_callback(snap)

    for fi in range(1, len(ordem)):
        random.seed(100 + fi)
        tables, tam = _fase_dimensional(
            tables, tam, niveis[ordem[fi - 1]], niveis[ordem[fi]], n_obj,
            pop_size, pop_size, gens_dim, toolbox, cxpb, evaluate,
            sel_fn, eta_cx, janela, snapshot_callback, arquivo)
        fases.append([x for mm in niveis[ordem[fi]] for x in tables[mm]])

    tables[0] = arquivo[0]
    if retorna_fases:
        return fases, tables
    return tables


def run_hierarquico(pop_ini, n_obj, pop_size, total_gens, toolbox,
                    cxpb=0.9, mutpb=1.0, janela=10, modo="cumulativo"):
    """Entrada da versao das anotacoes do professor.

    modo="cumulativo" -> run_cumulativo_desde_nivel1 (nada congela)
    modo="congelado"  -> run_staged_congelado_torneio
    modo="hibrido"    -> run_staged_congelado_hibrido
    """
    fn = {"cumulativo": run_cumulativo_desde_nivel1,
          "congelado": run_staged_congelado_torneio,
          "hibrido": run_staged_congelado_hibrido}[modo]
    return fn(pop_ini, n_obj, pop_size, total_gens, toolbox, cxpb, mutpb, janela=janela)

def run_todas_mascaras(n_obj, pop_size, total_gens, toolbox, evaluate,
                       cxpb=0.9, mutpb=1.0, eta_cx=2.0, frac_convergencia=0.6,
                       usar_grid=False, snapshot_callback=None, janela=10,
                       gens_vertices=None, frac_vertices=None):
    # janela, gens_vertices, frac_vertices: aceitos so por compatibilidade
    # de interface com run_tchebycheff (o wrapper do moeabench passa os
    # tres). Esta versao nao tem alocacao por derivada (janela) nem fase
    # de vertices separada -- todas as mascaras convergem com o MESMO
    # orcamento, definido por frac_convergencia. Se frac_vertices vier
    # preenchido, usa-o no lugar de frac_convergencia (mesmo papel: fracao
    # do orcamento para a etapa de convergencia por mascara).
    if frac_vertices is not None:
        frac_convergencia = frac_vertices
    """Alternativa a run_tchebycheff: em vez da cascata vertices->arestas->
    interior, converge as 2^n_obj - 1 mascaras TODAS por Tchebycheff, cada
    uma isolada, e so entao recombina tudo com sel_aresta.

    Cada mascara usa peso 1.0 nos objetivos ativos e residual (0.01) nos
    inativos -- a mesma receita da fase de vertices, so que aplicada a
    TODAS as mascaras (nao so as de nivel n-1). Uma mascara com k ativos
    nao cobre a regiao k-dimensional inteira (um Tchebycheff so converge
    pra um ponto), mas medido em DTLZ3/DTLZ5/DTLZ9 (n_obj=3) as maes
    acabam caindo em pontos espalhados o bastante ao longo da frente pra
    que o crossover final preencha os vaos -- sem now precisar da ordem
    dimensional nem do roteamento por interseccao de mascaras da cascata.

    ATENCAO -- pouco testado: uma seed por problema, so n_obj=3, sem
    comparacao de orcamento igual contra baselines. O custo cresce em
    2^n_obj - 1 mascaras, que fica caro rapido (31 pra n_obj=5, 255 pra
    n_obj=8) -- bem mais caro que as n mascaras de vertice do
    run_tchebycheff. Vale validar com mais seeds e em mais objetivos antes
    de usar como resultado.

    Retorna a populacao final (lista de individuos), apos a recombinacao.
    """
    setup_deap_classes(n_obj)
    todas_masks = list(range(1, 2 ** n_obj))
    n_masks = len(todas_masks)
    tam_m = max(5, pop_size // n_masks)
    gens_conv = max(1, int(total_gens * frac_convergencia))
    gens_final = max(1, total_gens - gens_conv)
    sel_fn = sel_grea if usar_grid else sel_aresta

    def _chave(ind, lam):
        return (_cv(ind), _tcheb(ind.fitness.values, lam))

    def _converge(mask, seed):
        random.seed(seed)
        ativos = get_active_objs(mask, n_obj)
        lam = [0.01] * n_obj
        for a in ativos:
            lam[a] = 1.0
        pop = toolbox.population()
        for ind in pop:
            ind.fitness.values = evaluate(ind)
        pop = sorted(pop, key=lambda i: _chave(i, lam))[:tam_m]
        for _ in range(gens_conv):
            off = []
            while len(off) < tam_m:
                c = random.sample(pop, 4)
                a = min(c[:2], key=lambda i: _chave(i, lam))
                b = min(c[2:], key=lambda i: _chave(i, lam))
                o1, o2 = toolbox.clone(a), toolbox.clone(b)
                if random.random() < cxpb:
                    toolbox.mate(o1, o2)
                    del o1.fitness.values, o2.fitness.values
                if random.random() < mutpb:
                    toolbox.mutate(o1); del o1.fitness.values
                if random.random() < mutpb:
                    toolbox.mutate(o2); del o2.fitness.values
                off += [o1, o2]
            for ind in off:
                if not ind.fitness.valid:
                    ind.fitness.values = evaluate(ind)
            pop = sorted(pop + off, key=lambda i: _chave(i, lam))[:tam_m]
        return pop

    pool = []
    for s, mask in enumerate(todas_masks):
        pool.extend(_converge(mask, seed=100 + s))

    # snapshot logo apos a convergencia por mascara, antes de recombinar --
    # mesma convencao do run_tchebycheff: chave 0 = arquivo, demais chaves
    # = as tabelas por mascara (aqui, o resultado congelado de cada uma).
    if snapshot_callback is not None:
        snap = {mask: pool[i * tam_m:(i + 1) * tam_m] for i, mask in enumerate(todas_masks)}
        snap[0] = list(pool)
        snapshot_callback(snap)

    random.seed(7)
    mask_full = (1 << n_obj) - 1
    pop = list(sel_fn(pool, mask_full, pop_size, n_obj))
    for _ in range(gens_final):
        off = []
        while len(off) < pop_size:
            a, b = random.choice(pop), random.choice(pop)
            o1, o2 = toolbox.clone(a), toolbox.clone(b)
            tools.cxSimulatedBinaryBounded(o1, o2, eta=eta_cx, low=0.0, up=1.0)
            del o1.fitness.values, o2.fitness.values
            off += [o1, o2]
        for ind in off:
            if not ind.fitness.valid:
                ind.fitness.values = evaluate(ind)
        pop = list(sel_fn(pop + off, mask_full, pop_size, n_obj))
        if snapshot_callback is not None:
            snap = {mask_full: pop, 0: pop}
            snapshot_callback(snap)

    tables = {mask: pool[i * tam_m:(i + 1) * tam_m] for i, mask in enumerate(todas_masks)}
    tables[0] = pop
    return tables