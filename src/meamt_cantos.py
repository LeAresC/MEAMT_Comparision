import os
import sys

import numpy as np
import moeabench as mb
from moeabench.progress import get_active_pbar

sys.path.append(os.path.abspath("."))
from src.meamt_core_cantos import executa


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


def _limites(valor, n_var):
    v = np.asarray(valor, dtype=float)
    return np.full(n_var, float(v)) if v.ndim == 0 else v.reshape(n_var)


class MEAMT_CANTOS(mb.moeas.BaseMoea):
    """MEAMT por Cantos: cantos selecionados na população para todas as máscaras
    de objetivos (critérios V0 e V2), protegidos na seleção por ângulo.

    Parâmetros de classe (ajuste por atributo antes de rodar):
      frac_fase1      -- fração do orçamento para o refino das máscaras de nível 1 e m-1
      mu_fase1        -- população de cada máscara na fase 1 (None = N // 2m)
      regra_cantos    -- "mascaras" (proposta) ou "maoeacs" (regra do MaOEA-CS, para comparação)
      niveis_mascaras -- níveis de máscara na fase 2 (None = todos até 10 objetivos)
      ancora_estavel  -- só cantos estáveis por K gerações viram âncoras (0 = todos)
    Orçamento total: population * (generations + 1) avaliações.
    """

    frac_fase1 = 0.3
    mu_fase1 = None
    regra_cantos = "mascaras"
    niveis_mascaras = None
    ancora_estavel = 0

    def __init__(self, problem=None, population=None, generations=None, seed=None):
        super().__init__(problem, population, generations, seed)
        self.name = "MEAMTCANTOS"

    def evaluation(self):
        # ==========================================
        # 1. CONTRATO DO MOEABENCH: Acesso ao Problema
        # ==========================================
        mop = self.get_problem()
        n_obj = mop.M
        n_var = mop.N
        n_constraints = _number_of_constraints(mop)
        xl = _limites(mop.xl, n_var)
        xu = _limites(mop.xu, n_var)

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
        # O núcleo trabalha em [0, 1]^n_var; aqui convertemos para os limites
        # reais do problema antes de avaliar.
        def avaliar(X01):
            X_eval = xl + np.asarray(X01) * (xu - xl)
            n = len(X_eval)

            # A chamada à evaluation_benchmark é OBRIGATÓRIA.
            # Ela processa penalidades, restrições e conta os FES para o framework.
            resultado = self.evaluation_benchmark(X_eval)
            F = _normalize_objectives(resultado["F"], n, n_obj)

            if "G" in resultado:
                G = _normalize_constraints(resultado["G"], n)
                CV = np.maximum(G, 0.0).sum(axis=1)
            elif n_constraints > 0:
                raise ValueError(
                    "problema restrito não retornou G em evaluation_benchmark"
                )
            else:
                CV = np.zeros(n, dtype=float)

            self.fes_gasto += n
            return F, CV

        # ==========================================
        # 3. CALLBACK DE HISTÓRICO (uma chamada por geração)
        # ==========================================
        # rank = 0 é a primeira frente com restrições: não-dominados entre os
        # factíveis (ou os de menor violação, se não houver factível).
        def snapshot_callback(X01, F, CV, rank, n_cantos):
            X = xl + X01 * (xu - xl)
            nd = rank == 0

            self.F_gens.append(np.asarray(F).copy())
            self.X_gens.append(X.copy())
            self.F_nd_gens.append(F[nd].copy())
            self.X_nd_gens.append(X[nd].copy())
            self.F_dom_gens.append(F[~nd].copy())
            self.X_dom_gens.append(X[~nd].copy())

            pbar = get_active_pbar()
            if pbar:
                generation = min(len(self.F_gens) - 1, self.generations)
                pbar.update_to(generation)

        # ==========================================
        # 4. MOTOR EVOLUTIVO
        # ==========================================
        X01, F, CV = executa(
            avaliar,
            n_obj=n_obj,
            n_var=n_var,
            N=self.population,
            n_ger=self.generations,
            seed=self.seed,
            frac_fase1=self.frac_fase1,
            mu_fase1=self.mu_fase1,
            regra_cantos=self.regra_cantos,
            niveis_mascaras=self.niveis_mascaras,
            ancora_estavel=self.ancora_estavel,
            callback=snapshot_callback,
        )

        # ==========================================
        # 5. FRONTEIRA FINAL: não-dominados da população final
        # ==========================================
        # Mesma regra de restrição do núcleo: se houver factíveis, só eles.
        fac = CV <= 0
        if fac.any():
            Ff = F[fac]
            le = (Ff[:, None, :] <= Ff[None, :, :]).all(2)
            lt = (Ff[:, None, :] < Ff[None, :, :]).any(2)
            F_final = Ff[~(le & lt).any(0)]
        else:
            F_final = F[CV <= CV.min()]

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