"""MEAMT por Cantos -- versao com cantos selecionados na populacao.

Nucleo em NumPy puro, independente do moeabench. O wrapper apenas fornece a
funcao de avaliacao e recebe o historico.

Algoritmo
---------
Fase 1 (fracao `frac_fase1` do orcamento):
    as 2m mascaras dos niveis 1 e m-1 sao refinadas isoladamente pelo criterio
    V2 (populacao de ~N/2m por mascara). A uniao delas, completada com
    individuos aleatorios, forma a populacao inicial da fase 2.

Fase 2, a cada geracao:
    1. N filhos por SBX + mutacao polinomial, pais sorteados na populacao;
    2. R = pais + filhos + arquivo de cantos (duplicatas removidas);
    3. cantos: para cada uma das 2^m - 1 mascaras, duas vagas -- o melhor de R
       por V0 e o melhor por V2 (um canto so e trocado por alguem melhor no
       mesmo criterio, porque o anterior esta em R);
    4. nadir estimado pelos cantos de nivel m-1; um canto de nivel 1 so entra se
       estender esse nadir;
    5. se houver mais de N/2 cantos distintos, a propria selecao por angulo
       escolhe entre eles o subconjunto mais espalhado, partindo dos de nivel m-1;
    6. selecao ambiental: dominancia com restricoes, solucoes dentro do nadir e
       selecao por angulo (ABS) com os cantos entrando primeiro.

Criterios das mascaras (S = objetivos ativos), com o ideal na origem:
    V0(x) = max_{i in S} f_i + 0.01 * sum_{todos} f_i
    V2(x) = max_{i in S} f_i + 0.01 * sum_{fora de S} f_i + 0.003 * sum_{i in S} f_i
Os dois sao estritamente crescentes em cada objetivo, entao o minimo deles sobre
qualquer conjunto nunca e dominado por outro membro do conjunto.

Restricoes: factivel sempre vence infactivel; entre infactiveis, vence a menor
violacao. Objetivos negativos: os criterios sao calculados sobre F - z_baixo,
com z_baixo = min(0, menor valor ja visto de cada objetivo), o que e o mesmo
que z = 0 quando os objetivos sao nao-negativos (DTLZ, MaF, WFG).
"""
import itertools
import numpy as np

PESO_ATIVOS_V2 = 3e-5
PESO_SOMA = 0.0001


# --------------------------------------------------------------------- utilitarios
def mascaras(M, niveis=None):
    ks = range(1, M + 1) if niveis is None else sorted(set(niveis))
    return [S for k in ks for S in itertools.combinations(range(M), k)]


def matriz_ativos(ms, M):
    A = np.zeros((len(ms), M), dtype=bool)
    for i, S in enumerate(ms):
        A[i, list(S)] = True
    return A


def criterio(F, ACT, var):
    """F: (n, M) nao-negativo; ACT: (K, M). Devolve (K, n)."""
    b = np.where(ACT[:, None, :], F[None], -np.inf).max(2)
    if var == "V0":
        return b + PESO_SOMA * F.sum(1)[None]
    inat = np.where(ACT[:, None, :], 0.0, F[None]).sum(2)
    at = np.where(ACT[:, None, :], F[None], 0.0).sum(2)
    return b + PESO_SOMA * inat + PESO_ATIVOS_V2 * at


def frentes(F):
    """Postos de nao-dominancia (0 = primeira frente), vetorizado."""
    n = len(F)
    if n == 0:
        return np.zeros(0, dtype=int)
    le = (F[:, None, :] <= F[None, :, :]).all(2)
    lt = (F[:, None, :] < F[None, :, :]).any(2)
    dom = le & lt
    cnt = dom.sum(0)
    rank = np.full(n, -1)
    r = 0
    atual = np.flatnonzero(cnt == 0)
    while len(atual):
        rank[atual] = r
        cnt = cnt - dom[atual].sum(0)
        cnt[rank >= 0] = -1
        atual = np.flatnonzero(cnt == 0)
        r += 1
    return rank


def frentes_com_restricao(F, CV):
    """Factiveis ordenados em frentes; infactiveis depois, por violacao crescente."""
    rank = np.empty(len(F), dtype=int)
    fac = CV <= 0
    if fac.any():
        rank[fac] = frentes(F[fac])
        base = rank[fac].max() + 1
    else:
        base = 0
    if (~fac).any():
        _, nivel = np.unique(CV[~fac], return_inverse=True)
        rank[~fac] = base + nivel
    return rank


def sbx(A, B, rng, eta=20.0, pc=0.9):
    """SBX com limites (Deb e Agrawal), no dominio [0, 1] -- a mesma forma de
    tools.cxSimulatedBinaryBounded do DEAP e do PlatEMO. A versao sem limites
    seguida de corte NAO chega as pontas do DTLZ9 (medido: 17-22 graus contra
    2-3 graus com esta versao, alvo 0)."""
    x1, x2 = np.minimum(A, B), np.maximum(A, B)
    dx = np.maximum(x2 - x1, 1e-14)
    u = rng.random(A.shape)

    def beta_q(beta):
        alpha = 2.0 - beta ** -(eta + 1)
        return np.where(u <= 1 / alpha, (u * alpha) ** (1 / (eta + 1)), (1 / (2 - u * alpha)) ** (1 / (eta + 1)))

    c1 = np.clip(0.5 * (x1 + x2 - beta_q(1 + 2 * x1 / dx) * dx), 0, 1)
    c2 = np.clip(0.5 * (x1 + x2 + beta_q(1 + 2 * (1 - x2) / dx) * dx), 0, 1)
    troca = rng.random(A.shape) < 0.5
    c1, c2 = np.where(troca, c2, c1), np.where(troca, c1, c2)
    faz = (rng.random(A.shape) < 0.5) & (np.abs(A - B) > 1e-14) & (rng.random(A.shape[:-1] + (1,)) < pc)
    return np.where(faz, c1, A), np.where(faz, c2, B)


def mutacao_polinomial(X, rng, eta=20.0):
    """Mutacao polinomial com limites (Deb), probabilidade 1/D por variavel --
    a mesma forma de tools.mutPolynomialBounded."""
    m = rng.random(X.shape) < 1.0 / X.shape[-1]
    u = rng.random(X.shape)
    dq1 = (2 * u + (1 - 2 * u) * (1 - X) ** (eta + 1)) ** (1 / (eta + 1)) - 1
    dq2 = 1 - (2 * (1 - u) + 2 * (u - 0.5) * X ** (eta + 1)) ** (1 / (eta + 1))
    return np.clip(np.where(m, X + np.where(u < 0.5, dq1, dq2), X), 0, 1)


def selecao_por_angulo(Fn, ancoras, n):
    """ABS: insere, um por vez, o candidato de maior angulo ao selecionado mais
    proximo (farthest-first). As ancoras entram primeiro. Fn: objetivos
    normalizados. Trabalha com cossenos: maximizar o angulo minimo = minimizar o
    maior cosseno."""
    U = Fn / np.maximum(np.linalg.norm(Fn, axis=1, keepdims=True), 1e-12)
    sel = list(dict.fromkeys(int(a) for a in ancoras))[:n]
    if len(sel) >= n:
        return np.array(sel[:n])
    if not sel:
        sel = [int(np.argmin(np.linalg.norm(Fn, axis=1)))]
    cmax = (U @ U[sel].T).max(1)
    cmax[sel] = 2.0
    while len(sel) < n:
        j = int(np.argmin(cmax))
        sel.append(j)
        cmax = np.maximum(cmax, U @ U[j])
        cmax[j] = 2.0
    return np.array(sel)


# --------------------------------------------------------------------- algoritmo
def executa(avaliar, n_obj, n_var, N, n_ger, seed=None, frac_fase1=0.3, mu_fase1=None,
            regra_cantos="mascaras", niveis_mascaras=None, normalizar=False, callback=None,
            ancora_estavel=0, callback_cantos=None):
    """
    avaliar(X01) -> (F, CV): X01 em [0,1]^n_var (o wrapper converte para os
        limites do problema); F (n, M); CV (n,) >= 0 (0 = factivel).
    N: tamanho da populacao. n_ger: numero de geracoes -- o orcamento total e
        N * (n_ger + 1) avaliacoes, contando a populacao inicial.
    frac_fase1: fracao do orcamento para a fase 1.
    mu_fase1: populacao de cada mascara na fase 1 (padrao: N // 2m). No DTLZ9, as
        pontas so foram atingidas com ~24 mil avaliacoes por mascara (por
        exemplo 60 individuos x 400 geracoes); com ~5 mil, nenhuma divisao chegou.
    normalizar: se True, os criterios das mascaras usam cada objetivo dividido
        pela faixa do conjunto nao-dominado atual (origem mantida em zero). Na
        fase 1 a faixa e estimada com as populacoes de todas as mascaras juntas.
        Necessario em problemas com objetivos em escalas diferentes (MaF4, MaF5).
    ancora_estavel: K > 0 faz so os cantos que continuam os mesmos por K geracoes
        seguidas (em alguma vaga) entrarem como ancoras da selecao por angulo;
        os demais continuam no arquivo e competem normalmente. 0 = todos ancoras.
    callback_cantos(F_cantos, idade): opcional, chamado a cada geracao da fase 2.
    regra_cantos: "mascaras" (proposta) ou "maoeacs" (distancia ao eixo e
        minimo de cada objetivo, para comparacao).
    niveis_mascaras: niveis usados na fase 2 (padrao: todos ate 10 objetivos;
        acima disso, 1, 2, m-2 e m-1, porque 2^m - 1 mascaras fica caro).
    callback(X01, F, CV, rank, n_cantos): chamado uma vez por geracao.
    Devolve (X01, F, CV) da populacao final.
    """
    M, D = n_obj, n_var
    rng = np.random.default_rng(seed)
    maxfe = N * (n_ger + 1)
    fes = 0
    z_baixo = np.zeros(M)

    def aval(X):
        nonlocal fes, z_baixo
        F, CV = avaliar(X)
        F = np.asarray(F, dtype=float)
        CV = np.asarray(CV, dtype=float).reshape(-1)
        fes += len(X)
        z_baixo = np.minimum(z_baixo, F.min(0))
        return F, CV

    escala = np.ones(M)

    def atualiza_escala(F, CV):
        nonlocal escala
        if not normalizar:
            return
        r = frentes_com_restricao(F, CV)
        Fn = F[r == 0] - z_baixo
        escala = np.maximum(Fn.max(0), 1e-12)

    def chave(F, CV, ACT, var):
        c = criterio((F - z_baixo) / escala, ACT, var)
        fac = CV <= 0
        if fac.any():                                  # factivel sempre vence
            c = np.where(fac[None, :], c, np.inf)
        else:                                          # ninguem factivel: menor violacao
            c = np.broadcast_to(CV[None, :], c.shape).copy()
        return c

    # ---------------- fase 1: refino isolado dos niveis 1 e m-1 por V2
    m1 = mascaras(M, {1, M - 1})
    A1 = matriz_ativos(m1, M)
    K = len(m1)
    mu = max(4, N // K) if mu_fase1 is None else int(mu_fase1)
    X = rng.random((K, mu, D))
    F, CV = aval(X.reshape(-1, D))
    F = F.reshape(K, mu, M); CV = CV.reshape(K, mu)
    ar = np.arange(K)[:, None]
    ger1 = max(0, int(frac_fase1 * maxfe) // (K * mu) - 1)

    def chaves_fase1(F, CV):
        atualiza_escala(F.reshape(-1, M), CV.reshape(-1))
        return np.stack([chave(F[i], CV[i], A1[i:i + 1], "V2")[0] for i in range(K)])

    for _ in range(ger1):
        if fes + K * mu > maxfe:
            break
        k = chaves_fase1(F, CV)

        def torneio():
            i, j = rng.integers(0, mu, (2, K, mu))
            return np.where(k[ar, i] <= k[ar, j], i, j)

        C, _ = sbx(X[ar, torneio()], X[ar, torneio()], rng)
        C = mutacao_polinomial(C, rng)
        FC, CVC = aval(C.reshape(-1, D))
        X2 = np.concatenate([X, C], 1)
        F2 = np.concatenate([F, FC.reshape(K, mu, M)], 1)
        CV2 = np.concatenate([CV, CVC.reshape(K, mu)], 1)
        o = np.argsort(chaves_fase1(F2, CV2), 1)[:, :mu]
        X, F, CV = X2[ar, o], F2[ar, o], CV2[ar, o]
        if callback is not None:
            PX, PF, PC = X.reshape(-1, D), F.reshape(-1, M), CV.reshape(-1)
            callback(PX, PF, PC, frentes_com_restricao(PF, PC), 0)

    PX, PF, PCV = X.reshape(-1, D), F.reshape(-1, M), CV.reshape(-1)
    if len(PX) < N and fes + (N - len(PX)) <= maxfe:
        XR = rng.random((N - len(PX), D))
        FR, CR = aval(XR)
        PX, PF, PCV = np.vstack([PX, XR]), np.vstack([PF, FR]), np.concatenate([PCV, CR])
    elif len(PX) > N:
        o = rng.choice(len(PX), N, replace=False)
        PX, PF, PCV = PX[o], PF[o], PCV[o]

    # ---------------- fase 2
    if niveis_mascaras is None:
        niveis_mascaras = range(1, M + 1) if M <= 10 else {1, 2, M - 2, M - 1}
    todas = mascaras(M, niveis_mascaras)
    AT = matriz_ativos(todas, M)
    nivel = AT.sum(1)
    arqX = np.zeros((0, D)); arqF = np.zeros((0, M)); arqC = np.zeros(0)
    n_cantos = 0
    idade_vaga = {}           # vaga -> (solucao que a ocupa, geracoes seguidas)

    while fes + N <= maxfe:
        # ----- reproducao
        pa, pb = rng.integers(0, len(PX), (2, (N + 1) // 2))
        c1, c2 = sbx(PX[pa], PX[pb], rng)
        QX = mutacao_polinomial(np.vstack([c1, c2]), rng)[:N]
        QF, QC = aval(QX)
        RX = np.vstack([PX, QX, arqX]); RF = np.vstack([PF, QF, arqF]); RC = np.concatenate([PCV, QC, arqC])
        RX, ui = np.unique(RX, axis=0, return_index=True)
        RF, RC = RF[ui], RC[ui]
        rank = frentes_com_restricao(RF, RC)
        nd = np.flatnonzero(rank == 0)
        F1 = RF[nd]

        # ----- cantos
        atualiza_escala(RF, RC)
        if regra_cantos == "mascaras":
            cand, niv = [], []
            for var in ("V0", "V2"):
                cand += list(np.argmin(chave(RF, RC, AT, var), 1))
                niv += list(nivel)
            cand = np.array(cand); niv = np.array(niv)
            nova = {}
            for v_, c_ in enumerate(cand):
                chave_x = RX[c_].tobytes()
                ant = idade_vaga.get(v_)
                nova[v_] = (chave_x, ant[1] + 1 if ant is not None and ant[0] == chave_x else 0)
            idade_vaga = nova
            idade_sol = {}
            for v_, c_ in enumerate(cand):
                idade_sol[int(c_)] = max(idade_sol.get(int(c_), 0), idade_vaga[v_][1])
            topo = cand[niv == M - 1] if (niv == M - 1).any() else cand
            nad = RF[topo].max(0)
            for f in RF[cand[niv == 1]]:               # nivel 1 so entra se estender o nadir
                if (f > nad + 1e-12).any():
                    nad = np.maximum(nad, f)
            cantos = np.array(list(dict.fromkeys([int(c) for c in topo] + [int(c) for c in cand])))
            n_anc = len(set(int(c) for c in topo))
        else:                                           # regra do MaOEA-CS
            dist = np.sqrt(np.maximum((F1 ** 2).sum(1, keepdims=True) - F1 ** 2, 0))
            P1 = nd[np.argmin(dist, 0)]; P2 = nd[np.argmin(F1, 0)]
            nad = RF[P1].max(0)
            extra = [int(p) for p in P2 if (RF[p] > nad + 1e-12).any()]
            for p in extra:
                nad = np.maximum(nad, RF[p])
            cantos = np.array(list(dict.fromkeys([int(p) for p in P1] + extra)))
            n_anc = M

        zst = F1.min(0)
        esc = np.maximum(nad - zst, 1e-12)
        if len(cantos) > N // 2:                        # teto de N/2 cantos
            sub = selecao_por_angulo((RF[cantos] - zst) / esc, list(range(min(n_anc, len(cantos)))), N // 2)
            cantos = cantos[sub]
        n_cantos = len(cantos)
        arqX, arqF, arqC = RX[cantos].copy(), RF[cantos].copy(), RC[cantos].copy()
        if regra_cantos == "mascaras":
            idades = np.array([idade_sol.get(int(c_), 0) for c_ in cantos])
        else:
            idades = np.full(len(cantos), 10 ** 9)
        if callback_cantos is not None:
            callback_cantos(RF[cantos], idades)
        ancoras = cantos[idades >= ancora_estavel] if ancora_estavel > 0 else cantos

        # ----- selecao ambiental
        if len(nd) <= N:
            escolha = list(nd)
            r = 1
            while len(escolha) < N:
                fr = np.flatnonzero(rank == r); r += 1
                if len(fr) == 0:
                    break
                if len(escolha) + len(fr) <= N:
                    escolha += list(fr)
                else:
                    pool = np.array(escolha + list(fr))
                    s = selecao_por_angulo((RF[pool] - zst) / esc, list(range(len(escolha))), N)
                    escolha = list(pool[s])
            idx = np.array(escolha)
        else:
            dentro = nd[(RF[nd] <= nad + 1e-9).all(1)]
            fora = np.setdiff1d(nd, dentro)
            base = np.array(list(dict.fromkeys(list(ancoras) + list(cantos) + list(dentro))))
            if len(base) >= N:
                s = selecao_por_angulo((RF[base] - zst) / esc, list(range(len(ancoras))), N)
                idx = base[s]
            else:
                dfo = np.linalg.norm((RF[fora] - zst) / esc, axis=1)
                idx = np.concatenate([base, fora[np.argsort(dfo)[:N - len(base)]]])
        PX, PF, PCV = RX[idx], RF[idx], RC[idx]
        if callback is not None:
            callback(PX, PF, PCV, frentes_com_restricao(PF, PCV), n_cantos)

    return PX, PF, PCV