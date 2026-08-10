import numpy as np

class MochilaMultiobjetivo:
    """
    Instância customizada do Problema da Mochila Multiobjetivo (Knapsack)
    adaptada para a interface de avaliação matricial do MoeaBench.
    """
    def __init__(self, n_itens=100, n_obj=6, seed=42):
        self.N = n_itens
        self.M = n_obj
        self.n_ieq_constr = 0 
        self.name = f"Mochila_{n_itens}itens_{n_obj}obj"
        
        self.xl = np.zeros(self.N)
        self.xu = np.ones(self.N)
        
        np.random.seed(seed)
        self.values = np.random.randint(10, 101, size=(n_itens, n_obj))
        self.weights = np.random.randint(10, 101, size=(n_itens, n_obj))
        self.capacities = np.sum(self.weights, axis=0) * 0.5
        self.max_profits = np.sum(self.values, axis=0)

    def get_M(self): return self.M
    def get_Nvar(self): return self.N
    def get_n_ieq_constr(self): return self.n_ieq_constr

    def evaluation(self, X, n_ieq_constr=0):
        X_eval = np.atleast_2d(X)
        
        # Arredondamento para lidar com o espaço contínuo dos algoritmos
        X_bin = np.round(X_eval)
        
        # Multiplicação Matricial 
        evals = np.dot(X_bin, self.values)  
        wgts = np.dot(X_bin, self.weights)  
        
        # Fitness base (quanto menor, melhor)
        evals_norm = evals / self.max_profits
        F = 1.0 - evals_norm
        
        # 1. Calcula o excesso de peso (se não estourou, fica 0)
        excesso_peso = np.maximum(0, wgts - self.capacities)
        
        # 2. Transforma o excesso em uma taxa normalizada (porcentagem de estouro)
        taxa_excesso = excesso_peso / self.capacities
        
        # 3. Aplica a penalidade diretamente na matriz F.
        # Usamos um multiplicador (ex: 2.0) para garantir que soluções inválidas 
        # fiquem matematicamente piores que as válidas, mas mantendo a inclinação.
        F = F + (taxa_excesso * 2.0)
        # =======================================================
        
        return {'F': F}