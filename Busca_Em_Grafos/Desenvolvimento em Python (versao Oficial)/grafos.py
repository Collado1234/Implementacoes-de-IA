import heapq
from collections import deque

class Grafo_MatrizAdjacencia:
    def __init__(self, vertices):
        self.vertices = vertices
        self.matriz = [[0 for _ in range(vertices)] for _ in range(vertices)]
        self.nomes = [f"V{i}" for i in range(vertices)]  # nomes padrão

    def definir_nome(self, vertice, nome):
        if 0 <= vertice < self.vertices:
            self.nomes[vertice] = nome

    def adicionar_aresta(self, origem, destino, peso=1):
        if 0 <= origem < self.vertices and 0 <= destino < self.vertices:
            self.matriz[origem][destino] = peso  # apenas de origem → destino

    def remover_aresta(self, origem, destino):
        if 0 <= origem < self.vertices and 0 <= destino < self.vertices:
            self.matriz[origem][destino] = 0  # apenas de origem → destino

    def exibir_matriz(self):
        print("   " + " ".join(f"{nome:3}" for nome in self.nomes))
        for i, linha in enumerate(self.matriz):
            print(f"{self.nomes[i]:3} " + " ".join(f"{peso:3}" for peso in linha))

    def caminho_com_nomes(self, caminho):
        return [self.nomes[v] for v in caminho]

    # ===================== DFS =====================
    def depth_first_search(self, inicio, objetivo):
        visitado = [False] * self.vertices
        pai = [-1] * self.vertices
        pilha = [inicio]
        visitado[inicio] = True
        pai[inicio] = inicio

        while pilha:
            vertice = pilha.pop()
            if vertice == objetivo:
                caminho = []
                v = objetivo
                while v != inicio:
                    caminho.append(v)
                    v = pai[v]
                caminho.append(inicio)
                caminho.reverse()
                return caminho, pai

            for vizinho in range(self.vertices):
                if self.matriz[vertice][vizinho] != 0 and not visitado[vizinho]:
                    pilha.append(vizinho)
                    visitado[vizinho] = True
                    pai[vizinho] = vertice

        return None, pai

    # ===================== BFS =====================
    def breadth_first_search(self, inicio, objetivo):
        visitado = [False] * self.vertices
        fila = deque([inicio])
        pai = [-1] * self.vertices
        visitado[inicio] = True
        pai[inicio] = inicio

        while fila:
            vertice = fila.popleft()
            if vertice == objetivo:
                caminho = []
                v = objetivo
                while v != inicio:
                    caminho.append(v)
                    v = pai[v]
                caminho.append(inicio)
                caminho.reverse()
                return caminho, pai

            for vizinho in range(self.vertices):
                if self.matriz[vertice][vizinho] != 0 and not visitado[vizinho]:
                    fila.append(vizinho)
                    visitado[vizinho] = True
                    pai[vizinho] = vertice

        return None, pai

    # ===================== UCS =====================
    def uniform_cost_search(self, inicio, objetivo):
        custo = [float('inf')] * self.vertices
        pai = [-1] * self.vertices
        visitado = [False] * self.vertices

        custo[inicio] = 0
        fila = [(0, inicio)]
        heapq.heapify(fila)

        while fila:
            custo_atual, vertice = heapq.heappop(fila)
            if visitado[vertice]:
                continue

            visitado[vertice] = True
            if vertice == objetivo:
                caminho = []
                v = objetivo
                while v != inicio:
                    caminho.append(v)
                    v = pai[v]
                caminho.append(inicio)
                caminho.reverse()
                return caminho, pai

            for vizinho in range(self.vertices):
                if self.matriz[vertice][vizinho] != 0:
                    novo_custo = custo_atual + self.matriz[vertice][vizinho]
                    if novo_custo < custo[vizinho]:
                        custo[vizinho] = novo_custo
                        pai[vizinho] = vertice
                        heapq.heappush(fila, (novo_custo, vizinho))

        return None, pai

    # ===================== DLS =====================
    def depth_limited_search(self, inicio, objetivo, limite):
        pai = [-1] * self.vertices
        visitado = [False] * self.vertices

        def dls(vertice, profundidade):
            if profundidade == 0 and vertice == objetivo:
                return [vertice]
            if profundidade > 0:
                visitado[vertice] = True
                for vizinho in range(self.vertices):
                    if self.matriz[vertice][vizinho] != 0 and not visitado[vizinho]:
                        pai[vizinho] = vertice
                        caminho = dls(vizinho, profundidade - 1)
                        if caminho:
                            return [vertice] + caminho
            return None

        for profundidade in range(limite + 1):
            visitado = [False] * self.vertices
            resultado = dls(inicio, profundidade)
            if resultado:
                return resultado, pai

        return None, pai

    # ===================== IDS =====================
    def iterative_deepening_search(self, inicio, objetivo):
        limite = 0
        while True:
            resultado, pai = self.depth_limited_search(inicio, objetivo, limite)
            if resultado is not None:
                return resultado, pai
            limite += 1
