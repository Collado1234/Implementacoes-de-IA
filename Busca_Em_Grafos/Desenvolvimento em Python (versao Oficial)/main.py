# main.py
from habilidades import construir_skill_tree, custo_do_caminho, imprimir_resultado
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

def plot_matriz_elemento(g, stats, elemento):
    # Pega os índices das magias do elemento
    indices = [i for i, s in stats.items() if s.tipo == elemento]
    n = len(indices)
    matriz = np.zeros((n, n))

    # Preenche a matriz com os custos
    for i_idx, u in enumerate(indices):
        for j_idx, v in enumerate(indices):
            if u == v:
                matriz[i_idx, j_idx] = -1  # marcar a diagonal
            else:
                matriz[i_idx, j_idx] = g.matriz[u][v]

    # Cria mapa de cores: diagonal cinza, resto viridis
    cmap = plt.cm.viridis
    cmap_diagonal = colors.ListedColormap(['gray'])
    bounds = [-1.5, -0.5, np.max(matriz)]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(6,6))
    cax = ax.matshow(matriz, cmap=cmap, vmin=0, vmax=np.max(matriz))
    
    # Marca a diagonal com cinza
    for i in range(n):
        ax.matshow(np.array([[matriz[i,i]]]), cmap=cmap_diagonal, vmin=-1, vmax=-1, extent=(i,i+1,i,i+1))

    fig.colorbar(cax)

    # Labels
    nomes = [g.nomes[i] for i in indices]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(nomes, rotation=90)
    ax.set_yticklabels(nomes)
    ax.set_title(f"Matriz de adjacência: {elemento}")
    plt.tight_layout()
    plt.show()


def testar_busca(g, stats, inicio_nome, objetivo_nome):
    idx = {nome: i for i, nome in enumerate(g.nomes)}
    inicio = idx[inicio_nome]
    objetivo = idx[objetivo_nome]

    print("\n" + "="*40)
    print(f"Caminho de {inicio_nome} até {objetivo_nome}")
    print("="*40)

    # ===== DFS =====
    print("\n[DFS - Depth First Search]")
    caminho, _ = g.depth_first_search(inicio, objetivo)
    imprimir_resultado(g, "DFS", caminho)

    # ===== BFS =====
    print("\n[BFS - Breadth First Search]")
    caminho, _ = g.breadth_first_search(inicio, objetivo)
    imprimir_resultado(g, "BFS", caminho)

    # ===== UCS =====
    print("\n[UCS - Uniform Cost Search]")
    caminho, _ = g.uniform_cost_search(inicio, objetivo)
    imprimir_resultado(g, "UCS", caminho)

    # ===== DLS =====
    limite = 4
    print(f"\n[DLS - Depth Limited Search (limite={limite})]")
    caminho, _ = g.depth_limited_search(inicio, objetivo, limite)
    imprimir_resultado(g, f"DLS (limite={limite})", caminho)

    # ===== IDS =====
    print("\n[IDS - Iterative Deepening Search]")
    caminho, _ = g.iterative_deepening_search(inicio, objetivo)
    imprimir_resultado(g, "IDS", caminho)


if __name__ == "__main__":
    g, stats, idx = construir_skill_tree()

    # ===== Plot das matrizes por elemento =====
    elementos = ["Terra", "Fogo", "Água", "Ar"]
    for elemento in elementos:
        plot_matriz_elemento(g, stats, elemento)

    # ===== Teste de buscas =====
    testar_busca(g, stats, "Pedregulho", "Meteoro de Terra")
    testar_busca(g, stats, "Brasa", "Inferno")
