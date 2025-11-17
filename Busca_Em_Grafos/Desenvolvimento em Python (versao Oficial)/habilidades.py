# habilidades.py
from grafos import Grafo_MatrizAdjacencia

class Stats:
    def __init__(self, tipo: str, nivel: int, dificuldade: int):
        self.tipo = tipo
        self.nivel = nivel
        self.dificuldade = dificuldade

def construir_skill_tree():
    """
    Skill tree multidimensional com Terra, Fogo, Água e Ar.
    Retorna (g, stats, idx)
    """
    nomes = [
        # ===== Terra =====
        "Pedregulho",        # 0
        "Nuvem de Poeira",   # 1
        "Poça de Lama",      # 2
        "Muralha de Pedra",  # 3
        "Tornado de Terra",  # 4
        "Inundação de Lama", # 5
        "Erupção Parcial",   # 6
        "Terremoto",         # 7
        "Avalanche de Rocha",# 8
        "Meteoro de Terra",  # 9

        # ===== Fogo =====
        "Brasa",             # 10
        "Chama Curta",       # 11
        "Labareda",          # 12
        "Explosão Ígnea",    # 13
        "Inferno",           # 14

        # ===== Água =====
        "Gota",              # 15
        "Jato de Água",      # 16
        "Poça Congelada",    # 17
        "Tsunami",           # 18
        "Maremoto",          # 19

        # ===== Ar =====
        "Brisa",             # 20
        "Rajada",            # 21
        "Redemoinho",        # 22
        "Tornado",           # 23
        "Furacão",           # 24
    ]

    g = Grafo_MatrizAdjacencia(vertices=len(nomes))
    for i, nome in enumerate(nomes):
        g.definir_nome(i, nome)

    stats = {
        # Terra
        0: Stats("Terra", 1, 1), 1: Stats("Terra", 1, 1), 2: Stats("Terra", 1, 1),
        3: Stats("Terra", 3, 3), 4: Stats("Terra", 3, 3), 5: Stats("Terra", 3, 4),
        6: Stats("Terra", 4, 5), 7: Stats("Terra", 6, 8), 8: Stats("Terra", 6, 7),
        9: Stats("Terra", 9, 12),

        # Fogo
        10: Stats("Fogo", 1, 1), 11: Stats("Fogo", 2, 2), 12: Stats("Fogo", 3, 3),
        13: Stats("Fogo", 5, 6), 14: Stats("Fogo", 8, 10),

        # Água
        15: Stats("Água", 1, 1), 16: Stats("Água", 2, 2), 17: Stats("Água", 3, 3),
        18: Stats("Água", 5, 6), 19: Stats("Água", 8, 10),

        # Ar
        20: Stats("Ar", 1, 1), 21: Stats("Ar", 2, 2), 22: Stats("Ar", 3, 3),
        23: Stats("Ar", 5, 6), 24: Stats("Ar", 8, 10),
    }

    def custo(u, v):
        su, sv = stats[u], stats[v]
        base = sv.dificuldade
        if su.tipo != sv.tipo:
            base += 2  # penalidade por mudar de elemento
        salto_nivel = max(0, sv.nivel - su.nivel)
        base += 0.3 * salto_nivel
        return round(base, 2)

    def link(u, v):
        g.adicionar_aresta(u, v, custo(u, v))

    # ===== Terra =====
    link(0, 3); link(1, 4); link(2, 5)
    link(0, 6); link(2, 6)
    link(3, 7); link(4, 7); link(6, 8)
    link(7, 9); link(8, 9)

    # ===== Fogo =====
    link(10, 11); link(11, 12)
    link(12, 13); link(13, 14)

    # ===== Água =====
    link(15, 16); link(16, 17)
    link(17, 18); link(18, 19)

    # ===== Ar =====
    link(20, 21); link(21, 22)
    link(22, 23); link(23, 24)

    # ===== Conexões alternativas / sinergias entre elementos =====
    link(3, 13)   # Terra intermediário pode desbloquear Fogo avançado
    link(12, 7)   # Fogo pode auxiliar Terra avançado
    link(16, 4)   # Água intermediário ajuda Terra
    link(21, 18)  # Ar intermediário ajuda Água
    link(23, 9)   # Ar avançado pode desbloquear Meteoro de Terra

    idx = {nome: i for i, nome in enumerate(nomes)}
    return g, stats, idx

# -----------------------------
# Utilitários de demonstração
# -----------------------------
def custo_do_caminho(g: Grafo_MatrizAdjacencia, caminho):
    if not caminho or len(caminho) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(caminho, caminho[1:]):
        total += g.matriz[a][b]
    return round(total, 2)

def imprimir_resultado(g, titulo, caminho):
    print(f"\n=== {titulo} ===")
    if caminho:
        nomes = g.caminho_com_nomes(caminho)
        custo = custo_do_caminho(g, caminho)
        print("Caminho:", " -> ".join(nomes))
        print("Custo total:", custo)
    else:
        print("Nenhum caminho encontrado.")

def listar_desbloqueaveis(g, adquiridas):
    """Mostra magias conectadas a alguma adquirida, ainda não aprendidas."""
    s = set(adquiridas)
    candidatos = set()
    for u in adquiridas:
        for v in range(g.vertices):
            if g.matriz[u][v] != 0 and v not in s:
                candidatos.add(v)
    print("\n=== Magias desbloqueáveis a partir do seu kit atual ===")
    for v in sorted(candidatos):
        print(f"- {g.nomes[v]} (custo vizinho: {g.matriz[adquiridas[0]][v] if g.matriz[adquiridas[0]][v]!=0 else 'varia'})")

# -----------------------------
# Exemplo de uso
# -----------------------------
