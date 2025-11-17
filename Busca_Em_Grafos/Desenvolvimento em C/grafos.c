#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define MAX_VERTICES 100
#define MAX_NAME_LENGTH 30

typedef enum Boolean {
    true = 1,
    false = 0
} boolean;

typedef struct Queue {
    int *data;      
    int front;      
    int rear;      
    int size;      
    int capacity;  
} Queue;

typedef struct Stack_node{
    int neighbor;
    struct Stack_node* prox;
}Stack_node;

typedef struct Stack {
    Stack_node* top;
    int size;      
} Stack;

typedef struct {
    int vertex;
    float cost;
} PQNode;

typedef struct {
    PQNode *data;
    int size;
    int capacity;
} PriorityQueue;


typedef Queue* p_queue;
typedef struct Stack_node* p_stack_node;
typedef Stack* p_stack;

//Queue functions
p_queue create_queue(int capacity);
int is_queue_empty(p_queue q);
int is_queue_full(p_queue q);
void enqueue(p_queue q, int value);
int dequeue(p_queue q);
void free_queue(p_queue q);

//Stack functions
p_stack create_stack();
int is_stack_empty(p_stack s);
void push(p_stack s, int neighbor);
int pop(p_stack s);
void free_stack(p_stack s);

// Priority Queue for Uniform Cost Search (UCS)
PriorityQueue* create_pq(int capacity);
void swap(PQNode *a, PQNode *b);
void pq_push(PriorityQueue *pq, int vertex, float cost);
PQNode pq_pop(PriorityQueue *pq);
int pq_empty(PriorityQueue *pq);
void free_pq(PriorityQueue *pq);

// Estrutura para ponto (vértice)
typedef struct Point {
    int pos_x;
    int pos_y;
    char name[30];
} Point;

// Estrutura para aresta (nó da lista)
typedef struct Node {
    int neighbor;
    float weight;
    struct Node* prox;
} Node;

typedef Node* p_node;

// Estrutura para grafo
typedef struct Graph {
    p_node *list_adjacency;
    int number_of_vertices;
    Point *coordinates;
    boolean *busy;  // <-- vetor para indicar quais índices já têm vértices
} Graph;

typedef Graph* p_graph;

typedef float (*DistanceFunction)(Point a, Point b);

float calculate_euclidean(Point a, Point b) {
    return sqrt(pow(a.pos_x - b.pos_x, 2) + pow(a.pos_y - b.pos_y, 2));
}

float calculate_manhattan(Point a, Point b) {
    return fabs(a.pos_x - b.pos_x) + fabs(a.pos_y - b.pos_y);
}

// Cria grafo
p_graph create_graph(int size) {
    p_graph new_g = malloc(sizeof(Graph));
    new_g->number_of_vertices = size;
    new_g->list_adjacency = malloc(size * sizeof(p_node));
    new_g->coordinates = malloc(size * sizeof(Point));
    new_g->busy = calloc(size, sizeof(boolean));  // Inicializa como false


    for (int i = 0; i < size; i++) {
        new_g->list_adjacency[i] = NULL;
    }


    return new_g;
}

// Define coordenadas para o primeiro índice disponível
void set_coordinates(p_graph g, int x, int y, char *name) {
    for (int i = 0; i < g->number_of_vertices; i++) {
        if (!g->busy[i]) {
            g->coordinates[i].pos_x = x;
            g->coordinates[i].pos_y = y;
            strcpy(g->coordinates[i].name, name);
            g->busy[i] = true;
            return;
        }
    }
    printf("Erro: Não há espaço livre no grafo.\n");
}

void free_list(p_node list) {
    p_node temp;
    while (list != NULL) {
        temp = list;
        list = list->prox;
        free(temp);
    }
}

void free_graph(p_graph g){
    for(int i = 0; i < g->number_of_vertices; i++) {
        free_list(g->list_adjacency[i]);
    }
    free(g->list_adjacency);
    free(g->coordinates);
    free(g->busy);
    free(g);
    g = NULL;
    printf("Grafo destruído com sucesso.\n");
    return;    
}

/*
    function to insert a node in the adjacency list
    This function creates a new node with the given neighbor and weight,
    and inserts it at the beginning of the list.
    @param list Pointer to the head of the adjacency list
    @param neighbor The index of the neighbor vertex
    @param weight The weight of the edge (distance)
    @return Pointer to the new head of the list
*/
p_node insert_list(p_node list, int neighbor, float weight) {
    p_node new_node = malloc(sizeof(Node));
    new_node->neighbor = neighbor;
    new_node->weight = weight;
    new_node->prox = list;
    return new_node;
}

/*
Function to insert an edge in the graph
This function checks if the source and destination vertices are valid,
and if they are, it inserts the edge in both directions (for an undirected graph).
@param g Pointer to the graph
@param src The index of the source vertex
@param dest The index of the destination vertex
@param distance_fn Function to calculate the distance between two points
*/
void insert_edge(p_graph g, int src, int dest, DistanceFunction distance_fn) {
    if (src >= g->number_of_vertices || dest >= g->number_of_vertices) {
        printf("Erro: vértice inválido.\n");
        return;
    }
    float distance = distance_fn(g->coordinates[src], g->coordinates[dest]);
    g->list_adjacency[src] = insert_list(g->list_adjacency[src], dest, distance);
    g->list_adjacency[dest] = insert_list(g->list_adjacency[dest], src, distance);
}

/*
    Function to print the graph
    This function prints the adjacency list of each vertex in the graph.
    @param g Pointer to the graph
*/
void print_graph(p_graph g) {
    for (int i = 0; i < g->number_of_vertices; i++) {
        if (g->busy[i]) {
            printf("%s (%d, %d): ", g->coordinates[i].name, g->coordinates[i].pos_x, g->coordinates[i].pos_y);
            p_node current = g->list_adjacency[i];
            while (current != NULL) {
                printf(" -> %s [%.2f]", g->coordinates[current->neighbor].name, current->weight);
                current = current->prox;
            }
            printf("\n");
        }
    }
}

void recursive_dfs_with_goal(p_graph g, int v, int goal, boolean *visited, int *dfs_steps, boolean *found) {
    if (*found || !g->busy[v] || visited[v]) return;

    visited[v] = true;
    (*dfs_steps)++;
    printf("Visitou: %s\n", g->coordinates[v].name);

    if (v == goal) {
        printf("Objetivo encontrado!\n");
        *found = true;
        return;
    }

    for (p_node u = g->list_adjacency[v]; u != NULL; u = u->prox) {
        if (!visited[u->neighbor]) {
            recursive_dfs_with_goal(g, u->neighbor, goal, visited, dfs_steps, found);
        }
    }
}

int dfs_recursive(p_graph g, int v, int goal, boolean *visited) {
    int dfs_steps = 0;
    boolean found = false;
    recursive_dfs_with_goal(g, v, goal, visited, &dfs_steps, &found);

    return dfs_steps;
}

void dfs(p_graph g, int start, int goal) {
    int visited[MAX_VERTICES] = {0};
    p_stack s = create_stack();


    push(s, start);
    visited[start] = 1;


    while (!is_stack_empty(s)) {
        int v = pop(s);
        printf("Visitou: %s\n", g->coordinates[v].name);


        if (v == goal) {
            printf("Objetivo encontrado: %s\n", g->coordinates[v].name);
            free_stack(s);
            return;
        }


        for (p_node u = g->list_adjacency[v]; u != NULL; u = u->prox){
            if (!visited[u->neighbor]) {
                visited[u->neighbor] = 1;
                push(s, u->neighbor);
            }
        }
    }


    printf("Objetivo não encontrado.\n");
    free_stack(s);
}

/*
    Function to perform a recursive depth-first search
    This function recursively visits all vertices in the graph starting from the given vertex.
    @param g Pointer to the graph
    @param v The index of the vertex to start the search
    @param visited Array to keep track of visited vertices
*/
int breadth_first_search(p_graph g, int start, int goal) {
    int *father = malloc(g->number_of_vertices * sizeof(int));
    int *visited = calloc(g->number_of_vertices, sizeof(int));
    int steps = 0;


    for (int i = 0; i < g->number_of_vertices; i++) {
        father[i] = -1;
    }


    p_queue q = create_queue(g->number_of_vertices);
    enqueue(q, start);
    visited[start] = 1;
    father[start] = start;


    while (!is_queue_empty(q)) {
        int v = dequeue(q);
        steps++;
        printf("Visitou: %s\n", g->coordinates[v].name);


        if (v == goal) {
            printf("Objetivo encontrado!\n");
            break;
        }


        for (p_node u = g->list_adjacency[v]; u != NULL; u = u->prox) {
            if (!visited[u->neighbor]) {
                visited[u->neighbor] = 1;
                father[u->neighbor] = v;
                enqueue(q, u->neighbor);
            }
        }
    }


    printf("Total de passos: %d\n", steps);


    // Se quiser imprimir o caminho de volta
    if (visited[goal]) {
        printf("Caminho: ");
        int p = goal;
        while (p != start) {
            printf("%s <- ", g->coordinates[p].name);
            p = father[p];
        }
        printf("%s\n", g->coordinates[start].name);
    } else {
        printf("Objetivo não alcançado.\n");
    }


    free(visited);
    free(father);
    free_queue(q);
    return steps;
}

void uniform_cost_search(p_graph g, int start, int goal) {
    float *cost = malloc(sizeof(float) * g->number_of_vertices);
    int *father = malloc(sizeof(int) * g->number_of_vertices);
    int *visited = calloc(g->number_of_vertices, sizeof(int));


    for (int i = 0; i < g->number_of_vertices; i++) {
        cost[i] = INFINITY;
        father[i] = -1;
    }
    cost[start] = 0;


    PriorityQueue *pq = create_pq(g->number_of_vertices);
    pq_push(pq, start, 0);


    while (!pq_empty(pq)) {
        PQNode node = pq_pop(pq);
        int v = node.vertex;


        if (visited[v]) continue;
        visited[v] = 1;


        printf("Visitou: %s (custo: %.2f)\n", g->coordinates[v].name, node.cost);


        if (v == goal) {
            printf("Objetivo encontrado!\n");
            break;
        }


        for (p_node u = g->list_adjacency[v]; u != NULL; u = u->prox) {
            float new_cost = cost[v] + u->weight;
            if (new_cost < cost[u->neighbor]) {
                cost[u->neighbor] = new_cost;
                father[u->neighbor] = v;
                pq_push(pq, u->neighbor, new_cost);
            }
        }
    }


    // Mostrar caminho
    if (visited[goal]) {
        printf("Caminho: ");
        int path[MAX_VERTICES], len = 0;
        for (int p = goal; p != -1; p = father[p]) {
            path[len++] = p;
        }
        for (int i = len - 1; i >= 0; i--) {
            printf("%s", g->coordinates[path[i]].name);
            if (i > 0) printf(" -> ");
        }
        printf("\nCusto total: %.2f\n", cost[goal]);
    } else {
        printf("Objetivo não alcançado.\n");
    }


    free(cost);
    free(father);
    free(visited);
    free_pq(pq);
}

void dls_recursive(p_graph g, int v, int visited[], int depth, int limit) {
    visited[v] = 1;
    printf("Visitou: %s (profundidade %d)\n", g->coordinates[v].name, depth);


    if (depth == limit) {
        return;
    }


    for (p_node u = g->list_adjacency[v]; u != NULL; u = u->prox) {
        if (!visited[u->neighbor]) {
            dls_recursive(g, u->neighbor, visited, depth + 1, limit);
        }
    }
}

void dls(p_graph g, int start, int limit) {
    int visited[MAX_VERTICES] = {0};
    printf("=== DLS (limite = %d) ===\n", limit);
    dls_recursive(g, start, visited, 0, limit);
}

void ids(p_graph g, int start, int max_limit) {
    printf("=== IDS (limite máximo = %d) ===\n", max_limit);
    for (int limit = 0; limit <= max_limit; limit++) {
        printf("\n--- Iteração com limite %d ---\n", limit);
        int visited[MAX_VERTICES] = {0};
        dls_recursive(g, start, visited, 0, limit);
    }
}

void list_vertices(p_graph g) {
    printf("Pontos disponíveis:\n");
    for (int i = 0; i < g->number_of_vertices; i++) {
        if (g->busy[i]) {
            printf("- %s\n", g->coordinates[i].name);
        }
    }
}

int get_vertex_index_by_name(p_graph g, const char *name) {
    for (int i = 0; i < g->number_of_vertices; i++) {
        if (g->busy[i] && strcmp(g->coordinates[i].name, name) == 0) {
            return i;
        }
    }
    return -1;  // não encontrado
}

int main() {
    // ===== 1) Criar grafo =====
    p_graph g = create_graph(4);


    // ===== 2) Inserir vértices (nome e coordenadas) =====
    set_coordinates(g, 0, 0, "Casa A");
    set_coordinates(g, 3, 4, "Casa B");
    set_coordinates(g, 5, 2, "Casa C");
    set_coordinates(g, 1, 1, "Casa D");


    // ===== 3) Distância exemplo =====
    float d = calculate_euclidean(g->coordinates[0], g->coordinates[1]);
    printf("Distância entre %s e %s: %.2f\n",
           g->coordinates[0].name,
           g->coordinates[1].name,
           d);


    // ===== 4) Inserir arestas (ponderadas pela distância escolhida) =====
    insert_edge(g, 0, 1, calculate_euclidean);
    insert_edge(g, 1, 2, calculate_euclidean);
    insert_edge(g, 0, 3, calculate_euclidean);
    insert_edge(g, 2, 3, calculate_euclidean);


    // ===== 5) Mostrar o grafo =====
    printf("\n=== Lista de Adjacência ===\n");
    print_graph(g);


    // ===== 6) Escolher origem e destino por nome =====
    printf("\n=== Pontos Disponíveis ===\n");
    list_vertices(g);


    char origem_nome[30], destino_nome[30];
    printf("\nDigite o nome do ponto de partida: ");
    scanf(" %[^\n]", origem_nome);
    printf("Digite o nome do ponto de destino: ");
    scanf(" %[^\n]", destino_nome);


    int origem = get_vertex_index_by_name(g, origem_nome);
    int destino = get_vertex_index_by_name(g, destino_nome);


    if (origem == -1 || destino == -1) {
        printf("Erro: ponto(s) não encontrado(s).\n");
        free_graph(g);
        return 0;
    }


    // ===== 7) BFS =====
    printf("\n=== Breadth First Search (BFS) — menor nº de arestas ===\n");
    breadth_first_search(g, origem, destino);


    // ===== 8) DFS (iterativa) =====
    printf("\n=== Depth First Search (DFS - Iterativa) ===\n");
    dfs(g, origem, destino);


    // ===== 9) DFS (recursiva) =====
    printf("\n=== Depth First Search (DFS - Recursiva) ===\n");
    boolean *visited = calloc(g->number_of_vertices, sizeof(boolean));
    int dfs_steps = dfs_recursive(g, origem, destino, visited);
    printf("Total de passos (DFS recursiva): %d\n", dfs_steps);
    free(visited);

    // ===== 10) DLS (Depth-Limited Search) =====
    int dls_limit;
    printf("\nDigite o limite de profundidade para DLS: ");
    scanf("%d", &dls_limit);
    dls(g, origem, dls_limit);

    // ===== 11) IDS (Iterative Deepening Search) =====
    int ids_max_limit;
    printf("\nDigite o limite máximo de profundidade para IDS: ");
    scanf("%d", &ids_max_limit);
    ids(g, origem, ids_max_limit);

    // ===== 12) UCS (Uniform Cost Search) =====
    printf("\n=== Uniform Cost Search (UCS) — menor custo acumulado ===\n");
    uniform_cost_search(g, origem, destino);

    // ===== 13) Fim =====
    free_graph(g);
    printf("\nPrograma finalizado.\n");
    return 0;
}

// Queue implementation
p_queue create_queue(int capacity) {
    p_queue q = malloc(sizeof(Queue));
    if (q == NULL) {
        printf("Erro ao alocar memória para a fila.\n");
        exit(1);
    }
    q->data = malloc(capacity * sizeof(int));
    if (q->data == NULL) {
        printf("Erro ao alocar memória para os dados da fila.\n");
        free(q);
        exit(1);
    }
    q->front = 0;
    q->rear = -1;
    q->size = 0;
    q->capacity = capacity;
    return q;
}

/*
    Function to check if the queue is empty
    @param q Pointer to the queue
    @return 1 if the queue is empty, 0 otherwise
*/
int is_queue_empty(p_queue q) {
    return q->size == 0;
}

/*
    Function to check if the queue is full
    @param q Pointer to the queue
    @return 1 if the queue is full, 0 otherwise
*/
int is_queue_full(p_queue q) {
    return q->size == q->capacity;
}

/*
    Function to add an element to the queue
    This function adds an element to the end of the queue.
    If the queue is full, it prints an error message and exits.
    @param q Pointer to the queue
    @param value The value to be added to the queue
*/
void enqueue(p_queue q, int value) {
    if (is_queue_full(q)) {
        printf("Erro: fila cheia!\n");
        exit(1);
    }
    q->rear = (q->rear + 1) % q->capacity; // Incrementa o índice do fim circularmente
    q->data[q->rear] = value;
    q->size++;
}

/*
    Function to remove an element from the queue
    This function removes an element from the front of the queue.
    If the queue is empty, it prints an error message and exits.
    @param q Pointer to the queue
    @return The value removed from the front of the queue
*/
int dequeue(p_queue q) {
    if (is_queue_empty(q)) {
        printf("Erro: fila vazia!\n");
        exit(1);
    }
    int value = q->data[q->front];
    q->front = (q->front + 1) % q->capacity; // Incrementa o índice do início circularmente
    q->size--;
    return value;
}

/*
    Function to destroy the queue
    This function frees the memory allocated for the queue and its data.
    @param q Pointer to the queue
*/
void free_queue(p_queue q) {
    free(q->data);
    free(q);
}

p_stack create_stack() {
    p_stack s = malloc(sizeof(Stack));
    s->top = NULL;
    s->size = 0;
    return s;
}

int is_stack_empty(p_stack s) {
    return s->top == NULL;
}

void push(p_stack s, int neighbor) {
    p_stack_node new_node = malloc(sizeof(Stack_node));
    new_node->neighbor = neighbor;
    new_node->prox = s->top;
    s->top = new_node;
    s->size++;
}

int pop(p_stack s) {
    if (is_stack_empty(s)) {
        printf("Erro: pilha vazia!\n");
        exit(1);
    }
    p_stack_node temp = s->top;
    int neighbor = temp->neighbor;
    s->top = s->top->prox;
    free(temp);
    s->size--;
    return neighbor;
}

void free_stack(p_stack s) {
    p_stack_node current = s->top;
    p_stack_node temp;
    while (current != NULL) {
        temp = current;
        current = current->prox;
        free(temp);
    }
    free(s);
}

PriorityQueue* create_pq(int capacity) {
    PriorityQueue *pq = malloc(sizeof(PriorityQueue));
    pq->data = malloc(sizeof(PQNode) * capacity);
    pq->size = 0;
    pq->capacity = capacity;
    return pq;
}

void swap(PQNode *a, PQNode *b) {
    PQNode temp = *a;
    *a = *b;
    *b = temp;
}

void pq_push(PriorityQueue *pq, int vertex, float cost) {
    pq->data[pq->size].vertex = vertex;
    pq->data[pq->size].cost = cost;
    int i = pq->size++;
    while (i > 0 && pq->data[i].cost < pq->data[(i-1)/2].cost) {
        swap(&pq->data[i], &pq->data[(i-1)/2]);
        i = (i - 1) / 2;
    }
}

PQNode pq_pop(PriorityQueue *pq) {
    PQNode min = pq->data[0];
    pq->data[0] = pq->data[--pq->size];
    int i = 0;
    while (1) {
        int left = 2*i + 1, right = 2*i + 2, smallest = i;
        if (left < pq->size && pq->data[left].cost < pq->data[smallest].cost) smallest = left;
        if (right < pq->size && pq->data[right].cost < pq->data[smallest].cost) smallest = right;
        if (smallest != i) {
            swap(&pq->data[i], &pq->data[smallest]);
            i = smallest;
        } else break;
    }
    return min;
}

int pq_empty(PriorityQueue *pq) {
    return pq->size == 0;
}

void free_pq(PriorityQueue *pq) {
    free(pq->data);
    free(pq);
}
