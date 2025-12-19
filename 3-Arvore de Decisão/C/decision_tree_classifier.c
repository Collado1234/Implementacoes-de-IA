#include<stdio.h>
#include<stdlib.h>
#include<math.h>

typedef double real;

typedef struct {
    int size;     // número de elementos
    real *data;   // memória contígua
} Vector;

/* Matriz em layout row-major */
typedef struct {
    int rows;
    int cols;
    real *data;
} Matrix;

/* Macro para acesso matricial
   Evita ponteiros duplos e melhora cache */
#define MAT(m, i, j) m.data[(i) * m.cols + (j)]

typedef struct TreeNode{
    int feature;  //indice da featura usada no split
    real threshold; // valor do corte
    int predicted_class;

    struct TreeNode *left;
    struct TreeNode *right;
}TreeNode;

typedef struct{
    int max_depth;
    int min_samples_split;
    int n_classes;
    TreeNode *root;
}DecisionTree;

real gini(Vector y, int n_classes){
    int *count = calloc(n_classes, sizeof(int));
    for(int i = 0; i < y.size; i++){
        count[(int)y.data[i]]++;
    }

    real impurity = 1.0;
    for(int c = 0; c < n_classes; c++){
        real p = (real)count[c] / y.size;
        impurity -= p * p;
    }

    free(count);
    return impurity;
}

int best_split(Matrix X, Vector y, int *best_features, real *best_threshold, int n_classes){
    real best_gain = 0.0;
    real parent_gini = gini(y, n_classes);

    for(int j = 0; j < X.cols; j++){
        for (int i = 0; i < X.rows; i++){
            real thr = MAT(X, i, j);

            Vector y_left = {0, malloc(y.size * sizeof(real))};
            Vector y_right = {0, malloc(y.size * sizeof(real))};

            for(int k = 0; k < X.rows; k++){
                if(MAT(X, k, j) <= thr){
                    y_left.data[y_left.size++] = y.data[k];
                } else {
                    y_right.data[y_right.size++] = y.data[k];
                }
            }

            if(y_left.size == 0 || y_right.size == 0){
                free(y_left.data);
                free(y_right.data);
                continue;
            }

            real g_left = gini(y_left, n_classes);
            real g_right = gini(y_right, n_classes);

            real gain = parent_gini 
                        - ((real)y_left.size / y.size) * g_left
                        - ((real)y_right.size / y.size) * g_right;
            
            if(gain > best_gain){
                best_gain = gain;
                *best_features = j;
                *best_threshold = thr;
            }

            free(y_left.data);
            free(y_right.data);
        }
    }
    return best_gain > 0;
}

TreeNode* build_tree(Matrix X, Vector y, int depth, DecisionTree *tree){
    TreeNode *node = malloc(sizeof(TreeNode));

    int *count = calloc(tree->n_classes, sizeof(int));

    for(int i = 0; i < y.size; i++){
        count[(int)y.data[i]]++;
    }

    int  best_class = 0;
    for(int c = 1; c < tree->n_classes; c++){
        if(count[c] > count[best_class]){
            best_class = c;
        }
    }

    node->predicted_class = best_class;
    node->left = node->right = NULL;
    free(count);

    if(depth >= tree->max_depth || y.size < tree->min_samples_split){
        return node;
    }

    int feature;
    real threshold;

    if(!best_split(X, y, &feature, &threshold, tree->n_classes)){
        return node;
    }

    Matrix Xl = {0, X.cols, malloc(X.rows * X.cols * sizeof(real))};
    Matrix Xr = {0, X.cols, malloc(X.rows * X.cols * sizeof(real))};
    Vector yl = {0, malloc(y.size * sizeof(real))};
    Vector yr = {0, malloc(y.size * sizeof(real))};

    for(int i = 0; i < X.rows; i++){
        if(MAT(X,i,feature) <= threshold){
            for(int j = 0; j < X.cols; j++){
                MAT(Xl, Xl.rows, j) = MAT(X, i, j);
            }
            yl.data[yl.size++] = y.data[i];
            Xl.rows++;
        }else{
            for(int j = 0; j < X.cols; j++){
                MAT(Xr, Xr.rows, j) = MAT(X, i, j);
            }
            yr.data[yr.size++] = y.data[i];
            Xr.rows++;
        }
    }

    node->feature = feature;
    node->threshold = threshold;
    node->left = build_tree(Xl, yl, depth + 1, tree);
    node->right = build_tree(Xr, yr, depth + 1, tree);

    free(Xl.data); free(Xr.data);
    free(yl.data); free(yr.data);

    return node;
}

void free_tree(TreeNode *node) {
    if (!node) return;
    free_tree(node->left);
    free_tree(node->right);
    free(node);
}


int tree_predict(TreeNode *node, Vector x){
    while(node->left && node->right){
        if(x.data[node->feature] < node->threshold){
            node = node->left;
        }else{
            node = node->right;
        }
    }
    return node->predicted_class;
}

int main() {

    /* =========================
       Dataset AND lógico
       ========================= */

    int m = 4;          // número de amostras
    int n = 2;          // número de features

    Matrix X;
    X.rows = m;
    X.cols = n;
    X.data = malloc(m * n * sizeof(real));

    Vector y;
    y.size = m;
    y.data = malloc(m * sizeof(real));

    /* Dados */
    real X_raw[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    real y_raw[4] = {0, 0, 0, 1};

    /* Copia para Matrix / Vector */
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            MAT(X, i, j) = X_raw[i][j];
        }
        y.data[i] = y_raw[i];
    }

    /* =========================
       Criação da árvore
       ========================= */

    DecisionTree tree;
    tree.max_depth = 3;
    tree.min_samples_split = 2;
    tree.n_classes = 2;
    tree.root = NULL;

    tree.root = build_tree(X, y, 0, &tree);

    /* =========================
       Teste de predição
       ========================= */

    printf("Testando arvore de decisao (AND logico):\n");

    for (int i = 0; i < m; i++) {
        Vector xi;
        xi.size = n;
        xi.data = &MAT(X, i, 0); // aponta para a linha da matriz

        int pred = tree_predict(tree.root, xi);

        printf("Entrada: [%g, %g] -> Predito: %d | Real: %g\n",
               MAT(X, i, 0),
               MAT(X, i, 1),
               pred,
               y.data[i]);
    }

    /* =========================
       Limpeza de memória
       ========================= */

    free(X.data);
    free(y.data);
    // OBS: idealmente você implementa uma função
    // para liberar recursivamente os nós da árvore

    return 0;
}
