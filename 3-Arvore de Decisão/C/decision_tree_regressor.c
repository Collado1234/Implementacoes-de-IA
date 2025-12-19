#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef double real;

/* =========================
   Estruturas básicas
   ========================= */

typedef struct {
    int size;
    real *data;
} Vector;

typedef struct {
    int rows;
    int cols;
    real *data;
} Matrix;

#define MAT(m, i, j) m.data[(i) * m.cols + (j)]

/* =========================
   Nó da árvore (Regressão)
   ========================= */

typedef struct TreeNode {
    int feature;              // índice da feature do split
    real threshold;           // valor de corte
    real predicted_value;     // média dos y no nó

    struct TreeNode *left;
    struct TreeNode *right;
} TreeNode;

/* =========================
   Modelo
   ========================= */

typedef struct {
    int max_depth;
    int min_samples_split;
    TreeNode *root;
} DecisionTreeRegressor;

/* =========================
   Variância (critério CART)
   ========================= */

real variance(Vector y) {
    real mean = 0.0;
    for (int i = 0; i < y.size; i++)
        mean += y.data[i];
    mean /= y.size;

    real var = 0.0;
    for (int i = 0; i < y.size; i++)
        var += (y.data[i] - mean) * (y.data[i] - mean);

    return var;
}

/* =========================
   Melhor split (regressão)
   ========================= */

int best_split(Matrix X, Vector y, int *best_feature, real *best_threshold) {

    real best_gain = 0.0;
    real parent_var = variance(y);

    for (int j = 0; j < X.cols; j++) {
        for (int i = 0; i < X.rows; i++) {

            real thr = MAT(X, i, j);

            Vector yl = {0, malloc(y.size * sizeof(real))};
            Vector yr = {0, malloc(y.size * sizeof(real))};

            for (int k = 0; k < X.rows; k++) {
                if (MAT(X, k, j) <= thr)
                    yl.data[yl.size++] = y.data[k];
                else
                    yr.data[yr.size++] = y.data[k];
            }

            if (yl.size == 0 || yr.size == 0) {
                free(yl.data);
                free(yr.data);
                continue;
            }

            real gain = parent_var
                        - ((real)yl.size / y.size) * variance(yl)
                        - ((real)yr.size / y.size) * variance(yr);

            if (gain > best_gain) {
                best_gain = gain;
                *best_feature = j;
                *best_threshold = thr;
            }

            free(yl.data);
            free(yr.data);
        }
    }

    return best_gain > 0.0;
}

/* =========================
   Construção da árvore
   ========================= */

TreeNode* build_tree(Matrix X, Vector y, int depth,
                     DecisionTreeRegressor *tree) {

    TreeNode *node = malloc(sizeof(TreeNode));
    node->left = node->right = NULL;

    /* Valor predito = média */
    real mean = 0.0;
    for (int i = 0; i < y.size; i++)
        mean += y.data[i];
    mean /= y.size;

    node->predicted_value = mean;

    /* Critérios de parada */
    if (depth >= tree->max_depth || y.size < tree->min_samples_split)
        return node;

    int feature;
    real threshold;

    if (!best_split(X, y, &feature, &threshold))
        return node;

    Matrix Xl = {0, X.cols, malloc(X.rows * X.cols * sizeof(real))};
    Matrix Xr = {0, X.cols, malloc(X.rows * X.cols * sizeof(real))};
    Vector yl = {0, malloc(y.size * sizeof(real))};
    Vector yr = {0, malloc(y.size * sizeof(real))};

    for (int i = 0; i < X.rows; i++) {
        if (MAT(X, i, feature) <= threshold) {
            for (int j = 0; j < X.cols; j++)
                MAT(Xl, Xl.rows, j) = MAT(X, i, j);
            yl.data[yl.size++] = y.data[i];
            Xl.rows++;
        } else {
            for (int j = 0; j < X.cols; j++)
                MAT(Xr, Xr.rows, j) = MAT(X, i, j);
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

/* =========================
   Predição
   ========================= */

real tree_predict(TreeNode *node, Vector x) {
    while (node->left && node->right) {
        if (x.data[node->feature] <= node->threshold)
            node = node->left;
        else
            node = node->right;
    }
    return node->predicted_value;
}

/* =========================
   Liberação de memória
   ========================= */

void free_tree(TreeNode *node) {
    if (!node) return;
    free_tree(node->left);
    free_tree(node->right);
    free(node);
}

/* =========================
   Main de teste
   ========================= */

int main() {

    /* Dataset: y = 2x + 1 */
    int m = 6;
    int n = 1;

    Matrix X = {m, n, malloc(m * n * sizeof(real))};
    Vector y = {m, malloc(m * sizeof(real))};

    for (int i = 0; i < m; i++) {
        MAT(X, i, 0) = i;
        y.data[i] = 2.0 * i + 1.0;
    }

    DecisionTreeRegressor tree;
    tree.max_depth = 3;
    tree.min_samples_split = 2;
    tree.root = NULL;

    tree.root = build_tree(X, y, 0, &tree);

    printf("Predicoes:\n");
    for (int i = 0; i < m; i++) {
        Vector xi = {1, &MAT(X, i, 0)};
        real pred = tree_predict(tree.root, xi);
        printf("x = %.1f -> y_real = %.1f | y_pred = %.3f\n",
               xi.data[0], y.data[i], pred);
    }

    free_tree(tree.root);
    free(X.data);
    free(y.data);

    return 0;
}
