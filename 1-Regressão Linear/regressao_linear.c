#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* =========================================================
   Definição de tipo numérico
   ---------------------------------------------------------
   Usar typedef facilita:
   - trocar double -> float ou aritmética intervalar
   - padronizar precisão numérica
   ========================================================= */
typedef double real;

/* =========================================================
   Estruturas básicas
   ========================================================= */

/* Vetor matemático */
typedef struct {
    int size;     // número de elementos
    real *data;   // ponteiro para memória contígua
} Vector;

/* Matriz em memória linear (row-major) */
typedef struct {
    int rows;
    int cols;
    real *data;
} Matrix;

/* Macro para acesso matricial
   Evita ponteiros duplos (**)
   e mantém memória contígua (mais rápida) */
#define MAT(m, i, j) m.data[(i) * m.cols + (j)]

/* =========================================================
   Modelo Linear (simula OO em C)
   ========================================================= */

typedef struct LinearModel LinearModel;

struct LinearModel {
    Vector coef;      // coeficientes (sem intercepto)
    real intercept;   // termo independente

    /* Ponteiros de função → polimorfismo manual */
    void (*fit)(LinearModel *self, Matrix X, Vector y);
    real (*predict)(LinearModel *self, Vector x);
};

/* =========================================================
   Gradiente do MSE
   ========================================================= */

Vector gradient(Matrix X, Vector y, Vector theta) {
    int m = y.size;

    /* Alocação dinâmica do gradiente */
    Vector grad;
    grad.size = theta.size;
    grad.data = calloc(theta.size, sizeof(real));

    for (int i = 0; i < m; i++) {
        real pred = 0.0;

        /* Produto escalar linha X[i] com theta */
        for (int j = 0; j < theta.size; j++)
            pred += MAT(X, i, j) * theta.data[j];

        real error = pred - y.data[i];

        /* Acumula gradiente */
        for (int j = 0; j < theta.size; j++)
            grad.data[j] += error * MAT(X, i, j);
    }

    /* Fator 2/m do MSE */
    for (int j = 0; j < grad.size; j++)
        grad.data[j] *= (2.0 / m);

    return grad; // retorna struct (cópia rasa do ponteiro)
}

/* =========================================================
   Treinamento por Gradient Descent
   ========================================================= */

void gd_fit(LinearModel *self, Matrix X, Vector y) {
    int n = X.cols;

    /* Vetor de parâmetros (theta) */
    Vector theta;
    theta.size = n;
    theta.data = calloc(n, sizeof(real));

    real lr = 0.01;
    int iters = 1000;

    for (int it = 0; it < iters; it++) {
        Vector grad = gradient(X, y, theta);

        /* Atualização dos parâmetros */
        for (int j = 0; j < n; j++)
            theta.data[j] -= lr * grad.data[j];

        /* Liberação obrigatória (C não tem GC) */
        free(grad.data);
    }

    /* Separação do intercepto */
    self->intercept = theta.data[0];

    /* Copiamos os coeficientes (1..n-1) */
    self->coef.size = n - 1;
    self->coef.data = malloc((n - 1) * sizeof(real));

    for (int i = 1; i < n; i++)
        self->coef.data[i - 1] = theta.data[i];

    free(theta.data);
}

/* =========================================================
   Predição para um único vetor x
   ========================================================= */

real gd_predict(LinearModel *self, Vector x) {
    real y = self->intercept;
    for (int i = 0; i < x.size; i++)
        y += self->coef.data[i] * x.data[i];
    return y;
}

/* =========================================================
   Inicialização do modelo
   ========================================================= */

LinearModel gd_model() {
    LinearModel model;
    model.coef.data = NULL;   // segurança
    model.fit = gd_fit;
    model.predict = gd_predict;
    return model;
}

/* =========================================================
   Métrica R²
   ========================================================= */

real r2_score(Vector y, Vector y_pred) {
    real mean = 0.0;

    for (int i = 0; i < y.size; i++)
        mean += y.data[i];
    mean /= y.size;

    real ss_tot = 0.0, ss_res = 0.0;

    for (int i = 0; i < y.size; i++) {
        ss_tot += pow(y.data[i] - mean, 2);
        ss_res += pow(y.data[i] - y_pred.data[i], 2);
    }

    return 1.0 - (ss_res / ss_tot);
}

/* =========================================================
   Main de teste
   ========================================================= */

int main() {
    /* Dataset simples: y = 2x + 1 */
    int m = 5;

    Matrix X;
    X.rows = m;
    X.cols = 2; // coluna de bias + x
    X.data = malloc(m * 2 * sizeof(real));

    Vector y;
    y.size = m;
    y.data = malloc(m * sizeof(real));

    for (int i = 0; i < m; i++) {
        MAT(X, i, 0) = 1.0;      // bias
        MAT(X, i, 1) = (real)i; // x
        y.data[i] = 2.5 * i + 1;
    }

    LinearModel model = gd_model();
    model.fit(&model, X, y);

    printf("Intercept: %.4f\n", model.intercept);
    printf("Coeficiente: %.4f\n", model.coef.data[0]);

    /* ===============================
       Predições para TODO o dataset
       =============================== */

    Vector y_pred;
    y_pred.size = m;
    y_pred.data = malloc(m * sizeof(real));

    for (int i = 0; i < m; i++) {
        Vector xi;
        xi.size = 1;
        xi.data = &MAT(X, i, 1); // aponta para o x da linha i
        y_pred.data[i] = model.predict(&model, xi);
    }

    real r2 = r2_score(y, y_pred);
    printf("R² do modelo: %.4f\n", r2);

    /* Limpeza de memória */
    free(X.data);
    free(y.data);
    free(y_pred.data);
    free(model.coef.data);

    return 0;
}
