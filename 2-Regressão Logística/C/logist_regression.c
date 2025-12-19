#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef double real;

/* =====================================================
   Estruturas básicas
   ===================================================== */

/* Vetor matemático */
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

/* =====================================================
   Função Sigmoide (estável numericamente)
   =====================================================
   Evita overflow de exp(z) para valores grandes
*/
real sigmoid(real z) {
    if (z >= 0) {
        return 1.0 / (1.0 + exp(-z));
    } else {
        real ez = exp(z);
        return ez / (1.0 + ez);
    }
}

/* =====================================================
   Modelo de Regressão Logística
   ===================================================== */
typedef struct {
    Vector theta;     // coeficientes (SEM bias)
    real intercept;  // termo bias
    real lr;         // learning rate
    int iters;       // iterações
} LogisticReg;

/* =====================================================
   Criação do modelo
   ===================================================== */
LogisticReg logistic_create(real lr, int iters, int n_features) {
    LogisticReg model;
    model.lr = lr;
    model.iters = iters;
    model.intercept = 0.0;

    /* Apenas os coeficientes (bias separado) */
    model.theta.size = n_features;
    model.theta.data = calloc(n_features, sizeof(real));

    return model;
}

/* =====================================================
   Treinamento por Gradient Descent
   ===================================================== */
void logistic_fit(LogisticReg *model, Matrix X, Vector y) {
    int m = X.rows;
    int n = X.cols;

    /* theta completo inclui bias → tamanho n+1 */
    Vector theta;
    theta.size = n + 1;
    theta.data = calloc(n + 1, sizeof(real));

    for (int it = 0; it < model->iters; it++) {

        /* Gradiente separado dos parâmetros */
        Vector grad;
        grad.size = n + 1;
        grad.data = calloc(n + 1, sizeof(real));

        for (int i = 0; i < m; i++) {
            /* z = bias + somatório */
            real z = theta.data[0];
            for (int j = 0; j < n; j++)
                z += MAT(X, i, j) * theta.data[j + 1];

            real p = sigmoid(z);
            real error = p - y.data[i];

            /* Gradiente do bias */
            grad.data[0] += error;

            /* Gradiente dos coeficientes */
            for (int j = 0; j < n; j++)
                grad.data[j + 1] += error * MAT(X, i, j);
        }

        /* Atualização dos parâmetros */
        for (int j = 0; j < theta.size; j++)
            theta.data[j] -= (model->lr / m) * grad.data[j];

        free(grad.data);
    }

    /* Separação final */
    model->intercept = theta.data[0];
    for (int j = 0; j < n; j++)
        model->theta.data[j] = theta.data[j + 1];

    free(theta.data);
}

/* =====================================================
   Probabilidade logística
   ===================================================== */
real logistic_predict_proba(LogisticReg *model, Vector x) {
    real z = model->intercept;
    for (int i = 0; i < x.size; i++)
        z += model->theta.data[i] * x.data[i];
    return sigmoid(z);
}

/* =====================================================
   Classe binária
   ===================================================== */
int logistic_predict(LogisticReg *model, Vector x, real threshold) {
    return logistic_predict_proba(model, x) >= threshold;
}

/* =====================================================
   Main de teste
   ===================================================== */
int main() {
    int m = 4;

    Matrix X = {m, 2, malloc(m * 2 * sizeof(real))};
    Vector y = {m, malloc(m * sizeof(real))};

    real data[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    real labels[4] = {0,0,0,1}; // AND lógico

    for (int i = 0; i < m; i++) {
        MAT(X,i,0) = data[i][0];
        MAT(X,i,1) = data[i][1];
        y.data[i] = labels[i];
    }

    LogisticReg model = logistic_create(0.1, 2000, 2);
    logistic_fit(&model, X, y);

    for (int i = 0; i < m; i++) {
        Vector xi = {2, &MAT(X,i,0)};
        int pred = logistic_predict(&model, xi, 0.5);
        printf("Entrada [%g, %g] -> %d\n",
               MAT(X,i,0), MAT(X,i,1), pred);
    }

    free(X.data);
    free(y.data);
    free(model.theta.data);

    return 0;
}
