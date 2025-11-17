if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    import k_means as km

    # 1. Gerar Dados Sintéticos
    # make_blobs gera grupos artificiais bem definidos (ótimos para testar K-Means)
    X, y_true = make_blobs(
        n_samples=300,       # número total de pontos
        centers=3,           # número de clusters verdadeiros
        n_features=4,        # número de dimensões (features)
        cluster_std=0.7,     # dispersão dos clusters
        random_state=42      # reprodutibilidade
    )

    # 2. Redução de Dimensionalidade (para visualização em 2D)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # 3. Treinar K-Means (sabemos que existem 3 grupos)
    K = 3
    print(f"\n--- Treinando K-Means (K={K}) ---")
    
    kmeans_manual = km.KMeans(K=K, max_iter=100, tolerance=1e-5)
    kmeans_manual.fit(X)
    
    # Obter os rótulos de cluster do modelo
    cluster_labels_manual = kmeans_manual.get_labels()

    # 4. Plotar Resultados: Comparação com os clusters verdadeiros
    plt.figure(figsize=(12, 5))
    
    # Gráfico 1: Resultado do seu K-Means
    plt.subplot(1, 2, 1)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels_manual, cmap='viridis', marker='o', s=50)
    centroids_2d = pca.transform(kmeans_manual.centroids)
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='X', s=200, label='Centroides')
    plt.title(f'K-Means Manual (K={K})')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend()
    
    # Gráfico 2: Grupos verdadeiros (Ground Truth do make_blobs)
    plt.subplot(1, 2, 2)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_true, cmap='viridis', marker='o', s=50)
    plt.title('Clusters Verdadeiros (make_blobs)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')

    plt.tight_layout()
    plt.show()
