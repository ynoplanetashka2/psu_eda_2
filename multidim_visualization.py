"""
Скрипт для визуализации многомерных данных с использованием различных техник.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors
import seaborn as sns
from sklearn.datasets import make_classification, make_blobs, make_moons
from sklearn.decomposition import PCA, TruncatedSVD 
from sklearn.manifold import MDS, Isomap, SpectralEmbedding, LocallyLinearEmbedding, TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
 
from sklearn.cluster import KMeans, DBSCAN
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


import warnings
warnings.filterwarnings('ignore')


try:
    import umap  # pip install umap-learn  
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Настройки отображения
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100

class HighDimVisualizer:
    """
    Класс для визуализации многомерных данных.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.class_names = {}  # Словарь для хранения названий классов
        np.random.seed(random_state)
        
    def generate_sample_data(self, n_samples=1000, n_features=10, n_classes=3):
        """
        Генерация тестовых данных.
        
        Parameters:
        -----------
        n_samples : int
            Количество образцов
        n_features : int
            Количество признаков
        n_classes : int
            Количество классов
        
        Returns:
        --------
        X : np.ndarray
            Матрица признаков
        y : np.ndarray
            Вектор меток
        feature_names : list
            Имена признаков
        """
        print(f"Генерация данных: {n_samples} образцов, {n_features} признаков, {n_classes} классов")
        
        # Генерируем данные с классами
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.8),
            n_redundant=int(n_features * 0.2),
            n_classes=n_classes,
            n_clusters_per_class=1,
            random_state=self.random_state
        )
        
        # Добавляем немного шума
        X += np.random.normal(0, 0.1, X.shape)
        
        # Создаем имена признаков
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
        
        # Масштабируем данные
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X, y, feature_names
    
    def generate_complex_data(self, n_samples=500):
        """
        Генерация более сложных данных для демонстрации.
        """
        print("Генерация комплексных данных...")
        
        # Создаем несколько наборов данных
        X1, y1 = make_blobs(n_samples=n_samples, centers=4, n_features=10, 
                           cluster_std=1.5, random_state=self.random_state)
        
        X2, y2 = make_moons(n_samples=n_samples, noise=0.1, random_state=self.random_state)
        
        # Объединяем данные
        X = np.vstack([X1, X2 + [5, 0]])
        y = np.hstack([y1, y2 + 4])
        
        # Добавляем дополнительные случайные признаки
        n_extra_features = 8
        extra_features = np.random.randn(X.shape[0], n_extra_features)
        X = np.hstack([X, extra_features])
        
        # Масштабируем
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        feature_names = [f'F{i+1}' for i in range(X.shape[1])]
        
        return X, y, feature_names
    
    def plot_pairplot(self, X, y, feature_names, n_features=5):
        """
        Парные диаграммы рассеяния для первых n признаков.
        """
        print("\n1. Парные диаграммы рассеяния (PairPlot)")
        
        # Выбираем первые n признаков для визуализации
        n_features = min(n_features, X.shape[1])
        indices = np.random.choice(X.shape[1], n_features, replace=False)
        
        # Создаем DataFrame для seaborn
        import pandas as pd
        df = pd.DataFrame(X[:, indices], columns=[feature_names[i] for i in indices])
        # Используем названия классов вместо числовых меток
        df['Class'] = [self._get_class_label(c) for c in y]
        
        # Строим pairplot
        g = sns.pairplot(df, hue='Class', palette='viridis', 
                        diag_kind='kde', plot_kws={'alpha': 0.6})
        g.fig.suptitle(f'Pairplot первых {n_features} признаков', y=1.02)
        plt.show()
        
    def plot_correlation_heatmap(self, X, feature_names):
        """
        Тепловая карта корреляций между признаками.
        """
        print("\n2. Тепловая карта корреляций")
        
        import pandas as pd
        
        # Вычисляем корреляционную матрицу
        df = pd.DataFrame(X, columns=feature_names)
        corr_matrix = df.corr()
        
        # Строим тепловую карту
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   annot=True, fmt=".2f")
        plt.title('Матрица корреляций признаков', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
    def _get_class_label(self, class_idx):
        """
        Получить название класса по индексу.
        """
        if self.class_names and class_idx in self.class_names:
            return self.class_names[class_idx]
        return f'Class {int(class_idx)}'
    
    def _get_legend_labels(self, unique_classes):
        """
        Получить список названий классов для легенды.
        """
        return [self._get_class_label(c) for c in unique_classes]
    
    def plot_pca_2d_3d(self, X, y):
        """
        Визуализация PCA в 2D и 3D.
        """
        print("\n3. Визуализация PCA (2D и 3D)")
        
        # Применяем PCA
        pca = PCA(n_components=3, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        
        # Объясненная дисперсия
        explained_variance = pca.explained_variance_ratio_
        print(f"Объясненная дисперсия: {explained_variance}")
        print(f"Суммарная объясненная дисперсия: {sum(explained_variance):.3f}")
        
        # 2D визуализация
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 2D plot
        scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, 
                                 cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.5)
        axes[0].set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
        axes[0].set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
        axes[0].set_title('PCA 2D проекция')
        axes[0].grid(True, alpha=0.3)
        
        # Легенда для классов с использованием названий
        unique_classes = np.unique(y)
        legend_labels = self._get_legend_labels(unique_classes)
        handles, _ = scatter.legend_elements()
        axes[0].legend(handles, legend_labels, title="Классы")
        
        # 3D визуализация
        ax3d = axes[1]
        ax3d.remove()
        ax3d = fig.add_subplot(122, projection='3d')
        
        scatter3d = ax3d.scatter3D(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                                  c=y, cmap='viridis', alpha=0.7, edgecolors='k', linewidth=0.5)
        ax3d.set_xlabel(f'PC1 ({explained_variance[0]:.2%})')
        ax3d.set_ylabel(f'PC2 ({explained_variance[1]:.2%})')
        ax3d.set_zlabel(f'PC3 ({explained_variance[2]:.2%})')
        ax3d.set_title('PCA 3D проекция')
        
        plt.tight_layout()
        plt.show()
        
        # Дополнительно: график кумулятивной объясненной дисперсии
        self.plot_explained_variance(pca, X.shape[1])
    
    def plot_explained_variance(self, pca_model, n_features):
        """
        График объясненной дисперсии PCA.
        """
        # Вычисляем PCA для всех компонент, если нужно
        if not hasattr(pca_model, 'explained_variance_ratio_'):
            pca_full = PCA().fit(X)
            explained_variance = pca_full.explained_variance_ratio_
        else:
            explained_variance = pca_model.explained_variance_ratio_
        
        plt.figure(figsize=(10, 6))
        
        # График объясненной дисперсии
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7)
        plt.xlabel('Номер главной компоненты')
        plt.ylabel('Доля объясненной дисперсии')
        plt.title('Объясненная дисперсия по компонентам')
        plt.grid(True, alpha=0.3)
        
        # Кумулятивный график
        plt.subplot(1, 2, 2)
        cumulative_variance = np.cumsum(explained_variance)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
                'b-o', linewidth=2, markersize=6)
        plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% дисперсии')
        plt.axhline(y=0.90, color='g', linestyle='--', alpha=0.7, label='90% дисперсии')
        plt.xlabel('Количество компонент')
        plt.ylabel('Кумулятивная объясненная дисперсия')
        plt.title('Кумулятивная объясненная дисперсия')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_tsne_visualization(self, X, y):
        """
        Визуализация t-SNE проекции.
        """
        print("\n4. Визуализация t-SNE")
        
        # Для больших датасетов можно использовать PCA для предобработки
        if X.shape[0] > 1000:
            print("  Применяем PCA для уменьшения размерности перед t-SNE...")
            # Адаптивное количество компонент: не более min(n_samples, n_features)
            n_components_pca = min(50, X.shape[1] - 1)  # -1 чтобы оставить хотя бы 1 компоненту
            pca = PCA(n_components=n_components_pca, random_state=self.random_state)
            X_reduced = pca.fit_transform(X)
        else:
            X_reduced = X
        
        # Параметры t-SNE
        perplexities = [5, 30, 50]
        
        fig, axes = plt.subplots(1, len(perplexities), figsize=(18, 5))
        
        for i, perplexity in enumerate(perplexities):
            tsne = TSNE(n_components=2, perplexity=perplexity, 
                       random_state=self.random_state, max_iter=1000)
            X_tsne = tsne.fit_transform(X_reduced)
            
            scatter = axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, 
                                     cmap='tab20', alpha=0.7, s=30, edgecolors='k', linewidth=0.3)
            axes[i].set_title(f't-SNE (perplexity={perplexity})')
            axes[i].set_xlabel('t-SNE 1')
            axes[i].set_ylabel('t-SNE 2')
            axes[i].grid(True, alpha=0.3)
            
            # Добавляем легенду только для первого графика
            if i == 0:
                unique_classes = np.unique(y)
                handles, _ = scatter.legend_elements()
                legend_labels = self._get_legend_labels(unique_classes[:len(handles)])
                axes[i].legend(handles, legend_labels, title="Классы", fontsize=8)
        
        plt.tight_layout()
        plt.show()
    
    def plot_umap_visualization(self, X, y):
        """
        Визуализация UMAP проекции.
        """
        print("\n5. Визуализация UMAP")
        
        try:
            # Пробуем разные параметры UMAP
            n_neighbors_list = [5, 15, 50]
            min_dists = [0.1, 0.5, 0.99]
            
            fig, axes = plt.subplots(3, 3, figsize=(18, 10))
            axes = axes.flatten()
            
            plot_idx = 0
            for n_neighbors in n_neighbors_list:
                for min_dist in min_dists:
                    umap_model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, 
                                    min_dist=min_dist, random_state=self.random_state)
                    X_umap = umap_model.fit_transform(X)
                    
                    scatter = axes[plot_idx].scatter(X_umap[:, 0], X_umap[:, 1], c=y, 
                                                    cmap='Spectral', alpha=0.7, s=30, 
                                                    edgecolors='k', linewidth=0.3)
                    axes[plot_idx].set_title(f'UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})')
                    axes[plot_idx].set_xlabel('UMAP 1')
                    axes[plot_idx].set_ylabel('UMAP 2')
                    axes[plot_idx].grid(True, alpha=0.3)
                    
                    # Добавляем легенду с названиями классов только для первого графика
                    if plot_idx == 0:
                        unique_classes = np.unique(y)
                        handles, _ = scatter.legend_elements()
                        legend_labels = self._get_legend_labels(unique_classes[:len(handles)])
                        axes[plot_idx].legend(handles, legend_labels, title="Классы", fontsize=7)
                    
                    plot_idx += 1
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("UMAP не установлен. Установите: pip install umap-learn")
    
    def plot_parallel_coordinates(self, X, y, feature_names, n_features=10):
        """
        Параллельные координаты.
        """
        print("\n6. Параллельные координаты")
        
        import pandas as pd
        
        # Выбираем подмножество признаков
        n_features = min(n_features, X.shape[1])
        indices = np.random.choice(X.shape[1], n_features, replace=False)
        
        # Создаем DataFrame
        df = pd.DataFrame(X[:, indices], columns=[feature_names[i] for i in indices])
        df['Class'] = y
        
        # Масштабируем данные для лучшей визуализации
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:, :-1]), 
                                columns=df.columns[:-1])
        df_scaled['Class'] = df['Class'].values
        
        plt.figure(figsize=(14, 8))
        
        # Выбираем подмножество точек для избежания перегруженности
        n_samples_plot = min(200, len(df_scaled))
        sample_indices = np.random.choice(len(df_scaled), n_samples_plot, replace=False)
        df_plot = df_scaled.iloc[sample_indices]
        
        # Получаем цвета для классов
        unique_classes = np.unique(y)
        colors_plt = cm.tab20(np.linspace(0, 1, len(unique_classes)))
        
        # Рисуем линии для каждого класса
        for i, class_val in enumerate(unique_classes):
            class_data = df_plot[df_plot['Class'] == class_val].iloc[:, :-1]
            for j in range(len(class_data)):
                plt.plot(range(n_features), class_data.iloc[j], 
                        color=colors_plt[i], alpha=0.3, linewidth=0.5)
        
        plt.xticks(range(n_features), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Признаки')
        plt.ylabel('Нормализованное значение')
        plt.title('Параллельные координаты (случайная выборка)')
        plt.grid(True, alpha=0.3)
        
        # Создаем легенду с названиями классов
        from matplotlib.lines import Line2D
        legend_labels = self._get_legend_labels(unique_classes)
        legend_elements = [Line2D([0], [0], color=colors_plt[i], lw=2, 
                                 label=legend_labels[i]) 
                          for i in range(len(unique_classes))]
        plt.legend(handles=legend_elements, title="Классы")
        
        plt.tight_layout()
        plt.show()
    
    def plot_radar_chart(self, X, y, feature_names, n_features=6):
        """
        Радар-чарты для центроидов классов.
        """
        print("\n7. Радар-чарты по классам")
        
        # Выбираем признаки
        n_features = min(n_features, X.shape[1])
        indices = np.random.choice(X.shape[1], n_features, replace=False)
        selected_features = [feature_names[i] for i in indices]
        
        # Вычисляем средние значения по классам
        unique_classes = np.unique(y)
        
        # Создаем фигуру
        fig, axes = plt.subplots(1, len(unique_classes), figsize=(16, 6), 
                                subplot_kw=dict(projection='polar'))
        
        if len(unique_classes) == 1:
            axes = [axes]
        
        # Углы для осей
        angles = np.linspace(0, 2 * np.pi, n_features, endpoint=False).tolist()
        angles += angles[:1]  # Замыкаем круг
        
        for idx, class_val in enumerate(unique_classes):
            # Средние значения для класса
            class_data = X[y == class_val][:, indices]
            class_mean = class_data.mean(axis=0)
            class_std = class_data.std(axis=0)
            
            # Нормализуем значения
            values = class_mean.tolist()
            values += values[:1]  # Замыкаем круг
            
            # Получаем название класса
            class_label = self._get_class_label(class_val)
            
            # Рисуем радар-чарт
            axes[idx].plot(angles, values, 'o-', linewidth=2, label=class_label)
            axes[idx].fill(angles, values, alpha=0.25)
            
            # Настройки
            axes[idx].set_xticks(angles[:-1])
            axes[idx].set_xticklabels(selected_features, fontsize=8)
            axes[idx].set_title(class_label, size=14, y=1.1)
            axes[idx].grid(True)
        
        plt.suptitle('Радар-чарты средних значений признаков по классам', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_interactive_3d(self, X, y, feature_names):
        """
        Интерактивная 3D визуализация с использованием Plotly.
        """
        print("\n8. Интерактивная 3D визуализация (откроется в браузере)")
        
        # Применяем PCA для 3D
        pca = PCA(n_components=3, random_state=self.random_state)
        X_pca = pca.fit_transform(X)
        
        # Создаем интерактивный график
        fig = go.Figure(data=[
            go.Scatter3d(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                z=X_pca[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=y,
                    colorscale='Viridis',
                    opacity=0.8,
                    showscale=True
                ),
                text=[f'Class: {self._get_class_label(label)}<br>Point: {i}' for i, label in enumerate(y)],
                hoverinfo='text'
            )
        ])
        
        explained_variance = pca.explained_variance_ratio_
        fig.update_layout(
            title=f'3D PCA проекция данных<br>Объясненная дисперсия: {sum(explained_variance):.2%}',
            scene=dict(
                xaxis_title=f'PC1 ({explained_variance[0]:.2%})',
                yaxis_title=f'PC2 ({explained_variance[1]:.2%})',
                zaxis_title=f'PC3 ({explained_variance[2]:.2%})'
            ),
            width=1000,
            height=800
        )
        
        # Сохраняем в HTML файл
        fig.write_html("interactive_3d_plot.html")
        print("  График сохранен в файл: interactive_3d_plot.html")
        
        # Показываем в ноутбуке или браузере
        fig.show()
    
    def compare_dim_reduction_methods(self, X, y):
        """
        Сравнение различных методов уменьшения размерности.
        """
        print("\n9. Сравнение методов уменьшения размерности")
        
        # Список методов для сравнения
        methods = [
            ('PCA', PCA(n_components=2, random_state=self.random_state)),
            ('t-SNE', TSNE(n_components=2, random_state=self.random_state, 
                          perplexity=30, max_iter=1000)),
            ('MDS', MDS(n_components=2, random_state=self.random_state)),
            ('Isomap', Isomap(n_components=2, n_neighbors=10)),
        ]
        
        # Добавляем UMAP если установлен
        if HAS_UMAP:
            import umap
            methods.append(('UMAP', umap.UMAP(n_components=2, random_state=self.random_state)))
        else:
            pass
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(methods):
            if idx >= len(axes):
                break
                
            print(f"  Применяем {name}...")
            
            try:
                # Применяем метод уменьшения размерности
                if name == 't-SNE' and X.shape[0] > 1000:
                    # Используем PCA предобработку для t-SNE на больших данных
                    n_components_pca = min(50, X.shape[1] - 1)
                    pca_pre = PCA(n_components=n_components_pca, random_state=self.random_state)
                    X_reduced = pca_pre.fit_transform(X)
                    X_emb = model.fit_transform(X_reduced)
                else:
                    X_emb = model.fit_transform(X)
                
                # Визуализируем
                scatter = axes[idx].scatter(X_emb[:, 0], X_emb[:, 1], c=y, 
                                           cmap='tab20', alpha=0.7, s=30, 
                                           edgecolors='k', linewidth=0.3)
                axes[idx].set_title(f'{name}')
                axes[idx].set_xlabel('Component 1')
                axes[idx].set_ylabel('Component 2')
                axes[idx].grid(True, alpha=0.3)
                
                # Добавляем легенду с названиями классов
                unique_classes = np.unique(y)
                handles, _ = scatter.legend_elements()
                legend_labels = self._get_legend_labels(unique_classes[:len(handles)])
                axes[idx].legend(handles, legend_labels, title="Классы", fontsize=7)
                
            except Exception as e:
                axes[idx].text(0.5, 0.5, f'Error:\n{str(e)}', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{name} (failed)')
        
        # Удаляем лишние оси
        for idx in range(len(methods), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.suptitle('Сравнение методов уменьшения размерности', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def run_full_visualization(self, X, y, feature_names, class_names=None):
        """
        Запуск полной визуализации.
        
        Parameters:
        -----------
        X : np.ndarray
            Матрица признаков
        y : np.ndarray
            Вектор меток классов (числовые)
        feature_names : list
            Имена признаков
        class_names : dict, optional
            Словарь соответствия числовых меток к названиям классов
            Пример: {0: 'Good', 1: 'Moderate', 2: 'Poor'}
        """
        # Сохраняем class_names как атрибут класса для использования в других методах
        self.class_names = class_names if class_names is not None else {}
        
        print("=" * 60)
        print("ПОЛНАЯ ВИЗУАЛИЗАЦИЯ МНОГОМЕРНЫХ ДАННЫХ")
        print("=" * 60)
        
        # Основные визуализации
        self.plot_pairplot(X, y, feature_names, n_features=5)
        self.plot_correlation_heatmap(X, feature_names)
        self.plot_pca_2d_3d(X, y)
        self.plot_tsne_visualization(X, y)
        
        # Дополнительные визуализации
        self.plot_parallel_coordinates(X, y, feature_names, n_features=8)
        self.plot_radar_chart(X, y, feature_names, n_features=6)
        self.compare_dim_reduction_methods(X, y)
        
        # Пробуем UMAP если установлен
        try:
            import umap
            self.plot_umap_visualization(X, y)
        except ImportError:
            print("\nПримечание: Для UMAP визуализации установите: pip install umap-learn")
        
        # Интерактивная визуализация
        self.plot_interactive_3d(X, y, feature_names)
        
        print("\n" + "=" * 60)
        print("ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА")
        print("=" * 60)

 