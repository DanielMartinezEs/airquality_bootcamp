import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import skew, kurtosis


def plot_hist_boxplot_skew_kurt_outliers(dataframe):
    """
    Esta función grafica histogramas y boxplots para cada columna del DataFrame,
    calcula y muestra la skewness y la kurtosis para los valores no nulos de cada columna,
    e identifica outliers asumiendo como tal a aquel dato que esté por fuera de 4 desviaciones estándar de la media.
    
    # Descipción corta: Graficar histogramas y boxplots, y calcular sesgo, curtosis y número de registros outliers
    """
    for column in dataframe.columns:
        plt.figure(figsize=(12, 5))
        
        # Histograma
        plt.subplot(1, 2, 1)
        sb.histplot(dataframe[column], kde=True, color='skyblue')
        plt.title(f'Histograma de {column}')
        
        # Boxplot
        plt.subplot(1, 2, 2)
        sb.boxplot(x=dataframe[column], color='lightgreen')
        plt.title(f'Boxplot de {column}')
        
        plt.show()

        # Filtrar valores no nulos
        non_null_values = dataframe[column].dropna()

        # Calcular y mostrar skewness y kurtosis
        skewness = skew(non_null_values)
        kurt = kurtosis(non_null_values)
        print(f'Variable: {column}')
        print(f'Skewness: {skewness:.2f}')
        print(f'Kurtosis: {kurt:.2f}')

        # Calcular el número de outliers
        mean = non_null_values.mean()
        std = non_null_values.std()
        outliers = non_null_values[(non_null_values < (mean - 4 * std)) | (non_null_values > (mean + 4 * std))]
        print(f'Número de outliers: {len(outliers)}')
        print()


def plot_correlation_heatmap(dataframe, title='Correlation Heatmap I', font_size=12):
    """
    Genera un mapa de calor de la matriz de correlación Spearman para el DataFrame dado.
    Nota: no retorna ningún valor, muestra un gráfico.
    
    Descripción corta: Se visualizan correlaciones entre variables para estudio y entendimiento de datos
    """
    plt.figure(figsize=(16, 6))
    heatmap = sb.heatmap(dataframe.corr(method='spearman'), vmin=-1, vmax=1, annot=True)
    heatmap.set_title(title, fontdict={'fontsize': font_size}, pad=12)
    plt.show()
