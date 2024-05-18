import argparse
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis
from typing import Text
import yaml
import os
import sys

# Ajusto el path directamente al directorio src
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

# Importamos directamente desde utils
from utils.logs import get_logger


def featurize(config_path: Text) -> None:
    """Create new features.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('FEATURIZE', log_level=config['base']['log_level'])

    logger.info('Load raw data cleaned')
    dataframe = pd.read_csv(config['data_load']['dataset_csv_cleaned']) #dataframe es trusted_data

    logger.info('Drop columns with no interest')
    cols_to_drop = config['featurize']['cols_to_drop']
    dataframe = dataframe.drop(cols_to_drop, axis=1) #dataframe es trusted_data

    
    """
    Ahora la función reemplaza los valores faltantes denominados con "-200" con NaN
    y luego aplica métodos de imputación específicos a algunas columnas
    
    """
    logger.info('Missing Values and Imputation tasks')
    
    dataframe.replace(to_replace=-200, value=np.NaN, inplace=True)

    # Se define criterio individual (métodos de imputación) según cada variable
    imputation_methods = {
        'CO(GT)': 'mean',
        'NOx(GT)': 'mean',
        'NO2(GT)': 'mean',
        'PT08.S4(NO2)': 'mean',
        'PT08.S5(O3)': 'mean',
        'T': 'mean',
        'RH': 'mean',
        'AH': 'mean'
    }

    # Se itera sobre el diccionario y se aplica el método de imputación correspondiente a cada columna
    for column, method in imputation_methods.items():
        if method == 'mean':
            mean_value = dataframe[column].mean()
            dataframe[column] = dataframe[column].fillna(mean_value)
        elif method == 'median':
            median_value = dataframe[column].median()
            dataframe[column] = dataframe[column].fillna(median_value)
        elif method == 'mode':
            mode_value = dataframe[column].mode()[0]
            dataframe[column] = dataframe[column].fillna(mode_value)
        else:
            # Imprime un error si el método de imputación no es reconocido
            print(f'Método de imputación no válido para la columna {column}: {method}')
    #return dataframe
    
    
    """
    Ahora se le da tratamiento a los outliers de las columnas numéricas.
    Lo que se hace es eliminar outliers del DataFrame de pandas basado en una regla endurecida de 4 desviaciones estándar.
    Todo esto previo a realizar transformaciones de datos para buscar distribuciones lo más simétricas posibles (cercanas a 0 en skew)    
    
    """
    logger.info('Removing Outliers')
    
    for column in dataframe.columns:
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            mean = dataframe[column].mean()
            std_dev = dataframe[column].std()
            
            # Calcular límites para identificar outliers
            cut_off = std_dev * 4
            lower, upper = mean - cut_off, mean + cut_off
            
            # Filtrar outliers en el DataFrame original
            dataframe = dataframe[(dataframe[column] > lower) & (dataframe[column] < upper)]
 
 
    
    
    """
    Se separa el conjunto de características del conjunto de datos dataframe
    para manipular más facilmente la transformación del conjunto de entrenamiento
    y no afectar a la variable objetivo.
    """
    
    logger.info('Separation of X and y prior to BoxCox transformation')

    X = dataframe.drop(columns=['CO(GT)'], axis=1)
    y = dataframe['CO(GT)'] 
    
 
    
    """
    Ahora se aplica tratamiento para mitigar sesgo.
    Para esto se propone propone la transformación boxcox para lidiar con el sesgo dañino previo a entrenamiento de modelos.
    
    Se aplica la transformación Box-Cox a las columnas de un DataFrame
    cuyo skew sea superior a 1.25 o menor a -1.25. Imprime las variables transformadas,
    su skew antes y después de la transformación, y el valor de lambda utilizado.
    Indica si no fue necesario realizar ninguna transformación.

    """

    logger.info('Box-Cox Transformation to feature set X')


    # Variable para rastrear si alguna columna fue transformada
    transformation_applied = False

    for column in X.columns:
        # Verificar que la columna sea de tipo numérico
        if pd.api.types.is_numeric_dtype(X[column]):
            skewness_before = X[column].skew()
            # Verificar el skewness de la columna
            if skewness_before > 1.25 or skewness_before < -1.25:
                # Asegurarse de que todos los valores sean positivos
                if all(X[column] > 0):
                    # Aplicar la transformación Box-Cox
                    transformed_data, best_lambda = stats.boxcox(X[column])
                    X[column] = transformed_data
                    skewness_after = X[column].skew()
                    print(f"Columna '{column}': Skew antes = {skewness_before:.2f}, Skew después = {skewness_after:.2f}, Lambda = {best_lambda:.2f}")
                    transformation_applied = True
                else:
                    print(f"La columna '{column}' contiene valores no positivos y no se transformará.")
        else:
            print(f"La columna '{column}' no es numérica y será ignorada.")

    if not transformation_applied:
        print("No fue necesario realizar ninguna transformación Box-Cox bajo el criterio de sesgo establecido.")
    
    
    """
    Se guardan en csv conjuntos "X_scaled" y "y" que servirán en etapas posteriores del pipeline.
    """
    
    logger.info('Save X_scaled and y sets')
    
    X_scaled_csv_path = config['featurize']['X_scaled_csv_path']
    y_csv_path = config['featurize']['y_csv_path']
    X.to_csv(X_scaled_csv_path, index=False)
    y.to_csv(y_csv_path, index=False)    
      
    logger.info('featurize complete')
    
    
if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    featurize(config_path=args.config)