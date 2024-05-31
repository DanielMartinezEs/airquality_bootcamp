import argparse
import pandas as pd
from typing import Text
import yaml
import os
import sys


# Ajusto el path directamente al directorio src
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

# Importamos directamente desde utils
from utils.logs import get_logger


def data_load(config_path: Text) -> None:
    """Load raw data.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('DATA_LOAD', log_level=config['base']['log_level'])

    logger.info('Get dataset')
    
    raw_data = pd.read_csv(config['data_load']['dataset_csv'], sep=';', decimal=',')
    
        # Se definen las columnas válidas que se esperan en el dataset
    valid_columns = [
        'Date', 'Time', 'CO(GT)', 'PT08.S1(CO)', 'NMHC(GT)', 
        'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 
        'NO2(GT)', 'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH'
    ]
   
    # Reemplazar los caracteres '(', ')', '.' por '_'
    valid_columns = [col.replace('(', '_').replace(')', '_').replace('.', '_') for col in valid_columns]

    # Cambiar también los nombres de las columnas en el DataFrame
    raw_data.columns = [col.replace('(', '_').replace(')', '_').replace('.', '_') for col in raw_data.columns]

    # Conservar solo las columnas válidas que también existen en el DataFrame
    raw_data = raw_data[valid_columns]

    
    # Se eliminan filas donde todos los campos relevantes son NaN
    raw_data.dropna(how='all', inplace=True)
    
    #Se visualiza que figuran dos columnas extra (nulas) al extremo derecho de la matriz tabular que hay que remover
    #raw_data = raw_data.iloc[:, :-2]
    #Procediendo a eliminar filas del dataset se asegura tener el dataset limpios de filas y columnas llenas de valores nulos y lograr la coincidencia de dimensión del dataset según la bibliografía.
    #raw_data = raw_data.head(9357)

    logger.info('Save raw data cleaned')
    raw_data.to_csv(config['data_load']['dataset_csv_cleaned'], index=False)


if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)