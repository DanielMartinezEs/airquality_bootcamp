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
    
    #Se visualiza que figuran dos columnas extra (nulas) al extremo derecho de la matriz tabular que hay que remover
    raw_data = raw_data.iloc[:, :-2]
    #Procediendo a eliminar filas del dataset se asegura tener el dataset limpios de filas y columnas llenas de valores nulos y lograr la coincidencia de dimensión del dataset según la bibliografía.
    raw_data = raw_data.head(9357)

    logger.info('Save raw data cleaned')
    raw_data.to_csv(config['data_load']['dataset_csv_cleaned'], index=False)



if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)