import argparse
import pandas as pd
from typing import Text
import yaml
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Ajusto el path directamente al directorio src
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

# Importamos directamente desde utils
from utils.logs import get_logger


def data_split(config_path: Text) -> None:
    """Data split.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Configurando el logger
    logger = get_logger('DATA_SPLIT', log_level=config['base']['log_level'])

    logger.info('Get featurize datasets')
    
    X = pd.read_csv(config['featurize']['X_scaled_csv_path'])
    y = pd.read_csv(config['featurize']['y_csv_path'])
    
    logger.info('Splitting data')    
    # División en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['data_split']['test_size'], random_state=config['base']['random_state'], shuffle=False)

    #print(X_train.shape)
    #print(X_test.shape)
    #print(y_train.shape)
    #print(y_test.shape)
    
    logger.info('Scaling features')
    # Escalado de características
    min_max_scaler = MinMaxScaler()
    X_train_scaled = min_max_scaler.fit_transform(X_train)
    X_test_scaled = min_max_scaler.transform(X_test)
    
    # Convirtiendo de numpy arrays a pandas DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    #print(X_train_scaled.shape)
    #print(X_test_scaled.shape)
    
    logger.info('Save train and test sets')
    
    X_train_csv_path = config['data_split']['X_train_csv_path']
    X_test_csv_path = config['data_split']['X_test_csv_path']
    X_train_scaled_csv_path = config['data_split']['X_train_scaled_csv_path']
    X_test_scaled_csv_path = config['data_split']['X_test_scaled_csv_path']
    y_train_csv_path = config['data_split']['y_train_csv_path']
    y_test_csv_path = config['data_split']['y_test_csv_path']
    
    X_train.to_csv(X_train_csv_path, index=False)
    X_test.to_csv(X_test_csv_path, index=False)
    X_train_scaled.to_csv(X_train_scaled_csv_path, index=False)
    X_test_scaled.to_csv(X_test_scaled_csv_path, index=False)
    y_train.to_csv(y_train_csv_path, index=False)
    y_test.to_csv(y_test_csv_path, index=False)
    
    logger.info('Data split completed')
    

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_split(config_path=args.config)




