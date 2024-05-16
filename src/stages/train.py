import argparse
import pandas as pd
from typing import Text
import yaml
import os
import sys

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import joblib

# Ajusto el path directamente al directorio src
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

# Importamos directamente desde utils
from utils.logs import get_logger


def train(config_path: Text) -> None:
    """Train.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Configurando el logger
    logger = get_logger('TRAIN', log_level=config['base']['log_level'])

    logger.info('Get data splitted')
    
    X_train_scaled = pd.read_csv(config['data_split']['X_train_scaled_csv_path'])
    y_train = pd.read_csv(config['data_split']['y_train_csv_path'])
    
    # Asegurando que y_train es un vector unidimensional
    y_train = y_train.squeeze()  # Esto funciona si y_train es un DataFrame con una sola columna
    
    logger.info('Train model')  
    
    # Función para encontrar los mejores parámetros con GridSearchCV
    # Prepare the parameter grid and other configurations from the loaded config
    param_grid = config['train']['estimators'][config['train']['estimator_name']]['param_grid']
    cv = config['train']['cv']
    random_state = config['base']['random_state']
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=random_state),
        param_grid,
        cv=cv,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)

    logger.info('Evaluating hiperparameters to find the best model')
    # Extract the best estimator and parameters
    best_forest = grid_search.best_estimator_
    best_params = grid_search.best_params_
    

    # Print and return the results
    logger.info(f'Best parameters found: {best_params}')
    
    logger.info('Saving best model')
    
    models_path = config['train']['model_path']
    joblib.dump(best_forest, models_path)
    
    logger.info('Train completed')
    

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train(config_path=args.config)
