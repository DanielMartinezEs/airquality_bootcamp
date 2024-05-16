import argparse
import pandas as pd
import numpy as np
from typing import Text
import yaml
import os
import sys

import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Ajusto el path directamente al directorio src
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

# Importamos directamente desde utils
from utils.logs import get_logger


def evaluate(config_path: Text) -> None:
    """Evaluate.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Configurando el logger
    logger = get_logger('EVALUATE', log_level=config['base']['log_level'])

    logger.info('Get data to evaluate')
        
    X_train_scaled = pd.read_csv(config['data_split']['X_train_scaled_csv_path'])
    X_test_scaled = pd.read_csv(config['data_split']['X_test_scaled_csv_path'])
    y_train = pd.read_csv(config['data_split']['y_train_csv_path'])
    y_test = pd.read_csv(config['data_split']['y_test_csv_path'])
        
    # Convertir y_train y y_test en matrices unidimensionales
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    
    logger.info('Load model')
    
    model_path = config['train']['model_path']
    model = joblib.load(model_path)
    
    logger.info('Evaluating with model')  
    # Entrenar el modelo
    model.fit(X_train_scaled, y_train)
    # Predecir los valores para los conjuntos de entrenamiento y prueba
    predicted_train_values = model.predict(X_train_scaled)
    predicted_test_values = model.predict(X_test_scaled)
    
    # Crear un DataFrame con las métricas de evaluación
    scores = pd.DataFrame({
        'Metric': ['R2', 'RMSE', 'MAE'],
        'Train': [r2_score(y_train, predicted_train_values),
                  np.sqrt(mean_squared_error(y_train, predicted_train_values)),
                  mean_absolute_error(y_train, predicted_train_values)],
        'Test': [r2_score(y_test, predicted_test_values),
                 np.sqrt(mean_squared_error(y_test, predicted_test_values)),
                 mean_absolute_error(y_test, predicted_test_values)]
    })

    # Redondear los valores en el DataFrame
    scores['Train'] = scores['Train'].round(4)
    scores['Test'] = scores['Test'].round(4)

    logger.info('Showing Evaluation Metrics')

    # Imprimir los resultados
    print(scores.to_string(index=False))
    
    logger.info('Evaluate completed')
    

if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    evaluate(config_path=args.config)