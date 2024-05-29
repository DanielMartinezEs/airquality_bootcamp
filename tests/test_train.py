import unittest
import pandas as pd
import yaml
import os
from sklearn.ensemble import RandomForestRegressor
import joblib
from src.stages.train import train

class TestTrain(unittest.TestCase):
    """
    Clase de pruebas unitarias para el módulo de entrenamiento (train.py).
    """
    def setUp(self):
        """
        Configuración inicial para las pruebas.
        """
        # Se obtiene el directorio base del proyecto
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        # Ruta del archivo params.yaml
        self.config_path = os.path.join(base_dir, 'params.yaml')

        # Cargar el archivo params.yaml
        with open(self.config_path) as f:
            self.config = yaml.safe_load(f)

    def test_read_config(self):
        """
        Verifica que el archivo de configuración se lea correctamente.
        """
        self.assertEqual(self.config['base']['log_level'], 'INFO')
        self.assertEqual(self.config['base']['random_state'], 42)

    def test_train_model(self):
        """
        Verifica que el modelo se entrene y se guarde correctamente.
        """
        train(self.config_path)
        self.assertTrue(os.path.exists(os.path.join(os.path.dirname(self.config_path), 'models', 'model.joblib')))

    def test_model_type(self):
        """
        Verifica que el modelo entrenado sea una instancia de RandomForestRegressor.
        """
        train(self.config_path)
        model = joblib.load(os.path.join(os.path.dirname(self.config_path), 'models', 'model.joblib'))
        self.assertIsInstance(model, RandomForestRegressor)


if __name__ == '__main__':
    unittest.main()