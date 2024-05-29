import unittest
import pandas as pd
import yaml
import os

from src.stages.data_split import data_split


class TestDataSplit(unittest.TestCase):

    def setUp(self):
        # Se carga el archivo de configuración params.yaml
        config_path = 'params.yaml'
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def test_data_split_shape(self):
        # Verifica que los conjuntos de datos generados tengan las formas esperadas
        data_split('params.yaml')
        X_train = pd.read_csv(self.config['data_split']['X_train_csv_path'])
        X_test = pd.read_csv(self.config['data_split']['X_test_csv_path'])
        y_train = pd.read_csv(self.config['data_split']['y_train_csv_path'])
        y_test = pd.read_csv(self.config['data_split']['y_test_csv_path'])
        self.assertEqual(X_train.shape[0] + X_test.shape[0], len(pd.read_csv(self.config['featurize']['X_scaled_csv_path'])))
        self.assertEqual(y_train.shape[0] + y_test.shape[0], len(pd.read_csv(self.config['featurize']['y_csv_path'])))
        
    def test_data_split_files_exist(self):
        # Verifica que se hayan generado los archivos de salida
        data_split('params.yaml')
        self.assertTrue(os.path.exists(self.config['data_split']['X_train_csv_path']))
        self.assertTrue(os.path.exists(self.config['data_split']['X_test_csv_path']))
        self.assertTrue(os.path.exists(self.config['data_split']['X_train_scaled_csv_path']))
        self.assertTrue(os.path.exists(self.config['data_split']['X_test_scaled_csv_path']))
        self.assertTrue(os.path.exists(self.config['data_split']['y_train_csv_path']))
        self.assertTrue(os.path.exists(self.config['data_split']['y_test_csv_path']))
        
    def test_data_split_shuffle(self):
        # Verifica que los datos no estén mezclados cuando shuffle=False
        self.config['data_split']['shuffle'] = False
        data_split('params.yaml')
        X_train = pd.read_csv(self.config['data_split']['X_train_csv_path'])
        X_test = pd.read_csv(self.config['data_split']['X_test_csv_path'])
        y_train = pd.read_csv(self.config['data_split']['y_train_csv_path'])
        y_test = pd.read_csv(self.config['data_split']['y_test_csv_path'])
        self.assertEqual(X_train.index.tolist(), list(range(X_train.shape[0])))
        self.assertEqual(X_test.index.tolist(), list(range(X_test.shape[0])))
        self.assertEqual(y_train.index.tolist(), list(range(y_train.shape[0])))
        self.assertEqual(y_test.index.tolist(), list(range(y_test.shape[0])))

    def test_data_split_train_size(self):
        # Verifica que el tamaño del conjunto de prueba sea correcto
        X = pd.read_csv(self.config['featurize']['X_scaled_csv_path'])
        test_size = self.config['data_split']['test_size']
        expected_train_size = int(X.shape[0] * (1 - test_size))

        X_train = pd.read_csv(self.config['data_split']['X_train_csv_path'])
        self.assertEqual(X_train.shape[0], expected_train_size)

if __name__ == '__main__':
    unittest.main()


