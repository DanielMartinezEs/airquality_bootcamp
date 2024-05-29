import unittest
import os
import json
import pandas as pd
from src.stages.evaluate import evaluate

class TestEvaluate(unittest.TestCase):
    """
    Clase de pruebas unitarias para el módulo de evaluación (evaluate.py).
    """
    def setUp(self):
        # Se carga el archivo de configuración params.yaml
        self.config_path = 'params.yaml'

    def test_evaluate_data(self):
        """
        Verifica que los datos se lean correctamente desde los archivos CSV.
        """
        evaluate(self.config_path)
        self.assertTrue(os.path.exists('reports/metrics.json'))

    def test_evaluate_metrics(self):
        """
        Verifica que se generen las métricas de evaluación correctamente.
        """
        evaluate(self.config_path)
        self.assertTrue(os.path.exists('reports/metrics.json'))
        
        # Lee el archivo JSON de métricas
        with open('reports/metrics.json') as json_file:
            metrics = json.load(json_file)

        # Verifica que las métricas estén en el formato esperado
        self.assertIsInstance(metrics, dict)
        
        # Obtenemos las claves y las ordenamos alfabéticamente
        sorted_keys = sorted(metrics.keys())
        
        # Comparamos con las claves esperadas
        expected_keys = ['MAE', 'R2', 'RMSE']
        self.assertListEqual(sorted_keys, expected_keys)

        # Verificamos que los valores de las métricas sean flotantes
        for key, values in metrics.items():
            self.assertIsInstance(values, dict)
            for subset, score in values.items():
                self.assertIsInstance(score, float)

if __name__ == '__main__':
    unittest.main()