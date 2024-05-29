import unittest
import pandas as pd
import numpy as np
import yaml

from src.stages.featurize import featurize

class TestFeaturize(unittest.TestCase):

    def setUp(self):
        # Se carga el archivo de configuración params.yaml en el método setUp para acceder a las rutas de los archivos.
        with open('params.yaml') as f:
            self.config = yaml.safe_load(f)

    def test_load_data_cleaned(self):
        # Prueba para verificar que los datos limpios se cargan correctamente.
        featurize('params.yaml')  # Ejecutamos la función featurize
        df = pd.read_csv(self.config['data_load']['dataset_csv_cleaned'])  # Leemos el archivo de datos limpios
        self.assertFalse(df.empty, "Dataset is empty")  # Verificamos que el DataFrame no esté vacío
        
    def test_drop_columns(self):
        # Verifica que las columnas no deseadas hayan sido eliminadas correctamente
        featurize('params.yaml')  # Ejecuta la función featurize
        df = pd.read_csv(self.config['featurize']['X_scaled_csv_path'])  # Lee el conjunto de características escaladas
        cols_to_drop = self.config['featurize']['cols_to_drop']
        for col in cols_to_drop:
            self.assertNotIn(col, df.columns, f"Column {col} was not dropped")

    def test_missing_values_imputation(self):
        # Prueba para verificar que se realizan las tareas de imputación de valores faltantes correctamente.
        featurize('params.yaml')  # Ejecutamos la función featurize
        df = pd.read_csv(self.config['data_load']['dataset_csv_cleaned'])  # Leemos el archivo de datos limpios
        self.assertFalse(df.isnull().values.any(), "Missing values still present")

    def test_save_X_y_sets(self):
        # Prueba para verificar que los conjuntos X y y se guardan correctamente en archivos CSV.
        featurize('params.yaml')  # Ejecutamos la función featurize
        X = pd.read_csv(self.config['featurize']['X_scaled_csv_path'])  # Leemos el archivo de características escaladas
        y = pd.read_csv(self.config['featurize']['y_csv_path'])  # Leemos el archivo de la variable objetivo
        self.assertFalse(X.empty, "X set is empty")
        self.assertFalse(y.empty, "y set is empty")

    def test_boxcox_transformation(self):
        # Prueba para verificar que se aplica correctamente la transformación Box-Cox a las columnas con sesgo alto.
        featurize('params.yaml')  # Ejecutamos la función featurize
        X = pd.read_csv(self.config['featurize']['X_scaled_csv_path'])  # Leemos el archivo de características escaladas
        for column in X.columns:
            if pd.api.types.is_numeric_dtype(X[column]):
                skewness_after = X[column].skew()
                self.assertTrue(abs(skewness_after) < 1.25, f"Skewness after Box-Cox transformation is still high for column {column}")

if __name__ == '__main__':
    unittest.main()