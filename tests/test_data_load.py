import unittest
import pandas as pd
import yaml
from src.stages.data_load import data_load

class TestDataLoad(unittest.TestCase):

    def setUp(self):
        # Se carga params.yaml para poder acceder a las rutas de los archivos.
        with open('params.yaml') as f:
            self.config = yaml.safe_load(f)
    
    def test_column_integrity(self):
        # Prueba para verificar la integridad de las columnas.
        data_load('params.yaml')
        df = pd.read_csv(self.config['data_load']['dataset_csv_cleaned'])
        expected_columns = ['Date', 'Time', 'CO_GT_', 'PT08_S1_CO_', 'NMHC_GT_', 'C6H6_GT_', 'PT08_S2_NMHC_',
                            'NOx_GT_', 'PT08_S3_NOx_', 'NO2_GT_', 'PT08_S4_NO2_', 'PT08_S5_O3_', 'T', 'RH', 'AH']
        # Se comparan las columnas del DataFrame con las columnas esperadas.
        self.assertListEqual(list(df.columns), expected_columns)

    def test_no_empty_rows(self):
        # Prueba para verificar que no haya filas vacías en el dataset limpio.
        data_load('params.yaml')
        df = pd.read_csv(self.config['data_load']['dataset_csv_cleaned'])
        # Se verifica que el DataFrame no esté vacío.
        self.assertFalse(df.empty, "Dataset is empty")

    def test_output_content(self):
        # Prueba para verificar el contenido y el número de filas en el dataset limpio.
        data_load('params.yaml')
        df = pd.read_csv(self.config['data_load']['dataset_csv_cleaned'])
        # Se verifica que el número de filas en el DataFrame sea igual a 9357.
        self.assertEqual(len(df), 9357, "Unexpected number of rows in the cleaned dataset")
        

if __name__ == '__main__':
    unittest.main()