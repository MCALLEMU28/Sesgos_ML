import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

class BiasExplorerModel:
    def __init__(self):
        self.models = {}
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.SEED = 42

    def load_data(self):
        """Carga el dataset y realiza una limpieza inicial."""
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                   'marital-status', 'occupation', 'relationship', 'race', 'sex', 
                   'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
        
        # Carga manejando espacios vacíos
        df = pd.read_csv(url, names=columns, na_values=' ?', skipinitialspace=True)
        
        # 1. Target a binario
        df['target'] = df['income'].apply(lambda x: 1 if x == '>50K' else 0)
        df.drop('income', axis=1, inplace=True)
        
        # 2. Eliminación de Duplicados (Limpieza básica)
        df.drop_duplicates(inplace=True)
        
        self.data = df
        return df

    def clean_outliers(self):
        """
        Elimina outliers usando Rango Intercuartílico (IQR).
        Solo se aplica a columnas numéricas clave como 'age' y 'hours-per-week'.
        """
        if self.data is None: return
        
        # Se define columnas donde un valor fuera de rango produce ruido o es sospechoso
        cols_to_clean = ['age', 'hours-per-week']
        
        Q1 = self.data[cols_to_clean].quantile(0.25)
        Q3 = self.data[cols_to_clean].quantile(0.75)
        IQR = Q3 - Q1
        
        # Filtro: Mantener solo lo que esté dentro del rango aceptable
        # Nota: Se usa ~ (negación) y .any(axis=1) para filtrar filas
        condition = ~((self.data[cols_to_clean] < (Q1 - 1.5 * IQR)) | 
                      (self.data[cols_to_clean] > (Q3 + 1.5 * IQR))).any(axis=1)
        
        rows_before = self.data.shape[0]
        self.data = self.data[condition]
        rows_after = self.data.shape[0]
        
        return rows_before - rows_after  # Retorno de cuántos se eliminaron

    def preprocess_and_split(self, test_size=0.2):
        """Prepara pipelines (Imputación + Escalado + Codificación) y divide."""
        X = self.data.drop('target', axis=1)
        y = self.data['target']
        
        # Identificación tipos de columnas
        numeric_features = ['age', 'education-num', 'hours-per-week', 'capital-gain', 'capital-loss']
        categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 
                                'relationship', 'race', 'sex', 'native-country']
        
        # Pipeline Numérico: Imputa la mediana y escala
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
            
        # Pipeline Categórico: Imputa el más frecuente y codifica (OneHot)
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])
            
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])
        
        # EL uso de Stratify es clave por el desbalance de clases
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.SEED, stratify=y
        )
        
        return self.X_train.shape, self.X_test.shape

    def train_models(self):
        pipe_lr = Pipeline(steps=[('preprocessor', self.preprocessor),
                                  ('classifier', LogisticRegression(random_state=self.SEED, max_iter=1000))])
        pipe_lr.fit(self.X_train, self.y_train)
        self.models['Regresión Logística'] = pipe_lr
        
        pipe_rf = Pipeline(steps=[('preprocessor', self.preprocessor),
                                  ('classifier', RandomForestClassifier(random_state=self.SEED, max_depth=10))])
        pipe_rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = pipe_rf

    def evaluate_model(self, model_name):
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(self.X_test)[:, 1]
        else:
            y_proba = model.decision_function(self.X_test)

        metrics = {
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "Recall": recall_score(self.y_test, y_pred),
            "F1 Weighted": f1_score(self.y_test, y_pred, average='weighted')
        }
        cm = confusion_matrix(self.y_test, y_pred)
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        return metrics, cm, (fpr, tpr, roc_auc), y_pred
    
    def get_bias_metrics(self, model_name, sensitive_column='sex'):
        # Se reutiliza la lógica de evaluación para obtener predicciones frescas
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        test_data = self.X_test.copy()
        test_data['true_label'] = self.y_test
        test_data['prediction'] = y_pred
        
        metrics_by_group = {}
        unique_groups = test_data[sensitive_column].unique()
        
        for group in unique_groups:
            group_data = test_data[test_data[sensitive_column] == group]
            recall = recall_score(group_data['true_label'], group_data['prediction'], zero_division=0)
            metrics_by_group[group] = recall
            
        return metrics_by_group
