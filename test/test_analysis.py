import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

# Añadir src al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.analysis.error_analysis import ErrorAnalyzer

def test_error_analyzer():
    print("Iniciando test de ErrorAnalyzer...")
    
    # 1. Crear datos dummy
    texts = [
        "love baby music", "guitar rock loud", "rap yo flow", 
        "love song heart", "heavy metal dark", "yo hip hop"
    ]
    labels = [0, 1, 2, 0, 1, 2] # 0: Pop, 1: Rock, 2: Rap
    id2label = {0: 'Pop', 1: 'Rock', 2: 'Rap'}
    
    df_test = pd.Series(texts, name='text')
    y_test = pd.Series(labels, name='label')
    
    # 2. Entrenar modelo dummy
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    
    clf = SGDClassifier(loss='log_loss', random_state=42)
    clf.fit(X, y_test)
    
    # 3. Instanciar Analyzer
    analyzer = ErrorAnalyzer(clf, vec, X, y_test, id2label, df_test)
    
    # 4. Probar métodos
    print("\nProbando get_misclassified_examples...")
    # Forzamos un error para probar (cambiamos una etiqueta real para que parezca error)
    # En este caso el modelo probablemente sobreajuste y tenga 100% acc, así que no habrá errores reales.
    # Vamos a simular un error manual en y_test para el analyzer
    y_test_w_error = y_test.copy()
    y_test_w_error.iloc[0] = 1 # Decimos que el primero era Rock, pero el modelo dirá Pop
    
    analyzer_err = ErrorAnalyzer(clf, vec, X, y_test_w_error, id2label, df_test)
    errors = analyzer_err.get_misclassified_examples(n=1)
    print(errors)
    
    print("\nProbando explain_prediction...")
    analyzer.explain_prediction(0) # Explicar el primero
    
    print("\nProbando analyze_subgroups...")
    analyzer.analyze_subgroups()
    
    print("\nTest completado con éxito.")

if __name__ == "__main__":
    test_error_analyzer()
