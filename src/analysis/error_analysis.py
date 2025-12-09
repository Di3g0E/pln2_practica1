import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ErrorAnalyzer:
    def __init__(self, model, X_test, y_test, y_pred, id2label, vectorizer=None, tokenizer=None):
        """
        Constructor de la clase ErrorAnalyzer para análisis de explicabilidad.
        Compatible con BaselineModel (SGD) y TransformerModel.

        Args:
            model: Modelo entrenado (BaselineModel o TransformerModel)
            X_test: Matriz de características de test (vectorizada o Dataset)
            y_test: Etiquetas reales de test
            y_pred: Predicciones del modelo
            id2label: Diccionario {id: etiqueta_texto}
            vectorizer: TfidfVectorizer (para modelos baseline)
            tokenizer: Tokenizer de Hugging Face (para modelos transformer)
        """
        self.model = model
        self.X_test = X_test
        self.y_test = np.array(y_test) if isinstance(
            y_test, (pd.Series, list)) else y_test
        self.y_pred = np.array(y_pred) if isinstance(
            y_pred, (pd.Series, list)) else y_pred
        self.id2label = id2label
        self.vectorizer = vectorizer
        self.tokenizer = tokenizer

        # Detectar tipo de modelo
        self.model_type = self._detect_model_type()

        # Obtener nombres de features del vectorizador
        self.feature_names = None
        if self.vectorizer is not None and hasattr(self.vectorizer, 'get_feature_names_out'):
            self.feature_names = self.vectorizer.get_feature_names_out()

    def _detect_model_type(self):
        """
        Detecta si es un BaselineModel (con .clf) o TransformerModel (con .trainer).
        """
        if hasattr(self.model, 'clf'):
            return 'baseline'
        elif hasattr(self.model, 'trainer'):
            return 'transformer'
        else:
            return 'unknown'

    def _reconstruct_text_from_vectorizer(self, idx):
        """
        Reconstruye el texto desde la matriz sparse TF-IDF.
        """
        if self.vectorizer is None or self.feature_names is None:
            return "[Texto no disponible]"

        x_vec = self.X_test[idx]

        if hasattr(x_vec, 'indices'):
            feature_indices = x_vec.indices
            feature_values = x_vec.data
        else:
            feature_indices = np.where(x_vec > 0)[0]
            feature_values = x_vec[feature_indices]

        words_with_scores = [(self.feature_names[idx], feature_values[i])
                             for i, idx in enumerate(feature_indices)]
        words_with_scores.sort(key=lambda x: x[1], reverse=True)

        top_words = [word for word, score in words_with_scores[:100]]
        return " ".join(top_words)

    def _reconstruct_text_from_tokenizer(self, idx):
        """
        Reconstruye el texto desde los token IDs usando el tokenizador.
        """
        if self.tokenizer is None:
            return "[Texto no disponible]"

        try:
            # Ensure idx is a standard Python int
            idx_int = int(idx)
            if hasattr(self.X_test, '__getitem__'):
                example = self.X_test[idx_int]  # Changed to idx_int

                if isinstance(example, dict):
                    if 'input_ids' in example:
                        token_ids = example['input_ids']
                    else:
                        token_ids = list(example.values())[0]
                else:
                    return "[Formato desconocido]"

                text = self.tokenizer.decode(
                    token_ids, skip_special_tokens=True)
                return text
        except Exception as e:
            return f"[Error: {str(e)}]"

        return "[Texto no disponible]"

    def get_text(self, idx):
        """Obtiene el texto reconstruido."""
        if self.tokenizer is not None:
            return self._reconstruct_text_from_tokenizer(idx)
        elif self.vectorizer is not None:
            return self._reconstruct_text_from_vectorizer(idx)
        return "[Texto no disponible]"

    def get_misclassified_examples(self, n=10):
        """
        Retorna los n ejemplos mal clasificados con más detalles.
        """
        errors_mask = self.y_test != self.y_pred
        error_indices = np.where(errors_mask)[0]

        if len(error_indices) == 0:
            return pd.DataFrame()

        selected_indices = np.random.choice(
            error_indices, min(n, len(error_indices)), replace=False)

        results = []
        for idx in selected_indices:
            real_id = self.y_test[idx]
            pred_id = self.y_pred[idx]
            text = self.get_text(idx)
            text_snippet = text[:250] + "..." if len(text) > 250 else text

            results.append({
                'índice': idx,
                'texto_muestra': text_snippet,
                'etiqueta_real': self.id2label[real_id],
                'etiqueta_predicha': self.id2label[pred_id],
            })

        return pd.DataFrame(results)

    def explain_prediction(self, text_idx, top_n=10):
        """
        Explica una predicción mostrando las palabras más influyentes.
        Compatible con ambos tipos de modelos.
        """
        pred_id = self.y_pred[text_idx]
        real_id = self.y_test[text_idx]

        print(f"- EXPLICACIÓN DE PREDICCIÓN - Ejemplo #{text_idx}")
        print(f"    - Predicción: {self.id2label[pred_id]}")
        print(f"    - Etiqueta Real: {self.id2label[real_id]}")
        print(f"    - Predicción acertada: {"Sí" if pred_id == real_id else "No"} ")

        # Mostrar texto reconstruido
        text = self.get_text(text_idx)
        print(f"    - Texto:\n{text[:500]}...\n")

        # Explicabilidad según tipo de modelo
        if self.model_type == 'baseline':
            self._explain_baseline(text_idx, pred_id, real_id, top_n)
        else:
            print("Modelo no soportado para explicabilidad")

        print(f"{'='*80}\n")

    def _explain_baseline(self, text_idx, pred_id, real_id, top_n):
        """
        Explicabilidad para BaselineModel (accede a model.clf.coef_).
        """
        x_vec = self.X_test[text_idx]

        if not hasattr(self.model.clf, 'coef_') or self.feature_names is None:
            print("- BaselineModel no tiene coeficientes disponibles")
            return

        if hasattr(x_vec, 'indices'):
            feature_indices = x_vec.indices
            feature_values = x_vec.data
        else:
            feature_indices = np.where(x_vec > 0)[0]
            feature_values = x_vec[feature_indices]

        self._print_top_features_explanation(
            self.model.clf.coef_, pred_id, feature_indices, feature_values,
            f"Palabras influyentes para '{self.id2label[pred_id]}'", top_n)

        if pred_id != real_id:
            self._print_top_features_explanation(
                self.model.clf.coef_, real_id, feature_indices, feature_values,
                f"Palabras influyentes para '{self.id2label[real_id]}'", top_n)

    def _print_top_features_explanation(self, coef, class_id, feature_indices, feature_values, title, top_n):
        """
        Imprime las features más importantes para una clase.
        """
        class_coefs = coef[class_id]
        contributions = class_coefs[feature_indices] * feature_values
        sorted_idx = np.argsort(contributions)[::-1]

        print(f"- {title}:")
        print(
            f"    {'Palabra':<25} {'Score TF-IDF':<15} {'Coeficiente':<15} {'Influencia':<15}")

        count = 0
        for i in sorted_idx:
            if count >= top_n or contributions[i] <= 0:
                continue

            feat_idx = feature_indices[i]
            word = self.feature_names[feat_idx]
            tfidf = feature_values[i]
            coef_val = class_coefs[feat_idx]
            contrib = contributions[i]

            print(
                f"    {word:<25} {tfidf:<15.6f} {coef_val:<15.6f} {contrib:<15.6f}")
            count += 1

    def analyze_error_patterns(self):
        """
        Analiza patrones en los errores cometidos por el modelo.
        """
        print("- ANÁLISIS DE PATRONES DE ERROR")

        errors_mask = self.y_test != self.y_pred
        total_errors = errors_mask.sum()
        total_samples = len(self.y_test)
        error_rate = total_errors / total_samples * 100

        print(
            f"    - Errores totales: {total_errors}/{total_samples} ({error_rate:.2f}%)")

        error_pairs = {}
        for i in np.where(errors_mask)[0]:
            real = self.id2label[self.y_test[i]]
            pred = self.id2label[self.y_pred[i]]
            key = f"{real} → {pred}"
            error_pairs[key] = error_pairs.get(key, 0) + 1

        for pair in sorted(error_pairs.items(), key=lambda x: x[1], reverse=True):
            print(
                f"        - {pair[0]:<30} {pair[1]:<10} {(pair[1]/total_errors)*100:.1f}%")

    def analyze_subgroups(self):
        """
        Analiza el rendimiento según la longitud del texto reconstruido.
        """
        print("- ANÁLISIS DE RENDIMIENTO POR LONGITUD DE TEXTO")

        # Calcular longitud de textos (limitado para eficiencia con transformers)
        text_lengths = []
        limit = min(len(self.y_test), 1000)
        for idx in range(limit):
            text = self.get_text(idx)
            text_lengths.append(len(text.split()))

        # Crear DataFrame
        df_analysis = pd.DataFrame({
            'length': text_lengths,
            'correct': (self.y_test[:len(text_lengths)] == self.y_pred[:len(text_lengths)])
        })

        # Definir rangos
        bins = [0, 20, 50, 100, 200, 500, 10000]
        labels = ['Muy Corto', 'Corto', 'Medio',
                  'Largo', 'Muy Largo', 'Extremadamente Largo']

        df_analysis['length_group'] = pd.cut(
            df_analysis['length'], bins=bins, labels=labels)

        # Agrupar y mostrar
        grouped = df_analysis.groupby('length_group', observed=True)['correct'].agg(
            ['count', 'sum', 'mean'])
        grouped.columns = ['Total', 'Correctas', 'Accuracy']
        grouped['Accuracy %'] = (
            grouped['Accuracy'] * 100).apply(lambda x: f"{x:.1f}%")

        print(grouped[['Total', 'Correctas', 'Accuracy %']])

        # Visualizar
        fig, ax = plt.subplots(figsize=(12, 6))
        accuracy_values = grouped['Accuracy'].values
        grouped_labels = grouped.index.astype(str)
        colors = ['green' if x > 0.7 else 'orange' if x >
                  0.5 else 'red' for x in accuracy_values]

        ax.bar(range(len(grouped_labels)),
               accuracy_values, color=colors)
        ax.set_xticks(range(len(grouped_labels)))
        ax.set_xticklabels(grouped_labels, rotation=45, ha='right')
        ax.set_title('Accuracy por Longitud del Texto')
        ax.set_xlabel('Rango de Longitud (palabras)')
        ax.set_ylabel('Accuracy')
        ax.grid(axis='y')
        ax.set_ylim([0, 1.05])

        plt.tight_layout()
        plt.show()

        return fig
