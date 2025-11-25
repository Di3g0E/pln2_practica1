import json
import os

notebook_path = r"c:/Users/diego/OneDrive - Universidad Rey Juan Carlos/Documentos/GIA_URJC/Curso 2025-26/PLN2/Practicas/practica1/pln2_practica1/pruebas.ipynb"

new_cells = [
  {
   "cell_type": "markdown",
   "id": "tf_md1",
   "metadata": {},
   "source": [
    "## 4. Modelado con Transformers (BERT/RoBERTa)\n",
    "\n",
    "Implementamos un modelo basado en Transformers para capturar mejor la semántica profunda.\n",
    "Usaremos `roberta-base` con la librería `transformers` de HuggingFace."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "tf_init",
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.models.transformer_model import TransformerTrainer\n",
    "\n",
    "# Inicializar el trainer con RoBERTa\n",
    "# Nota: Se puede cambiar a 'bert-base-uncased' u otros\n",
    "tf_trainer = TransformerTrainer(model_name='roberta-base', num_labels=6, seed=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "tf_prep",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Preparar datos (Tokenización)\n",
    "# Usamos el mismo DataFrame que ya tenemos cargado en 'dp'\n",
    "tf_trainer.prepare_data(dp.get_df())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "tf_train",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Entrenar el modelo\n",
    "# Ajustar epochs y batch_size según la capacidad de la máquina\n",
    "tf_trainer.train(epochs=3, batch_size=16)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "tf_eval",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluación final\n",
    "tf_trainer.evaluate()"
   ]
  }
]

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the index of the cell with "## 4."
target_index = -1
for i, cell in enumerate(nb['cells']):
    if cell.get('id') == '78beba29' or (cell['cell_type'] == 'markdown' and len(cell['source']) > 0 and "## 4." in cell['source'][0]):
        target_index = i
        break

if target_index != -1:
    print(f"Found target cell at index {target_index}. Replacing...")
    # Replace the target cell with the new cells
    nb['cells'][target_index:target_index+1] = new_cells
else:
    print("Target cell not found. Appending new cells...")
    nb['cells'].extend(new_cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
