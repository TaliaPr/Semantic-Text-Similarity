
# README

El objetivo de este trabajo es entrenar modelos que sean capaces de predecir la similitud entre dos textos en base a los embeddings de sus palabras. 
Hay dos ficheros .ipynb principales:

- Sequence_model.ipynb: 
1. One-hot encoding de las palabras (con Jaccard y Cosine como métricas de similitud).
2. Modelo con red neuronal secuencial para predecir la similitud entre dos textos y Atención para construir los embeddings de las palabras.
3. Modelo con red neuronal secuencial para predecir la similitud entre dos textos y Atención con ponderaciones de TF-IDF para construir los embeddings de las palabras.
4. Modelo ampliado con LSTM para predecir la similitud entre dos textos.

- 