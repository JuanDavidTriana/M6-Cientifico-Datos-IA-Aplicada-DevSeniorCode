"""Utilidades para crear una red minima con Embedding Layer (compatible con Keras 3)."""

from __future__ import annotations


def build_embedding_model(vocab_size: int, max_len: int, embedding_dim: int = 32, mask_zero: bool = True):
    """Crea un modelo minimo Embedding -> GlobalAveragePooling1D -> Dense.

    - ``keras.Input(shape=(max_len,))`` reemplaza al argumento ``input_length``
      (obsoleto en Keras 3) y permite que ``model.summary()`` muestre los parametros.
    - ``mask_zero=True`` hace que el padding (indice 0) no participe en el promedio.
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        keras.Input(shape=(max_len,)),
        layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=mask_zero),
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
