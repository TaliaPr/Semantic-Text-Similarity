from typing import Tuple, List, Optional, Set, Dict, Union, Any
import numpy as np
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict

def map_corpus(corpus):
    """
    Preprocesses a dataset corpus by extracting sentence pairs and their similarity scores.
    
    Args:
        corpus: A dataset containing 'sentence_1', 'sentence_2', and 'label' fields
        
    Returns:
        List of tuples containing (preprocessed_sentence_1, preprocessed_sentence_2, score)
    """
    sentences_1_preproc = [simple_preprocess(d["sentence_1"]) for d in corpus] #lista de listas que son oraciones lematizadas
    sentences_2_preproc = [simple_preprocess(d["sentence_2"]) for d in corpus]
    scores = [d["label"] for d in corpus]
    sentence_pairs = list(zip(sentences_1_preproc, sentences_2_preproc, scores))
    return sentences_1_preproc, sentences_2_preproc, scores, sentence_pairs

def one_hot_evaluation(sent1: List[Union[str, Set[str]]], sent2: List[Union[str, Set[str]]]) -> float:
    """
    Calcular la similitud de Jaccard entre dues oracions
    
    Args:
        sent1: Primera oració tokenitzada com una llista de paraules o conjunts de paraules
        sent2: Segona oració tokenitzada com una llista de paraules o conjunts de paraules
        
    Returns:
        float: Puntuació de similitud basada en la distància de Jaccard
    """
    scores = []
    for i in range(len(sent1)):
        # Convertir a conjunts si no ho són ja
        set1 = set(sent1[i]) if not isinstance(sent1[i], set) else sent1[i]
        set2 = set(sent2[i]) if not isinstance(sent2[i], set) else sent2[i]
        
        # Calcular la similitud de Jaccard
        score = len(set1.intersection(set2)) / len(set1.union(set2))
        scores.append(score)
    
    # Retornar la puntuació mitjana si tenim puntuacions vàlides
    return scores

def one_hot_cosine_similarity(sent1: List[Union[str, Set[str]]], sent2: List[Union[str, Set[str]]], vocabulary: Optional[Dict] = None) -> List[float]:
    """
    Calcula la similitud del coseno entre pares de oracions utilizando representación one-hot encoding
    
    Args:
        sent1: Lista de oraciones tokenizadas (primera oración de cada par)
        sent2: Lista de oraciones tokenizadas (segona oración de cada par)
        vocabulary: Diccionario opcional para mapear palabras a índices
        
    Returns:
        List[float]: Lista de puntuaciones de similitud basadas en el coseno
    """
    scores = []
    
    for i in range(len(sent1)):
        # Convertir a conjunts per facilitar el manejo
        set1 = set(sent1[i]) if not isinstance(sent1[i], set) else sent1[i]
        set2 = set(sent2[i]) if not isinstance(sent2[i], set) else set2[i]
        
        if vocabulary:
            # Usar el vocabulario proporcionado
            vocab_size = len(vocabulary.token2id)
            vec1 = np.zeros(vocab_size)
            vec2 = np.zeros(vocab_size)
            
            for word in set1:
                if word in vocabulary.token2id:
                    vec1[vocabulary.token2id[word]] = 1
                    
            for word in set2:
                if word in vocabulary.token2id:
                    vec2[vocabulary.token2id[word]] = 1
        else:
            # Crear un vocabulario ad-hoc per a aquest pare
            all_words = set1.union(set2)
            word_to_idx = {word: idx for idx, word in enumerate(all_words)}
            
            vec1 = np.zeros(len(all_words))
            vec2 = np.zeros(len(all_words))
            
            for word in set1:
                vec1[word_to_idx[word]] = 1
                
            for word in set2:
                vec2[word_to_idx[word]] = 1
        
        # Calcular similitud del coseno
        # Si los vectores son cero, asignamos una similitud de 0
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            scores.append(0.0)
        else:
            cos_sim = np.dot(vec1, vec2) / (norm1 * norm2)
            scores.append(cos_sim)
    
    return scores

def map_word_embeddings(
        sentence: Union[str, List[str]],
        sequence_len: int = 32,
        fixed_dictionary: Optional[Dictionary] = None,
        wv_model = None
        ) -> np.ndarray:
    """
    Map to word-embedding indices
    :param sentence:
    :param sequence_len:
    :param fixed_dictionary:
    :param wv_model:
    :return:
    """
    if not isinstance(sentence, list):
        sentence_preproc = simple_preprocess(sentence)
    else:
        sentence_preproc = sentence
    _vectors = np.zeros(sequence_len, dtype=np.int32)
    index = 0
    for word in sentence_preproc:
        if fixed_dictionary is not None:
            if word in fixed_dictionary.token2id:
                # Sumo 1 perquè el valor 0 està reservat a padding
                _vectors[index] = fixed_dictionary.token2id[word] + 1
                index += 1
        else:
            if word in wv_model.key_to_index:
                _vectors[index] = wv_model.key_to_index[word] + 1
                index += 1
    return _vectors

def map_pairs(
        sentence_pairs: List[Tuple[str, str, float]],
        sequence_len: int = 32,
        fixed_dictionary: Optional[Dictionary] = None,
        wv_model = None
) -> List[Tuple[Tuple[np.ndarray, np.ndarray], float]]:
    """
    Mapea els triplets d'oracions a llistes de (x, y), (pares de vectors, score)
    :param sentence_pairs:
    :param sequence_len:
    :param fixed_dictionary:
    :param wv_model:
    :return:
    """
    # Mapeig dels paquets d'oracions a paquets de vectors
    pares_vectores = []
    for i, (sentence_1, sentence_2, similitud) in enumerate(sentence_pairs):
        vector1 = map_word_embeddings(sentence_1, sequence_len, fixed_dictionary, wv_model)
        vector2 = map_word_embeddings(sentence_2, sequence_len, fixed_dictionary, wv_model)
        # Afegir a la llista
        pares_vectores.append(((vector1, vector2), similitud))
    return pares_vectores

def pair_list_to_x_y(pair_list: List[Tuple[Tuple[np.ndarray, np.ndarray], int]]) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Otiene las matrices X_1 (N x d) , X_2 (N x d), e Y (n) a partir de listas de parelles de vectors d'oracions - Llistes de (d, d, 1)
    :param pair_list:
    :return:
    """
    _x, _y = zip(*pair_list)
    _x_1, _x_2 = zip(*_x)
    return (np.row_stack(_x_1), np.row_stack(_x_2)), np.array(_y) / 5.0

class SimpleAttention(tf.keras.layers.Layer):
    def __init__(self, units: int, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)
        self.units = units
        self.dropout_s1 = tf.keras.layers.Dropout(0.3)
        self.dropout_s2 = tf.keras.layers.Dropout(0.2)
        self.W_s1 = tf.keras.layers.Dense(units, activation='tanh', use_bias=True, name="attention_transform")
        # Dense layer to compute attention scores (context vector)
        self.W_s2 = tf.keras.layers.Dense(1, use_bias=False, name="attention_scorer")
        self.supports_masking = True  # Declare that this layer supports masking

    def call(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        # inputs shape: (batch_size, sequence_length, embedding_dim)
        # mask shape: (batch_size, sequence_length) boolean tensor

        # Attention hidden states
        hidden_states = self.dropout_s1(self.W_s1(inputs))

        # Compute attention scores
        scores = self.dropout_s2(self.W_s2(hidden_states))

        if mask is not None:
            # Apply the mask to the scores before softmax
            expanded_mask = tf.expand_dims(tf.cast(mask, dtype=tf.float32), axis=-1)
            # Add a large negative number to masked (padded) scores
            scores += (1.0 - expanded_mask) * -1e9

        # Compute attention weights
        attention_weights = tf.nn.softmax(scores, axis=1)

        # Compute the context vector (weighted sum of input embeddings)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)

        return context_vector

    def get_config(self) -> dict:
        config = super(SimpleAttention, self).get_config()
        config.update({"units": self.units})
        return config

    def compute_mask(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None) -> Optional[tf.Tensor]:
        return None

def build_and_compile_model_2(
        input_length: int = 32,
        dictionary_size: int = 1000,
        embedding_size: int = 300,
        learning_rate: float = 0.001,
        trainable_embedding: bool = False,
        pretrained_weights: Optional[np.ndarray] = None,
        attention_units: int = 4,
) -> tf.keras.Model:
    input_1 = tf.keras.Input((input_length,), dtype=tf.int32, name="input_1")
    input_2 = tf.keras.Input((input_length,), dtype=tf.int32, name="input_2")

    # Determine effective embedding parameters
    if pretrained_weights is not None:
        effective_dictionary_size = pretrained_weights.shape[0]
        effective_embedding_size = pretrained_weights.shape[1]
        embedding_initializer = tf.keras.initializers.Constant(pretrained_weights)
        is_embedding_trainable = trainable_embedding
        embedding_layer_name = "embedding_pretrained"
    else:
        effective_dictionary_size = dictionary_size
        effective_embedding_size = embedding_size
        embedding_initializer = 'uniform'
        is_embedding_trainable = True
        embedding_layer_name = "embedding"

    # Shared Embedding Layer
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=effective_dictionary_size,
        output_dim=effective_embedding_size,
        input_length=input_length,
        mask_zero=True,
        embeddings_initializer=embedding_initializer,
        trainable=is_embedding_trainable,
        name=embedding_layer_name
    )

    # Apply embedding layer to both inputs
    embedded_1 = embedding_layer(input_1)  # Shape: (batch_size, input_length, effective_embedding_size)
    embedded_2 = embedding_layer(input_2)  # Shape: (batch_size, input_length, effective_embedding_size)

    # Shared Attention Layer
    # Input: (batch_size, input_length, effective_embedding_size) with a mask
    # Output: (batch_size, effective_embedding_size)
    sentence_attention_layer = SimpleAttention(units=attention_units, name="sentence_attention")
    # sentence_attention_layer = tf.keras.layers.GlobalAveragePooling1D(name="sentence_attention_layer")

    sentence_vector_1 = sentence_attention_layer(embedded_1)
    sentence_vector_2 = sentence_attention_layer(embedded_2)

    # Projection layer
    first_projection_layer = tf.keras.layers.Dense(
        effective_embedding_size,
        activation='tanh',
        kernel_initializer=tf.keras.initializers.Identity(),
        bias_initializer=tf.keras.initializers.Zeros(),
        name="projection_layer"
    )
    dropout = tf.keras.layers.Dropout(0.2, name="projection_dropout")
    projected_1 = dropout(first_projection_layer(sentence_vector_1))
    projected_2 = dropout(first_projection_layer(sentence_vector_2))

    # Normalize the projected vectors (L2 normalization)
    normalized_1 = tf.keras.layers.Lambda(
        lambda x: tf.linalg.l2_normalize(x, axis=1), name="normalize_1"
    )(projected_1)
    normalized_2 = tf.keras.layers.Lambda(
        lambda x: tf.linalg.l2_normalize(x, axis=1), name="normalize_2"
    )(projected_2)

    # Compute Cosine Similarity = X * Y / (||X|| * ||Y||) 
    similarity_score = tf.keras.layers.Lambda(
        lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=True), name="cosine_similarity"
    )([normalized_1, normalized_2])

    # Scale similarity from [-1, 1] to [0, 1]
    output_layer = tf.keras.layers.Lambda(
        lambda x: 0.5 * (1.0 + x), name="output_scaling"
    )(similarity_score)

    # Define the Keras Model
    model = tf.keras.Model(
        inputs=[input_1, input_2],
        outputs=output_layer,
        name="sequence_similarity_attention_model"
    )

    # Compile the model
    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['mae'],
    )

    return model

class TFIDFAttention(tf.keras.layers.Layer):
    """
    An attention layer that incorporates TF-IDF weights to enhance the importance
    of distinctive words while reducing the influence of common words.
    """
    def __init__(
        self, 
        units: int, 
        tfidf_matrix: Optional[np.ndarray] = None,
        vocabulary: Optional[Dict[str, int]] = None,
        sentences: Optional[List[List[str]]] = None,
        dictionary: Optional[Dictionary] = None,
        **kwargs
    ):
        super(TFIDFAttention, self).__init__(**kwargs)
        self.units = units
        self.dropout_s1 = tf.keras.layers.Dropout(0.3)
        self.dropout_s2 = tf.keras.layers.Dropout(0.2)
        self.W_s1 = tf.keras.layers.Dense(units, activation='tanh', use_bias=True, name="attention_transform")
        self.W_s2 = tf.keras.layers.Dense(1, use_bias=False, name="attention_scorer")
        self.supports_masking = True
        
        # TF-IDF related attributes
        self.tfidf_weights = None
        self.vocabulary = vocabulary
        self.tfidf_tensor = None
        
        # Compute TF-IDF weights if not provided but sentences are
        if tfidf_matrix is None and sentences is not None:
            self._compute_tfidf_weights(sentences, dictionary)
        elif tfidf_matrix is not None and vocabulary is not None:
            # Use provided TF-IDF matrix
            self.tfidf_weights = tfidf_matrix
            self.vocabulary = vocabulary
    
    def _compute_tfidf_weights(self, sentences: List[List[str]], dictionary: Optional[Dictionary] = None):
        """
        Compute TF-IDF weights for the vocabulary based on the provided sentences.
        
        Args:
            sentences: List of tokenized sentences
            dictionary: Optional Dictionary object from gensim
        """
        # Convert tokenized sentences to strings for sklearn's TfidfVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        sentence_texts = [' '.join(sentence) for sentence in sentences]
        
        # Initialize and fit the TF-IDF vectorizer
        vectorizer = TfidfVectorizer(lowercase=False)  # Tokens are already preprocessed
        tfidf_matrix = vectorizer.fit_transform(sentence_texts)
        
        # Create a mapping between vectorizer's vocabulary and our dictionary
        self.vocabulary = vectorizer.vocabulary_
        
        # Convert sparse matrix to dictionary for easier lookup
        self.tfidf_weights = defaultdict(float)
        for word, idx in self.vocabulary.items():
            # Get the average TF-IDF score for this word across all documents
            col = tfidf_matrix[:, idx].toarray().flatten()
            avg_tfidf = np.mean(col[col > 0]) if np.any(col > 0) else 0
            self.tfidf_weights[word] = avg_tfidf
        
        # If dictionary is provided, map the TF-IDF weights to dictionary indices
        if dictionary is not None:
            # Create a tensor of TF-IDF weights that aligns with the embedding indices
            # Start with zeros (for padding and unknown words)
            self.tfidf_tensor = np.zeros(len(dictionary.token2id) + 1, dtype=np.float32)
            
            # Fill in known words
            for word, dict_idx in dictionary.token2id.items():
                if word in self.tfidf_weights:
                    # Add 1 to dict_idx because index 0 is reserved for padding
                    self.tfidf_tensor[dict_idx + 1] = self.tfidf_weights[word]
            
            # Convert to tensor
            self.tfidf_tensor = tf.convert_to_tensor(self.tfidf_tensor, dtype=tf.float32)
    
    def call(self, inputs: tf.Tensor, word_indices: Optional[tf.Tensor] = None, mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Apply TF-IDF weighted attention to the input sequence.
        
        Args:
            inputs: Input tensor of shape (batch_size, sequence_length, embedding_dim)
            word_indices: Optional tensor of word indices of shape (batch_size, sequence_length)
            mask: Optional mask tensor of shape (batch_size, sequence_length)
            
        Returns:
            Context vector of shape (batch_size, embedding_dim)
        """
        # Standard attention mechanism first
        hidden_states = self.dropout_s1(self.W_s1(inputs))
        scores = self.dropout_s2(self.W_s2(hidden_states))
        
        # Apply mask if provided
        if mask is not None:
            expanded_mask = tf.expand_dims(tf.cast(mask, dtype=tf.float32), axis=-1)
            scores += (1.0 - expanded_mask) * -1e9
        
        # Apply TF-IDF weighting if available and word indices are provided
        if self.tfidf_tensor is not None and word_indices is not None:
            # Gather TF-IDF weights for each word in the batch
            tfidf_weights = tf.gather(self.tfidf_tensor, word_indices)
            # Expand dimensions to match scores shape
            tfidf_weights = tf.expand_dims(tfidf_weights, axis=-1)
            # Multiply attention scores by TF-IDF weights
            scores = scores * tfidf_weights
        
        # Compute softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=1)
        
        # Compute context vector (weighted sum)
        context_vector = tf.reduce_sum(inputs * attention_weights, axis=1)
        
        return context_vector
    
    def get_config(self) -> dict:
        config = super(TFIDFAttention, self).get_config()
        config.update({"units": self.units})
        return config
    
    def compute_mask(self, inputs: tf.Tensor, mask: Optional[tf.Tensor] = None) -> Optional[tf.Tensor]:
        return None

class SentenceComparisonLayer(tf.keras.layers.Layer):
    """
    A layer that compares two sentence embeddings using element-wise operations
    and concatenates the results.
    """
    def __init__(self, **kwargs):
        super(SentenceComparisonLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        # Unpack the inputs
        vec1, vec2 = inputs
        
        # Compute element-wise operations
        diff = tf.math.abs(vec1 - vec2)  # Absolute difference
        mult = vec1 * vec2  # Element-wise multiplication
        
        # Concatenate the vectors and their element-wise operations
        concat = tf.keras.layers.Concatenate()([vec1, vec2, diff, mult])
        return concat
    
    def compute_output_shape(self, input_shape):
        shape1, shape2 = input_shape
        return (shape1[0], shape1[1] * 4)
    
    def get_config(self):
        config = super(SentenceComparisonLayer, self).get_config()
        return config

def create_reduced_embeddings(original_model, dimensions=[50, 100, 150]):
    """
    Crea versiones reducidas de los word embeddings originales.
    
    Args:
        original_model: Modelo de word embeddings original
        dimensions: Lista de dimensiones a las que reducir los embeddings
        
    Returns:
        dict: Diccionario con los modelos reducidos (key: dimensión, value: modelo)
    """
    import time
    reduced_models = {}
    
    print(f"Creando versiones reducidas de los embeddings ({len(original_model.index_to_key)} palabras)...")
    start_time = time.time()
    
    for dim in dimensions:
        print(f"Creando modelo de {dim} dimensiones...")
        # Utilizamos defaultdict para evitar KeyError cuando busquemos palabras
        reduced_model = defaultdict(lambda: np.zeros(dim))
        
        # Solo procesamos las primer
        for word in original_model.index_to_key:
            reduced_model[word] = original_model[word][:dim]
        
        reduced_models[dim] = reduced_model
        print(f"Modelo de {dim}d creado. Ejemplo: 'casa' tiene forma {reduced_model['casa'].shape}")
    
    elapsed_time = time.time() - start_time
    print(f"Tiempo total para crear modelos reducidos: {elapsed_time:.2f} segundos")
    
    return reduced_models

def build_simplified_advanced_model(
        input_length: int = 32,
        dictionary_size: int = 1000,
        embedding_size: int = 300,
        learning_rate: float = 0.001,
        trainable_embedding: bool = False,
        pretrained_weights: Optional[np.ndarray] = None,
        rnn_units: int = 128,
        dropout_rate: float = 0.1
) -> tf.keras.Model:
    """
    Build a simplified but advanced model for semantic text similarity.
    Removes the complex attention mechanisms that may cause mask incompatibility issues.
    """
    # Define model inputs
    input_1 = tf.keras.Input((input_length,), dtype=tf.int32, name="input_1")
    input_2 = tf.keras.Input((input_length,), dtype=tf.int32, name="input_2")

    # Determine effective embedding parameters
    if pretrained_weights is not None:
        effective_dictionary_size = pretrained_weights.shape[0]
        effective_embedding_size = pretrained_weights.shape[1]
        embedding_initializer = tf.keras.initializers.Constant(pretrained_weights)
        is_embedding_trainable = trainable_embedding
        embedding_layer_name = "embedding_pretrained"
    else:
        effective_dictionary_size = dictionary_size
        effective_embedding_size = embedding_size
        embedding_initializer = 'uniform'
        is_embedding_trainable = True
        embedding_layer_name = "embedding"

    # Shared Embedding Layer
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=effective_dictionary_size,
        output_dim=effective_embedding_size,
        input_length=input_length,
        mask_zero=True,
        embeddings_initializer=embedding_initializer,
        trainable=is_embedding_trainable,
        name=embedding_layer_name
    )

    # Apply embedding layer to both inputs
    embedded_1 = embedding_layer(input_1)
    embedded_2 = embedding_layer(input_2)

    # Create encoders for each branch with unique names
    # Branch 1
    bilstm_1 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            rnn_units, 
            return_sequences=False,  # Get the final state directly
            dropout=dropout_rate,
            recurrent_dropout=0.0
        ),
        name="branch1_bilstm"
    )(embedded_1)
    
    dense_1 = tf.keras.layers.Dense(
        effective_embedding_size // 2,
        activation='relu',
        name="branch1_dense"
    )(bilstm_1)
    
    sentence_vector_1 = tf.keras.layers.Dropout(
        dropout_rate, 
        name="branch1_dropout"
    )(dense_1)
    
    # Branch 2
    bilstm_2 = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            rnn_units, 
            return_sequences=False,  # Get the final state directly
            dropout=dropout_rate,
            recurrent_dropout=0.0
        ),
        name="branch2_bilstm"
    )(embedded_2)
    
    dense_2 = tf.keras.layers.Dense(
        effective_embedding_size // 2,
        activation='relu',
        name="branch2_dense"
    )(bilstm_2)
    
    sentence_vector_2 = tf.keras.layers.Dropout(
        dropout_rate, 
        name="branch2_dropout"
    )(dense_2)

    # Compare sentence vectors using our custom comparison layer
    merged = SentenceComparisonLayer(name="sentence_comparison")(
        [sentence_vector_1, sentence_vector_2]
    )
    
    # Final dense layers
    x = tf.keras.layers.Dense(effective_embedding_size, activation='relu', name="dense1")(merged)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    x = tf.keras.layers.Dense(effective_embedding_size // 2, activation='relu', name="dense2")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    
    # Final prediction
    prediction = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(x)

    # Create and compile model
    model = tf.keras.Model(
        inputs=[input_1, input_2],
        outputs=prediction,
        name="simplified_advanced_model"
    )

    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['mae']
    )

    return model

def build_tfidf_enhanced_model(
        input_length: int = 32,
        dictionary_size: int = 1000,
        embedding_size: int = 300,
        learning_rate: float = 0.001,
        trainable_embedding: bool = False,
        pretrained_weights: Optional[np.ndarray] = None,
        attention_units: int = 4,
        tfidf_weights: Optional[np.ndarray] = None,
        sentences: Optional[List[List[str]]] = None,
        dictionary: Optional[Dictionary] = None,
) -> tf.keras.Model:
    """
    Build a model with TF-IDF enhanced attention for semantic similarity.
    
    Args:
        input_length: Maximum sequence length
        dictionary_size: Size of the vocabulary
        embedding_size: Dimensionality of embeddings
        learning_rate: Learning rate for optimizer
        trainable_embedding: Whether embeddings should be trainable
        pretrained_weights: Optional pretrained embedding weights
        attention_units: Number of units in attention layer
        tfidf_weights: Optional precomputed TF-IDF weights
        sentences: Optional sentences for computing TF-IDF
        dictionary: Optional Dictionary object from gensim
        
    Returns:
        A compiled Keras model
    """
    input_1 = tf.keras.Input((input_length,), dtype=tf.int32, name="input_1")
    input_2 = tf.keras.Input((input_length,), dtype=tf.int32, name="input_2")

    # Determine effective embedding parameters
    if pretrained_weights is not None:
        effective_dictionary_size = pretrained_weights.shape[0]
        effective_embedding_size = pretrained_weights.shape[1]
        embedding_initializer = tf.keras.initializers.Constant(pretrained_weights)
        is_embedding_trainable = trainable_embedding
        embedding_layer_name = "embedding_pretrained"
    else:
        effective_dictionary_size = dictionary_size
        effective_embedding_size = embedding_size
        embedding_initializer = 'uniform'
        is_embedding_trainable = True
        embedding_layer_name = "embedding"

    # Shared Embedding Layer
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=effective_dictionary_size,
        output_dim=effective_embedding_size,
        input_length=input_length,
        mask_zero=True,
        embeddings_initializer=embedding_initializer,
        trainable=is_embedding_trainable,
        name=embedding_layer_name
    )

    # Apply embedding layer to both inputs
    embedded_1 = embedding_layer(input_1)  # Shape: (batch_size, input_length, effective_embedding_size)
    embedded_2 = embedding_layer(input_2)  # Shape: (batch_size, input_length, effective_embedding_size)

    # TF-IDF Enhanced Attention Layer
    sentence_attention_layer = TFIDFAttention(
        units=attention_units, 
        tfidf_matrix=tfidf_weights,
        sentences=sentences,
        dictionary=dictionary,
        name="tfidf_attention"
    )
    
    # Apply attention to get sentence vectors
    sentence_vector_1 = sentence_attention_layer(embedded_1, input_1)
    sentence_vector_2 = sentence_attention_layer(embedded_2, input_2)

    # Projection layer
    first_projection_layer = tf.keras.layers.Dense(
        effective_embedding_size,
        activation='tanh',
        kernel_initializer=tf.keras.initializers.Identity(),
        bias_initializer=tf.keras.initializers.Zeros(),
        name="projection_layer"
    )
    dropout = tf.keras.layers.Dropout(0.2, name="projection_dropout")
    projected_1 = dropout(first_projection_layer(sentence_vector_1))
    projected_2 = dropout(first_projection_layer(sentence_vector_2))

    # Normalize vectors
    normalized_1 = tf.keras.layers.Lambda(
        lambda x: tf.linalg.l2_normalize(x, axis=1), name="normalize_1"
    )(projected_1)
    normalized_2 = tf.keras.layers.Lambda(
        lambda x: tf.linalg.l2_normalize(x, axis=1), name="normalize_2"
    )(projected_2)

    # Compute cosine similarity
    similarity_score = tf.keras.layers.Lambda(
        lambda x: tf.reduce_sum(x[0] * x[1], axis=1, keepdims=True), name="cosine_similarity"
    )([normalized_1, normalized_2])

    # Scale from [-1,1] to [0,1]
    output_layer = tf.keras.layers.Lambda(
        lambda x: 0.5 * (1.0 + x), name="output_scaling"
    )(similarity_score)

    # Create and compile model
    model = tf.keras.Model(
        inputs=[input_1, input_2],
        outputs=output_layer,
        name="tfidf_enhanced_similarity_model"
    )

    model.compile(
        loss='mean_squared_error',
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=['mae'],
    )

    return model

def build_and_compile_model_with_reduced_embeddings(reduced_models, train_dataset, val_dataset, diccionario, num_epochs, x_val, y_val, dimensions=[50, 100, 150]):
    """
    Builds and trains models with different embedding dimensions.
    
    Args:
        reduced_models: Dictionary of reduced embedding models
        train_dataset: Training dataset
        val_dataset: Validation dataset
        diccionario: Dictionary object
        num_epochs: Number of epochs to train
        dimensions: List of dimensions to test
        
    Returns:
        Dictionary of Pearson correlations by dimension
    """
    results = {}
    
    for dim in dimensions:
        print(f"\nEntrenando modelo con embeddings de {dim} dimensiones...")
        
        # 1. Create the reduced embedding weights for this dimension
        pretrained_weights = np.zeros((len(diccionario.token2id) + 1, dim), dtype=np.float32)
        for token, _id in diccionario.token2id.items():
            if token in reduced_models[dim]:
                # +1 because index 0 is reserved for padding
                pretrained_weights[_id + 1] = reduced_models[dim][token]
        
        # 2. Build the model with the reduced dimensions
        model = build_and_compile_model_2(
            input_length=32,
            dictionary_size=len(diccionario.token2id) + 1,  # +1 for padding
            embedding_size=dim,
            pretrained_weights=pretrained_weights,
            trainable_embedding=True,
            attention_units=4
        )
        
        # 3. Train the model
        model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)
        
        # 4. Evaluate the model
        y_pred = model.predict(x_val)
        correlation, _ = pearsonr(y_pred.flatten(), y_val.flatten())
        print(f"Correlación de Pearson para {dim}d: {correlation}")
        
        # Store results
        results[dim] = correlation
    
    return results

def visualize_pearson_correlations(model, x_train, y_train, x_val, y_val, x_test=None, y_test=None, baseline_methods=None):
    """
    Creates a visualization of Pearson correlation scores for different datasets and methods.
    
    Args:
        model: The trained model to evaluate
        x_train: Training input data
        y_train: Training target data
        x_val: Validation input data
        y_val: Validation target data
        x_test: Test input data (optional)
        y_test: Test target data (optional)
        baseline_methods: Dictionary of baseline method results (optional)
            Format: {'method_name': {'train': score, 'val': score, 'test': score}}
    
    Returns:
        None (displays plot)
    """
    # Create a figure with appropriate size
    plt.figure(figsize=(12, 6))
    
    # Get predictions for each dataset
    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    
    # Calculate Pearson correlations
    correlations = {
        'Neural Model': {
            'Train': pearsonr(train_pred.flatten(), y_train.flatten())[0],
            'Validation': pearsonr(val_pred.flatten(), y_val.flatten())[0]
        }
    }
    
    # Add test results if provided
    if x_test is not None and y_test is not None:
        test_pred = model.predict(x_test)
        correlations['Neural Model']['Test'] = pearsonr(test_pred.flatten(), y_test.flatten())[0]
    
    # Add baseline methods if provided
    if baseline_methods:
        for method_name, scores in baseline_methods.items():
            correlations[method_name] = scores
    
    # Prepare data for plotting
    methods = list(correlations.keys())
    datasets = list(correlations[methods[0]].keys())
    
    # Set bar width
    bar_width = 0.8 / len(methods)
    
    # Setup x positions
    x_pos = np.arange(len(datasets))
    
    # Plot bars for each method
    for i, method in enumerate(methods):
        method_values = [correlations[method][dataset] for dataset in datasets]
        offset = (i - len(methods)/2 + 0.5) * bar_width
        bars = plt.bar(x_pos + offset, method_values, width=bar_width, 
                       label=method, alpha=0.8)
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Add labels and title
    plt.xlabel('Dataset')
    plt.ylabel('Pearson Correlation')
    plt.title('Pearson Correlation by Method and Dataset')
    plt.xticks(x_pos, datasets)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Show plot
    plt.tight_layout()
    plt.show()
