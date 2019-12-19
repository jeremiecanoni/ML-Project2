from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Bidirectional
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.keras import optimizers




def build_model_lstm_emb_(dim_output, embedding_layer):
    """
    Build a LSTM network with a given embedding layer. The model is built with a binary cross entropy loss
    and the Adam optimizer.

    :param dim_output: Dimension of the output in the LSTM layer
    :param embedding_layer: Embedding layer of type tensorflow.python.keras.layers.embedding
    :return: LSTM network
    """

    model = Sequential([
        embedding_layer,
        LSTM(dim_output),
        Dense(1, activation='relu')])

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model



def build_model_lstm_emb(filters, vocab_size, n_dim, embedding_matrix, len_max_tweet, lr):
    """
    Build a LSTM network with a given embedding matrix. The model is built with a binary cross entropy loss
    and the Adam optimizer.

    :param dim_output: Dimension of the output in the LSTM layer
    :param embedding_layer: Embedding layer of type keras.layers.embedding
    :return: LSTM network
    """
    model = Sequential([
        Embedding(vocab_size, n_dim, weights=[embedding_matrix], input_length=len_max_tweet,
                  trainable=False),
        LSTM(filters),
        Dense(1, activation='sigmoid')])
        
    opt = optimizers.Adam(learning_rate=lr)

    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
                        
    return model


