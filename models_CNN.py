import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D


def build_model_emb(emb_w2v, filters, kernel_size, hidden_dims, len_max_tweet, size_emb, learning_rate=0.001, dropout=0.2):
    """
    Build a Convolutional network using a given embedding layer

    :param emb_w2v: Embedding layer of type keras.layers.embeddings
    :param filters: Number of filters in the 1D convolutional layer
    :param kernel_size: Size of the kernel for the 1D convolutional layer
    :param hidden_dims: Dimension of the hidden layer in the fully connected layer
    :param len_max_tweet: Maximum length of the tweet in the dataset
    :param size_emb: Dimension of the embedding
    :param learning_rate: Learning rate for the optimization method
    :param dropout: Dropout rate
    :return: Convolutional neural network with embedding
    """
    model = Sequential([
        emb_w2v,
        Dropout(dropout),
        Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1, input_shape=(len_max_tweet, size_emb)),
        GlobalMaxPooling1D(),
        Dense(hidden_dims),
        Dropout(dropout),
        Activation('relu'),
        Dense(1, activation='sigmoid')]
    )

    adam = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])

    return model



