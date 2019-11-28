import gensim
import numpy as np
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D


def build_model(filters, kernel_size, hidden_dims, len_max_tweet):

    model = Sequential([
        Conv1D(filters, kernel_size, padding='valid', activation='relu', strides=1, input_shape=(len_max_tweet, 400)),
        GlobalMaxPooling1D(),
        Dense(hidden_dims),
        Dropout(0.2),
        Activation('relu'),
        Dense(1, activation='sigmoid')]
    )

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def convert_w2v(model_wv, data, len_max_tweet):
    size_w2v = model_wv.vector_size
    x = np.empty((data.shape[0], len_max_tweet, size_w2v))

    for idx_t, tweet in enumerate(data):
        if len(tweet) == 0:
            x[idx_t, :, :] = np.zeros_like(x[idx_t, :, :])
        else:
            vec = model_wv[tweet]
            x[idx_t, :, :] = np.transpose(sequence.pad_sequences(vec.T, maxlen=len_max_tweet, dtype=np.float32))

    return x


class W2VGenerator(keras.utils.Sequence):

    def __init__(self, data_names, labels_names, batch_size, n_samples_tot, shuffle=True):
        super(W2VGenerator, self).__init__()
        print("In __init__ function", flush=True)
        self.data_names = np.asarray(data_names)
        self.labels_names = np.asarray(labels_names)
        self.batch_size = batch_size
        self.count = 0
        self.n_samples_tot = n_samples_tot

        if len(data_names) < 1 or len(self.labels_names) < 1:
            raise "Wrong number of files"
        else:
            print("Loading File in init, self.count =", self.count, flush=True)
            tmp = np.load(self.data_names[self.count], allow_pickle=True)
            self.data = tmp['arr_0']
            tmp.close()
            self.labels = np.load(self.labels_names[self.count], allow_pickle=True)

        self.n_samples = self.data.shape[0]
        self.indexes = np.arange(self.n_samples)
        self.n_batches = self.data.shape[0] // batch_size
        self.shuffle = shuffle
        self.ind = 0

    def __len__(self):
        out = int(self.n_samples_tot//self.batch_size)
        return out


    def __getitem__(self, idx):
        if self.ind < self.n_batches - 1:
            indexes = self.indexes[self.ind * self.batch_size: (self.ind + 1) * self.batch_size]
            self.ind = self.ind + 1

            batch_x = self.data[indexes]
            batch_y = self.labels[indexes]
        else:
            indexes = self.indexes[self.ind * self.batch_size:]
            self.count = self.count + 1

            batch_x = self.data[indexes]
            batch_y = self.labels[indexes]

            if self.count < len(self.data_names):
                print("\nself.count =", self.count)
                print("Opening new file", flush=True)
                tmp = np.load(self.data_names[self.count], allow_pickle=True)
                self.data = tmp['arr_0']
                tmp.close()
                self.labels = np.load(self.labels_names[self.count], allow_pickle=True)

                # Permutations !
                perm = np.random.permutation(self.data.shape[0])
                self.data = self.data[perm]
                self.labels = self.labels[perm]

            self.n_samples = self.data.shape[0]
            self.n_batches = self.data.shape[0] // self.batch_size
            self.indexes = np.arange(self.n_samples)

            self.ind = 0

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        print("\nEpochs end", flush=True)
        # Updates indexes after each epoch
        samples_idx = np.arange(self.data_names.shape[0])
        self.count = 0

        print("self.count set to 0\n", flush=True)

        if self.shuffle:
            np.random.shuffle(samples_idx)

        self.data_names = self.data_names[samples_idx]
        self.labels_names = self.labels_names[samples_idx]

        tmp = np.load(self.data_names[self.count], allow_pickle=True)
        self.data = tmp['arr_0']
        tmp.close()
        self.labels = np.load(self.labels_names[self.count], allow_pickle=True)

        # Permutations !
        perm = np.random.permutation(self.data.shape[0])
        self.data = self.data[perm]
        self.labels = self.labels[perm]
