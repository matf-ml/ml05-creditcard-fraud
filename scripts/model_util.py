from keras.layers import Dense, LeakyReLU, BatchNormalization, Dropout
import keras.backend as K


def dense_relu(units, alpha=0., bn=False, dropout=0.):
    def layer(x):
        x = Dense(units)(x)
        x = LeakyReLU(alpha=alpha)(x)
        if bn:
            x = BatchNormalization()(x)
        if dropout:
            x = Dropout(rate=dropout)(x)
        return x
    return layer