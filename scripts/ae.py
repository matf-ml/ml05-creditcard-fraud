from scripts.data_util import build_dataset
from scripts.model_util import dense_relu
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

LR = 1e-4
EPOCHS = 50
BATCH_SIZE = 128


class AE():
    def __init__(self, input_shape, encoder_units, z_unit, decoder_units):
        self.input_shape = input_shape
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.z_unit = z_unit

        input = Input((*self.input_shape,))
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.ae = Model(input, self.decoder(self.encoder(input)), name='autoencoder')

        optimizer = Adam(LR)

        # self.encoder.compile(optimizer=optimizer, loss=mean_squared_error)
        self.ae.compile(optimizer=optimizer, loss=mean_squared_error, metrics=['mse'])

    def build_encoder(self):
        input = x = Input(shape=(*self.input_shape,))
        for unit in self.encoder_units:
            x = dense_relu(unit,0.2,True,0)(x)
        z = Dense(self.z_unit, activation='linear', name='embedding')(x)

        return Model(input, z, name='encoder')

    def build_decoder(self):
        input = x = Input(shape=(self.z_unit,))
        for unit in self.decoder_units:
            x = dense_relu(unit,0.2,True,0)(x)
        out = Dense(*(self.input_shape), activation='linear')(x)

        return Model(input,out, name='dencoder')

    def train(self, train_x, epochs=10, batch_size=64, callbacks=None):
        history = self.ae.fit(train_x, train_x, batch_size=batch_size, epochs=epochs, callbacks=callbacks, verbose=2)

        return history


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = build_dataset('../data/old/creditcard_train.csv', '../data/old/creditcard_test.csv')

    nind = np.argwhere(train_y==0)
    train_x_neg = np.squeeze(train_x[nind], axis=1)

    checkpoint_callback = ModelCheckpoint('models/checkpoints/ae_best.h5', monitor='loss')

    structure = {
        'input_shape': (28,),
        'encoder_units': (256,128,64),
        'z_unit' : 16,
        'decoder_units': (64,128,256)
    }

    model = AE(**structure)

    history = model.train(train_x_neg, epochs=EPOCHS, batch_size=BATCH_SIZE)

    model.ae.save('models/AE/ae'+str(structure['encoder_units'])+str(str(structure['z_unit']))+str(structure['decoder_units'])+'_epochs'+str(EPOCHS)+'.h5')
    model.encoder.save('models/AE/encoder'+str(structure['encoder_units'])+str(str(structure['z_unit']))+str(structure['decoder_units'])+'_epochs'+str(EPOCHS)+'.h5')

    plt.plot(np.arange(EPOCHS), history.history['loss'])
    plt.show()

    reconstructions = model.ae.predict(test_x, batch_size=1)
    losses = np.mean((reconstructions - test_x) * (reconstructions - test_x), axis=1)

    auc_pr = average_precision_score(test_y, losses)
    print(f'AUC-PR: {auc_pr}')


    test_nind = np.argwhere(test_y==0)
    test_pind = np.argwhere(test_y)
    plt.scatter(test_nind, losses[test_nind], c='b', s=5)
    plt.scatter(test_pind, losses[test_pind], c='r', s=5)
    plt.show()

