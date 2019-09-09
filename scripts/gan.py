import numpy as np
import pandas as pd
from tqdm import tqdm
from scripts.data_util import build_dataset
from keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU
from keras.optimizers import Adam
from keras import Model
import matplotlib.pyplot as plt

EPOCHS = 50
LR = 1e-4
BATCH_SIZE = 128
LATENT_DIM = 3
INPUT_SHAPE = 28

N_SAMPLES = 10000

ALPHA = 0.2

class GAN():
    def __init__(self, input_shape, latent_dim):
        self.input_shape = input_shape
        self.latent_dim = latent_dim

        optimizer = Adam(lr=LR, beta_1=0.5)

        print(f'DESCRIMINATOR: ')
        self.descriminator = self.build_descriminator()
        self.descriminator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.descriminator.summary()

        self.generator = self.build_generator()
        # print(f'GENERATOR: ')
        # self.generator.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        # self.generator.summary()
        z = Input((self.latent_dim,))
        pred = self.generator(z)

        print(f'COMBINED: ')
        self.descriminator.trainable = False


        self.combined = Model(z, self.descriminator(pred))
        self.combined.compile(optimizer=optimizer, loss='binary_crossentropy')
        self.combined.summary()

    # add batchnorm
    def build_generator(self):
        input = Input((self.latent_dim,))

        # x = Dense(64, activation='relu')(input)
        # x = BatchNormalization(momentum=0.8)(x)
        # x = Dense(64, activation='relu')(x)
        # x = BatchNormalization(momentum=0.8)(x)
        # x = Dense(128, activation='relu')(x)
        # x = BatchNormalization(momentum=0.8)(x)

        x = Dense(200, activation=LeakyReLU(ALPHA))(input)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(100, activation=LeakyReLU(ALPHA))(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(50, activation=LeakyReLU(ALPHA))(x)
        x = BatchNormalization(momentum=0.8)(x)

        output = Dense(self.input_shape)(x)

        return Model(input, output)
        # return output

    # add batchnorm and dropout
    def build_descriminator(self):
        input = Input((self.input_shape,))

        # x = Dense(32, activation='relu')(input)
        # x = BatchNormalization(momentum=0.8)(x)
        # x = Dropout(0.25)(x)
        # x = Dense(64, activation='relu')(x)
        # x = BatchNormalization(momentum=0.8)(x)
        # x = Dropout(0.25)(x)
        # x = Dense(128, activation='relu')(x)
        # x = BatchNormalization(momentum=0.8)(x)
        # x = Dropout(0.25)(x)
        # x = Dense(128, activation='relu')(x)
        # x = BatchNormalization(momentum=0.8)(x)
        # x = Dropout(0.25)(x)

        x = Dense(100, activation=LeakyReLU(ALPHA))(input)
        x = BatchNormalization(momentum=0.8)(x)
        # x = Dropout(0.25)(x)
        x = Dense(50, activation=LeakyReLU(ALPHA))(x)
        x = BatchNormalization(momentum=0.8)(x)
        # x = Dropout(0.25)(x)

        output = Dense(1, activation='sigmoid')(x)

        return Model(input, output)

    def train(self, train_x, epochs, batch_size=64):
        valid_y = np.ones((batch_size, 1))
        fake_y = np.zeros((batch_size, 1))

        gen_losses = np.ones((epochs))
        des_losses = np.ones((epochs))
        des_acc = np.ones((epochs))

        idx = np.shuffle(np.arange(len(train_x)))

        for i in tqdm(range(epochs)):
            inds = np.random.randint(0, train_x.shape[0], batch_size)
            real = train_x[inds].squeeze(axis=1)

            # print(f'Real shape: {real.shape}')

            prior = np.random.normal(0, 1, (batch_size, self.latent_dim))
            generated = self.generator.predict(prior)

            d_loss_real = self.descriminator.train_on_batch(real, valid_y)
            d_loss_generated = self.descriminator.train_on_batch(generated, fake_y)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_generated)

            des_losses[i] = d_loss[0]
            des_acc[i] = d_loss[1]

            g_loss = self.combined.train_on_batch(prior, valid_y)
            gen_losses[i] = g_loss

        return des_losses, gen_losses, des_acc

    def generate_sample(self, n=1):
        latent_sample = np.random.normal(0, 1, (n, self.latent_dim))

        return self.generator.predict(latent_sample)

def plot_losses(des_loss, gen_loss):
    plt.figure(figsize=(20,10))
    plt.plot(range(len(des_loss)), des_loss)
    plt.plot(range(len(gen_loss)), gen_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Descriminator', 'Generator'])
    plt.title('Losses')
    plt.show()

def plot_acc(des_acc):
    plt.figure(figsize=(20,10))
    plt.plot(range(len(des_acc)), des_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Descriminator Accuracy')
    plt.show()

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = build_dataset('../data/creditcard_train.csv', '../data/creditcard_test.csv', 'standard')

    pind = np.argwhere(train_y)
    train_x_pos = train_x[pind]

    print(f'{len(train_x)} - {len(train_x_pos)}')

    gan = GAN(INPUT_SHAPE, LATENT_DIM)

    des_losses, gen_losses, des_acc = gan.train(train_x_pos, 6000, batch_size=128)

    plot_losses(des_losses, gen_losses)
    plot_acc(des_acc)

    generated_samples = pd.DataFrame(gan.generate_sample(n=N_SAMPLES), columns = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'])

    print(generated_samples.size)

    generated_samples.to_csv('../data/small_gen_l'+str(LATENT_DIM)+'_frauds'+str(N_SAMPLES)+'.csv', index=False)
