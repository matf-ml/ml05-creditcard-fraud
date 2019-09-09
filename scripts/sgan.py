import numpy as np
import pandas as pd
from tqdm import tqdm
from scripts.data_util import build_dataset, augment_with_positives
from keras.layers import Input, Dense, BatchNormalization, Dropout, LeakyReLU
from keras.optimizers import Adam
from keras import Model
from keras.utils import to_categorical
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score

EPOCHS = 20000
LR = 1e-4
BETA_1 = 0.5
BATCH_SIZE = 128
LATENT_DIM = 3
INPUT_SHAPE = 28

N_SAMPLES = 10000

ALPHA = 0.2

class SGAN():
    def __init__(self, input_shape, latent_dim, num_classes):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.num_clasess = num_classes

        optimizer = Adam(lr=LR, beta_1=BETA_1)

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

        valid, _ = self.descriminator(pred)

        self.combined = Model(z, valid)
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

        x = Dense(200)(input)
        x = LeakyReLU(ALPHA)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(200)(x)
        x = LeakyReLU(ALPHA)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(100)(x)
        x = LeakyReLU(ALPHA)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Dense(50)(x)
        x = LeakyReLU(ALPHA)(x)
        x = BatchNormalization(momentum=0.8)(x)

        output = Dense(self.input_shape)(x)

        return Model(input, output)
        # return output

    # add batchnorm and dropout
    def build_descriminator(self):
        input = Input((self.input_shape,))
        label = Input((1,), dtype='int32')

        x = Dense(128)(input)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        # x = Dropout(0.25)(x)
        x = Dense(128)(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        # x = Dropout(0.25)(x)
        x = Dense(64)(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        # x = Dropout(0.25)(x)
        x = Dense(128)(x)
        x = LeakyReLU(0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        # x = Dropout(0.25)(x)

        # x = Dense(100)(input)
        # x = LeakyReLU(ALPHA)(x)
        # x = BatchNormalization(momentum=0.8)(x)
        # x = Dropout(0.25)(x)
        # x = Dense(100)(x)
        # x = LeakyReLU(ALPHA)(x)
        # x = BatchNormalization(momentum=0.8)(x)
        # x = Dropout(0.25)(x)
        # x = Dense(50)(x)
        # x = LeakyReLU(ALPHA)(x)
        # x = BatchNormalization(momentum=0.8)(x)
        # x = Dropout(0.25)(x)

        fakeVreal = Dense(1, activation='sigmoid')(x)
        label = Dense(self.num_clasess+1, activation='softmax')(x)

        return Model(input, [fakeVreal, label])

    def train(self, train_x, train_y, epochs, batch_size=64):
        half_batch = batch_size//2
        cw1 = {0: 1, 1: 1}
        # tweak this to balance class out
        # cw2 = {i: 2 / half_batch for i in range(self.num_clasess)}
        cw2 = {0: 2/half_batch, 1: 10/half_batch}
        cw2[self.num_clasess] = 1 / half_batch

        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        gen_losses = np.ones((epochs))
        des_losses = np.ones((epochs))
        des_acc = np.ones((epochs))

        for i in tqdm(range(epochs)):
            inds = np.random.randint(0, train_x.shape[0], batch_size)
            real = train_x[inds]#.squeeze(axis=1)

            # print(f'Real shape: {real.shape}')

            prior = np.random.normal(0, 1, (batch_size, self.latent_dim))
            generated = self.generator.predict(prior)

            #encode labels
            labels = to_categorical(train_y[inds], num_classes=self.num_clasess+1)
            fake_labels = to_categorical(np.full((batch_size, 1), self.num_clasess), num_classes=self.num_clasess+1)

            d_loss_real = self.descriminator.train_on_batch(real, [valid, labels], class_weight=[cw1, cw2])
            d_loss_generated = self.descriminator.train_on_batch(generated, [fake, fake_labels], class_weight=[cw1, cw2])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_generated)

            des_losses[i] = d_loss[0]
            des_acc[i] = d_loss[1]

            g_loss = self.combined.train_on_batch(prior, valid, class_weight=[cw1, cw2])
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
    train_x, train_y, test_x, test_y = build_dataset('../data/old/creditcard_train.csv', '../data/old/creditcard_test.csv', scaler='standard')

    train_x, train_y = augment_with_positives(train_x, train_y, 10000)


    pind = np.argwhere(train_y)
    train_x_pos = train_x[pind]

    print(f'{len(train_x)} - {len(train_x_pos)}')

    sgan = SGAN(INPUT_SHAPE, LATENT_DIM, 2)

    des_losses, gen_losses, des_acc = sgan.train(train_x, train_y, EPOCHS, batch_size=BATCH_SIZE)

    plot_losses(des_losses, gen_losses)
    plot_acc(des_acc)

    sgan.descriminator.save('sgan_des_full_tunning_nodrop.h5')
    sgan.descriminator.save_weights('sgan_des_weights_tunning.h5')

    predictions = sgan.descriminator.predict(test_x, batch_size=1)

    auc_pr = average_precision_score(test_y.to_numpy(), predictions[1][:, 1])
    print(f'AUC-PR: {auc_pr}')

    # descr = load_model('sgan_des_full.h5')
    # predictions = descr.predict(test_x, batch_size=1)
    #
    # auc_pr = average_precision_score(test_y.to_numpy(), predictions[1][:,1])
    #
    # print(f'AUC-PR: {auc_pr}')