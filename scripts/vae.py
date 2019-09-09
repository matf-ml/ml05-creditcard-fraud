from keras.layers import Dense, Input
from keras.layers import Lambda, BatchNormalization
from keras.models import Model, load_model
from keras.losses import binary_crossentropy, mse
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import average_precision_score
from imblearn import ensemble

from scripts.data_util import build_dataset

INPUT_DIM = 28

EPOCHS = 50
LR = 1e-3
BATCH_SIZE = 128
LATENT_DIM = 20
BETA = 1

# N_SAMPLES = 500

def build_model():
    def reparameterization_trick(args):
        mean, log_var = args
        batch = K.shape(mean)[0]
        dim = K.int_shape(mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return mean + K.exp(0.5 * log_var) * epsilon

    def vae_loss(y_true, y_pred):
        reconstruction_loss = INPUT_DIM*mse(y_true, y_pred)#binary_crossentropy(y_true, y_pred)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(reconstruction_loss + BETA*kl_loss)

    inputs = Input(shape=(INPUT_DIM,))

    encoding = Dense(256, activation='relu')(inputs)
    encoding = BatchNormalization()(encoding)
    encoding = Dense(128, activation='relu')(encoding)
    encoding = BatchNormalization()(encoding)
    encoding = Dense(64, activation='relu')(encoding)
    encoding = BatchNormalization()(encoding)

    z_mean = Dense(LATENT_DIM)(encoding)
    z_log_var = Dense(LATENT_DIM)(encoding)

    z = Lambda(reparameterization_trick, output_shape=(LATENT_DIM,))([z_mean, z_log_var])

    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()

    latent_inputs = Input(shape=(LATENT_DIM,))

    decoding = Dense(64, activation='relu')(latent_inputs)
    decoding = BatchNormalization()(decoding)
    decoding = Dense(128, activation='relu')(decoding)
    decoding = BatchNormalization()(decoding)
    decoding = Dense(256, activation='relu')(decoding)
    decoding = BatchNormalization()(decoding)

    outputs = Dense(INPUT_DIM)(decoding)

    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()

    outputs = decoder(encoder(inputs)[2])

    vae = Model(inputs, outputs, name='vae')

    vae.compile(optimizer=Adam(LR), loss=vae_loss)
    vae.summary()

    return encoder, decoder, vae

NUM_TESTS = 20

def main_hybrid():
    encoder, decoder, vae = build_model()

    encoder.load_weights('models/wamencoder16.h5')

    train_x, train_y, test_x, test_y = build_dataset()

    etrain_x = encoder.predict(train_x)[0]
    etest_x = encoder.predict(test_x)[0]

    seeds = np.random.randint(low=np.iinfo(np.int32).max, size=NUM_TESTS)

    performance_scores = []

    for i, seed in enumerate(seeds):
        classifier = ensemble.BalancedRandomForestClassifier(random_state=seed)
        classifier.fit(etrain_x, train_y)

        classifier_whole = ensemble.BalancedRandomForestClassifier(random_state=seed)
        classifier_whole.fit(train_x, train_y)

        pred_test_y = classifier.predict_proba(etest_x)[:,1]
        # pred_class = classifier.predict(test_x_with_scores)

        pred_test_y_whole = classifier_whole.predict_proba(test_x)[:,1]

        auc_pr = average_precision_score(test_y, pred_test_y)
        auc_pr_whole = average_precision_score(test_y, pred_test_y_whole)

        performance_scores.append({
            'auc_pr': auc_pr,
            'auc_pr_whole': auc_pr_whole
        })

        print(f'[Iteration {i + 1}/{NUM_TESTS}] AUC-PR: {auc_pr:0.4f} whole: {auc_pr_whole:0.4f}')

    auc_pr_scores = [score['auc_pr'] for score in performance_scores]
    mean_auc_pr = np.mean(auc_pr_scores)
    auc_pr_std = np.std(auc_pr_scores)

    auc_pr_scores_whole = [score['auc_pr_whole'] for score in performance_scores]
    mean_auc_pr_whole = np.mean(auc_pr_scores_whole)
    auc_pr_std_whole = np.std(auc_pr_scores_whole)

    print(f'avg AUC-PR: {mean_auc_pr} (\u00B1{auc_pr_std}) whole: {mean_auc_pr_whole} (\u00B1{auc_pr_std_whole})')


def main_gen():
    encoder, decoder, vae = build_model()

    train_x, train_y, test_x, test_y = build_dataset('../data/old/creditcard_train.csv', '../data/old/creditcard_test.csv',
                                                     'standard')

    pind = np.argwhere(train_y)
    train_x_pos = np.squeeze(train_x[pind], axis=1)

    print(f'{len(train_x)} - {len(train_x_pos)}')
    print(train_x_pos.shape)

    tb_callback = TensorBoard(log_dir='logdef generate_sample(self, n=1):s/', batch_size=1)
    # es_callback = EarlyStopping(monitor='val_loss', patience=5)
    # mc_callback = ModelCheckpoint('models/vae.h5', save_best_only=True, save_weights_only=False)
    rlr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

    # train_x_neg = train_x[train_y==0]
    # res_x_pos = train_x[train_y==1]
    vae.fit(train_x_pos, train_x_pos,
            # validation_data=(test_x, test_x),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[tb_callback, rlr_callback])#, mc_callback, es_callback])

    # vae.save_weights('models/wamvae'+str(LATENT_DIM)+'.h5')
    # encoder.save_weights('models/wamencoder'+str(LATENT_DIM)+'.h5')
    decoder.save_weights('models/POSdecoder' + str(LATENT_DIM) + '.h5')

    for n in [500, 1000, 5000, 10000]:
        latent_sample = np.random.normal(0, 1, (n, LATENT_DIM))

        generated_samples = pd.DataFrame(decoder.predict(latent_sample),
                                         columns=['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',
                                                  'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
                                                  'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28'])

        print(generated_samples.size)

        generated_samples.to_csv('../data/vae_frauds' + str(n) + '.csv', index=False)


def main():
    encoder, decoder, vae = build_model()

    train_x, train_y, test_x, test_y = build_dataset('../data/old/creditcard_train.csv', '../data/old/creditcard_test.csv',
                                                     'standard')

    pind = np.argwhere(train_y)
    train_x_pos = train_x[pind]

    print(f'{len(train_x)} - {len(train_x_pos)}')

    tb_callback = TensorBoard(log_dir='logs/', batch_size=1)
    es_callback = EarlyStopping(monitor='val_loss', patience=5)
    mc_callback = ModelCheckpoint('models/vae.h5', save_best_only=True, save_weights_only=False)
    rlr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

    train_x_neg = train_x[train_y==0]
    res_x_pos = train_x[train_y==1]
    vae.fit(train_x_neg, train_x_neg,
            validation_data=(test_x, test_x),
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            callbacks=[tb_callback, mc_callback, rlr_callback, es_callback])

    # vae.save_weights('models/wamvae'+str(LATENT_DIM)+'.h5')
    # encoder.save_weights('models/wamencoder'+str(LATENT_DIM)+'.h5')
    # decoder.save_weights('models/POSdecoder'+str(LATENT_DIM)+'.h5')

    if LATENT_DIM == 2:
        embedding = encoder.predict(test_x)
        # res_embed = encoder.predict(res_x_pos)

        test_y = np.array(test_y)
        plt.scatter(embedding[0][:,0][test_y==0], embedding[0][:,1][test_y == 0], c='b', s=5)
        plt.scatter(embedding[0][:,0][test_y==1], embedding[0][:,1][test_y==1], c='r', s=5)
        # plt.scatter(res_embed[0][:,0], res_embed[0][:,1], c='y', marker='x', alpha=0.5, s=10)

        plt.show()

        plt.clf()

    ev = vae.predict(test_x,batch_size=1)
    ev_res = vae.predict(res_x_pos,batch_size=1)
    losses = np.mean((ev - test_x)*(ev - test_x),axis=1)
    losses_res = np.mean((ev_res - res_x_pos)*(ev_res - res_x_pos),axis=1)

    pind = np.argwhere(test_y)
    nind = np.argwhere(test_y==0)

    plt.scatter(nind, losses[nind], c='b', s=5)
    plt.scatter(np.linspace(0,70000,len(losses_res)), losses_res, c='r', s=5)
    plt.scatter(pind, losses[pind], c='r', s=5)

    plt.show()

    losses_pred = np.append(losses,losses_res)
    losses_res_y = np.ones(losses_res.shape)
    # print(f'test_y: {test_y.shape}\nlosses_res_y: {losses_res_y.shape}')
    classes = np.append(test_y,losses_res_y)

    print(f'losses_pred: {losses_pred.shape}\nclasses: {classes.shape}')

    # losses_pred_normalized = normalizer.fit_transform(X=losses_pred.reshape(-1,1))
    auc_pr = average_precision_score(classes, losses_pred)
    # auc_pr_n = average_precision_score(classes, losses_pred_normalized)
    # auc_pr = average_precision_score(test_y, losses)

    print(f'AUC_PR: {auc_pr}')

if __name__ == '__main__':
    main()
    # main_hybrid()
    # main_gen()