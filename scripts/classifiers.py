import pandas as pd
from sklearn import metrics
import numpy as np
from sklearn import linear_model
from keras.layers import Dense, Input, LeakyReLU, Dropout
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from scripts.outlier_scores import *
from scripts.data_util import build_dataset, augment_with_positives, train_val_split

SEED = 123
NUM_TESTS = 1

INPUT_DIMENSION = 28

EPOCHS = 100
LR = 1e-4
BETA_1 = 0.5
BATCH_SIZE = 64
ALPHA = 0.1

WEIGHT_NORMAL = 1
WEIGHT_FRAUD = 10

def build_net_decriminator(descriminator):
    # descriminator.summary()
    # descriminator._layers.pop(-2)
    # descriminator._layers.pop(-1)
    input = descriminator.input
    dense1 = descriminator.get_layer('dense_1')
    bn1 = descriminator.get_layer('batch_normalization_1')
    dense2 = descriminator.get_layer('dense_2')
    bn2 = descriminator.get_layer('batch_normalization_2')
    dense3 = descriminator.get_layer('dense_3')
    bn3 = descriminator.get_layer('batch_normalization_3')
    dense4 = descriminator.get_layer('dense_4')
    bn4 = descriminator.get_layer('batch_normalization_4')

    x = dense1(input)
    x = LeakyReLU(ALPHA)(x)
    x = bn1(x)
    # x = Dropout(0.25)(x)
    x = dense2(x)
    x = LeakyReLU(ALPHA)(x)
    x = bn2(x)
    x = Dropout(0.25)(x)
    x = dense3(x)
    x = LeakyReLU(ALPHA)(x)
    x = bn3(x)
    # x = Dropout(0.25)(x)
    x = dense4(x)
    x = LeakyReLU(ALPHA)(x)
    x = bn4(x)

    output = Dense(1, activation='sigmoid', name='output_layer')(x)
    # output = Dense(1, activation='sigmoid', name='Output_Layer')(descriminator.layers[-3].output)
    # descriminator.summary()
    return Model(input, output)


def threshold_proba(pred_prob,num=10):
    class_arr = []
    for t in np.linspace(0,1,num):
        cs = [int(x) for x in pred_prob >= t]
        class_arr.append(cs)
    return class_arr

def evaluate(model, test_x, test_y, save_model=False, save_name=''):
    pred_test_y = model.predict(test_x, batch_size=1)
    # print(pred_test_y)
    # pred_test_y = classifier.predict_proba(test_x_with_scores)[:, 1]
    # pred_class = classifier.predict(test_x_with_scores)

    auc_pr = metrics.average_precision_score(test_y, pred_test_y)

    if save_model:
        model.save(f'{save_name}{auc_pr:0.4f}.h5')

    pr_classes = threshold_proba(pred_test_y, 21)
    f1s = [metrics.f1_score(test_y, pred_y) for pred_y in pr_classes]

    performance_scores = {
        'auc_pr': auc_pr,
        'f1': f1s
    }

    return performance_scores
    # print(f'[Iteration {i + 1}/{NUM_TESTS}] AUC-PR: {auc_pr:0.4f}')


if __name__ == '__main__':
    np.random.seed(123)

    train_x, train_y, test_x, test_y = build_dataset('../data/old/creditcard_train.csv', '../data/old/creditcard_test.csv', scaler='standard')#, generated_frauds_path='../data/small_gen_l3_frauds10000.csv')

    train_x, train_y = augment_with_positives(train_x, train_y, 10000)

    train_x = pd.DataFrame(train_x)
    test_x = pd.DataFrame(test_x)


    # scaler = preprocessing.StandardScaler()
    # scaler.fit(train_x)
    #
    # train_x = scaler.transform(train_x)
    # test_x = scaler.transform(test_x)

    outlier_model_combinations = [
        [],
        # [(ZScore, {})],
        # [(PCA, {'m': 1, 'whiten': True})],
        # [(PCA_RE, {'m': 1, 'whiten': True})],
        # [(IF, {})],
        # [(GM, {'m': 1})],
        # [
        #     (ZScore, {}),
        #     (PCA, {'m': 1, 'whiten': True}),
        #     (PCA_RE, {'m': 1, 'whiten': True}),
        #     (IF, {}),
        #     (GM, {'m': 1})
        # ]
        # [
        #     (PCA_RE, {'m': 1, 'whiten': True}),
        #     (GM, {'m': 1})
        # ],
        # [
        #     (IF, {}),
        #     (GM, {'m': 1})
        # ]
    ]

    # combinations_two = itertools.combinations(outlier_model_combinations, 2)
    # comb = [l+s for l,s in combinations_two]

    seeds = np.random.randint(low=np.iinfo(np.int32).max, size=NUM_TESTS)
    history = []

    for outlier_models_signatures in outlier_model_combinations:
        outlier_models_names = [model(**params).name for model, params in outlier_models_signatures]

        print(f'current outlier models: {outlier_models_names}')

        performance_scores = []

        for i, seed in enumerate(seeds):
            outlier_models = [outlier_model(**{**params, 'random_state': seed})
                              for outlier_model, params in outlier_models_signatures]

            train_outlier_scores_data = {}
            test_outlier_scores_data = {}

            for outlier_model in outlier_models:
                # fit the outlier model on only the non-fraudulent transactions
                outlier_model.fit(train_x.loc[train_y == 0])
                train_outlier_scores = outlier_model.score(train_x)
                test_outlier_scores = outlier_model.score(test_x)
                train_outlier_scores_data[outlier_model.name] = train_outlier_scores
                test_outlier_scores_data[outlier_model.name] = test_outlier_scores

            train_x_with_scores = train_x.join(pd.DataFrame(train_outlier_scores_data))
            train_x_with_scores, train_y = augment_with_positives(train_x_with_scores.to_numpy(), train_y, 15000)
            test_x_with_scores = test_x.join(pd.DataFrame(test_outlier_scores_data))

            train_x_with_scores, train_y, val_x, val_y = train_val_split(train_x_with_scores, train_y, stratify=train_y)

            name = 'trained_sgan_des_full_tunning_1drop'

            chk_callback = ModelCheckpoint(f'best{name}.h5', monitor='val_loss', save_best_only=True)
            es_callback = EarlyStopping(monitor='val_loss', patience=5)
            rlr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

            descriminator = load_model('sgan_des_full_tunning.h5')
            classifier = build_net_decriminator(descriminator)
            classifier.summary()
            classifier.compile(optimizer=Adam(LR, beta_1=BETA_1), loss='binary_crossentropy', metrics=['acc'])
            classifier.fit(train_x_with_scores, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[es_callback, rlr_callback, chk_callback], validation_data=(val_x, val_y), class_weight={0: WEIGHT_NORMAL, 1: WEIGHT_FRAUD}, verbose=2)
            # classifier = ensemble.BalancedRandomForestClassifier(random_state=seed, n_   estimators=100)
            # classifier = linear_model.LogisticRegression(solver='lbfgs', max_iter=5000)
            # classifier.fit(train_x_with_scores, train_y)
            best_classifier = load_model(f'best{name}.h5')

            performance_scores.append(evaluate(classifier, test_x, test_y, save_model=True, save_name=name))
            performance_scores.append(evaluate(best_classifier, test_x, test_y))

        # auc_pr_scores = [score['auc_pr'] for score in performance_scores]
        # mean_auc_pr = np.mean(auc_pr_scores)
        # auc_pr_std = np.std(auc_pr_scores)
        #
        # print(f'avg AUC-PR: {mean_auc_pr} (\u00B1{auc_pr_std})')
        #
        # print(f'F1: {performance_scores}')
        print(performance_scores)
