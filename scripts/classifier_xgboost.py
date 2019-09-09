import pandas as pd
from imblearn import combine
from sklearn import metrics, model_selection
import xgboost as xgb
from scripts.data_util import build_dataset, train_val_split

from scripts.outlier_scores import *

SEED = 123
NUM_TESTS = 1


def to_dataset(creditcard_data):
    x = creditcard_data.drop(columns=['Time', 'Class'])
    y = creditcard_data['Class']
    return x, y

def threshold_proba(pred_prob,num=10):
    class_arr = []
    for t in np.linspace(0,1,num):
        cs = [int(x) for x in pred_prob >= t]
        class_arr.append(cs)
    return class_arr


if __name__ == '__main__':
    np.random.seed(123)

    train_x, train_y, test_x, test_y = build_dataset('../data/old/creditcard_train.csv', '../data/old/creditcard_test.csv',
                                                     scaler='standard')#,
                                                     #generated_frauds_path='../data/gen_frauds10000.csv')

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
    ]

    seeds = np.random.randint(low=np.iinfo(np.int32).max, size=NUM_TESTS)
    history = []

    params = {
        'eta': 0.3,
        'max_depth': 5,
        'num_class': 2,
        # 'n_estimators': 200,
        'learning_rate': 0.1,
        'objective': 'multi:softprob'
    }

    steps = 500

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
                # print(train_y == 0)
                # print(train_x.loc[train_y == 0)])

                outlier_model.fit(train_x.loc[train_y == 0])
                train_outlier_scores = outlier_model.score(train_x)
                test_outlier_scores = outlier_model.score(test_x)
                train_outlier_scores_data[outlier_model.name] = train_outlier_scores
                test_outlier_scores_data[outlier_model.name] = test_outlier_scores

            train_x, train_y, val_x, val_y = train_val_split(train_x, train_y, stratify=train_y)

            train = xgb.DMatrix(train_x, label=train_y)
            test = xgb.DMatrix(test_x, label=test_y)
            val = xgb.DMatrix(val_x, label=val_y)

            gs_params = {
                'max_depth': range(3,15,2),
                'min_child_weight': range(1,6,2)
            }

            classifier = xgb.train(params,train,steps)
            # classifier = xgb.XGBClassifier(seed=seed, n_jobs=4)
            # classifier.fit(train_x_with_scores, train_y)


            pred_test_y = classifier.predict(test)[:,1]
            pred_val_y = classifier.predict(val)[:, 1]
            # pred_test_y = classifier.predict_proba(test_x_with_scores)[:,1]
            print(pred_test_y)

            auc_pr = metrics.average_precision_score(test_y, pred_test_y)

            pr_classes = threshold_proba(pred_val_y,51)
            f1s = [metrics.f1_score(val_y, pred_y) for pred_y in pr_classes]
            print(f'MAX F1 {max(f1s)} index: {f1s.index(max(f1s))}')

            performance_scores.append({
                'auc_pr': auc_pr,
                'f1' : f1s
            })

            print(f'[Iteration {i + 1}/{NUM_TESTS}] AUC-PR: {auc_pr:0.4f}')

        auc_pr_scores = [score['auc_pr'] for score in performance_scores]
        mean_auc_pr = np.mean(auc_pr_scores)
        auc_pr_std = np.std(auc_pr_scores)

        print(f'avg AUC-PR: {mean_auc_pr} (\u00B1{auc_pr_std})')

        print(f"F1: {performance_scores}")
