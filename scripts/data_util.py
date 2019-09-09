import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def build_dataset(train_path, test_path, scaler='standard', generated_frauds_path=None):
    '''
        :param train_path - path to train set csv file
        :param test_path - path to test set csv file
        :param scaler - 'standard', 'minmax', None

        :returns train_x, train_y, test_x, test_y
    '''
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    train_x = train.drop(['Time', 'Amount', 'Class'], axis=1)
    test_x = test.drop(['Time', 'Amount', 'Class'], axis=1)
    train_y = train['Class']
    test_y = test['Class']

    if generated_frauds_path != None:
        gen_x = pd.read_csv(generated_frauds_path)
        gen_y = pd.DataFrame(np.ones((len(gen_x.index),))).squeeze(axis=1)
        # print(f'Columns train_x: {train_x.columns}\nColumns gen_x: {gen_x.columns}' )
        # print(f'pre shape x: {train_x.shape}')
        train_x = train_x.append(gen_x)
        # print(f'post shape x: {train_x.shape}')

        # print(f'pre shape y: {train_y.shape}')
        train_y = train_y.append(gen_y)
        # print(f'post shape y: {train_y.shape}')

    if scaler != None:
        if scaler == 'standard':
            sc = StandardScaler().fit(train_x)
        elif scaler == 'minmax':
            sc = MinMaxScaler().fit(train_x)
        train_x = sc.transform(train_x)
        test_x = sc.transform(test_x)

    return train_x, train_y, test_x, test_y

def augment_with_positives(train_x, train_y, n):
    '''
        :param train_x - samples
        :param train_y - labels
        :param n - aprox. number of positives in augmented set

        :returns train_x, train_y
    '''

    pind = np.argwhere(train_y)
    train_x_pos = train_x[pind]

    rno = (n - len(pind)) // len(pind)
    train_x_pos_augmented = np.repeat(train_x_pos, rno, axis=0).squeeze(axis=1)
    train_y_pos_augmented = np.ones((len(train_x_pos_augmented),))
    np.random.shuffle(train_x_pos_augmented)

    train_x = np.append(train_x, train_x_pos_augmented, axis=0)
    train_y = np.append(train_y, train_y_pos_augmented)

    return train_x, train_y

def train_val_split(train_x, train_y, val_percent=.15, stratify=None):
    x_train, x_val, y_train, y_val = train_test_split(train_x,train_y,test_size=val_percent,stratify=stratify)

    return x_train, y_train, x_val, y_val