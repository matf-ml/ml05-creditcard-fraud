import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sn

if __name__ == '__main__':
    h = pd.read_csv('all_history.txt', sep=';')
    for m, results in h.groupby('outlier_models'):
        print(m, 'AUC-PR: {avg:0.4f} (\u00b1{std:0.4f})'.format(
            avg=results.auc_pr.mean(),
            std=results.auc_pr.std()))

    # fig = plt.figure(1,figsize=(10,12))
    ax = sn.catplot(x='outlier_models', y='auc_pr', data=h, kind='box')


    ax.set_xticklabels(rotation=90)


    plt.show()
