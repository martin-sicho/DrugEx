#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This script is used for drawing figures shown in the manuscript."""

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from rdkit import Chem, rdBase
from rdkit.Chem import Draw
from sklearn import metrics

from drugex.core.metric import converage, dimension, diversity, logP_mw, properties, substructure, training_process

# configuration for drawing figures on linux
plt.switch_backend('Agg')
# ignoring the warning output by RDkit
rdBase.DisableLog('rdApp.error')


def fig4():
    """Performance of the predictor
        1. ROC curve
        2. Barplot for MCC, Sensitivity, Specificity and Accuracy
    """
    pair = ['LABEL', 'SCORE']
    legends = ['NB', 'RF', 'KNN', 'SVM', 'DNN']
    fnames = ['v2/NB_cls_ecfp6.cv.txt', 'v2/RF_cls_ecfp6.cv.txt',
              'v2/KNN_cls_ecfp6.cv.txt', 'v2/SVM_cls_ecfp6.cv.txt',
              'v2/DNN_cls_ecfp6.cv.txt']
    preds = []
    for fname in fnames:
        df = pd.read_csv(fname)
        preds.append(df[pair].values)
    fig = plt.figure(figsize=(10, 5))
    # ROC curve plot
    ax1 = fig.add_subplot(121)
    lw = 1.5
    for i, pred in enumerate(preds):
        fpr, tpr, ths = metrics.roc_curve(pred[:, 0], pred[:, 1])
        auc = metrics.auc(fpr, tpr)
        ax1.plot(fpr, tpr, lw=lw, label=legends[i] + '(AUC=%.3f)' % auc)
    for i in range(1, 10):
        plt.plot([i * 0.1, i * 0.1], [0, 1], color='gray', lw=lw, linestyle='--')
        plt.plot([0, 1], [i * 0.1, i * 0.1], color='gray', lw=lw, linestyle='--')
    plt.plot([0, 1], [0, 1], color='gray', lw=lw, linestyle='--')
    ax1.set(xlim=[0.0, 1.0], ylim=[0.0, 1.0], xlabel='False Positive Rate', ylabel='True Positive Rate')
    ax1.legend(loc="lower right")

    # Bar plot
    th = 0.5
    ax2 = fig.add_subplot(122)
    for j, pred in enumerate(preds):
        label, score = pred[:, 0], pred[:, 1]
        square = np.zeros((2, 2), dtype=int)
        for i, value in enumerate(score):
            row, col = int(label[i]), int(value > th)
            square[row, col] += 1
        mcc = metrics.matthews_corrcoef(label, score > th)
        sn = square[1, 1] / (square[1, 0] + square[1, 1])
        sp = square[0, 0] / (square[0, 0] + square[0, 1])
        acc = metrics.accuracy_score(label, score > th)
        ax2.bar(np.arange(4) + 0.17 * j, (mcc, sn, sp, acc), 0.17, label=legends[j])
    ax2.set_xticks(np.arange(4) + 0.34)
    ax2.set_xticklabels(('MCC', 'Sensitivity', 'Specificity', 'Accuracy'))
    ax2.set_xlabel('Metric')
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(loc='upper left')

    fig.tight_layout()
    fig.savefig('Figure_4.tif', dpi=300)


def fig5():
    """Training curve plot for pre-trained and fine-tuned model
        1. Pre-trained model
        2. fine-tuned model
    """
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    valid, loss = training_process('v2/net_p.log')
    ax1.plot(valid, label='SMILES validation rate')
    ax1.plot(loss, label='Value of Loss function')
    ax1.set_xlabel('Training Epochs')
    ax1.legend(loc='center right')
    ax1.set(ylim=(0, 1.0), xlim=(0, 1000))

    ax2 = fig.add_subplot(122)
    valid, loss = training_process('net_ex.log')
    ax2.plot([value for i, value in enumerate(valid) if i % 2 == 0], label='SMILES validation rate')
    ax2.plot([value for i, value in enumerate(loss / 100) if i % 2 == 0], label='Value of Loss function')
    ax2.set_xlabel('Training Epochs')
    ax2.legend(loc='center right')
    ax2.set(ylim=(0, 1.0), xlim=(0, 1000))
    fig.tight_layout()
    fig.savefig('Figure_5.tif', dpi=300)


def fig6():
    """ violin plot for the physicochemical proerties comparison.
        A: molecules generated by pre-trained model v.s. ZINC set.
        B: molecules generated by fine-tuned model v.s. A2AR set.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    sns.set(style="white", palette="pastel", color_codes=True)
    df = properties(['data/ZINC_B.txt', 'mol_p.txt'], ['ZINC Dataset', 'Pre-trained Model'])
    sns.violinplot(x='Property', y='Number', hue='Set', data=df, linewidth=1, split=True, bw=1)
    sns.despine(left=True)
    plt.ylim([0.0, 18.0])
    plt.xlabel('Structural Properties')

    plt.subplot(122)
    df = properties(['data/CHEMBL251.txt', 'mol_ex.txt'], ['A2AR Dataset', 'Fine-tuned Model'])
    sns.set(style="white", palette="pastel", color_codes=True)
    sns.violinplot(x='Property', y='Number', hue='Set', data=df, linewidth=1, split=True, bw=1)
    sns.despine(left=True)
    plt.ylim([0.0, 18.0])
    plt.xlabel('Structural Properties')
    plt.tight_layout()
    plt.savefig('Figure_6.tif', dpi=300)


def fig7(*, colors):
    """Chemical space comparison based on logP ~ MW, PCA (with 19D physchem descriptors)
    and t-SNE (with 4096D ECFP6 fingerprints)
    """
    fig = plt.figure(figsize=(12, 12))
    lab = ['ZINC Dataset', 'Pre-trained Model']
    ax1 = fig.add_subplot(221)
    df = logP_mw(['data/ZINC.txt', 'v1/mol_p.txt'])

    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    ax1.scatter(group0.MWT, group0.LOGP, s=10, marker='o', label=lab[0], c='', edgecolor=colors[1])
    ax1.scatter(group1.MWT, group1.LOGP, s=10, marker='o', label=lab[1], c='', edgecolor=colors[3])
    ax1.set(ylabel='LogP', xlabel='Molecular Weight')
    ax1.legend(loc='lower right')

    ax2 = fig.add_subplot(222)
    df, ratio = dimension(['data/ZINC.txt', 'v1/mol_p.txt'], fp='physchem')
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    ax2.scatter(group0.X, group0.Y, s=10, marker='o', label=lab[0], c='', edgecolor=colors[1])
    ax2.scatter(group1.X, group1.Y, s=10, marker='o', label=lab[1], c='', edgecolor=colors[3])
    ax2.set(ylabel='Principal Component 2 (%.2f%%)' % (ratio[1] * 100),
            xlabel='Principal Component 1 (%.2f%%)' % (ratio[0] * 100), ylim=[-10, 15], xlim=[-10, 10])
    ax2.legend(loc='lower right')

    ax2 = fig.add_subplot(233)
    df = dimension(['data/ZINC.txt', 'v1/mol_p.txt'], alg='TSNE')
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    ax2.scatter(group0.X, group0.Y, s=10, marker='o', label=lab[0], c='', edgecolor=colors[1])
    ax2.scatter(group1.X, group1.Y, s=10, marker='o', label=lab[1], c='', edgecolor=colors[3])
    ax2.set(ylabel='Component 2', xlabel='Component 1')
    ax2.legend(loc='lower right')

    lab = ['A2AR Dataset', 'Fine-tuned Model']
    ax3 = fig.add_subplot(223)
    df = logP_mw(['data/CHEMBL251.txt', 'v1/mol_ex.txt'])
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    ax3.scatter(group0.MWT, group0.LOGP, s=10, marker='o', label=lab[0], c='', edgecolor=colors[1])
    ax3.scatter(group1.MWT, group1.LOGP, s=10, marker='o', label=lab[1], c='', edgecolor=colors[3])
    ax3.set(ylabel='LogP', xlabel='Molecular Weight', xlim=[0, 1000], ylim=[-5, 10])
    ax3.legend(loc='lower right')

    ax4 = fig.add_subplot(224)
    df, ratio = dimension(['data/CHEMBL251.txt', 'v1/mol_ex.txt'], fp='physchem')
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    ax4.scatter(group0.X, group0.Y, s=10, marker='o', label=lab[0], c='', edgecolor=colors[1])
    ax4.scatter(group1.X, group1.Y, s=10, marker='o', label=lab[1], c='', edgecolor=colors[3])
    ax4.set(ylabel='Principal Component 2 (%.2f%%)' % (ratio[1] * 100),
            xlabel='Principal Component 1 (%.2f%%)' % (ratio[0] * 100), xlim=[-10, 15], ylim=[-5, 10])
    ax4.legend(loc='lower right')

    ax4 = fig.add_subplot(236)
    df = dimension(['data/CHEMBL251.txt', 'v1/mol_ex.txt'], alg='TSNE')
    group0, group1 = df[df.LABEL == 0], df[df.LABEL == 1]
    ax4.scatter(group0.X, group0.Y, s=10, marker='o', label=lab[0], c='', edgecolor=colors[1])
    ax4.scatter(group1.X, group1.Y, s=10, marker='o', label=lab[1], c='', edgecolor=colors[3])
    ax4.set(ylabel='Component 2', xlabel='Component 1')
    ax4.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('Figure_7.png', dpi=300)


def fig8(*, log_paths, labels, log_paths1):
    """ Training process curve for reinforcement learning
    """
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    df = converage(log_paths)
    for i, label in enumerate(labels):
        ax1.plot(df[df.LABEL == i].SCORE.values, label=label)
    ax1.set(ylabel='Average Score', xlabel='Training Epochs')

    ax2 = fig.add_subplot(122)
    df = converage(log_paths1)
    for i, label in enumerate(labels):
        ax2.plot(df[df.LABEL == i].SCORE.values, label=label)
    ax2.set(ylabel='Average Score', xlabel='Training Epochs')
    fig.tight_layout()
    fig.savefig('Figure_8.tif', dpi=300)


def fig9(*, mol_paths, real_path, labels, real_label, mol_paths1):
    """ violin plot for the physicochemical proerties comparison.
            1: molecules generated by DrugEx with pre-trained model as exploration network.
            2: molecules generated by DrugEx with fine-tuned model as exploration network.
        """
    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(211)
    sns.set(style="white", palette="pastel", color_codes=True)
    df = properties(mol_paths + real_path, labels + real_label, is_active=True)
    sns.violinplot(x='Property', y='Number', hue='Set', data=df, linewidth=1, bw=0.8)
    sns.despine(left=True)
    ax1.set(ylim=[0.0, 15.0], xlabel='Structural Properties')

    ax2 = fig.add_subplot(212)
    df = properties(mol_paths1 + real_path, labels + real_label, is_active=True)
    sns.set(style="white", palette="pastel", color_codes=True)
    sns.violinplot(x='Property', y='Number', hue='Set', data=df, linewidth=1, bw=0.8)
    sns.despine(left=True)
    ax2.set(ylim=[0.0, 15.0], xlabel='Structural Properties')
    fig.tight_layout()
    fig.savefig('Figure_9.tif', dpi=300)


def fig10(*, real_path, colors):
    """Chemical space comparison based on logP ~ MW, PCA (with 19D physchem descriptors)
        and t-SNE (with 4096D ECFP6 fingerprints)
    """
    fnames = real_path + ['v2/mol_e_10_1_500x10.txt', 'v2/mol_a_10_1_500x10.txt',
                          'v2/mol_REINVENT_p_ex.txt', 'v1/mol_gan_5_0_500x10.txt']
    fig = plt.figure(figsize=(15, 20))
    legends = ['Active Ligands', 'DrugEx(Fine-tuned)', 'DrugEx(Pre-trained)', 'REINVENT', 'ORGANIC']

    df = logP_mw(fnames, is_active=True)
    group0 = df[df.LABEL == 0]
    for i in range(1, len(legends)):
        ax = fig.add_subplot(4, 3, i*3-2)
        group1 = df[df.LABEL == i]
        ax.scatter(group1.MWT, group1.LOGP, s=10, marker='o', label=legends[i], c='', edgecolor=colors[i])
        ax.scatter(group0.MWT, group0.LOGP, s=10, marker='o', label=legends[0], c='', edgecolor=colors[0])
        ax.set(ylabel='LogP', xlabel='Molecular Weight', xlim=[0, 1000], ylim=[-5, 10])
        ax.legend(loc='lower right')

    df, ratio = dimension(fnames, is_active=True, fp='phychem')
    group0 = df[df.LABEL == 0]
    for i in range(1, len(legends)):
        ax = fig.add_subplot(4, 3, i*3-1)
        group1 = df[df.LABEL == i]
        ax.scatter(group1.X, group1.Y, s=10, marker='o', label=legends[i], c='', edgecolor=colors[i])
        ax.scatter(group0.X, group0.Y, s=10, marker='o', label=legends[0], c='', edgecolor=colors[0])
        ax.set(ylabel='Principal Component 2 (%.2f%%)' % (ratio[1] * 100),
               xlabel='Principal Component 1 (%.2f%%)' % (ratio[0] * 100),
               xlim=[-20, 30], ylim=[-10, 10])
        ax.legend(loc='lower right')

    df = dimension(fnames, is_active=True, alg='TSNE')
    group0 = df[df.LABEL == 0]
    for i in range(1, len(legends)):
        ax = fig.add_subplot(4, 3, i*3)
        group1 = df[df.LABEL == i]
        ax.scatter(group1.X, group1.Y, s=10, marker='o', label=legends[i], c='', edgecolor=colors[i])
        ax.scatter(group0.X, group0.Y, s=10, marker='o', label=legends[0], c='', edgecolor=colors[0])
        ax.set(ylabel='Component 2', xlabel='Component 1', xlim=[-100, 100], ylim=[-100, 100])
        ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig('Figure_10.png', dpi=300)


def fig11():
    """ Generated molecules exhibition based on probability score (X-axis) and
        Tanimoto distance (Y-axis).
    """
    dist = diversity('mol_e_10_1_500x10.txt', 'data/CHEMBL251.txt')
    dist.to_csv('distance.txt', index=None, sep='\t')

    df = pd.read_table('distance.txt')
    dists = [0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    scores = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mols = []
    for i, dist in enumerate(dists):
        if i == len(dists) - 1: continue
        samples = df[(df.DIST > dist) & (df.DIST < dists[i + 1])].sort_values("SCORE")
        for j, score in enumerate(scores):
            if j == len(scores) - 1: continue
            sample = samples[(samples.SCORE > score) & (samples.SCORE < scores[j+1])]
            if len(sample) > 0:
                sample = sample.sample(1)
                print(sample.values)
                mols += [Chem.MolFromSmiles(smile) for smile in sample.CANONICAL_SMILES]
    img = Draw.MolsToGridImage(mols, molsPerRow=5, subImgSize=(400, 300))
    img.save('Figure_11_%f.tif' % (dist))


def figS1(*, colors):
    """ Chemical space exhibation (logP ~ Molecular weight ) for generated molecules
    trained with canonical REINFORCE algorith.
    """
    df = logP_mw(['mol_p.txt', 'v2/sample_agent_without_ex.txt', 'data/CHEMBL251.txt'])
    labs = ['Pre-trained model', 'Reinforced model', 'Active Ligands']
    plt.figure(figsize=(6, 6))
    groups = df.groupby('LABEL')
    for i, group in groups:
        plt.scatter(group.MWT, group.LOGP, s=10, marker='o', label=labs[i], c='', edgecolor=colors[i])
    plt.ylabel('LogP')
    plt.xlabel('Molecular Weight')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('Figure_S1.tif', dpi=300)


def figS2():
    """Line plot for the performance of DrugEx with a range of exploration rate"""
    df = pd.read_table('epsilon.txt')
    fig = plt.figure(figsize=(8, 8))
    columns = ['valid', 'desired', 'unique', 'diversity']
    labels = ['Valid SMILES (%)', 'Desired SMILES (%)', 'Unique desired SMILES (%)', 'Diversity']
    for i, column in enumerate(columns):
        ax = fig.add_subplot(221 + i)
        for a in ['Pre-trained', 'Fine-tuned']:
            data = df[df['exploration'] == a]
            baselines = data.groupby('baseline')
            for b, baseine in baselines:
                ax.plot(baseine.epsilon, baseine[column], label='%s(β = %.1f)' % (a, b))
        ax.legend(loc='lower left')
        ax.set(ylabel=labels[i], xlabel='Epsilon', xlim=[0, 0.26], ylim=[0.65, 0.85] if column == 'diversity' else [30, 100])
    fig.tight_layout()
    fig.savefig('Figure_S2.tif', dpi=300)


def figS3():
    """ K-means clustering algorithm on ECFP6 fingerprints to separate
    generated molecules and known active ligands into 20 groups. The bar
    plot shows the percentage of molecules in each clusters.
    1. ECFP6 of full compound structure;
    2. ECFP6 of Murcko scaffold;
    3. ECFP6 of Murcko topological scaffold.
    """
    df = pd.read_table('cluster.txt')
    columns = ['FULL_COMPOUND', 'MURCKO_SCAFFOLD', 'MURCKO_TOPOLOGICAL_SCAFFOLD']
    labels = ['Full Compound (%)', 'Murcko Scaffold (%)', 'Topological Murcko Scaffold (%)']
    legends = ['DrugEx(Fine-tuned)', 'DrugEx(Pre-trained)', 'Active Ligands', 'REINVENT', 'ORGANIC']
    fig = plt.figure(figsize=(12, 9))
    for i, column in enumerate(columns):
        ax = fig.add_subplot(3, 1, i + 1)
        for j, legend in enumerate(legends):
            data = df[df['LABEL'] == legend]
            ax.bar(np.arange(20) + 0.15 * j, data[column], 0.15, label=legend)
            ax.set_xticks(np.arange(20)+0.30)
            ax.set_xticklabels(np.arange(1, 21))
            ax.set_xlabel('Clusters')
            ax.set_ylabel(labels[i])
            ax.set_xlim(-0.5, 20)
            ax.set_ylim(0.0, 50.0)
            ax.legend(loc='upper right')
            plt.xticks()
    fig.tight_layout()
    fig.savefig('Figure_S3.tif', dpi=300)


def main():
    colors = ['#ff7f0e', '#1f77b4', '#d62728', '#2ca02c', '#9467bd']  # orange, blue, green, red, purple
    pkg_paths = ['net_a_1_0_500x10.pkg', 'net_a_1_1_500x10.pkg',
                 'net_a_10_0_500x10.pkg', 'net_a_10_1_500x10.pkg']
    log_paths = ['net_a_1_0_500x10.log', 'net_a_1_1_500x10.log',
                 'net_a_10_0_500x10.log', 'net_a_10_1_500x10.log', ]
    mol_paths = ['mol_a_1_0_500x10.txt', 'mol_a_1_1_500x10.txt',
                 'mol_a_10_0_500x10.txt', 'mol_a_10_1_500x10.txt', ]
    pkg_paths1 = ['net_e_1_0_500x10.pkg', 'net_e_1_1_500x10.pkg',
                  'net_e_10_0_500x10.pkg', 'net_e_10_1_500x10.pkg']
    log_paths1 = ['net_e_1_0_500x10.log', 'net_e_1_1_500x10.log',
                  'net_e_10_0_500x10.log', 'net_e_10_1_500x10.log', ]
    mol_paths1 = ['mol_e_1_0_500x10.txt', 'mol_e_1_1_500x10.txt',
                  'mol_e_10_0_500x10.txt', 'mol_e_10_1_500x10.txt', ]
    labels = ["ε = 0.01, β = 0.0", "ε = 0.01, β = 0.1",
              "ε = 0.1, β = 0.0", "ε = 0.1, β = 0.1"]
    real_path = ['data/CHEMBL251.txt']
    real_label = ['Active Ligands']

    fig4()
    fig5()
    fig6()
    fig7(colors=colors)
    fig8(labels=labels, log_paths=log_paths, log_paths1=log_paths1)
    fig9(labels=labels, mol_paths=mol_paths, mol_paths1=mol_paths1, real_label=real_label, real_path=real_path)
    fig10(colors=colors, real_path=real_path)
    fig11()

    # Table 2
    for sub in ['[R2][R2]', 'c1cocc1', 'c1ccccc1']:
        substructure(['mol_l.txt', 'data/CHEMBL251.txt', 'mol_v.txt'] +
            ['mol_dv_20_1_500x10.txt', 'mol_dv_30_1_500x10.txt',
             'mol_REINVENT.txt', 'mol_gan_5_0_500x10.txt'], sub)
    figS2()
    figS3()


if __name__ == '__main__':
    main()
