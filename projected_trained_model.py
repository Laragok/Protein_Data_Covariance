#! /usr/bin/env python3

"""Read in protein data files [related to bla bla], extract a column,
and train a model on the projected [bla bla bla]."""

import os
import sys
import math

import numpy as np
import matplotlib.pyplot as plt

import pylab
import scipy.stats as stats

import pandas as pd
#import numpy as np
import seaborn as sns
import sklearn
from sklearn import datasets, linear_model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score   
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.svm import SVC
from sklearn.metrics import r2_score
from scipy.spatial.distance import pdist, squareform
from Bio import SeqIO
from scipy.cluster import hierarchy
import h5py
# vector graphics display (makes things less blurry)
#get_ipython().run_line_magic('config', "InlineBackend.figure_formats = ['svg']")
#get_ipython().run_line_magic('matplotlib', 'inline')

aminoacids_ordered = pd.read_csv(os.path.join("Projected_Protein_Data_Covariance",
                                              "sorted_amino_acids.txt"),
                                 names=["amino_acids"]).amino_acids.to_numpy().astype(str)

# def match_aminoacids_ordered(protein_sequence, chars=aminoacids_ordered):
def match_aminoacids_ordered(protein_sequence, chars=aminoacids_ordered):
    """
    compares each pair of amino-acid at position in protein against
    our library of amino-acids (returns one hot)
    """
    # return np.equal.outer(chars, protein_sequence).astype(np.float64)
    return np.equal.outer(chars, protein_sequence, dtype=object).astype(np.float64)


def main():
    """Main flow of the program: load the data, then invoke the training
    on the specific column we are interested in.
    """
    column_name = "states_0_brightness"
    if len(sys.argv) > 2:
        raise Exception(f'usage: {sys.argv[0]} [column_name]')
    elif len(sys.argv) == 2:
        column_name = sys.argv[1]
    else:
        print('# using default column name')
    print(f'##COLUMN_NAME: {column_name}')
    V512, df_orig = load_protein_data()
    do_training(V512, df_orig, column_name)
    

def load_protein_data():
    ###reading the data file
    fpath = os.path.join("Projected_Protein_Data_Covariance",
                         "ohe_projection_matrices_truncated_512positions.npz")
    assert(os.path.exists(fpath))
    data = np.load(fpath)

    with np.load(os.path.join("Projected_Protein_Data_Covariance",
                               "ohe_projection_matrices_truncated_512positions.npz")) as data:
        print("keys in file:")
        for k in data.keys():
            print(f"   {k}")
        total_number_of_proteins = data["total_number_of_proteins"].item()
        evals = data["evals"]
        V_512 = data["evecs_512"]

    print('total_proteins:', total_number_of_proteins)

    print(V_512.shape) # features by components

    print('aminoacids_ordered:', aminoacids_ordered)
    # check that equality of the types is correct A == A
    assert("A" == aminoacids_ordered[0])

    # df = pd.read_csv(os.path.join(ddir , "basic-monomeric.csv"), delimiter='\t')
    df_orig = pd.read_csv(os.path.join("Projected_Protein_Data_Covariance",
                                       "basic-monomeric.csv"), delimiter='\t')
    print('df_orig.shape:', df_orig.shape)
    df_orig.to_csv('df_orig.csv')
    return V_512, df_orig

def do_training(V_512, df_orig, col_name):
    df = df_orig.fillna(0)
    print('df.shape:', df.shape)
    print('JUST_AFTER_FILLNA:', df[col_name].to_string())
    # df = df_orig.dropna(subset=["states_0_brightness"])
    # print('df.shape:', df.shape)
    # print('JUST_AFTER_DROPNA:', df["states_0_brightness"].to_string())
    df.to_csv('df.csv')

    # array of amino acids in a certain protein ( sirius aequorea victoria row(1) )
    aminoacids = set(df.seq[0])   ###sets the aminoacids variable to the 1 row of seq column
    aminoacids.add('*')
    aminoacids = np.array(list(aminoacids))   ###turns the 1 line of seq column into a python list
    aminoacids.sort()
    aminoacids # all amino acids plus termination

    ### generating matrices
    #                      add termination char        left justify    split all chars  to numpy array
    leftjustified_seqs = (df.seq.astype(str) + "*").str.ljust(512, " ").apply(list).apply(np.array)
    # vertically concatenate all proteins
    leftjustified_seqs = np.vstack(leftjustified_seqs)

    ###left_justified_seqs are fixed to the size of 512 with this function
    print('leftjustified_seqs.shape:', leftjustified_seqs.shape)
    ###227 proteins with 512 amino acid positions each

    # print('leftjustified_seqs:', leftjustified_seqs)
    # print('leftjustified_seqs.shape:', leftjustified_seqs.shape)
    # print('seqs:', leftjustified_seqs[1])
    print('df.shape:', df.shape)
    print('leftjustified_seqs.shape:', leftjustified_seqs.shape)
    # protein_ohe = np.apply_along_axis(match_aminoacids_ordered, 1,
    #                                   leftjustified_seqs).reshape(leftjustified_seqs.shape[0], -1)

    protein_ohe_ordered = np.apply_along_axis(match_aminoacids_ordered, 1,
                                              leftjustified_seqs).reshape(df.shape[0], -1)


    print('df.seq[0]:', df.seq[0])
    
    # # Confirm that our ohe features are in the correct locations to extract the correct feature names 
    # feature_names_512.feature_names[np.nonzero(protein_ohe_ordered[0, :])[0]].sort_values()
    
    # proteins x features (512 position * 21 amino-acids)
    print('protein_ohe_ordered.shape:', protein_ohe_ordered.shape)
    print('V_512.shape', V_512.shape)
    
    proteins_projected_pc_coords = protein_ohe_ordered @ V_512
    # protein in same order as fpbase x principal component
    print('proteins_projected_pc_coords.shape:', proteins_projected_pc_coords.shape)
    
    f, ax = plt.subplots()
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.scatter(proteins_projected_pc_coords[:,0], proteins_projected_pc_coords[:,1]);
    fname = f'{col_name}_proteins_projected_pc_coords_before_change.png'
    print(f'##SAVE_PLOT_TO: {fname}')
    plt.savefig(fname)
    # plt.draw()
    plt.clf()
    
    # u, s, vt = np.linalg.svd(protein_ohe)
    u, s, vt = np.linalg.svd(protein_ohe_ordered)
    # matrix multiplication (this is just scaling each left singular vector, by it singular value) 
    proteins_projected_pc_coords = u @ np.diag(s)
    print('proteins_projected_pc_coords.shape:', proteins_projected_pc_coords.shape)

    # rows correspond to each protein in same order as 
    print('proteins_projected_pc_coords.shape:', proteins_projected_pc_coords.shape)

    f, ax = plt.subplots()
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.scatter(proteins_projected_pc_coords[:,0], proteins_projected_pc_coords[:,1]);
    fname = f'{col_name}_proteins_projected_pc_coords_post_change.png'
    print(f'##SAVE_PLOT_TO: {fname}')
    plt.savefig(fname)
    plt.clf()
    
    kf = KFold(n_splits=10)
    print('kf:', kf)
    
    ###setting up the k-fold cross validation
    X = proteins_projected_pc_coords
    print('X.shape:', X.shape)
    y = df["states_0_brightness"]
    print('df.shape:', df.shape)
    print('y.shape:', y.shape)

    # prepares train and test sets
    for seq_no, (train_index, test_index) in enumerate(kf.split(X)):
        print("SEQ:", seq_no, "TRAIN.shape:", train_index.shape, "TEST:", test_index.shape)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        make_predictions(X_train, X_test, y_train, y_test, col_name, seq_no)

def make_predictions(X_train, X_test, y_train, y_test, col_name, seq_no):
    """Take the training and testing data we have been given, and use the
    sklearn "lasso" trainer to predict results.  Then we also compare
    predictions to test data.
    """
    ###predicting r2 value for test set
    lasso = linear_model.Lasso()
    print('X_train.shape:', X_train.shape)
    print('y_train.shape:', y_train.shape)
    lasso.fit(X_train,y_train)
    y_pred = lasso.predict(X_test)
    score = lasso.score(X_test ,y_test)
    print('training set score:', score)
    rsq = r2_score(y_test, y_pred)
    print('test set rsq:', rsq)

    plot_predictions(y_pred, y_test, col_name, seq_no)

    ###analytical analyis of the predictions vs real value
    print("Mean Absolute Error: " , mean_absolute_error(y_test, y_pred))
    print("Mean Squared Error: " , mean_squared_error(y_test, y_pred))
    print("RMSE: " , math.sqrt(mean_squared_error(y_test, y_pred)))

    plot_residuals(y_pred, y_test, col_name, seq_no)
        

def plot_predictions(y_pred, y_test, col_name, seq_no):
    """plotting the predictions"""
    sns.scatterplot(x=y_pred , y=y_test)
    plt.xlabel("Predictions")
    # plt.ylabel("Brightness ")
    plt.ylabel(col_name)
    plt.title("Predictions Of Brightness Using A Trained Model")
    fname = f'{col_name}_saved_prediction_{seq_no}.png'
    print(f'##SAVE_PLOT_TO: {fname}')
    plt.savefig(fname)
    plt.clf()

def plot_residuals(y_pred, y_test, col_name, seq_no):
    """residuals(errors)"""
    residuals = y_test - y_pred
    sns.displot(residuals, bins=30)
    plt.title("Residual Distribution")
    plt.xlabel(col_name)
    fname = f'{col_name}_residual_plot_{seq_no}.png'
    print(f'##SAVE_PLOT_TO: {fname}')
    pylab.savefig(fname)
    pylab.clf()
    ###checking for bias in residuals(errors)
    ###plotting residuuals
    stats.probplot(residuals , dist="norm", plot=pylab)
    fname = f'{col_name}_probplot_{seq_no}.png'
    print(f'##SAVE_PLOT_TO: {fname}')
    pylab.savefig(fname)
    # pylab.show()
    pylab.clf()


if __name__ == '__main__':
    main()
