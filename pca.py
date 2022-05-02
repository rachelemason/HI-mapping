#!/usr/bin/env python3
#pca.py
#REM 2022-05-02

"""
Contains functions for running PCA on extracted VSWIR spectra, and interpreting the
output
"""

import pickle


def unpickle_spectra(path):
    """
    Read in a file of spectra pickled by spectral_utils.Extract.extract
    """

    with open(path, 'rb') as f:
        spectra = pickle.load(f)

    return spectra

if __name__ == "__main__":
    pass
