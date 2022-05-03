#!/usr/bin/env python3
#pca.py
#REM 2022-05-02

"""
Contains classes for running PCA on extracted VSWIR spectra, and interpreting the
output
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as skl_PCA


class PCA():
    """
    Read and standardize spectra
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.spectra = None #defined in unpickle_spectra()


    def unpickle_spectra(self):
        """
        Read in a file of spectra pickled by spectral_utils.Extract.extract. The
        resulting self.spectra is a dictionary where each key is a tuple containing a
        feature ID number and category label (e.g. (1, 'Roof')) and each value is
        the feature's spectrum as a list of floats.
        """

        with open(self.data_path, 'rb') as f:
            self.spectra = pickle.load(f)


    def run_pca(self):
        """
        For each band in each spectrum, subtract the mean of all spectra in
        that band, and divide by the std. dev. of all spectra in that band (i.e,
        create z-score spectra)
        """

        labels = [label[1] for label in self.spectra.keys()]

        #stack the spectra into an array with len(self.spectra) rows and 214 columns
        X = np.vstack(list(self.spectra.values()))

        #remove columns that contain only NaNs
        #these are bands with poor H2O correction, so are NaN in all spectra;
        #we don't seem to have bad pixels where not all spectra would contain NaN
        X = X[:,~np.all(np.isnan(X), axis=0)]

        #standardize (create z-score spectra)
        X = StandardScaler().fit_transform(X)

        #run the PCA
        pca = skl_PCA()

        #pca.fit() to get properties such as explained_variance_ratio_
        properties = pca.fit(X)

        #pca.fit_transform to get the n_components components for each spectrum
        comps = pca.fit_transform(X)

        return [properties, comps, labels]


class Diagnostics():
    """
    Methods to make plots etc. that illustrate/explain the PCA results
    """

    def __init__(self, pca_results, plot_path):
        self.pc_props = pca_results[0]
        self.comps = pca_results[1]
        self.labels = pca_results[2]
        self.plot_path = plot_path


    def explained_variance(self, fname=None):
        """
        Plot the variance explained by each component
        """

        varex = self.pc_props.explained_variance_ratio_[:-1]

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        ax1.plot(varex, '-*', color='k', label='Explained variance')
        ax1.plot(np.cumsum(varex), '-*', color='blue', label='Cumulative')
        ax1.legend()
        ax1.set_xlabel('Component')
        ax1.set_ylabel('(Cumulative) variance explained')

        ax2.semilogy(varex, '-*', color='k') 
        ax2.set_xlabel('Component')
        ax2.set_ylabel('Variance explained')

        if fname is not None:
            plt.savefig(self.plot_path+fname)


    @staticmethod
    def colordict():
        """
        Dictionary containing key=spectrum label, value=color (e.g. 'Kukui': 'yellowgreen').
        Used by self.biplot() to color PCA biplots.
        """


        return {'Roof': 'k', 'Pool': 'cyan', 'Solar panels': 'magenta', 'Road': '0.7',\
                'Lava': 'k', 'Probably grass': 'khaki', 'Kukui': 'yellowgreen',\
                'Orchard crop': 'lime', 'Patch of trees': 'red',\
                'Individual tree, unknown species': 'green',\
                'White paint': 'white',\
                'Blue-green-purple paint (with cobalt)': 'blue',\
                'Blue-green paint (no cobalt)': 'cornflowerblue',\
                'Brown or brick-red paint': 'saddlebrown',\
                'Yellow-red paint': 'orangered',\
                'Rusty (and/or dirty?)': 'firebrick',\
                'Some kind of paint, often off-white': 'wheat',\
                'Another kind of pale-ish paint': 'darkseagreen',\
                'Asphalt shingle?': '0.2',\
                'Not sure ("Camel humps")': 'hotpink'}


    @classmethod
    def get_color(cls, colordict, label):
        """
        """

        try:
            color = colordict[label]
        except KeyError:
            #print(f"{label} not found in color dictionary; assigning yellow")
            color = 'yellow'

        return color


    def biplot_loadings(self, pc_x, pc_y, fname=None):
        """
        Create a figure that shows PCA biplots and loadings, for several combinations
        of components pc_x and pc_y
        """

        unique = list(set(self.labels))

        colors = self.colordict()

        _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

        legend_data =[]
        for _, label in enumerate(unique):
            x = [self.comps[j,pc_x] for j in range(len(self.comps[:,pc_x]))\
                 if self.labels[j] == label]
            y = [self.comps[j,pc_y] for j in range(len(self.comps[:,pc_y]))\
                 if self.labels[j] == label]

            if ':' in label:
                label=label.split(':')[1].strip()
                color = self.get_color(colors, label)
                if 'paint' in label:
                    marker = 'o'
                    size = 60
                else:
                    marker = '*'
                    size = 80
                legend_data.append(ax1.scatter(x, y, color=color, edgecolor='k', alpha=0.5, s=size,\
                                               marker=marker, label=label))
            else:
                color = self.get_color(colors, label)
                legend_data.append(ax1.scatter(x, y, color=color, edgecolor='k', label=label,\
                                               s=60, marker='^'))

        #label x and y axes with principal component numbers
        ax1.set_xlabel(f"PC{pc_x+1}")
        ax1.set_ylabel(f"PC{pc_y+1}")

        #loadings for pc_x
        for component, color in zip([pc_x, pc_y], ['k', 'b']):
            loadings = np.transpose(self.pc_props.components_[component, :])
            ax2.plot(loadings, color, label=f'Loading for PC{component+1}')

        ax2.axhline(0, color='gray', lw=1, ls='--')
        ax2.legend()
        ax2.set_xlabel('Band, but need to add NaNs back in')

        #put the biplot legend in a separate axis to avoid overcrowding
        ax3.legend(handles=legend_data, loc='center left')
        ax3.axis('off')

        plt.tight_layout()

        if fname is not None:
            plt.savefig(self.plot_path+fname)


if __name__ == "__main__":
    pass
