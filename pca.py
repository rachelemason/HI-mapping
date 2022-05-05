#!/usr/bin/env python3
#pca.py
#REM 2022-05-04

"""
Contains classes for running PCA on extracted VSWIR spectra, and interpreting the
output
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as skl_PCA
import spectral_utils


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

        return [properties, comps]


class Diagnostics():
    """
    Methods to make plots etc. that illustrate/explain the PCA results
    """

    def __init__(self, pca_results, spectra, plot_path):
        self.pc_props = pca_results[0]
        self.comps = pca_results[1]
        self.spectra = spectra
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
    def styledict():
        """
        Dictionary containing key=spectrum label, value=color (e.g. 'Kukui': 'yellowgreen').
        Used by self.biplot() to color PCA biplots.
        """


        return {'Pool': ('cyan', 's', 40),\
                'Lava': ('k', '*', 80), 'Probably grass': ('khaki', '^', 80),\
                'Kukui': ('yellowgreen', '^', 80),\
                'Orchard crop': ('lime', '^', 80), 'Patch of trees': ('red', '^', 80),\
                'Individual tree, unknown species': ('green', '^', 80),\
                'White paint': ('white', 'o', 60),\
                'Blue-green-purple paint (with cobalt)': ('blue', 'o', 60),\
                'Blue-green paint (no cobalt)': ('cornflowerblue', 'o', 60),\
                'Brown or brick-red paint': ('saddlebrown', 'o', 60),\
                'Yellow-red paint': ('orangered', 'o', 60),\
                'Rusty (and/or dirty?)': ('firebrick', '*', 80),\
                'Some kind of paint, often off-white': ('wheat', 'o', 60),\
                'Another kind of pale-ish paint': ('darkseagreen', 'o', 60),\
                'Asphalt shingle?': ('Yellow', '*', 80),\
                'Not sure ("Camel humps")': ('hotpink', 'o', 60)}
                #'Road': ('0.7') Replace once have proper road spectra extracted
                #'Solar panels': ('magenta', 's', 40),


    def add_pcs_to_plot(self, pc_x, pc_y, ax, start, num, annotate=False):
        """
        Helper method for self.pair_plot(). Actually adds the data to the axes.
        """

        style = self.styledict()
        labels = list(self.spectra.keys())
        unique = list({label[1] for label in labels})

        for _, label in enumerate(unique):
            x = [self.comps[j,pc_x] for j in range(len(self.comps[:,pc_x]))\
                 if labels[j][1] == label]
            y = [self.comps[j,pc_y] for j in range(len(self.comps[:,pc_y]))\
                 if labels[j][1] == label]
            id_num = [labels[j][0] for j in range(len(self.comps[:,pc_y]))\
                 if labels[j][1] == label]

            if ':' in label:
                label=label.split(':')[1].strip()

            try:
                color = style[label][0]
                marker = style[label][1]
                size = style[label][2]
            except KeyError:
                #print(f'{label} not found in style dictionary, excluding from plot')
                continue

            ax.scatter(x, y, color=color, edgecolor='k', marker=marker, s=size,\
                       alpha=0.5, label=label)
            if annotate:
                for n, i, j in zip(id_num, x, y):
                    ax.annotate(n, (i, j))

            #label x and y axes with principal component numbers
            if pc_x == start:
                ax.set_ylabel(f"PC{pc_y+1}")
            if pc_y == start + (num - 1):
                ax.set_xlabel(f"PC{pc_x+1}")

            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


    def pair_plot(self, start=0, num=5, annotate=False, fname=None):
        """
        Create a figure that shows
        """

        fig, _ = plt.subplots(num, num, figsize=(16, 16))

        i = start
        j = start
        for ax in fig.axes:
            self.add_pcs_to_plot(pc_x=i, pc_y=j, ax=ax, start=start, num=num,\
                                 annotate=annotate)
            if i == start + (num - 1):
                j += 1
                i = start
            else:
                i += 1

        plt.subplots_adjust(wspace=0, hspace=0)

        if fname is not None:
            plt.savefig(self.plot_path+fname)


    def make_legend(self):
        """
        Make a 'plot' that just shows a legend for the pair plot. There are too
        many classes for the legend to fit on the actual plots.
        """

        _, ax = plt.subplots(1, 1)

        legend_elements = []
        for label, style in self.styledict().items():
            legend_elements.append(Line2D([0], [0], color=style[0], marker=style[1],
                                          markersize=style[2]/7, ls='', mec='k',\
                                          label=label))

        ax.legend(handles=legend_elements, ncol=2, loc='upper center')
        ax.axis('off')


    def find_nans(self):
        """
        Return the band numbers that contain NaNs, i.e. those with poor telluric
        cancellation. Those bands were excluded from the PCA but we need to re-insert
        them when we make the spectral loading plots in self.loadings() so as to have
        a sensible wavelength scale.
        """

        nan_idx = []
        #use the first spectrum to find where the NaN bands are
        #(all spectra should have them at the same locations)
        for spectrum in self.spectra.values():
            for i, band in enumerate(spectrum):
                if np.isnan(band):
                    nan_idx.append(i)
            break

        return nan_idx


    def loadings(self, header_path, header_file):
        """
        Make a plot that shows PC loadings for the first 12 components
        """

        prep = spectral_utils.Prepare(data_path=None, header_path=header_path, out_path=None)
        wavelengths = prep.read_wavelengths(hdr_file=header_file)

        #Find where NaNs were excluded from PCA
        #(regions of poor H2O cancellation)
        nan_locs = self.find_nans()

        fig, _ = plt.subplots(4, 3, figsize=(16, 12))
        for comp, ax in enumerate(fig.axes):
            #find loadings and convert to list in order to re-insert NaNs
            loadings = list(np.transpose(self.pc_props.components_[comp, :]))
            for idx in nan_locs:
                loadings.insert(idx, np.nan)
            ax.plot(wavelengths, loadings, 'k-')
            ax.axhline(0, color='gray', lw=1, ls='--')
            ax.axvline(750, color='gray', lw=1, ls=':')
            ax.text(0.95, 0.95, f'PC{comp+1}', va='top', ha='right', transform=ax.transAxes)
            ax.set_ylim(-0.3, 0.3)
            if comp == 10:
                ax.set_xlabel('Wavelength, nm')


if __name__ == "__main__":
    pass
