#!/usr/bin/env python3
#spectral_utils.py
#REM 2022-05-02

"""
Module containing various classes to open and manipulate GAO data. Intended to be
called from ipynb notebooks.
"""

import os
import sys
import shutil
import copy
import warnings
import pickle
from collections import OrderedDict
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import shapes
import matplotlib.pyplot as plt
from skimage import exposure
import scipy
from shapely.geometry import shape, mapping
import fiona
import geopandas as gpd
import sampleraster #spectral extraction code (written by NV)
warnings.simplefilter(action='ignore', category=FutureWarning) #for unfixable Pandas FutureWarnings



class Prepare():
    """
    Class for doing simple things like reading raster files and brightness-normalizing spectra
    """

    def __init__(self, data_path, header_path, out_path):
        self.data_path = data_path
        self.out_path = out_path
        if not os.path.isdir(self.out_path):
            os.makedirs(self.out_path)
        self.header_path = header_path


    def read_raster(self, file, roi=None):
        """
        Read all bands of region of interest <roi> in raster file, <file>, and return
        a 3D numpy array (bands, x, y) and the metadata needed for later writing the
        array to file.
        """

        with rasterio.open(self.data_path+file, 'r') as f:

            #read the entire file or the ROI
            print(f'Reading {self.data_path+file}')

            meta = f.meta.copy()

            if roi is None:
                arr = f.read()

            else:
                window = Window(roi[0], roi[1], roi[2], roi[3])
                arr = f.read(window=window)

                #get the correct metadata for this ROI
                meta.update({
                'height': window.height,
                'width': window.width,
                'transform': f.window_transform(window)})

            #convert array to float so as to be able to set bad values
            #(e.g. regions affected by telluric H2O) and ignore them in calculations

            arr = arr.astype(float)

        return arr, meta


    @classmethod
    def apply_shade_mask(cls, arr, shade_mask, show=False):
        """
        Returns a version of the input <arr> in which shaded pixels are set to NaN using
        <shade_mask>. In the GAO shade masks, shaded pixels are defined partly by ray
        tracing and partly by classifying all pixels below a certain height above ground as
        shaded. This is appropriate for taller things like trees and buildings, but will
        classify bare ground, roads, short grass etc. as shaded.
        """

        #reshape mask from 3D (1, X, Y) to 2D (X, Y)
        shade_mask = shade_mask.reshape(shade_mask.shape[1], shade_mask.shape[2])

        #report the percentage of shaded pixels. From NV:
        #"not sunlit vegetation" = 127, "sunlit vegetation" = 255 (nodata = 0)
        shaded = (shade_mask == 127).sum()
        frac = shaded / (shade_mask.shape[0] * shade_mask.shape[1])
        print(f"{frac*100 :.1f}% of pixels are classed as shaded")

        #set all pixels not in the "sunlit vegetation" category to NaN
        masked = np.where((shade_mask != 255), np.nan, arr)

        if show:
            #display a single (blue, arbitrarily) band that shows the shaded pixels
            _ = plt.figure(figsize=(12, 12))
            cmap = copy.copy(plt.cm.get_cmap('Oranges'))
            cmap.set_bad(color='0.5')
            _ = plt.imshow(masked[11, :, :], cmap=cmap, vmax=1000)

        return masked


    def read_wavelengths(self, hdr_file):
        """
        Returns the a list containing the wavelength of each band in a reflectance data cube,
        read from its corresponding header file.
        """

        wavelengths = []
        with open(self.header_path+hdr_file, 'r', encoding='utf8') as f:
            read = False
            for line in f:
                if 'wavelength = ' in line:
                    read = True
                if read is True and 'wavelength' not in line:
                    wavelengths.append(line.split())
                    if '}' in line:
                        read = False

        wavelengths = [item.strip(',').strip('}') for sublist in wavelengths for item in sublist]
        wavelengths = [np.round(float(w), 1) for w in wavelengths]

        return wavelengths


    def write_raster(self, arr, meta, filename):
        """
        Write a raster: an ROI read by self.read_master, or an ROI or full frame
        that has been modified by e.g. self.apply_shade_mask.
        """

        #self.read_raster converts arrays to float, so need to also write as float
        #to avoid small numbers being rounded to zero
        meta.update({'dtype': 'float32', 'driver': 'GTiff'})
        with rasterio.open(self.out_path+filename, 'w', **meta) as f:
            f.write(arr)


    def select_polygons(self, arr, feature_coords, filename):
        """
        Create a shapefile and mask in which image features are identified.
        Intended as a substitute for selecting features in QGIS.
        Parameters:
            - arr
              the numpy array containing the features. Currently this code
              expects a single-band array such as an NDVI image
            - feature_coords
              list of lists containing the corners of the (square) feature(s)
              to be defined (order is y1, y2, x1, x2?). This list can be created
              by displaying <arr> using Display.display_single_band and gradually
              refining the list entires until they cover the features of interest
            - filename
              the name of the shapefile into which the features are to be written as
              separate Polygons
        Returns:
            - mask
              a numpy array of the same dimensions as <arr> in which features selected
              by <feature_coords> are set to 1, and all other pixels are set to 0. Can
              be used as an overlay in Display.display_single_band to refine/verify
              the definition of <feature_coords>
        """

        mask = np.zeros(arr.shape, dtype='int16')
        #identify regions we want to extract by setting their value to 1
        for feature in feature_coords:
            mask[:, feature[0]:feature[1], feature[2]:feature[3]] = 1

        polyz=[]
        for vec in shapes(mask):
            #vec is a (0, 1) tuple in which vec[0] is the shape and vec[1] is the data value
            #write only polygons where vec[1] = 1 (regions we want to extract)
            if vec[1] == 1:
                polyz.append(shape(vec[0]))

        #define a polygon feature geometry with one attribute
        schema = {'geometry': 'Polygon', 'properties': {'id': 'int'}}

        #write the features into a shapefile
        with fiona.open(self.out_path+filename, 'w', 'ESRI Shapefile', schema) as shp:
            for poly, num in zip(polyz, range(len(polyz))):
                shp.write({'geometry': mapping(poly), 'properties': {'id': num}})

        return mask


class Display():
    """
    Methods for displaying single-band and colour images
    """

    def __init__(self, figure_path):
        self.figure_path = figure_path # path for saving figures


    def display_single_band(self, arr, subset=None, overlay=None, filename=None):
        """
        Display an image with a single band (e.g. an NDVI image), optionally
        overlaying another image and/or writing the plot to a file. Intended
        for use with Prepare.select_polygons to define features for later
        spectral extraction. However, it can also be used to just display
        single bands of a data cube using something like arr = refl[10:11, :, :]
        """

        _ = plt.figure(figsize=(12, 12))
        if subset is not None:
            arr = arr[:, subset[0]:subset[1], subset[2]:subset[3]]
            if overlay is not None:
                overlay = overlay[:, subset[0]:subset[1], subset[2]:subset[3]]
        plt.imshow(np.dstack(arr))

        if overlay is not None:
            cmap = copy.copy(plt.cm.get_cmap('Greys_r'))
            cmap.set_bad(alpha=0)
            plt.imshow(np.dstack(overlay), alpha=0.3, cmap=cmap)

        if filename is not None:
            plt.savefig(self.figure_path+filename, dpi=450)


    @classmethod
    def make_rgb(cls, img, bands):
        """
        Given the 3D image <img> and a list of three bands, return a color image that can be
        displayed with imshow
        """

        red = img[bands[0], :, :]
        green = img[bands[1], :, :]
        blue = img[bands[2], :, :]

        rgb = np.dstack([red, green, blue])

        return rgb


    @classmethod
    def fancy_rgb(cls, rgb, clip_lim=1, ax=None):
        """
        Display a color image using the Contrast Limited Adaptive Histogram Equalization
        algorithm for local contrast enhancement. Gives weird results if the image contains
        NaN values.
        """

        rgb = rgb / np.amax(rgb)
        img_nonan = np.ma.masked_invalid(rgb)
        img_adapteq = exposure.equalize_adapthist(img_nonan, clip_limit=clip_lim)
        if ax is None:
            plt.imshow(img_adapteq)
        else:
            ax.imshow(img_adapteq)


    @classmethod
    def show_rgb(cls, rgb, meta, ax, percentile=2):
        """
        Display a color image using histogram clipping. Works OK with images containing
        NaN values.
        """

        #Set nodata values to NaN to avoid -9999.0 dominating displayed images
        tmp = rgb.copy()
        tmp[abs(tmp - float(meta['nodata'])) < 0.1] = np.nan

        p_low, p_high = np.percentile(tmp[~np.isnan(tmp)], (percentile, 100 - percentile))
        img_rescale = exposure.rescale_intensity(tmp, in_range=(p_low, p_high))
        ax.imshow(img_rescale, interpolation=None)


class Extract():
    """
    Methods for extracting spectra from GAO data reflectance data cubes. Includes methods to
    save parts of numpy array containing features (e.g. buildings) to shapefiles to be read by
    the spectral extraction code, sampleraster.py.
    """

    def __init__(self, feature_path, output_path):
        self.feature_path = feature_path #path to shapefiles containing features to be extracted
        self.output_path = output_path #directory where spectra and plots will live
        if not os.path.isdir(self.output_path):
            os.makedirs(self.output_path)


    def extract(self, feature_file, raster, brightness_norm=True):
        """
        Call sampleraster.py to extract feature spectra from a shapefile, return a
        dictionary of feature spectra.
        Parameters:
            - feature_file, str
              the name of a shapefile containing polygons representing features
              whose spectra are to be extracted; may have been created in QGIS,
              for example
            - raster, str
              the name of the raster from which spectra are to be extracted, such
              as an ROI read by Prepare.read_roi and written by Prepare.write_roi.
        Returns:
            - spectra, dict
              dictionary of feature spectra in which key=feature ID number (arbitrary),
              value=spectrum of that feature.
        """

        #For some reason sampleraster.py can't open the shapefiles if they're in
        #/data/gdcsdata/test_directory/rmason_test/ (something to do with it being mapped
        #from gdcs-remote?), so copy them to a temp dir here. In addition to .shp, .shx
        #is needed by sampleraster and .dbf is needed for geopandas to read feature
        #labels/attributes as well as geometries

        if not os.path.isdir('temp'):
            os.makedirs('temp')
        tmp_name = self.feature_path+feature_file
        for suffix in ['shp', 'shx', 'dbf']:
            shutil.copyfile(tmp_name.replace('shp', suffix),\
                            f"temp/{feature_file.replace('shp', suffix)}")

        #Extract the spectra
        #sampleraster.py outputs a spectrum for every pixel in a feature, and that
        #gets confusing if there are multiple features per shapefile. We could deal
        #with that in postprocessing, but instead we'll handle it here by writing
        #each feature into a separate shapefile and extracting each one separately.
        #Then, the mean spectrum of each feature is calculated and added to a dictionary.

        #read the shapefile that contains the feature(s) to be extracted
        spectra = {}
        features = gpd.read_file(f"temp/{feature_file}")

        #for each feature (polygon) in the shapefile, find the mean spectrum
        #of its pixels
        for i in range(len(features)):
            key = (features.loc[i, 'id'], features.loc[i, 'Descr.'])

            #delete existing input and output files/dir - code barfs or hangs if they're present
            if os.path.isfile('temp.shp'):
                os.remove('temp.shp')
            shutil.rmtree('temp', ignore_errors=True)

            #write the polygon for this feature into its own (new) file in the current dir,
            #which sampleraster will use for the extraction of this feature
            features[i: i+1].to_file('temp.shp')

            #extract the spectrum of each pixel in this feature
            sys.argv = ['./sampleraster.py', 'temp.shp', 'temp', raster]
            sampleraster.main()

            #get the spectrum of each pixel, optionally brightness-normalize, add it to a list
            #of spectra, create mean spectrum of all pixels in the feature
            feature_pix = 0 #count number of pixels in feature
            for point in fiona.open("temp/temp.shp"):
                values = []
                #reflectance values are kept in point['properties'] dictionary
                for wav, value in point['properties'].items():
                    if wav[0].isdigit(): #find the dict keys that contain wavelengths
                        values.append(value) #build up the spectrum of this pixel

                #brightness-normalize the spectrum for this pixel (also takes care of setting
                #water bands to NaN)
                if brightness_norm:
                    values = self.brightness_norm(values)
                #or just set water bands to NaN
                else:
                    self.identify_water_bands(values, bandval=np.nan)

                #add the spectrum to any existing spectra for this feature, to build up the
                #summed spectrum
                if key not in spectra:
                    spectra[key] = values
                else:
                    spectra[key] = [x + y for (x, y) in zip(spectra[key], values)]
                feature_pix += 1

            #divide by the number of pixels in this feature to get the mean spectrum for the feature
            spectra[key] = [y / feature_pix  for y in spectra[key]]

        #Remove the temp directory and anything in it (but usually fails because files are open)
        shutil.rmtree('temp', ignore_errors=True)
        #Remove remaining temp files
        for f in ['temp.shp', 'temp.shx', 'temp.dbf', 'temp.cpg']:
            if os.path.isfile(f):
                os.remove(f)

        return spectra


    @classmethod
    def identify_water_bands(cls, spectrum, bandval):
        """
        Set the spectral regions in <arr> badly affected by telluric H2O to <bandval>. The
        locations of the water bands were first defined by stepping through
        bands in ENVI and looking for the blank images.
        """

        water_bands = [list(range(0,5)), list(range(99, 108)), list(range(142, 179)), [213]]
        for badbit in water_bands:
            for band, _ in enumerate(spectrum):
                if band in badbit:
                    spectrum[band] = bandval

        return spectrum


    def brightness_norm(self, arr):
        """
        Brightness normalise a 1D spectrum, i.e., find the vector norm (sum of absolute values
        at each wavelength), and divide all pixels by this number.
        This effectively removes the effect of things like viewing angle that would otherwise
        make some pixels brighter than others regardless of their intrinsic spectra.
        This method also sets regions of poor atmospheric H20 correction to NaN
        (see self.identify_water_bands).
        """

        #We want to exclude telluric H2O regions from calculation, but
        #NaNs and masked arrays don't seem to work with np.linalg.norm, so we
        #temporarily set those spectral regions to 0
        arr = self.identify_water_bands(arr, bandval=0)

        norm = arr / scipy.linalg.norm(arr, axis=0)

        #Now set the H20 regions to np.nan
        norm = self.identify_water_bands(norm, bandval=np.nan)

        return norm


    @classmethod
    def find_wavelength(cls, wavelengths, value):
        """
        Helper method for plot_spectra. Given a wavelength in nm, return the
        index of the point in a spectrum that is closest to that wavelength. For example,
        given wavelength value=650 (nm), return 30; the 30th pixel in the spectrum
        is closest to 650 nm.
        """

        diffs = []
        for wav in wavelengths:
            diffs.append(abs(wav - value))
        idx = np.argmin(diffs)

        return idx


    def plot_spectra(self, spectra, wavelengths, include=None, exclude=None, ylabel='',\
                     normalize_at=None, indicate_wavs=None, figname=None, ylims=None):
        """
        Display a simple plot of all the spectra in the input dictionary
        of feature spectra. Requires a list of wavelengths for each pixel;
        see Prepare.read_wavelengths.
        """

        classes = list({x[1] for x in spectra.keys()})
        if exclude is not None:
            classes = [c for c in classes if c not in exclude]
        if include is not None:
            classes = include
        if include is None and exclude is None:
            classes = sorted(classes)

        rows = int(np.ceil(len(classes) / 3))
        fig, _ = plt.subplots(rows, 3, figsize=(16, 4*rows))

        if normalize_at is not None:
            idx = self.find_wavelength(wavelengths, normalize_at)

        for i, ax in enumerate(fig.axes):
            try:
                clazz = classes[i]
            except IndexError:
                ax.axis('off')
                continue

            for feature, spectrum in spectra.items():
                if feature[1] == clazz:

                    #sampleraster seems to set (most) water bands to 0;
                    #change to NaN to omit from plots. Also, divide reflectance
                    #by 100 to get into % (per advice from Robin, 2022-03-29)
                    #BUT ONLY RELEVANT IF SPECTRA NOT BRIGHTNESS NORMALIZED
                    #spectrum = [np.nan if y == 0 else y/100 for y in spectrum ]

                    if normalize_at is not None:
                        spectrum = [s/spectrum[idx] for s in spectrum]

                    ax.plot(wavelengths, spectrum, label=feature[0])

                #use vertical lines to indicate wavelengths of R, G, B colors
                if indicate_wavs is not None:
                    for wav in indicate_wavs:
                        ax.axvline(wav, ls='dotted', color='gray', lw=0.5)

                ax.axvline(750, ls=':', color='0.5')
                if ylims is not None:
                    ax.set_ylim(ylims)
                ax.set_xlim(390, 2500)
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                ax.set_ylabel(ylabel)
                ax.set_xlabel('Wavelength, nm')
                try:
                    ax.set_title(clazz.split(':')[1].replace('\\', ''))
                except IndexError:
                    ax.set_title(clazz.replace('\\', ''))

            ax.legend(ncol=3)

        plt.tight_layout()
        if figname is not None:
            fig.savefig(self.output_path+figname)


    @classmethod
    def order_spectra(cls, spectra, key_list):
        """
        Create an ordered dictionary of spectra to be passed to self.plot_single_category()
        """

        return OrderedDict((k, spectra[k]) for k in key_list if k in spectra)


    def plot_single_category(self, spectra, wavelengths, **kwargs):
        """
        A lot like self.plot_spectra(), but for plotting only one category of spectra, each one
        in its own set of axes. This was written for classifying roof spectra by eye, because
        there were quite a lot of them.
        """

        kwargs.setdefault('color', 'k')
        kwargs.setdefault('indicate', None)
        kwargs.setdefault('rows', 11)
        kwargs.setdefault('cols', 8)
        kwargs.setdefault('figsize', (16, 18))
        kwargs.setdefault('fname', None)

        fig, _ = plt.subplots(kwargs['rows'], kwargs['cols'], figsize=kwargs['figsize'])

        count = 0
        for feature, spectrum in spectra.items():
            ax = fig.axes[count]
            ax.plot(wavelengths, spectrum, label=feature[0], color=kwargs['color'], lw=1)
            if kwargs['indicate'] is not None:
                ax.axvline(kwargs['indicate'], ls=':', lw=0.75, color='gray')
            ax.tick_params(axis='both', which='both', bottom=False, left=False,\
                              labelbottom=False, labelleft=False)
            ax.text(0.98, 0.98, feature[0], transform=ax.transAxes, ha='right', va='top')
            count += 1

        for num in range(count, len(fig.axes)):
            ax = fig.axes[num]
            ax.axis('off')

        try:
            plt.suptitle(kwargs['title'])
        except KeyError:
            pass
        plt.subplots_adjust(wspace=0, hspace=0)

        if kwargs['fname'] is not None:
            plt.savefig(self.output_path+kwargs['fname'], dpi=450)


    def index_plot(self, spectra, wavelengths, index_wavs):
        """
        Make a plot showing a normalized difference index using two
        wavelengths in <index_wav>.
        """

        idx = []
        for wav in index_wavs:
            idx.append(self.find_wavelength(wavelengths, wav))

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        classes = sorted(list({x[1] for x in spectra.keys()}))

        #plot
        for i, clazz in enumerate(classes):
            for feature, spectrum in spectra.items():
                if feature[1] in clazz:
                    ndvi = (spectrum[idx[0]] - spectrum[idx[1]]) / (spectrum[idx[0]] +\
                                                                    spectrum[idx[1]])
                    ax1.scatter(i, ndvi, color='k', alpha=0.5)
        ax1.axhline(0.0, ls=':', color='0.5')
        ax1.set_xticks(range(len(classes)), classes, rotation=20)
        ax1.set_ylabel('Mean Index of all pixels in feature')
        ax1.set_title(f"({index_wavs[0]} nm - {index_wavs[1]} nm) / ({index_wavs[0]} nm +\
        {index_wavs[1]} nm)")

        labels = {}
        colors = ['k', 'b', 'cyan', 'r', 'gray', 'limegreen']
        for i, clazz in enumerate(classes):
            for feature, spectrum in spectra.items():
                if feature[1] in clazz:
                    style = ax2.scatter(spectrum[idx[0]], spectrum[idx[1]], color=colors[i])
                    labels[clazz] = style
        ax2.legend(labels.values(), labels.keys(), loc='upper left', ncol=2)
        ax2.set_xlim(0, 6500)
        ax2.set_ylim(0, 6500)

        ax2.set_xlabel(f'Reflectance at {index_wavs[0]} nm')
        ax2.set_ylabel(f'Reflectance at {index_wavs[1]} nm')


    def combine_and_save(self, all_spectra, categorized):
        """
        Replace generic Roof spectra with categorized spectra in dictionary of all
        spectra, and pickle for later use
        """

        #Remove the generic roofing spectra
        keys = [key for key in all_spectra.keys() if key[1] == 'Roof']
        for key in keys:
            all_spectra.pop(key)
       
        #Replace with categorized roofing spectra
        for key, value in categorized.items():
            all_spectra[key] = value

        #Save for use as training dataset in other notebooks
        with open(self.output_path+'spectra.pkl', 'wb') as f:
            pickle.dump(all_spectra, f)


if __name__ == "__main__":
    pass
