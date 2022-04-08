#!/usr/bin/env python3
#spectral_utils.py
#REM 2022-04-08

"""
Module containing various classes to open and manipulate GAO data. Intended to be
called from ipynb notebooks.
"""

import os
import sys
import shutil
import copy
import warnings
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
warnings.simplefilter(action='ignore', category=FutureWarning) #for unfixable Pandas FutureWarnings
import sampleraster #spectral extraction code (written by NV)


class Prepare():
    """
    Class for doing simple things like reading raster files and brightness-normalizing spectra
    """

    def __init__(self, data_path, out_path):
        self.data_path = data_path
        self.out_path = out_path


    def read_roi(self, file, roi):
        """
        Read all bands of region of interest <roi> in raster file, <file>, and return
        a 3D numpy array (bands, x, y) and the metadata needed for later writing the
        array to file.
        """

        with rasterio.open(self.data_path+file, 'r') as f:

            #read the ROI
            window = Window(roi[0], roi[1], roi[2], roi[3])
            arr = f.read(window=window)

            #convert array to float so as to be able to set bad values
            #(e.g. regions affected by telluric H2O) and ignore them in calculations

            arr = arr.astype(float)
            #set bad/nodata values to np.nan (although I haven't seen any so far)
            try:
                #nodata values are -9999.0 for some files, so do an approx. check
                arr[abs(arr - float(f.meta['nodata'])) < 0.1] = np.nan
            except TypeError:
                #nodata values are not defined for some (e..g shade masks)
                pass

            #get the correct metadata for this ROI
            meta = f.meta.copy()
            meta.update({
            'height': window.height,
            'width': window.width,
            'transform': f.window_transform(window)})

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

        #set all pixels in the "shaded" category to NaN
        masked = np.where((shade_mask == 127), np.nan, arr)

        if show:
            #display a single (blue, arbitrarily) band that shows the shaded pixels
            _ = plt.figure(figsize=(12, 12))
            cmap = copy.copy(plt.cm.get_cmap('Oranges'))
            cmap.set_bad(color='0.5')
            _ = plt.imshow(masked[11, :, :], cmap=cmap, vmax=1000)

        return masked


    @classmethod
    def identify_water_bands(cls, arr, bandval):
        """
        Set the spectral regions in <arr> badly affected by telluric H2O to <bandval>. The
        locations of the water bands were first defined by stepping through
        bands in ENVI and looking for the blank images.
        """

        water_bands = [list(range(0,5)), list(range(101, 107)), list(range(146, 179)), [213]]
        for badbit in water_bands:
            for index, _ in enumerate(arr):
                if index in badbit:
                    arr[index, :, :] = bandval

        return arr


    def brightness_norm(self, arr):
        """
        Brightness normalise spectra of all locations in a data cube. In pseudo-code, this
        means for an x,y image with z spectral bands:
        for x,y location in array:
            find the vector norm of the z direction (sum of absolute values at each wavelength)
            divide all z pixels by this number
        This effectively removes the effect of things like viewing angle that would otherwise
        make some pixels brighter than others regardless of their intrinsic spectra.
        This method also sets regions of poor atmospheric H2) correction to NaN
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


    def read_wavelengths(self, hdr_file):
        """
        Returns the a list containing the wavelength of each band in a reflectance data cube,
        read from its corresponding header file.
        """

        wavelengths = []
        with open(self.data_path+hdr_file, 'r', encoding='utf8') as f:
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


    def write_roi(self, arr, meta, filename):
        """
        Write the roi created by read_roi, optionally modified by methods such as
        self.brightness_norm(), to a raster.
        """

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
    def show_rgb(cls, rgb, ax, percentile=2):
        """
        Display a color image using histogram clipping. Works OK with images containing
        NaN values.
        """

        p_low, p_high = np.percentile(rgb[~np.isnan(rgb)], (percentile, 100 - percentile))
        img_rescale = exposure.rescale_intensity(rgb, in_range=(p_low, p_high))
        ax.imshow(img_rescale, interpolation=None)


class Extract():
    """
    Methods for extracting spectra from GAO data reflectance data cubes. Includes methods to
    save parts of numpy array containing features (e.g. buildings) to shapefiles to be read by
    the spectral extraction code, sampleraster.py.
    """

    def __init__(self, feature_path, plot_path):
        self.feature_path = feature_path #path to shapefiles containing features to be extracted
        self.plot_path = plot_path #directory where plots will live
        if not os.path.isdir(self.plot_path):
            os.makedirs(self.plot_path)


    def extract(self, feature_file, raster):
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

        #And for some reason sampleraster.py can't open the shapefiles if they're in
        #/data/gdcsdata/test_directory/rmason_test/ (something to do with it being mapped
        #from gdcs-remote?), so copy them to a temp dir here

        temp_name = self.feature_path+feature_file
        if not os.path.isdir('temp'):
            os.makedirs('temp')
        shutil.copyfile(temp_name, f"temp/{feature_file}")
        #copy the associated files, too - .shx needed by sampleraster, .dbf needed
        #for geopandas to read feature labels/attributes as well as just geometries
        for suffix in ['shx', 'dbf']:
            shutil.copyfile(temp_name.replace('shp', suffix),\
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

            #write the polygon for this feature into its own (new) file
            features[i: i+1].to_file('temp.shp')

            #extract the spectrum of each pixel in this feature (quietly)
            sys.argv = ['./sampleraster.py', 'temp.shp', 'temp', raster]
            sampleraster.main()

            #get the spectrum of each pixel, add it to a list of spectra, create mean spectrum
            #of all pixels
            feature_pix = 0 #count number of pixels in feature
            for point in fiona.open("temp/temp.shp"):
                values = []
                spectrum = point['properties'] #this is where the reflectance values are kept
                for wav, value in spectrum.items():
                    if wav[0].isdigit(): #some dict keys don't contain wavelengths but other info
                        values.append(value) #build up the spectrum of this pixel

                #add the spectrum to any existing spectra for this feature, to build up the
                #summed spectrum
                if key not in spectra:
                    spectra[key] = values
                else:
                    sumlists = [x + y for (x, y) in zip(spectra[key], values)]
                    spectra[key] = sumlists
                feature_pix += 1

            #divide by the number of pixels in this feature to get the mean spectrum for the feature
            spectra[key] = [y / feature_pix  for y in spectra[key]]

        #Remove the temp directory and anything in it
        #This doesn't seem to work - why?
        shutil.rmtree('temp/', ignore_errors=True)
        #Remove remaining temp files
        for f in ['temp.shp', 'temp.shx', 'temp.dbf', 'temp.cpg']:
            if os.path.isfile(f):
                os.remove(f)

        return spectra


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


    def plot_spectra(self, spectra, wavelengths, ylabel, normalize_at=None, indicate_wavs=None,\
                     figname=None):
        """
        Display a simple plot of all the spectra in the input dictionary
        of feature spectra. Requires a list of wavelengths for each pixel;
        see Prepare.read_wavelengths.
        """

        classes = sorted(list({x[1] for x in spectra.keys()}))
        fig, _ = plt.subplots(2, 3, figsize=(16, 8))

        if normalize_at is not None:
            idx = self.find_wavelength(wavelengths, normalize_at)

        for i, ax in enumerate(fig.axes):
            try:
                clazz = classes[i]
            except IndexError:
                ax.axis('off')
                break

            for feature, spectrum in spectra.items():
                if feature[1] == clazz:

                    #sampleraster seems to set (most) water bands to 0;
                    #change to NaN to omit from plots. Also, divide reflectance
                    #by 100 to get into % (per advice from Robin, 2022-03-29)
                    spectrum = [np.nan if y == 0 else y/100 for y in spectrum ]

                    if normalize_at is not None:
                        spectrum = [s/spectrum[idx] for s in spectrum]

                    ax.plot(wavelengths, spectrum, label=feature[0])

                #use vertical lines to indicate wavelengths of R, G, B colors
                if indicate_wavs is not None:
                    for wav in indicate_wavs:
                        ax.axvline(wav, ls='dotted', color='gray', lw=0.5)

                if normalize_at is None:
                    ax.set_ylim(0, 80)
                ax.set_xlim(390, 2500)
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                ax.set_ylabel(ylabel)
                ax.set_xlabel('Wavelength, nm')
                ax.set_title(clazz.replace('\\', ''))
            ax.legend(ncol=2)

        plt.tight_layout()
        if figname is not None:
            fig.savefig(self.plot_path+figname)


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


if __name__ == "__main__":
    pass
