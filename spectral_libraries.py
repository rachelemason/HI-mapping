#!/usr/bin/env python3
#spectral_libraries.py
#REM 2022-04-29

"""
Module containing various functions for reading spectral library data
"""

import glob
from string import ascii_uppercase
import pandas as pd
import matplotlib.pyplot as plt

#pylint:disable=unsupported-assignment-operation
#pylint:disable=unsubscriptable-object
#pylint:disable=no-member


def plot_library_spex(spectra, exclude=None):
    """
    Simple function to plot a set of library spectra in a single figure
    """

    _ = plt.figure()

    for (material, _), spectrum in spectra.items():
        if exclude is None:
            plt.plot(spectrum['Wavelength, nm'], spectrum['Reflectance, %'], label=material)
        else:
            if material not in exclude:
                plt.plot(spectrum['Wavelength, nm'], spectrum['Reflectance, %'], label=material)
    plt.xlim(390, 2500)
    plt.legend()
    plt.xlabel('Wavelength, nm')
    plt.ylabel('Reflectance, %')


def get_xy(library, keys, refl='Reflectance, %'):
    """
    Return simple, plottable x, y lists from dictionaries of spectral library
    data frames
    """

    exes = []
    whys = []
    for key in keys:
        exes.append(library[key]['Wavelength, nm'].values)
        whys.append(library[key][refl])

    return exes, whys


def show_features(ax, c_h=True, fe_ox=True, ymin=0.9, ymax=0.97):
    """
    Indicate the approx wavelengths of features that are due to:
    1. 'binders' in paints/pigments (Levinson et al. 2005; sec 3.1),
        which I think are hydrocarbons
    or
    2. iron oxides (Herold et al. 2004; sec 4.1)
    """


    if c_h:
        for wav in [1180, 1450, 1715, 1940, 2130, 2260]:
            ax.axvline(wav, ymin=ymin, ymax=ymax, color='k', ls='-', lw=0.75)
    if fe_ox:
        for wav in [520, 670, 870]:
            ax.axvline(wav, ymin=ymin, ymax=ymax, color='r', ls='-', lw=0.75)


def axis_stuff(fig, textpos):
    """
    Helper function for doing things with axes in plot_by_type and plot_painted
    """

    for i, ax in enumerate(fig.axes):
        show_features(ax)
        ax.set_xlim(390, 2499)
        ax.set_ylim(0, 1.15)
        #approx. division between Vis/IR
        ax.axvline(750, ls=':', color='0.5')
        ax.legend()
        ax.text(textpos[0], textpos[1], ascii_uppercase[i], transform=ax.transAxes, va='bottom',\
                ha='left', weight='bold')
    for num, ax in enumerate(fig.axes):
        if num in [1, 2, 4, 5]:
            ax.tick_params(which='both', left=False, labelleft=False)
        if num == 4:
            ax.set_xlabel('Wavelength, nm')

    plt.subplots_adjust(wspace=0)


def plot_by_type(ecostress, levinson_metals, herold):
    """
    Plot roofing materials from different libraries, sorted into various categories
    """

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 8))

    #ALUMINIUM
    #Ecostress
    exes, whys = get_xy(ecostress, [('Aluminum Metal', '0384UUUALM')])
    for x, y in zip(exes, whys):
        ax1.plot(x, y, color='k', label='Al, weathered + scratched')
    #Levinson et al. (2007)
    x = levinson_metals['Wavelength, nm']
    y = levinson_metals['Al foil']
    ax1.plot(x, y, color='gray', label='Al foil')
    ax1.text(150, -0.5, 'Normalized reflectance', rotation='vertical', ha='center', fontsize=12)

    #GALVANISED STEEL
    #Levinson et al. (2007)
    columns = [c for c in levinson_metals.columns if c not in ['Al foil', 'Wavelength, nm']]
    for column, style, color, label in zip(columns, ['--', '-', '--', '-'],\
                                           ['k', 'k', 'gray', 'gray'],\
                                           ['Bare Zincalume steel', 'Treated Zincalume steel',\
                                            'Bare HDG steel', 'Treated HDG steel']):
        x = levinson_metals['Wavelength, nm']
        y = levinson_metals[column]
        ax2.plot(x, y, color=color, ls=style, label=label)

    #OXIDISED GALVANISED STEEL
    #Ecostress
    exes, whys = get_xy(ecostress, [('Oxidized Galvanized Steel Metal', '0526UUUSTLa'),\
                                    ('Galvanized Steel Metal', '0525UUUSTLb'),\
                                    ('Galvanized Steel Metal', '0525UUUSTLa')])
    for x, y, color, label in zip(exes, whys, ['k', 'gray', '0.7'],\
                                  ['Completely oxidized galvanized steel',\
                                   'Weathered galvanized steel',\
                                   'Weathered galvanized steel']):
        ax3.plot(x, y, color=color, label=label)

    #SHINGLE - SPECTRA THAT SLOPE DOWN THROUGH IR
    #Ecostress; grey weathered, white (sand-covered); reddish (unweathered)
    exes, whys = get_xy(ecostress, [('Asphalt roofing shingle', '0597UUUASP'),\
                                    ('Asphalt Shingle', '0490UUUASP'),\
                                    ('Reddish asphalt Shingle', '0683UUUASP')])
    labels = ['Gray, weathered, soiled shingle', 'White sand over construction asphalt',\
              'Reddish, unweathered shingle']
    colors = ['k', '0.2', 'firebrick']
    for x, y, color, label in zip(exes, whys, colors, labels):
        ax4.plot(x, y, color=color, label=label)
    #Herold
    x = herold['Wavelength']
    for material, color, label in zip(['lt tan', 'lt grey'], ['tan', '0.8'],\
                                      ['Light tan shingle', 'Light gray shingle']):
        y = herold[material]
        ax4.plot(x, y, color=color, label=label)

    #SHINGLE - SPECTRA THAT SLOPE UP/ARE ~CONSTANT THROUGH IR
    #Ecostress (weathered reddish asphalt)
    exes, whys = get_xy(ecostress, [('Reddish Asphalt roofing shingle', '0672UUUASP')])
    for x, y in zip(exes, whys):
        ax5.plot(x, y, color='firebrick', label='Reddish, weathered shingle')
    #Herold
    x = herold['Wavelength']
    materials = ['dark grey', 'dark brown', 'med grey; blue sprinkles',\
                     'dark grey; green sprinkles']
    labels = ['Dark gray shingle', 'Dark Brown shingle', 'Medium gray shingle + blue sprinkles',\
              'Dark gray shingle + green sprinkles']
    colors = ['0.3', 'saddlebrown', 'blue', 'green']
    for material, color, label in zip(materials, colors, labels):
        y = herold[material]
        ax5.plot(x, y, color=color, label=label)

    #TILE
    #Ecostress terra-cotta
    exes, whys = get_xy(ecostress, [('Terra cotta Tiles', '0484UUUPOT')])
    for x, y in zip(exes, whys):
        ax6.plot(x, y, color='orangered', label='Terra Cotta tile')
    #Herold 'red tile' and 'grey composite shingle'
    materials = ['red tile', 'grey composite shingle']
    colors = ['firebrick', 'gray']
    labels = ['Red tile', 'Gray composite shingle']
    for material, color, label in zip(materials, colors, labels):
        ax6.plot(herold['Wavelength'], herold[material], color=color, label=label)

    axis_stuff(fig, textpos=(0.015, 0.015))


def plot_painted(painted, pigments, pvc):
    """
    Plot spectra of painted materials, from Levinson et al. (2007) figure 2, and
    the Levinson et al. (2005) pigment collection. Approximate hex codes for the
    paint colors were determined by searching in https://encycolorpedia.com/, or
    """

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(16, 8))

    #White pigment over grey background, various thicknesses
    x = painted['Wavelength, nm']
    for microns, color in zip([50, 100, 200], ['lightgray', 'gray', 'k']):
        y = painted[f'{microns} um TiO2 rutile white']
        ax1.plot(x, y, color=color, label=fr'{microns} $\mu$m TiO$_2$ rutile white over gray')
    ax1.text(150, -0.5, 'Reflectance', rotation='vertical', ha='center', fontsize=12)

    #White pigment over white and black BG
    exes, whys = get_xy(pigments, keys=['W01-masstone-spectral-data.txt'],\
                        refl='Reflectance, white BG')
    labels = ['Inorganic Oxide White over white']
    for x, y, label in zip(exes, whys, labels):
        ax2.plot(x, y, color='k', label=label)
    exes, whys = get_xy(pigments, keys=['W01-masstone-spectral-data.txt'],\
                        refl='Reflectance, black BG')
    labels = ['Inorganic Oxide White over black']
    for x, y, label in zip(exes, whys, labels):
        ax2.plot(x, y, ls='--', color='k', label=label)

    #Blue, green, purple w/o cobalt
    exes, whys = get_xy(pigments, keys=['U12-masstone-spectral-data.txt',\
                                        'U14-masstone-spectral-data.txt',\
                                        'G01-masstone-spectral-data.txt'],\
                        refl='Reflectance, white BG')
    labels = ['Phthalo Blue', 'Dioxazine Purple', 'Chroma Green']
    colors = ['#000f89', '#54348f', '#558e47']
    for x, y, label, color in zip(exes, whys, labels, colors):
        ax3.plot(x, y, label=label, color=color)

    #Blue, green, w/ cobalt
    exes, whys = get_xy(pigments, keys=['U05-masstone-spectral-data.txt',\
                                        'G07-masstone-spectral-data.txt'],\
                                        refl='Reflectance, white BG')
    labels = ['Cobalt Blue', 'Cobalt Teal']
    colors = ['#2a508f', '#00ced1']
    for x, y, label, color in zip(exes, whys, labels, colors):
        ax4.plot(x, y, label=label, color=color)

    #Red/orange, with and without Fe oxide
    exes, whys = get_xy(pigments, keys=['R01-masstone-spectral-data.txt',\
                                        'R02-masstone-spectral-data.txt',\
                                        'R07-masstone-spectral-data.txt'],\
                        refl='Reflectance, white BG')
    labels = ['Red Iron Oxide I', 'Red Iron Oxide II', 'Cadmium Orange']
    colors = ['saddlebrown', '#9b2f1c', '#f17c21']
    for x, y, label, color in zip(exes, whys, labels, colors):
        ax5.plot(x, y, label=label, color=color)

    #PVC (not painted, but shows hydrocarbon bands)
    for x in [0, 1, 2]:
        ax6.plot(pvc[f'Wavelength {x}'], pvc[f'Reflectance {x}'], color='k', label='PVC')

    axis_stuff(fig, textpos=(0.96, 0.015))


def get_levinson_pigments(path):
    """
    Return a dictionary containing the full set of pigment spectra
    from Levinson et al. (2005). See spectral_datafile_guide.pdf.
    """

    spectra = {}

    files = glob.glob(path+'*.txt')
    for f in files:
        #skip wavelengths outside the range of our data
        df = pd.read_csv(f, skiprows=list(range(20)), header=None, sep='\t')

        #keep and rename only necessary columns
        #Wavelength, 'r.tilde.fw', 'r.tilde.fb' (reflectance over white/black background)
        df = df[[23, 28, 29]]
        df.rename(columns={23: 'Wavelength, nm', 28: 'Reflectance, white BG',\
                           29: 'Reflectance, black BG'}, inplace=True)

        #Add pigment spectrum to dict using its original filename
        spectra[f.split('Levinson/')[1]] = df

    return spectra


def get_levinson_fig2(path):
    """
    Returns a data frame of spectra from Levinson et al. (2007) Figure 2
    """

    data = pd.read_excel(path, sheet_name='Spectral Data', skiprows=[0, 1], header=None)

    #remove extraneous rows and columns
    data = data.head(441)
    data = data[[0, 8, 9, 10]]

    #rename columns
    data.rename(columns={0: 'Wavelength, nm', 8: '50 um TiO2 rutile white',\
                         9: '100 um TiO2 rutile white', 10: '200 um TiO2 rutile white'},\
                inplace=True)

    return data


def get_levinson_fig3(path, norm=True):
    """
    Returns a data frame of spectra from Levinson et al. (2007) Figure 3
    """

    data = pd.read_excel(path, sheet_name='Spectral Data', skiprows=[0, 1], header=None)

    #remove extraneous rows and columns
    data = data.head(441).drop(columns=[1, 2, 8, 9, 10, 11])

    #Normalize so max reflectance = 1
    if norm:
        for column in [3, 4, 5, 6, 7]:
            data[column] = data[column] / data[column].max()

    #rename columns
    data.rename(columns={0: 'Wavelength, nm', 3: 'Al foil', 4: 'Bare Zincalume',\
                         5: 'Treated Zincalume', 6: 'Bare HDG', 7: 'Treated HDG'}, inplace=True)

    return data


def get_herold(path, norm=True):
    """
    Returns a dataframe containing spectra of materials from the Herold et al.
    (2004) Santa Barbara library of urban material spectra. Downloaded from
    www.geo-informatie.nl/Projects/Santa_Barbara_Urban_Spectral_Library/urbanspec/sburbspec.htm
    on 2022-04-21
    """

    #Figure 2c (tile + others)
    df = pd.read_excel(path, sheet_name='fig2_c_major_roofs')

    #Figure 2d (various asphalt shingles)
    df2 = pd.read_excel(path, sheet_name='fig2_d_comp_sh_color')

    #Merge; rename some columns
    df = df.merge(df2, on='Wavelength')
    df.rename(columns={'med grey (with colored sprinkles - blue)': 'med grey; blue sprinkles',\
                       'dk grey with green sprinkles': 'dark grey; green sprinkles',\
                       'red tile ': 'red tile'}, inplace=True)

    #Normalize so max. reflectance = 1
    if norm:
        for column in df:
            if column != 'Wavelength':
                df[column] = df[column] / df[column].max()

    return df


def get_ecostress(path, kind, norm=True, describe=False):
    """
    Returns a dictionary of spectra from the ecostress spectral library
    Parameters:
        - path, str
          the path to the directory containing the spectra
        - kind, str
          the prefix denoting the type of spectra to be read. For example,
          kind="manmade.roofingmaterial.metal" will read all spectra whose
          filenames begin with this string.
    Returns:
        - spectra, dict
          dictionary whose keys are the name of the material (e.g. Aluminum metal)
          and its sample number (so, key=(material, sample)) and whose values are
          dataframes of wavelengths and reflectance values
    """

    spectra = {}

    #find the relevant files
    files = glob.glob(path+kind+'*')
    for f in files:

        #find the name of the material within the file's header info
        with open(f, 'r', encoding='utf8') as info:
            for line in info:
                if 'Name' in line:
                    material = line.split(':')[1].strip()
                if 'Sample No.' in line:
                    sample = line.split(':')[1].strip()
                if 'Description' in line:
                    description = line.split(':')[1].split('. Original')[0]

        #read the spectrum
        if describe:
            print(f"Reading spectrum of {material},{description}")
        df = pd.read_csv(f, header=20, sep='\t', names=['Wavelength, microns', 'Reflectance, %'])

        #create wavelength in nm, drop microns
        df['Wavelength, nm'] = df['Wavelength, microns'] * 1000
        df.drop(columns=['Wavelength, microns'], inplace=True)

        #normalize so max reflectance = 1, or convert from % to fractional reflectance
        for column in df.columns:
            if column != 'Wavelength, nm':
                if norm:
                    df[column] = df[column] / df[column].max()
                else:
                    df[column] = df[column] / 100

        #add the spectrum to the dictionary, identifying it by both its name and
        #sample number to avoid having duplicate keys and thus excluding some spectra
        spectra[(material, sample)] = df

    return spectra


def get_moreira(path):
    """
    Returns a dataframe containing the PVC spectrum from Moreira & Galvao (2010),
    fig. 5
    """

    #Find the spectra of the 3 separate spectral segments
    files = glob.glob(path+'moreira*')

    #Put into df with 6 columns, 2 for each wavelength-reflectance segment pair
    #need to sort <files> or lose data from segment 1
    df = pd.DataFrame()
    for n, f in enumerate(sorted(files)):
        data = pd.read_csv(f, header=None)
        df[f'Wavelength {n}'] = data[0]
        df[f'Reflectance {n}'] = data[1]

    return df

if __name__ == "__main__":
    pass
