
#----------------------------------------------------------------------------------------
## DESCRIPTION
#----------------------------------------------------------------------------------------
# This script contains the class pixel_counts, which is used to calculate the Relative Spectral Similarity (RSS) index.

# Author: Sabrina Paes Leme Passos Corrêa
# Creation: 2024-06-03
# Modified: 2024-06-04


#----------------------------------------------------------------------------------------
## IMPORTS
#----------------------------------------------------------------------------------------
from pathlib import Path
from typing import Any
import rasterio
import rasterio.mask
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from shapely import box
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
#----------------------------------------------------------------------------------------



#----------------------------------------------------------------------------------------
## RSS CLASS
#----------------------------------------------------------------------------------------
class RSS_CandidateSamples(object):

    def __init__(self,
                 coarse_file: Path | str,
                 fine_file: Path | str = None,
                 n_classes: int = 4, 
                 n_jobs: int = 1, 
                 
                 ):
        """
        Initialize the RSS_CandidateSamples class.
        Args:
            n_classes (int): Number of classes to be sampled.
            n_samples (int): Number of samples to be taken from each class.
            n_jobs (int): Number of parallel jobs to run.
        """
        self.bands          = ['green', 'red', 'blue', 'nir', 'ndvi']
        self.fine_file      = fine_file 
        self.n_classes      = n_classes
        self.n_jobs         = n_jobs


        with rasterio.open(coarse_file) as src:
            self.coarse_profile = src.profile # Get the profile of the coarse raster

        # Getting information from the coarse profile
        self.coarse_x  = self.coarse_profile['transform'][2] # x-coordinate of the top-left corner
        self.coarse_y  = self.coarse_profile['transform'][5] # y-coordinate of the top-left corner
        self.stepx     = self.coarse_profile['transform'][0] # pixel size in the x direction
        self.stepy     = self.coarse_profile['transform'][4] # pixel size in the y direction
        self.width     = self.coarse_profile['width']        # width of the raster (number of pixels in the x direction)
        self.height    = self.coarse_profile['height']       # height of the raster (number of pixels in the y direction)
        self.crs        = self.coarse_profile['crs']        # coordinate reference system of the raster

    def _get_pixel_counts(self, uniques):
        """
        Get the counts of each class in a coarse pixel.
        Args:
            uniques (tuple): A tuple containing unique values and their counts from the fine raster.
            n_classes (int): Number of classes in the coarse raster.
        Returns:
            np.ndarray: An array of counts for each class in the coarse pixel as well as the sum of pixels.
        """
            # Getting the counts of each class
        counts_tmp = np.zeros((1, self.n_classes), dtype=np.int32)  # +1 for the total count
        for i in range(1,self.n_classes+1):
            if i in uniques[0]:
                id = np.where(uniques[0]==i)[0][0]
                counts_tmp[0][i-1] = uniques[1][id]
            else: 
                counts_tmp[0][i-1] = 0
            id = None
        return counts_tmp[0]

    def get_samples(self, m, n):
        """
        Get the samples from the coarse raster for a specific pixel.
        Args:
            m (int): Row index of the coarse pixel.
            n (int): Column index of the coarse pixel.
        Returns:
            tuple: A tuple containing the centroid of the coarse pixel and the counts of each class in that pixel.
        """
        coarse_bbox = box(self.coarse_x + n*self.stepx,
                          self.coarse_y + m*self.stepy,
                          self.coarse_x + n*self.stepx + self.stepx,
                          self.coarse_y + m*self.stepy + self.stepy)
        coarse_pixel_gdf = gpd.GeoDataFrame(geometry=[coarse_bbox], crs=self.crs)

        with rasterio.open(self.fine_file) as src:
            fine_image, _ = rasterio.mask.mask(src, coarse_pixel_gdf.geometry, crop=True, all_touched=True)

        uniques = np.unique(fine_image[0], return_counts=True)
        centroid = coarse_pixel_gdf.centroid[0]
        counts = self._get_pixel_counts(uniques)

        return centroid, counts
    
    def direct_process(self):
        """
        Direct processing to get samples from the coarse raster.
        Returns:
            tuple: A tuple containing two numpy arrays:
                - centroids: Array of centroids for each coarse pixel.
                - counts: Array of pixel counts for each class in each coarse pixel.
        """
        centroids = []
        counts = []

        for n in tqdm(range(self.width), desc='Processing coarse pixels'):
            for m in range(self.height):
                centroid, count = self.get_samples(n, m)
                centroids.append(centroid)
                counts.append(count)

        return np.array(centroids), np.array(counts)
    
    def paralel_process(self, n_jobs = 4):
        """
        Parallel processing to get samples from the coarse raster.
        Args:
            n_jobs (int): Number of parallel jobs to run.
        Returns:
            tuple: A tuple containing two numpy arrays:
                - centroids: Array of centroids for each coarse pixel.
                - counts: Array of pixel counts for each class in each coarse pixel.
        """
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.get_samples)(n, m)
            for n in range(self.width) for m in range(self.height)
        )
        centroids, counts = zip(*results)
        return np.array(centroids), np.array(counts)
    
    def rss_candidate_samples(self, paralel: bool = False):
        """
        Calculate the Reference Sample Selection (RSS) index for candidate samples.
        Returns:
            pd.DataFrame: A DataFrame containing the pixel counts, centroids, modal class, proportion, and weight for each candidate sample.
        """
        # Counting the number of pixels in each class for each coarse pixel
        if paralel:
            centroids, counts = self.paralel_process(self.n_jobs)
        else:
            centroids, counts = self.direct_process()

        # Convert the results to a DataFrame
        df = pd.DataFrame(counts, columns=[f'class_{i}' for i in range(self.n_classes)])

        df['sum_pixels'] = df.sum(axis=1)  # Total number of pixels in each coarse pixel
        modal_class = np.argmax(counts, axis=1)  # Get the index of the modal class for each coarse pixel

        df['modal_class'] = modal_class +1  # Add 1 to match the class numbering starting from 1 instead of 0
        
        n_samples = np.max(df['sum_pixels'])  # Get the maximum number of pixels across all coarse pixels

        df['n_modal_pixels'] = df.apply(lambda row: row[f'class_{row["modal_class"] - 1}'], axis=1)  # Get the number of pixels in the modal class for each coarse pixel
        df['prop']   = df['n_modal_pixels'] / df['sum_pixels'] # Proportion of the modal class relative to the total count
        df['weight'] = df['sum_pixels']  / n_samples # Calculate the weight of each sample based on the total count and the number of samples

        df['weight'] = df['weight'].clip(upper=1.0)  # Ensure weights do not exceed 1.0

        df['centroid'] = centroids # x-coordinate of the centroid

        df['centroid'] = df['centroid'].apply(lambda x: (x.x, x.y))  # Convert centroid to tuple of (x, y)
        return df
        
