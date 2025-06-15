
#%%-----------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------
from pathlib import Path

import pandas as pd
import numpy as np
import random

import sklearn
# import shuffle 
from sklearn.model_selection import StratifiedGroupKFold

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# %%------------------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------------------

class RSS_Bagging(object):
    """
    Class to handle the RSS bagging process.
    """
    def __init__(self, df: pd.DataFrame, n_bag:int , n_classes:int ,seed: int = 42):
        self.df = df
        self.seed = seed
        self.n_bag = n_bag
        self.n_classes = n_classes

        self.classes = [f"class_{i}" for i in range(1, self.n_classes)]

        # Define the proportions (lower value) for the reference samples
        self.props = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]

        # Training and validation sets
        self.training_sets = [1, # [prop, prop + 5[
                              2, # [prop, 100[
                              3  # [100]
                              ]
        self.validation_sets = [1, # [50, 100]
                                2, # [100]
                                3  # [prop, 100]
                                ]

        # Define the setups for training and validation
        self.Setups = {
            "Setup_1": 
            { "training_set": self.training_sets[0],
            "validation_set": self.validation_sets[0]
            },
            
            "Setup_2": 
            { "training_set": self.training_sets[0],
            "validation_set": self.validation_sets[1]
            },
            
            "Setup_3": 
            { "training_set": self.training_sets[1],
            "validation_set": self.validation_sets[0]
            },

            "Setup_4": 
            { "training_set": self.training_sets[1],
            "validation_set": self.validation_sets[1]
            },
            "Setup_5":
            { "training_set": self.training_sets[0],
            "validation_set": self.validation_sets[2]
            },
            "Setup_6":
            { "training_set": self.training_sets[1],
            "validation_set": self.validation_sets[2]
            },
            "Setup_7":
            { "training_set": self.training_sets[2],
            "validation_set": self.validation_sets[1]
            }
        }

    def create_bag(self):
        """
        Create a bag of samples based on the proportions defined in the class.
        The bagging process will sample from the dataframe based on the defined proportions and classes.
        Returns:
            pd.DataFrame: A dataframe containing the sampled bag of data.
        """
        df_bag = self.df.copy()

        df_bag['prop_class'] = (df_bag['prop']*100//5)*5

        df_bag = df_bag.groupby(['prop_class', 'modal_class']).apply(lambda x: x.sample(n=self.n_bag, random_state=self.seed)).reset_index(drop =True)
        #df_bag = df_bag.drop(columns=['prop_class'])

        return df_bag
    
    def create_folds(self, df_bag:pd.DataFrame, setup:int, prop:float, random_state:int = 42):
        """
        Create folds for the bagging process based on the setup and proportion.
        Args:
            setup (int): The setup number to use for creating folds.
            prop (float): The proportion of samples to use for the training set.
        Returns:
            pd.DataFrame: A dataframe containing the folds for the bagging process.
        """

        # Define the training and validation sets based on the setup
        training_set = self.Setups[f"Setup_{setup}"]['training_set']
        validation_set = self.Setups[f"Setup_{setup}"]['validation_set']

        n_train = int(self.n_bag * 0.7) +1
        n_val = self.n_bag - n_train

        if training_set == 1:
            df_train = df_bag[ (df_bag['prop'] >= prop) & (df_bag['prop'] < prop + 0.05) ]
            df_train = df_train.groupby('modal_class').apply(lambda x: x.sample(n=n_train, random_state=random_state)).reset_index(drop=True)
        elif training_set == 2:
            df_train = df_bag[ (df_bag['prop'] >= prop) & (df_bag['prop'] <= 1) ]
            df_train = df_train.groupby('modal_class').apply(lambda x: x.sample(n=n_train, random_state=random_state)).reset_index(drop=True)


        if validation_set == 1:
            df_val = df_bag[ (df_bag['prop'] >= 0.5) & (df_bag['prop'] <= 1) ]
            df_val = df_val.groupby('modal_class').apply(lambda x: x.sample(n=n_val, random_state=random_state)).reset_index(drop=True)
        elif validation_set == 2:
            df_val = df_bag[ (df_bag['prop'] == 1) ]
            df_val = df_val.groupby('modal_class').apply(lambda x: x.sample(n=n_val, random_state=random_state)).reset_index(drop=True)
        elif validation_set == 3:
            df_val = df_bag[ (df_bag['prop'] >= prop) & (df_bag['prop'] <= 1) ]
            df_val = df_val.groupby('modal_class').apply(lambda x: x.sample(n=n_val, random_state=random_state)).reset_index(drop=True)

        return df_train, df_val




            












#%%

