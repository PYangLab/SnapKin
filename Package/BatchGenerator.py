import pandas as pd
import numpy as np
import re
from functools import reduce # Test output
from itertools import combinations # Pseudosampling

class BatchGenerator(): 
    '''
        Class for generating subsamples from an input data file.
    '''  
    def __init__(self, X, y, pseudo=True):
        '''
            X :: pandas dataframe consisting of sequence motif information and phosphoproteomic data 
            y :: binary list of positive (1) and negative (0) sites
            pseudo :: boolean denoting whether pseudo-positive sites are generated
        '''
        self.X = X 
        self.y = y 
        self.pseudo = True
        self.set_data()


    def set_data(self):
        self.X_pos = self.X[self.y == 1]
        self.X_neg = self.X[self.y == 0]

        if self.pseudo:
            self.X_pseudo = self.get_pseudo(self.X_pos)

    def get_batch(self):
        '''
            Generate a batch by subsampling
        '''
        num_pos = self.X_pos.shape[0] + self.X_pseudo.shape[0]
        neg_idx = np.random.choice(self.X_neg.index, size=num_pos, replace=False)

        neg_df = self.X_neg.loc[neg_idx] 

        # Generate Labels 
        labels = [1] * num_pos + [0] * num_pos 
        df = self.X_pos.append(self.X_pseudo).append(neg_df)

        # Convert to numpy data type 
        df, labels = df.to_numpy(),  np.array(labels)
        
        return (df, labels)

    ## Helper Functions ##
    def get_pseudo(self, dat):
        '''
            Generate pseudos and return dataframe of pseudos
            - Pseudo generation via pairwise means
        '''
        if (dat.shape[0] == 1): # No Pseudos
            return None
        pseudos = []
        for pair in combinations(dat.index, 2):
            pair_avg = dat.loc[list(pair)].mean()
            pseudos.append(pair_avg)
            
        pseudos = pd.DataFrame(pseudos)
        return pseudos

    