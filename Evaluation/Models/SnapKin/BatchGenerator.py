import pandas as pd
import numpy as np
import re
from functools import reduce # Test output
from itertools import combinations # Pseudosampling

class BatchGenerator():    
    def __init__(self, df, fold='F1', fold_cols = ['F1', 'F2', 'F3', 'F4', 'F5'], exclude=1,
                 label='y.MAPK1', class_columns=['y.MAPK1', 'y.MTOR'], pseudo=True):
        self.df = df
        self.fold = fold
        self.fold_cols = fold_cols
        self.exclude = exclude
        self.label = label
        self.class_columns = list(filter(re.compile('^y\..*').match, list(df[0].columns)))      
        self.pseudo = pseudo                                                         
        
        self.X_train_pos, self.X_train_neg = [], []
        self.X_test_pos, self.X_test_neg = [], []
        self.X_test_pos_full = []
        
        # Retrieve Positive and Negative train sites
        self.train_pos_sites, self.train_neg_sites = set(), set()       
        self.test_pos_sites, self.test_neg_sites = set(), set()     
        self.test_pos_sites_full = set()
        
        # Get sites (id) for training and test sets
        for dat in df:
            self.train_pos_sites.update(dat.loc[(dat[fold] != exclude) & (dat[label] == 1),'site'])
            self.train_neg_sites.update(dat.loc[(dat[fold] != exclude) & (dat[label] == 0),'site'])
            
            self.test_pos_sites.update(dat.loc[(dat[fold] == exclude) & (dat[label] == 1),'site'])
            self.test_neg_sites.update(dat.loc[(dat[fold] == exclude) & (dat[label] == 0),'site'])
            self.test_pos_sites_full.update(dat.loc[(dat[fold] == -1),'site'])
            
            
        # Set training and test set
        self.train_pos_sites, self.train_neg_sites = list(self.train_pos_sites), list(self.train_neg_sites)
        self.set_train()

        self.test_pos_sites, self.test_neg_sites = list(self.test_pos_sites), list(self.test_neg_sites)
        self.set_test()

    def generate_batch(self, batch_size=None, replace=False):
        '''
            Generate batch with or without replacement sampling sites.
        '''
        if batch_size == None:
            batch_size = self.num_pos
        # Generate a training batch sample
        if (self.num_pos < batch_size):
            raise Exception('Batch size larger than size of positive training set.')
        num_pos = np.random.randint(batch_size+1)
        # Sample positives
        pos_idx = np.random.choice(self.train_pos_sites, num_pos, replace=replace)
        X_pos, y_pos = BatchGenerator.get_filtered(pos_idx, self.X_train_pos, isPositive=True)
        
        # Sample negatives
        neg_idx = np.random.choice(self.train_neg_sites, batch_size-num_pos, replace=replace)
        X_neg, y_neg = BatchGenerator.get_filtered(neg_idx, self.X_train_neg, isPositive=False)
        
        X = [pos.append(neg) for pos, neg in zip(X_pos, X_neg)]
        y = y_pos + y_neg        
        
        return (X, y)
    
    def generate_batch_full(self):
        '''
            Generate batch consisting of 
            - All positives + pseudopositive sites
            - Negative sites (of equal number to positives)
        '''
        # Positives
        pos_idx = self.train_pos_sites
        X_pos, y_pos = BatchGenerator.get_filtered(pos_idx, self.X_train_pos, isPositive=True)
        
        # Sample negatives
        neg_idx = np.random.choice(self.train_neg_sites, self.num_pos)
        X_neg, y_neg = BatchGenerator.get_filtered(neg_idx, self.X_train_neg, isPositive=False)
        
        X = [pos.append(neg) for pos, neg in zip(X_pos, X_neg)]
        y = y_pos + y_neg                                          

        return (X, y)
    
    def get_train(self):
        '''
            Retrieve train set.
        '''
        return [df.loc[df[self.fold] != self.exclude]   for df in self.df]
    
    def get_test(self, dataset=None, individual=False, returnSites=False ):
        '''
            Retrieve test set.
        
            Param
            dataset     ::Obtain sites only found in dataset (list index)
            individual  ::If other datasets (not dataset specified) should be 0 (unused in model)
        '''
        X, y = [], []
        sites = None
        if dataset != None:
            if dataset >= len(self.df):
                raise Exception("Dataset does not exist")   
                
            sites = self.X_test_pos[dataset].append(self.X_test_neg[dataset])['site'].to_list()
            if individual: # Dataset Test Set and 0s for other datasets
                individual_dat = self.X_test_pos[dataset].append(self.X_test_neg[dataset]).drop(columns=['site'])
                y = [1]*self.X_test_pos[dataset].shape[0] + [0]*self.X_test_neg[dataset].shape[0]
                num_obs = individual_dat.shape[0]
                for i in range(len(self.df)):
                    if i == individual:
                        X.append(individual_dat)
                    else:
                        tmp_dat = self.X_test_pos[i].head(0).drop(columns=['site'])
                        tmp_dat = tmp_dat.append([pd.Series(0, index=tmp_dat.columns) for _ in range(num_obs)])
                        X.append(tmp_dat)
            else: # Only get sites from specific individual test set and any found in others
                comb_test = [pos.append(neg) for pos, neg in zip(self.X_test_pos, self.X_test_neg)]
                X, _ = BatchGenerator.get_filtered(sites, comb_test)
                y = [1]*self.X_test_pos[dataset].shape[0] + [0]*self.X_test_neg[dataset].shape[0]
        else:
            sites = self.test_pos_sites + self.test_neg_sites
            comb_test = [pos.append(neg) for pos, neg in zip(self.X_test_pos, self.X_test_neg)]
            X, _ = BatchGenerator.get_filtered(sites, comb_test)
            y = [1]*len(self.test_pos_sites) + [0]*len(self.test_neg_sites)
        if returnSites:
            return (X, y, sites)
        return (X, y)
    
    def get_test_selective(self, dataset=None, individual=False, isPositive=True, returnSites=False, subsample=False):
        X, y = [], []
        sites = None
        if dataset != None:
            if dataset >= len(self.df):
                raise Exception("Dataset does not exist")   
                
            sites = self.X_test_pos[dataset]['site'].to_list() if isPositive else self.X_test_neg[dataset]['site'].to_list()
            if individual: # Dataset Test Set and 0s for other datasets
                individual_dat = self.X_test_pos[dataset].drop(columns=['site']) if isPositive else self.X_test_neg[dataset].drop(columns=['site'])
                y = [1]*individual_dat.shape[0] if isPositive else [0]*individual_dat.shape[0]
                num_obs = individual_dat.shape[0]
                for i in range(len(self.df)):
                    if i == individual:
                        X.append(individual_dat)
                    else:
                        tmp_dat = self.X_test_pos[i].head(0).drop(columns=['site'])
                        tmp_dat = tmp_dat.append([pd.Series(0, index=tmp_dat.columns) for _ in range(num_obs)])
                        X.append(tmp_dat)
            else: # Only get sites from specific individual test set and any found in others
                comb_test = [pos.append(neg) for pos, neg in zip(self.X_test_pos, self.X_test_neg)]
                X, _ = BatchGenerator.get_filtered(sites, comb_test)
                y = [1]*len(sites) if isPositive else [0]*len(sites)
        else:
            sites = self.test_pos_sites if isPositive else self.test_neg_sites
            # Subsample equal size of positives
            if subsample:
                sites = np.random.choice(sites, len(self.test_pos_sites), replace=False).tolist()
            comb_test = [pos.append(neg) for pos, neg in zip(self.X_test_pos, self.X_test_neg)]
            X, _ = BatchGenerator.get_filtered(sites, comb_test)
            y = [1]*len(sites) if isPositive else [0]*len(sites)
        
        if returnSites:
            return (X,y, sites)
        return (X, y)
    
    def get_test_selective_full(self, dataset=None, individual=False):
        X, y = [], []
        sites = None
        if dataset != None:
            if dataset >= len(self.df):
                raise Exception("Dataset does not exist")   
                
            sites = self.X_test_pos_full[dataset]['site'].to_list() 
            if individual: # Dataset Test Set and 0s for other datasets
                individual_dat = self.X_test_pos_full[dataset].drop(columns=['site']) 
                y = [1]*individual_dat.shape[0] 
                num_obs = individual_dat.shape[0]
                for i in range(len(self.df)):
                    if i == individual:
                        X.append(individual_dat)
                    else:
                        tmp_dat = self.X_test_pos_full[i].head(0).drop(columns=['site'])
                        tmp_dat = tmp_dat.append([pd.Series(0, index=tmp_dat.columns) for _ in range(num_obs)])
                        X.append(tmp_dat)
            else: # Only get sites from specific individual test set and any found in others
                comb_test = self.X_test_pos_full
                X, _ = BatchGenerator.get_filtered(sites, comb_test)
                y = [1]*len(sites) 
        else:
            sites = self.test_pos_sites_full 
            comb_test = self.X_test_pos_full
            X, _ = BatchGenerator.get_filtered(sites, comb_test)
            y = [1]*len(sites) 

        return (X,y,sites)
    
    def set_train(self):
        for dat in self.df:
            tmp = dat.loc[dat['site'].isin(self.train_pos_sites)]
            tmp_X = tmp.drop(columns=self.fold_cols + self.class_columns).reset_index(drop=True)
            if self.pseudo:
                tmp_pseudos = self.get_pseudo(tmp_X)
                self.X_train_pos.append(tmp_X.append(tmp_pseudos))
            else:
                self.X_train_pos.append(tmp_X)
            tmp_X_neg = dat.loc[dat['site'].isin(self.train_neg_sites)].drop(columns=self.fold_cols + self.class_columns)
            self.X_train_neg.append(tmp_X_neg)
        self.num_pos, self.num_neg = len(self.train_pos_sites), len(self.train_neg_sites) 
        
    def set_test(self):
        for dat in self.df:
            tmp = dat.loc[dat['site'].isin(self.test_pos_sites)]
            tmp_X = tmp.drop(columns=self.fold_cols + self.class_columns)
            self.X_test_pos.append(tmp_X)
            tmp_X_neg =  dat.loc[dat['site'].isin(self.test_neg_sites)].drop(columns=self.fold_cols + self.class_columns)
            self.X_test_neg.append(tmp_X_neg)
            tmp_full = dat.loc[dat['site'].isin(self.test_pos_sites_full)].drop(columns=self.fold_cols + self.class_columns)
            self.X_test_pos_full.append(tmp_full)
        
    ## Helper Functions ##
    def get_pseudo(self, df):
        '''
            Generate pseudos and return dataframe of pseudos
            - Pseudo generation via pairwise means
        '''
        
        if (df.shape[0] == 1): # No Pseudos
            return None
        
        dat = df.drop(columns=['site'])
        pseudos = []
        for pair in combinations(dat.index, 2):
            site_pair = df.loc[list(pair),'site']
            pseudo_site = ''.join(map(lambda x: x[1], sorted([(site.split(';')[0], site) for site in site_pair], key=lambda x: x[0])))
            self.train_pos_sites.append(pseudo_site)
            pair_avg = dat.loc[list(pair)].mean()
            pair_avg['site'] = pseudo_site
            pseudos.append(pair_avg)
            
        pseudos = pd.DataFrame(pseudos)
        return pseudos
        
    def get_filtered(sites, df, isPositive=True):
        '''
            Generate list of dataframes filtered by sites, and 0 rows for missing sites.
        '''
        # Dataframe of sites for merge
        site_df = pd.DataFrame(sites, columns=['site'])
        # Datasets array of only sites
        filtered_df = [dat.loc[dat['site'].isin(sites)] for dat in df] 
        # Add missing sites and drop sites column
        X = [site_df.merge(dat, on=['site'], how='left').fillna(0).drop(columns=['site']) for dat in filtered_df]
        out = 1 if isPositive else 0
        num_obs = len(sites)
        y = [out]* num_obs 
        return (X, y)