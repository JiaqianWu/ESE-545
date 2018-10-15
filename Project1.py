# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 20:02:21 2018

@author: wujiaqian
"""

import numpy as np
import pandas as pd
import csv
import time
import matplotlib.pyplot as plt
import logging

# Setting up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("project1.log")
fh.setLevel(logging.DEBUG)
# Set streamhandler to print log，level debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# Set logging format
formatter = logging.Formatter("%(asctime)s - %(name)s %(funcName)s - %(lineno)d - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# Add handler to each logger
logger.addHandler(ch)
logger.addHandler(fh)


class MinHashLSH:
    def __init__(self,num_perm=500,threshold=0.35,data_array=np.array([])):
        self.data_ls = [] # Read in data as list
        self.data_arr = data_array # Transform data to numpy array
        self.arr_rows = 0 # Number of rows in array
        self.arr_cols = 0 # Number of cols in array
        self.sig_matrix = np.array([]) # Signature matrix
        self._integration_precision = 0.001 # Stride for integration
        self.nearest_neighbors = set() # Nearest neighbors 
        self.lsh_candidates = set() # Candidates 
        self.pst_dict = dict()  # Positional array recording row, col for array with 1
        self.threshold = threshold # Jaccard distance threshold
        self.num_perm = num_perm # Number of permutations
        self.coeff_a = np.zeros(self.num_perm) # Coefficient a for hash function (a*x+b)/prime
        self.coeff_b = np.zeros(self.num_perm) # Coefficient b for hash function (a*x+b)/prime
        self.prime_num = (1<<61)-1 # Parameter prime for hash function (a*x+b)/prime
        self.buckets = [] # Bucket to hold similar groups
        self.movie_ls = np.array([]) # The order of movie_id in pivot dataframe
    
    def read_data(self,filepath="Netflix_data.txt"):
        """
        Read data from .txt file and format in to a 1-d list
        
        """
        logger.info('Reading data from Netflix data') 
        NF_data_ls =[]
        movie_id = ''
        with open(filepath) as file:
#            count = 0
            for i in file.readlines():
#                count+=1
#                if count>10000:
#                    break
                if i[-2] == ':':
                    movie_id = i[:-2]
                else:
                    content = [movie_id]+i[:-2].split(',')
                    if int(content[2]) >= 3:
                        NF_data_ls.append(content) 
        self.data_ls = NF_data_ls
        logger.info('Reading completed')
    
    
    def _init_array(self):
        """
        Initiate a 2-d numpy array from 1-d list
        """
        logger.info('Creating pandas dataframe. Drop column DATE')
        NF_df = pd.DataFrame(self.data_ls,columns=['Movie_id','User_id','Rating','Date']).drop(['Date'],axis=1)
        logger.info('Setting all rating to 1')
        NF_df['Rating']=1
        logger.info('Count users rated less than 20 movies')
        count_df = NF_df.groupby(['User_id']).count()['Movie_id'].reset_index()
        count_df = count_df[count_df['Movie_id']<=20]
        count_ls = list(count_df['User_id'])
        logger.info('Select users rated less than 20 movies')
        NF_df = NF_df[NF_df['User_id'].isin(count_ls)]
        logger.info('Create Movieid by userid matrix')
        NF_pivot = pd.pivot_table(NF_df, values='Rating', index=['Movie_id'],columns=['User_id'], aggfunc=np.sum)
        NF_pivot = NF_pivot.fillna(value=0)
        self.movie_ls = NF_pivot.index.values
        logger.info('Get np.array from dataframe')
        self.data_arr = NF_pivot.values.astype(int)
        self.arr_rows, self.arr_cols = self.data_arr.shape
        logger.info('Matrix created {} * {}'.format(self.arr_rows, self.arr_cols))
        
        
    def save_array(self,outfile_path="NF_array.npy",movie_lspath = "movie_ls.npy"):
        # save array to .npy file, default "NF_array.npy"
        if outfile_path[-4:] != ".npy":
            raise ValueError("Please output to file in format of ***.npy")
            logging.exception("Output file format error")
        np.save(outfile_path, self.data_arr)
        np.save(movie_lspath, self.movie_ls)
        
        
    def load_array(self,infile_path="NF_array.npy",movie_lspath = "movie_ls.npy"):
        # load array from .npy file, default "NF_array.npy"
        if infile_path[-4:] != ".npy":
            raise ValueError("Please input from file in format of ***.npy")
            logging.exception("Input file format error")
        self.data_arr = np.load(infile_path)
        self.movie_ls = np.load(movie_lspath)
        self.arr_rows,self.arr_cols = self.data_arr.shape
        
        
    def _hamming_dist(self,a,b):
        
        """
        Calculate hamming distance of two vectors using bitwise 'xor' and 'or'
        Args:
            a,b: Two 1-d vectors of same length
        """
        
        if len(a) != len(b):
            raise ValueError("Vector of different size")
                
        return (a^b).sum()/(a|b).sum()
    
    
    def rand_users(self,num=10000):
        
        """
        Randomly select num pairs of users and calculate their Jaccard distance
        
        Args:
            num: The number of user we randomly select
        """
        
        logger.info('Randomly select {} users'.format(num))
        result = []
        for i in range(num):
            r,n = np.random.randint(0,self.arr_cols,size=2)
            result.append(self._hamming_dist(self.data_arr[:,r],self.data_arr[:,n]))
        logger.info('Calculating average distance')
        print("Average distance: " + str(np.mean(result)))
        logger.info('Calculating minimum distance')
        print("Minimum distance: " + str(np.min(result)))
        return result
    
    
    def rand_users_plot(self):
        
        """
        Plot histogram according to the randomly picked users
        """
        
        logger.info('Plotting histogram according to the randomly picked users')
        plot_data = self.rand_users()
        fig, ax = plt.subplots(figsize=[8,6])
        ax.set_title("Jaccard Distances")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Number")

        plt.hist(plot_data, bins=100)  # arguments are passed to np.histogram
        plt.title("Jaccard Distances")
        plt.show()
        
        
    def _find_next_prime(self,n):
        
        """
        Find the closest prime number larger than n
        
        """
        
        for p in range(n, 2*n):
            for i in range(2, p):
                if p % i == 0:
                    break
            else:
                return p
        return None
    
    def _pick_coeffs(self,n):
        
        """
        Generate an 1 by n array filled with random integers range from 0 to self.arr_rows-1
        
        Args:
            n: Output array dimension
        """
        coeff_List = []
        for i in range(n):
            randIndex = np.random.randint(0, self.arr_rows) 
            coeff_List.append(randIndex)
        return np.array(coeff_List)
    
    
    def position_dict(self):
        
        """
        Generate a dict to store all the positions with 1 in self.data_arr
        """
        
        logger.info("Generating position dict")
        # Create dict recording the row positions of 1's in each column
        position_dict = dict()
        for col in range(self.arr_cols):
            position_dict.update({col:np.where(self.data_arr[:,col]==1)})
        self.pst_dict = position_dict
        logger.info("Position dict generated")
                
    
    def _pickle_dump_position_dict(self,filepath='position_dict.pkl'):
        
        import pickle
        logger.info("Pickle dump dict into pkl files")
        with open('position_dict.pkl', 'wb') as handle:
            pickle.dump(self.pst_dict,handle)
        logger.info("Pickle dump finished")

    def _pickle_load_position_dict(self,filepath='position_dict.pkl'):
        
        import pickle
        logger.info("Pickle load dict from pkl files")
        with open('position_dict.pkl', 'rb') as handle:
            self.pst_dict = pickle.load(handle)
        logger.info("Pickle load finished")
    
    
    
    def MinHash(self,num_perm=500):
        
        """
        Generate a signature matrix of the array num_perm by self.arr_cols
        
        Args:
            num_perm (int, optional): Number of random permutation functions.
            
        Return:
            num_perm by self.arr_cols 2d array
        """
               
        # Find a prime num larger than number of rows
        logger.info("Finding prime number")
        self.prime_num = self._find_next_prime(self.arr_rows)
        # Generate a matrix of dimension num_perm by self.arr_cols, fill with infinity
        logger.info("Generating zero matrix")
        sig_mtrx = np.zeros((num_perm,self.arr_cols))
        # Get coefficients for hash function (ax+b)%prime
        self.coeff_a = self._pick_coeffs(num_perm)
        self.coeff_b = self._pick_coeffs(num_perm)
        logger.info("MinHashing start")
        time1=time.time()      
        for key, value in self.pst_dict.items():
            # Perform (a*x+b)%prime
            sig_mtrx[:,key] = np.amin((self.coeff_a[:,None]*value + self.coeff_b[:,None]) % self.prime_num,axis=1)
        self.sig_matrix = sig_mtrx        
        time2=time.time()
        logger.info("MinHashing finished. Used {} seconds".format(time2-time1))
    
    
    def _integration(self,f, a, b):
        
        """
        Calculate integration of function f range from a to b.
        Stride is set default as 0.01 for this problem
        
        Args:
            f: Function to be integrated
            a,b(int | float): Start point and end point
            
        """
        
        p = self._integration_precision
        area = 0.0
        x = a
        while x < b:
            area += f(x+0.5*p)*p
            x += p
        return area, None
    

    def _false_positive_probability(self,threshold, b, r):
        _probability = lambda s : 1 - (1 - s**float(r))**float(b)
        a, err = self._integration(_probability, 0.0, threshold) 
        return a
    
    
    def _false_negative_probability(self,threshold, b, r):
        _probability = lambda s : 1 - (1 - (1 - s**float(r))**float(b))
        a, err = self._integration(_probability, threshold, 1.0)
        return a
    
    
    def _optimal_param(self,threshold, num_perm, false_positive_weight=0.5, 
                       false_negative_weight=0.5):
        '''
        Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
        of probabilities of false positive and false negative.
        
        Args:
            threshold: Jaccard similarity
            num_perm: Number of permutations
            false_positive_weight(float, 0 to 1, optional)
            false_negative_weight(float, 0 to 1, optional)
            
        '''
        min_error = float("inf")
        opt = (0, 0)
        for b in range(1, num_perm+1):
            max_r = int(num_perm / b)
            for r in range(1, max_r+1):
                fp = self._false_positive_probability(threshold, b, r)
                fn = self._false_negative_probability(threshold, b, r)
                error = fp*false_positive_weight + fn*false_negative_weight
                if error < min_error:
                    min_error = error
                    opt = (b, r)
        return opt
    
    def RB_plot(self, num_perm = 500,threshold=0.35):
        
        """
        Plot band-row figures according to number of permutations, in order to 
        decide the optimal band-row value for LSH
        """
        
        fig, ax = plt.subplots(figsize=[20,15])
        ax.set_title("{} hash functions band-row selection".format(num_perm))
        ax.set_xlabel("Similarity")
        ax.set_ylabel("Pr(hit)")
        ax.grid()
        _probability = lambda s : 1 - (1 - s**float(r))**float(b)
        for r in range(5,15):
            b = num_perm // r
            x = np.arange(0,1,0.001)
            y = []
            for i in x:
                y.append(_probability(i))
            y = np.array(y)
            plt.plot(x,y,label='b={},r={}'.format(b,r))
        plt.axvline(x=0.65,label='threshold 0.65')
        plt.legend()
        plt.show()
    
    def _initialize_array_bucket(self,b):
        
        """
        Generate buckets for each band, buckets are empty dicts
        """
        
        array_buckets = []
        for i in range(b):
            array_buckets.append(dict())
        return array_buckets


    def _LSH_buckets(self,sig_mtrx,b,r):
        
        """
        Hash bands with col to buckets
        """
        
        logger.info("Hashing to buckets started")
        array_buckets = self._initialize_array_bucket(b)
        i = 0
        for j in range(b):
            buckets = array_buckets[j]
            band = sig_mtrx[i:(i+r),:]
            for col in range(self.arr_cols):
                key = hash(band[:,col].tostring())
                if key in buckets:
                    buckets[key].append(col)
                else:
                    buckets.update({key:[col]})
            i+=r
        logger.info("Hashing to buckets finished")
        self.buckets = array_buckets
        return self.buckets
    
    
    def LSH_candidates(self):
        """
        From buckets select candidate pairs
        """
        logger.info("Selecting candidate pairs")
        b,r = self._optimal_param(1-self.threshold, self.num_perm,0.7,0.3)
        LSH = self._LSH_buckets(self.sig_matrix,b,r)
        pair_set = set()
        for i in range(len(LSH)):
            for key, val in LSH[i].items():
                if len(val) > 1:
                    for j in range(len(val)):
                        for k in range(j,len(val)):
                            a,b = val[j],val[k]
                            if a > b:
                                a,b = b,a
                            pair_set.add((a,b))
        self.lsh_candidates = pair_set
        logger.info("Candidate pairs selection finished")
        return self.lsh_candidates
    

    def hamming_filter(self, item):
        
        """
        Filter the candidate pairs whose Jaccard distance is greater than threshold
        
        Args:
            item(tuple): Tuple of two candidates in one pair
            threshold(float,optional): The largest Jaccard distance that can be accepted
        """
        
        i,j = item
        h_dist = self._hamming_dist(self.data_arr[:,i],self.data_arr[:,j])
        return h_dist < self.threshold
    
    def _LSH_dump(self):
        
        """
        Save nearest neighbors to csv file
        """
        
        with open('similarPairs.csv','w') as writeFile:
            similarWriter = csv.writer(writeFile, delimiter=',')
            for i in self.nearest_neighbors:
                similarWriter.writerow(i)
    
    def LSH(self):
        
        """
        Filter out paired candidates with distance larger than threshold
        """
        
        self.nearest_neighbors = set(filter(self.hamming_filter, self.lsh_candidates))
        self._LSH_dump()
        return self.nearest_neighbors

    
    def quiry_person(self,user_mvlist):
        
        """
        Add a new user with list of Movies. Query the nearestest neighbors from buckets
        
        Args:
            user_mvlist: A list of movie_ids(string)  
            threshold: Threshold of jaccard distance from user_mvlist
        """
        logger.info("Querying a new user")
        time1 = time.time()
        pos_list = []
        mv_set = set(user_mvlist)
        logger.info("Creating position list")
        for i in range(self.arr_rows):
            if self.movie_ls[i] in mv_set:
                pos_list.append(i)
        pos_set = set(pos_list)
        most_similar = []
        largest_similarity = 0
        logger.info("Comparing user similarities")
        for key,val in self.pst_dict.items():
            similarity = len(pos_set&set(val[0]))/len(pos_set|set(val[0]))
            if similarity > largest_similarity:
                largest_similarity = similarity
                most_similar = [key]
            elif similarity == largest_similarity:
                most_similar.append(key)
        time2 = time.time()
        logger.info("Querying user completed, {} seconds consumed".format(time2-time1))
        return most_similar
    
    def execute(self,filepath="Netflix_data.txt",num_randusers=10000,num_perm=500):
        
        """
        Excecute code for project requirements
        """
        
        self.read_data(filepath)    
        self._init_array()
        self.rand_users(num_randusers)
        self.rand_users_plot()
        self.position_dict()
        self._pickle_dump_position_dict()
        self.MinHash(num_perm)
        self.RB_plot()
        self.LSH_candidates()
        self.LSH()
        
if __name__ == "__main__":
    ML = MinHashLSH()
    ML.execute()
    mv_input = 'default'
    user_mvlist = []
    while mv_input:
        mv_input = input("Please input a movieid: ")
        user_mvlist.append(mv_input)
    print(user_mvlist[:-1])
    most_similar = ML.quiry_person(user_mvlist)
    print("Most similar user is: ", most_similar)
  
    
    
    
    