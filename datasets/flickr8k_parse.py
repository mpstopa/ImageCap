from PIL import Image
from tqdm import tqdm_notebook as tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_filenames_with_all_captions(captions_file, images_path):
    """ 
    Generates dictionary contatining image filenames with corresponding captions
    
    Parameters
    -----------
    captions_file
        File from Flckr8k dataset containing all the filenames with captions
        
    images_path
        Path to folder with Flickr8k images
    -----------
    """
    initial_df = pd.read_csv(captions_file, sep='\t', header=None) # \t is regex for tab
    captions_df = pd.DataFrame([filename[:-2] for filename in initial_df[0]], columns=['filename'])
    captions_df['caption'] = initial_df.iloc[:, 1]
    
    filenames_with_all_captions = {}
    for i, row in captions_df.iterrows():
        tmp_full_path = os.path.join(images_path, row[0]) # I think row[0] is the jpg name in Flickr8k.token.txt
        if tmp_full_path in filenames_with_all_captions:
            filenames_with_all_captions[tmp_full_path].append(row[1]) # row[1] must be the caption.
        else:
            filenames_with_all_captions[tmp_full_path] = [row[1]]
    return filenames_with_all_captions


# In[41]:


def generate_set(txt, filenames_with_all_captions, images_path):
    """
    For a given txt file with image filenames, creates a subsample of the dictionary with all the filenames
    
    Parameters
    -----------
    txt
        txt file with filenames for a certain dataset
        
    filenames_with_all_captions
        dictionary contatining image filenames with corresponding captions
        
    images_path
        Path to folder with Flickr8k images
    -----------
    """
    with open(txt, 'r') as set_file:
        tmp_set = set_file.readlines() # txt is either Flickr8k.trainImages.txt,devImages or testImages.
        tmp_set = [os.path.join(images_path, filename.replace('\n', '')) for filename in tmp_set]
    dict_set = {}
    for filename in tmp_set:
        dict_set[filename] = filenames_with_all_captions[filename]
        
    return dict_set # this is a dictionary that is a subset of the filenames_with_all_captions dictionary

def make_list_of_captions(filenames_with_all_captions):
    """
    Extracts captions from the list of dictionaries with filenames and captions
    
    Parameters:
    -----------
    filenames_with_all_captions: list
        List of dictionaries containing images with the corresponding captions
    -----------
    """
    captions = []
    for _, val in filenames_with_all_captions.items(): # items is a python method on dictionaries
        captions.append(val)
    return captions
