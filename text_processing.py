import numpy as np
import os
import pickle
import re # re is regex for python


class Vocabulary(object):
    def __init__(self):
        self.number_of_words = 1
        self.word_to_id = dict()
        self.id_to_word = dict()
        self.word_counter = dict()
        self.word_to_id['<un>'] = 0
        self.id_to_word[0] = '<un>'
        self.word_counter['<un>'] = 0 
    
    def add_word(self, word):
        """
        Adds a word in the vocabulary

        :param word: the word to add
        """
        if word not in self.word_to_id:
            self.word_to_id[word] = self.number_of_words
            self.id_to_word[self.number_of_words] = word
            self.word_counter[word] = 1
            self.number_of_words += 1
        else:
            self.word_counter[word] += 1
                
    def get_id_by_word(self, word):
        """
        Returns id for an input word

        :param word: The word corresponding to the requested id
        :return: requested id
        """
        return self.word_to_id[word]
    
    def get_word_by_id(self, idx):
        """
        Returns a word for an input id

        :param idx: The id of the requested word
        :return: requested word
        """
        return self.id_to_word[idx]
    
    def get_word_frequency(self, word):
        """
        Return a frequncy for a given word
        
        :param word: input word
        :return: frequency
        """
        return self.word_counter[word]
    
    def get_most_frequent_words(self, word_num, pct=False):
        """
        Return word_num most frequent words
        
        :param word_num: the number of words to return 
        :return: dictionary of size word_num
        """
        tmp_dict = self.word_counter
        if '<sos>' and '<eos>' in tmp_dict:
            tmp_dict.pop('<sos>')
            tmp_dict.pop('<eos>')
        freq_list = sorted(tmp_dict.items(), key=lambda x: x[1], reverse=True)
        if pct:
            sum_words = 0
            freq_list_pct = []
            for pair in freq_list:
                sum_words += pair[1]
            for i, pair in enumerate(freq_list):
                freq_list_pct.append(str(pair[0]) + ' - ' + str(round(pair[1] / sum_words * 100, 2)) + '%')
            return freq_list_pct[:word_num]
        else:
            return freq_list[:word_num]
    
    def save_vocabulary(self, filename_word_to_id='word_to_id.pickle', filename_id_to_word='id_to_word.pickle', filename_counter='word_counter.pickle'):
        """
        Saves vocabulary dictionaries to pickle files

        :param filename_word_to_id: The filename for word_to_id dictionary
        :param filename_id_to_word: The filename for id_to_word dictionary
        """
        if not os.path.exists('./vocabulary'):
            os.mkdir('./vocabulary')
        path_word_to_id = os.path.join('./vocabulary/', filename_word_to_id)
        path_id_to_word = os.path.join('./vocabulary/', filename_id_to_word)
        path_word_counter = os.path.join('./vocabulary/', filename_counter)
        
        with open(path_word_to_id, 'wb') as writer:
            pickle.dump(self.word_to_id, writer)
            
        with open(path_id_to_word, 'wb') as writer:
            pickle.dump(self.id_to_word, writer)
            
        with open(path_word_counter, 'wb') as writer:
            pickle.dump(self.word_counter, writer)
            
    def load_vocabulary(self, path_word_to_id, path_id_to_word, path_word_counter):
        """
        Loads vocabulary dictionaries from pickle files

        :param path_word_to_id: The path to file with word_to_id dictionary
        :param path_id_to_word: The path to file with id_to_word dictionary
        """
        with open(path_word_to_id, 'rb') as reader:
            self.word_to_id = pickle.load(reader)
            
        with open(path_id_to_word, 'rb') as reader:
            self.id_to_word = pickle.load(reader)
            
        with open(path_word_counter, 'rb') as reader:
            self.word_counter = pickle.load(reader)
            
        self.number_of_words = len(self.word_to_id)


def preprocess_captions(all_captions):
    """
    Replaces all the signs by whitespaces and transforms words to lowercase inplace

    :param all_captions: List of lists with all the captions
    """
    for captions_list in all_captions:
        for i, caption in enumerate(captions_list):
            captions_list[i] = re.sub('\W+', ' ', caption.lower())
            
# see https://docs.python.org/3/library/re.html#:~:text=Regular%20Expression%20Syntax,down%20to%20the%20same%20thing).
# on regex (i.e. re) for python


def add_start_and_end_to_captions(all_captions, start_str='<SOS>', end_str='<EOS>'):
    """
    Adds start and end of caption markers with inplace

    :param all_captions: List of lists with all the captions
    :param start_str: Start of caption marker
    :param end_str: End of caption marker
    """
    for captions in all_captions:
        for i in range(len(captions)):
            captions[i] = '{} {} {}'.format(start_str, captions[i], end_str)
            captions[i] = captions[i].replace('  ', ' ').lower()


def tokenise_captions(list_captions, vocabulary):
    """
    Transforms the list of captions for all images in a dataset

    :param list_captions: list of captions for images in the dataset
    :param vocabulary: Vocabulary instance for the dataset
    :return: list of transformed captions
    """
    captions_tokens = []
    for captions in list_captions:
        tmp_captions_for_img = []
        for caption in captions:
            caption_words = caption.split()
            tmp = []
            for word in caption_words:
                if word in vocabulary.word_to_id:
                    tmp.append(vocabulary.get_id_by_word(word))
                else:
                    tmp.append(0)
            tmp_captions_for_img.append(tmp)
        captions_tokens.append(tmp_captions_for_img)
    return captions_tokens
