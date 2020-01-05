from optparse import OptionParser
import multiprocessing
#import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import zipfile
import os
import json
import zlib
import seaborn as sns
import resource
import sys
import regex as re
from bs4 import BeautifulSoup
from tqdm import tqdm
import nltk.data
import time
import json
from multiprocessing import Pool
from nltk.tokenize import word_tokenize
from numpy.testing import assert_array_equal
import threading
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from mlxtend.preprocessing import TransactionEncoder
from datasketch import MinHash, MinHashLSH
import pdb
from ast import literal_eval
#nltk.download('punkt')
sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


def build_ref_index(tex_ref_mappings, reference):

    '''
    Function to get bx -> referencing text mapping from referencing text -> bx map.
    '''

    inverse_tex_ref = {}
    
    for tex_refs in tex_ref_mappings:
        
        text, refs = list(tex_refs.items())[0]
        
        for ref in refs: 
            if ref in inverse_tex_ref:
                inverse_tex_ref[ref].append(text)
            else:
                inverse_tex_ref[ref] = [text]
    
    return inverse_tex_ref


def read_files():
    
    directory = 'data/intermediate/'
    tex_ref_mappings = []
    references       = []
    
    tex_ref_files = os.listdir(os.path.join(directory, 'tex_ref_mappings'))
    references_files = os.listdir(os.path.join(directory, 'references'))
    
    if len(tex_ref_files) != len(references_files):
        print('Not all text files have a references counterpart.\n'
              'Continuing with the files that do have a mapping')
    
    #file = tex_ref_files[1]
    
    for file in tqdm(tex_ref_files):
        #print(file)
        #print(tex_ref_mappings)
        
        with open(os.path.join(directory, 'tex_ref_mappings', file), 'r', encoding = 'UTF-8') as f0:

            try:
                with open(os.path.join(directory, 'references', file), 'r',  encoding = 'UTF-8') as f1:

                    tex_ref_mappings.append(json.loads(str(f0.read()),  encoding = 'UTF-8'))
                    references.append(json.loads(str(f1.read()),  encoding = 'UTF-8'))

            except IOError as e:
                print('References file %f not found.' % file)
    
    return tex_ref_mappings, references

def get_reference_map_1(get_reference_map_1_input): 
    
    """
    NOTE: Improvements Todo: Force 1-1 mapping, add year and journal to increase map accuracy
    """
    
    reference, tex_ref_mapping = get_reference_map_1_input
    
    if 'doi' in reference['metadata']:
        doi = reference['metadata']['doi']

        references_0 = get_reference_map_0(nature2references, nature_references_info, doi).fillna('')

        references_1 = pd.DataFrame.from_dict(reference['references'], orient = 'index').fillna('')

        if (len(references_1) > 0) and (len(references_0) > 0):

            if 'title' in references_1:

                references_1['text'] = references_1.index.map(build_ref_index(tex_ref_mapping, reference))

                vect = TfidfVectorizer(min_df=1, stop_words="english")
                tfidf = vect.fit(list(references_0['Title']) + list(references_1['title']))

                vect_0 = vect.transform(list(references_0['Title']))
                vect_1 = vect.transform(list(references_1['title']))

                pairwise_similarity = vect_0 * vect_1.T 

                references_0['match_index'] = [i[0] for i in np.argmax(pairwise_similarity, axis = 1).tolist()]

                test_match = pd.merge(references_0,
                                      references_1.reset_index(),
                                      left_on = 'match_index',
                                      right_index = True, how = 'left').rename(columns = {'Title': 'title0',
                                                                                         'title': 'title1'})

                return test_match


def get_reference_map_0(nature2references, nature_references_info, doi):
    
    try:
        article_id = nature_references_info[nature_references_info.index == doi]['ArticleID'].values[0]

        references = nature2references[nature2references['CitingArticleID'] == article_id]

        references = pd.merge(references, nature_references_info,
                          how = 'left', left_on = 'CitedArticleID', right_on = 'ArticleID')
    
    except Exception as e:
        
        print('doi: %s' % doi)
        print(e)
        references = pd.DataFrame()
    

    return references


def onur_json(tex_ref_mappings, references):
    
    '''
    Merge text_ref_mapping information into references.
    '''
    
    references_w_text = references.copy()
    
    for i in tqdm(range(len(references_w_text))):
        
        paper = tex_ref_mappings[i]

        #print (paper)
        for tex_ref_mapping in paper:

            #print(tex_ref_mapping.values())

            for ref_id in list(tex_ref_mapping.values())[0]:

                if 'references' in references_w_text[i]:
                    
                    if ref_id in references_w_text[i]['references']:

                        references_w_text[i]['references'][ref_id].update({'text' :list(tex_ref_mapping.keys())[0]})
                        
                    else:
                        references_w_text[i]['references'][ref_id] = {'text' :list(tex_ref_mapping.keys())[0]}

    return references



def get_reference_map_onur(get_reference_map_1_input): 
    
    """
    NOTE: Improvements Todo: Force 1-1 mapping, add year and journal to increase map accuracy
    """
    
    reference, tex_ref_mapping = get_reference_map_1_input
    
    if 'doi' in reference['metadata']:
        doi = reference['metadata']['doi']

        references_0 = get_reference_map_0(nature2references, nature_references_info, doi).fillna('')

        references_1 = pd.DataFrame.from_dict(reference['references'], orient = 'index').fillna('')

        if (len(references_1) > 0) and (len(references_0) > 0):

            references_1['text'] = references_1.index.map(build_ref_index(tex_ref_mapping, reference))
            
            references_1['citing_doi'] = doi 

            vect = TfidfVectorizer(min_df=1, stop_words="english")
            
            tfidf = vect.fit(list(references_0['Title']) + list(references_1['title']))

            vect_0 = vect.transform(list(references_0['Title']))
            vect_1 = vect.transform(list(references_1['title']))

            pairwise_similarity = vect_0 * vect_1.T 

            references_0['match_index'] = [i[0] for i in np.argmax(pairwise_similarity, axis = 1).tolist()]

            test_match = pd.merge(references_0,
                                  references_1.reset_index(),
                                  left_on = 'match_index',
                                  right_index = True, how = 'left').rename(columns = {'Title': 'Title - matched_to',
                                                                                     'title': 'Title - matched_from'})
            
            return test_match.drop(['match_index'], axis = 1)

    else:

        return pd.DataFrame()


def add_metadata(papers, references):

    for reference in references: 
    
        if 'doi' in reference['metadata']:
            
            if reference['metadata']['doi'] in papers:
                
                papers[reference['metadata']['doi']] = {'metadata' : reference['metadata'],
                                                    'references': papers[reference['metadata']['doi']]}

    return papers


def flatten(l):

    '''
    Function to flatten a list of list
    '''

    flat_list = []
    for sublist in l:
        if type(sublist) == list:
            for item in sublist:
                flat_list.append(item)
    return flat_list


def clean_df(matched_dfs):
    
    '''
    This function cleans the text -> reference map dataframe and groups them by reference such that 

    text_1(paper_1), text_2(paper_2), text_n(paper_m) -> reference_i

    '''

    matched_dfs['text'] = matched_dfs['text'].fillna('[]')
    matched_dfs['text'] = matched_dfs['text'].apply(literal_eval)
    matched_dfs = pd.DataFrame(matched_dfs.groupby('ArticleID')\
                                 .apply(lambda x: list(x['text'].values))).rename(columns = {0: 'text'})


    #pdb.set_trace()
    matched_dfs['clean_text'] = matched_dfs['text'].apply(flatten)
    matched_dfs['clean_text'] = matched_dfs['clean_text'].apply(lambda x: \
                                                    [BeautifulSoup(i, "lxml").text for i in x])
    
    #print(matched_dfs.head())
    matched_dfs['count']      = matched_dfs['clean_text'].apply(lambda x: len(x))
    matched_dfs               = matched_dfs[matched_dfs['count'] != 0]

    return matched_dfs


if __name__ == '__main__':


    """
    This modeule serves two purposes:
    
    1: if out_format == csv:
    	
    	Groups together all
    	referring sentences to the paper that they are referring to
    	in order to make the process easier for downstream analysis.

	2: if out_format == json:

		Create a file of new line seperated jsons with each json consisting of all parsed 
		information of a given paper.

    """

    parser = OptionParser()
    
    parser.add_option("-p", "--parallel",
                        action="store", type="string", dest="parallel",
                         default = multiprocessing.cpu_count() - 1)

    parser.add_option("-f", "--out_format",
                        action = "store", type = "string", dest="out_format",
                        default = "csv")

    parser.add_option("-t", "--test",
                        action = "store_true", dest="test",
                        default = False)

    parser.add_option("-c", "--clean_only",
                        action = "store_true", dest="clean_only",
                        default = False)



    (options, args) = parser.parse_args()


    if options.clean_only:

        matched_df = pd.read_csv('data/intermediate/matched_dfs_raw.csv')


        matched_df = clean_df(matched_df)
        matched_df.to_csv('data/intermediate/matched_dfs.csv')
        exit()


    nature_references_info = pd.read_hdf('data/nature/preprocessed/NatureReferencesInformation.hdf').set_index('DOI')
    nature2references      = pd.read_hdf('data/nature/preprocessed/Nature2References.hdf')

    nature_references_info['DOI'] = nature_references_info.index

    global TEST
    TEST = options.test
    
    tex_ref_mappings, references = read_files()
    references                   = onur_json(tex_ref_mappings, references)

    if TEST:
        
        matched_dfs = pd.DataFrame()

        print('Testing on 50 inputs:')
        for i in tqdm(range(50)):
            
            #matched_dfs = matched_dfs.append(get_reference_map_1((references[i], tex_ref_mappings[i])))

            try:
                matched_dfs = matched_dfs.append(get_reference_map_1((references[i], tex_ref_mappings[i])))
                pass
            except Exception as e:
                print(e)
                print('\nreferences:')
                print (references[i])
                print('\ntex_ref_mappings')
                print (tex_ref_mappings[i])
                #raise(e)

        print('Test succeeded on 50 items')
        print('Cross check the matched_dfs for expected output format:')
        print(matched_dfs)
        matched_dfs.to_csv('data/intermediate/matched_dfs_test.csv')
        exit()

    mp_input                     = zip(references, tex_ref_mappings)
    p                            = Pool(options.parallel)

    #pdb.set_trace()
    #matched_dfs = p.map(get_reference_map_onur, mp_input)
    
    matched_dfs = p.map(get_reference_map_1, mp_input)
    matched_dfs = pd.concat(matched_dfs)
    matched_dfs.to_csv('data/intermediate/matched_dfs_raw.csv')
    #matched_dfs = pd.concat(matched_dfs).set_index('citing_doi')

    if out_format == 'csv':
        
        matched_dfs = clean_df(matched_dfs)
        matched_dfs.to_csv('data/intermediate/matched_dfs.csv')

    else:

        papers      = {k: g.to_dict(orient='records') for k, g in matched_dfs.groupby(level=0)}
        papers      = add_metadata(papers, references)
        with open('data/intermediate/matched_onur_format.jsons', 'w') as f:
            
            for index, paper in papers.iteritems():
            
                f.write('\n')
                f.write(json.dumps({index:reference}))


