import scispacy
import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import zipfile
import os
import xmltodict
import json
import zlib
import resource
import sys
import regex as re
from bs4 import BeautifulSoup
from tqdm import tqdm
import nltk.data
from multiprocessing import Pool
from optparse import OptionParser


def get_reference_mapping(filename, content):
    
    """
    Given XML filename and XML file, extract rid mappings and attribute data
    """
    
    mappings  = {}
    extracted = {}
    references = {}
    parsed    = xmltodict.parse(content.decode('UTF-8'))
    soup      = BeautifulSoup(content)

    if '@id' in parsed['article']:
        extracted['id']       = str(parsed['article']['@id'])
    if '@language' in parsed['article']:
        extracted['language'] = str(parsed['article']['@language'])
    if '@publish' in parsed['article']:
        extracted['publish']  = str(parsed['article']['@publish'])
    if '@relation' in parsed['article']:
        extracted['relation'] = str(parsed['article']['@relation'])
    
    if ('article' in parsed):
        if ('pubfm' in parsed['article']):
            if ('jtl' in parsed['article']['pubfm']):
                extracted['jtl']   = str(parsed['article']['pubfm']['jtl'])
    
    if ('article' in parsed):
        if ('pubfm' in parsed['article']):
            if ('vol' in parsed['article']['pubfm']):
                extracted['vol']   = str(parsed['article']['pubfm']['vol'])
    
    if ('article' in parsed):
        if ('pubfm' in parsed['article']):
            if ('issue' in parsed['article']['pubfm']):
                extracted['issue'] = str(parsed['article']['pubfm']['issue'])

    if ('article' in parsed):
        if ('pubfm' in parsed['article']):
            if ('vol' in parsed['article']['pubfm']):
                extracted['doi']   = str(parsed['article']['pubfm']['doi'])
    
    if ('article' in parsed):
        if ('fm' in parsed['article']):
            if ('atl' in parsed['article']['fm']):
                extracted['title']   = str(parsed['article']['fm']['atl'])
     
    
    del parsed
    
    for bib in soup.find_all("bib"):
        
        try:
            reference_attr = {}
            
            reference_attr['title']   = str(bib.atl.contents[0])
            reference_attr['snm']     = str([i.contents[0] for i in bib.find_all('snm')])
            reference_attr['fnm']     = str([i.contents[0] for i in bib.find_all('fnm')])
            reference_attr['journal'] = str(bib.jtl.contents[0])
            reference_attr['year']    = str(bib.find_all('cd')[0].contents[0])

            references.update({bib.attrs['id']: reference_attr})
        
        except Exception as e:
            
            if DEBUG == True:
            
                print('='*50)
                print('Something is wrong with BeatifulSoup Tags: %s' % str(bib))
                for i in ['snm', 'fnm', 'jtl', 'year', 'atl']:
                    if len(bib.find_all(i)) == 0:
                        print('%s attribute is missing.' %i)
            
            else: pass
            
    
    references = {'metadata': extracted, 'references': references}
    
    try:
        with open('data/intermediate/references/%s' % (filename + '.json'), 'w') as f:
            f.write(json.dumps(references))
        return True
    
    except TypeError as e:
        #print('Some contents of the file %s is not serializable' % filename)
        raise e
    

def get_reference_text(filename, content):
    
    """
    Given XML filename and XML file, extract referencing text and reference metadata
    TODO: Compile all regex to make it faster
    
    Returns  {str(unique_paper_indentifier), list(preceeding_text)}
    """
    
    content   = content.decode('UTF-8')
    bibid     = re.findall(r'<bibr\srid=\"(.*?)\"\s*\/>', content)
    #bibtext  = re.findall(r"\s.*?<bibr\s", content)
    
    bibtext_intermediate = [s for s in sent_detector.tokenize(content)\
                               if re.search(r'<bibr\srid', s)]

    bibtext_intermediate = [s.split('</p>') for s in bibtext_intermediate]
    bibtext_intermediate = [item for sublist in bibtext_intermediate for item in sublist]

    bibtext_intermediate = [s for s in bibtext_intermediate if re.search(r'<bibr\srid', s)]
    text_counter         = [len(re.findall(r'<bibr\srid', s)) for s in bibtext_intermediate]

    bibtext = []

    for i, s in enumerate(bibtext_intermediate):
        while text_counter[i] != 0:
            bibtext.append(s)
            text_counter[i] -= 1
    
    assert(len(bibid) == len(bibtext)), "The bibid's and preceeding text don't match: " +\
                                        "for article %s\n" % filename +\
                                        "bibid: %s\n" %bibid +\
                                        "bibtext: %s\n" %bibtext
    
    with open('data/intermediate/tex_ref_mappings/%s' % (filename + '.json'), 'w') as f:

        #Only write sentences with 1 reference if MULTI_REF flag is on.

        if MULTI_REF:
            f.write(json.dumps([{i[1]:i[0].split(' ')} for i in zip(bibid, bibtext)]))
           
        else:
            f.write(json.dumps([{i[1]:i[0].split(' ')} for i in zip(bibid, bibtext) \
                                                            if len(i[0].split(' ')) == 1]))
        
    return 


def get_zips_parallel_mapper(directory):
    
    """
    Function to extract relevant files from the filebase
    """
    
    try:
        
        zip_list  = [i for i in os.listdir(directory) if 'supp_xml' not in i]
        errored   = []
        finfos    = []
        contents  = []

        for zfile in tqdm(zip_list):

            zfile     = zipfile.ZipFile(os.path.join(directory, zfile))

            for finfo in zfile.infolist():

                if 'nature' in finfo.filename:
                    
                    ifile = zfile.open(finfo)
                    content = ifile.read()

                    finfos.append(finfo)
                    contents.append(zlib.compress(content))

        return finfos, contents
    
    except Exception as e:
        
        print('Could not read file from zip %s, file %s' % (zfile, finfo.name))


def get_zips_parallel_reducer(arg):
    
    try:
        finfo, content = arg[0], zlib.decompress(arg[1])
        del arg

        #citations = {}
        #tex_ref_map = get_reference_text(finfo.filename, content)
        #ref_id_ref_map = get_reference_mapping(finfo.filename, content)
        #citations[finfo.filename] = {'tex_ref_map': tex_ref_map,
        #                            'ref_id_ref_map': ref_id_ref_map}
        
        get_reference_text(finfo.filename, content)
        get_reference_mapping(finfo.filename, content)
        
        return True
    
    except Exception as e:
        
        #print('Could not extract references from %s' % finfo)
        return str(Exception)


if __name__ == '__main__':

    """

    This module is written to create the map between the citing article and it's 
    reference sentences and the cited article.
    
    Input: The Nature Dataset of XML Files.
    Output:
        - tex_ref_mappings: This file consists of sentences associated with any given citation.
        - references: A list of jsons with each item consisting of references and refereces details 
                        in the paper.

    """


    #nltk.download('punkt')
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    directory     = 'data/nature/raw xml/'


    parser = OptionParser()
    
    parser.add_option("-m", "--include_multi_ref",
                        action="store_true", dest="include_multi_ref",
                         default = True, help = 'Include text that maps to multiple references')

    parser.add_option("-t", "--test",
                        action = "store_true", dest="test",
                        default = False, help = 'Run on first 100 files only')

    parser.add_option("-v", "--verbose",
                        action = "store_true", dest="verbose",
                        default = False, help = 'Print failures and other debug info')

    (options, args) = parser.parse_args()

    global DEBUG, MULTI_REF
    
    DEBUG = options.verbose
    MULTI_REF = options.include_multi_ref

    print(MULTI_REF)

    print('Reading Files..')
    finfos, contents = get_zips_parallel_mapper(directory)
    p                = Pool(6)
    
    print('Processing in parallel..')
    if options.test: 
        print ('Warning: Test is ON')
        finfos = finfos[:100]
        contents = contents[:100]


    success          = p.map(get_zips_parallel_reducer, zip(finfos, contents))
    success_rate      = sum([1 if i == True else 0 for i in success])/len(success)
    
    print ('%f of all processed files Succeeded' % success_rate)

    #if success_rate < 0.8:
    #    print(success)
