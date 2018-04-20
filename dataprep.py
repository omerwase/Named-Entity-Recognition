"""dataprep.py

Author: Omer Waseem
Description: functions for extracting sentences or words from CoNLL and NEEL data files
"""

import pandas as pd
from nltk.tokenize import word_tokenize


def neel_sentences(gs_file, tsv_file):
    """NEEL2006 sentences from gs and tsv files
    
    Seperating NEEL data into individual sentences with corresponding tags
    
    arguments: gs_file, tsv_file
    returns: sentences, entities, unknown tweet IDs
    """
    
    gs_col_names=['tweet_id','start','end','uri', 'confidence', 'entity']
    tsv_col_names=['tweet_id','text']
    tweets_dict = {}
    data_dict = {}
    seen_ids = set()
    sent = []
    entity = []
    unknown_indicies = set()
    
    gs_df = pd.read_table(gs_file, sep = '\t', header=None, names=gs_col_names)
    # fixes entity label at index 4805 that is incorrect
    if len(gs_df['entity']) > 4805 and gs_df['entity'][4805] == 'Organization373937812812615000':
        gs_df.at[4805, 'entity'] = 'Organization'
    
    tsv_df = pd.read_table(tsv_file, sep = ',', header=None, names=tsv_col_names)
    # strip '|' character from the edges of tsv_df column values
    tsv_df['tweet_id'] = tsv_df['tweet_id'].apply(lambda x: str(x).strip('|'))
    tsv_df['text'] = tsv_df['text'].apply(lambda x: str(x).strip('|'))

    for index, row in tsv_df.iterrows():
        tweets_dict[row['tweet_id']] = row['text']
    
    for index, row in gs_df.iterrows():
        tweet_id = str(row['tweet_id'])
        start = row['start']
        end = row['end']
        old_ent = row['entity']
        
        # Rename entity values as PER, LOC, ORG, MISC, O
        if old_ent in ('Character', 'Person'):
            new_ent = 'PER'
        elif old_ent == 'Location':
            new_ent = 'LOC'
        elif old_ent == 'Organization':
            new_ent = 'ORG'
        else:
            new_ent = 'MISC'
        
        try:
            text = tweets_dict[tweet_id]
            if tweet_id not in seen_ids:
                seen_ids.add(tweet_id)
                words = word_tokenize(text)
                labels = ['O']*len(words)
            else:
                words = data_dict[tweet_id]['words']
                labels = data_dict[tweet_id]['labels']
            assert(len(words)==len(labels))
            ent_words = word_tokenize(text[start:end])
            for e in ent_words:
                for i in range(len(words)):
                    if e == words[i]:
                        labels[i] = new_ent
            data_dict[tweet_id] = {'words': words, 'labels': labels}
        except KeyError:
            unknown_indicies.add(tweet_id)
    
    for key in data_dict:
        sent.append(data_dict[key]['words'])
        entity.append(data_dict[key]['labels'])
    
    return sent, entity, unknown_indicies


def neel_words(gs_file, tsv_file):
    """NEEL2006 words from gs and tsv files
    
    Seperating NEEL data into individual words with corresponding tags
    
    arguments: gs_file, tsv_file
    returns: words, entities, unknown tweet IDs
    """
    
    all_words = []
    all_entities = []
    all_errors = set()
    
    sent, entity, errors = neel_sentences(gs_file, tsv_file)

    for se in sent:
        for w in se:
            all_words.append(w)
    for en in entity:
        for e in en:
            all_entities.append(e)
    for er in errors:
        all_errors.add(er)
            
    return all_words, all_entities, all_errors


def conll_sentences(conll_file):
    """CoNLL2003 sentences from file
    
    Seperating CoNLL data into individual sentences with corresponding tags
    
    arguments: conll_file
    returns: sentences, POS tags, chunk tags, entities
    """
    
    sent = []
    pos = []
    chunk = []
    entity = []
    temp_sent = []
    temp_pos = []
    temp_chunk = []
    temp_entity = []
    
    with open(conll_file) as f:
        conll_raw_data = f.readlines()
    conll_raw_data = [x.strip() for x in conll_raw_data]

    for line in conll_raw_data:
        if line != '':
            split_line = line.split()
            if len(split_line) == 4:
                if split_line[0] != '-DOCSTART-':
                    temp_sent.append(split_line[0])
                    temp_pos.append(split_line[1])
                    temp_chunk.append(split_line[2])
                    
                    # Rename entity values as PER, LOC, ORG, MISC, O
                    old_ent = split_line[3]
                    if old_ent in ('I-ORG', 'B-ORG'):
                        new_ent = 'ORG'
                    elif old_ent in ('I-LOC', 'B-LOC'):
                        new_ent = 'LOC'
                    elif old_ent in ('I-MISC', 'B-MISC'):
                        new_ent = 'MISC'
                    elif old_ent in ('I-PER', 'B-PER'):
                        new_ent = 'PER'
                    else:
                        new_ent = 'O'
                    temp_entity.append(new_ent)
            else:
                raise IndexError('Line split length does not equal 4.')
        else:
            if len(temp_sent) > 0:
                assert(len(sent) == len(pos))
                assert(len(sent) == len(chunk))
                assert(len(sent) == len(entity))
                sent.append(temp_sent)
                pos.append(temp_pos)
                chunk.append(temp_chunk)
                entity.append(temp_entity)
                temp_sent = []
                temp_pos = []
                temp_chunk = []
                temp_entity = []
    
    return sent, pos, chunk, entity


def conll_words(conll_file):
    """CoNLL2003 words from file
    
    Seperating CoNLL data into individual words with corresponding tags
    
    arguments: conll_file
    returns: words, POS tags, chunk tags, entities
    """
    
    all_words = []
    all_pos = []
    all_chunk = []
    all_entities = []
    
    sent, pos, chunk, entity = conll_sentences(conll_file)

    for se in sent:
        for w in se:
            all_words.append(w)
    for po in pos:
        for p in po:
            all_pos.append(p)
    for ch in chunk:
        for c in ch:
            all_chunk.append(c)
    for en in entity:
        for e in en:
            all_entities.append(e)
            
    return all_words, all_pos, all_chunk, all_entities