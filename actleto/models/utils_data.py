import pandas as pd
import random

from bert_sequence_tagger.bert_utils import make_bert_tag_dict_from_flair_corpus


def convert_y_to_dict_format(X, y):
    dict_annots = []
    for sent_x, sent_y in zip(X, y):
        offsets = []
        curr_offset = 0
        for index, word in enumerate(sent_x):
            offsets.append(curr_offset)
            curr_offset += len(word) + 1

        sent_dict_annots = []
        start_offset = -1
        last_offset = -1
        entity_tag = ''
        for i, tag in enumerate(sent_y):
            if tag.split('-')[0] == 'O':
                if start_offset != -1:
                    sent_dict_annots.append({'tag' : entity_tag, 
                                            'start' : start_offset, 
                                            'end' : last_offset})
                start_offset = -1
                
            if tag.split('-')[0] == 'B':
                if start_offset != -1:
                    sent_dict_annots.append({'tag' : entity_tag, 
                                            'start' : start_offset, 
                                            'end' : last_offset})
                
                start_offset = offsets[i]
                entity_tag = tag.split('-')[1]
                last_offset = offsets[i] + len(sent_x[i])
            elif tag.split('-')[0] == 'I':
                last_offset = offsets[i] + len(sent_x[i])
        
        if start_offset != -1:
            sent_dict_annots.append({'tag' : entity_tag,
                             'start' : start_offset,
                             'end' : last_offset})
        
        dict_annots.append(sent_dict_annots)
    
    return dict_annots


def create_helper(X_train):
    X_helper = pd.DataFrame([' '.join(e) for e in X_train], columns=['texts'])
    return X_helper



def sample_seed_elements_for_al(y_train_dict, negative_size, positive_size, random_seed):
    random.seed(random_seed)
#     negative_size = 100
#     positive_size = 25

    random_sample = []

    positive_indexes = [i for i, e in enumerate(y_train_dict) if e]
    random_sample += random.sample(positive_indexes, positive_size)

    negative_indexes = [i for i, e in enumerate(y_train_dict) if (not e)]
    random_sample += random.sample(negative_indexes, negative_size)

    y_seed_dict = [None for _ in range(len(y_train_dict))]

    for elem in random_sample:
        y_seed_dict[elem] = y_train_dict[elem]
    
    return y_seed_dict
