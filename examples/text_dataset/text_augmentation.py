import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
from nltk.corpus import wordnet as wn
from snorkel.preprocess.nlp import SpacyPreprocessor
from snorkel.augmentation import MeanFieldPolicy, RandomPolicy, ApplyAllPolicy, PandasTFApplier, transformation_function
# Need to install: snorkel==0.9.6

RANDOM_STATE_SEED = 0
np.random.seed(RANDOM_STATE_SEED)
spacy = SpacyPreprocessor(text_field='text', doc_field='doc', memoize=True)


@transformation_function(pre=[spacy])
def replace_word_with_synonym(x):
    pos_allowed = {'NOUN': 'n', 'ADJ': 'a', 'VERB': 'v', 'ADV': 'r'}
    # Get indices of noun tokens in sentence.
    word_idxs = [i for i, token in enumerate(x.doc) if token.pos_ in pos_allowed]

    for i in range(len(word_idxs)):
        # Pick random noun idx to replace.
        idx = np.random.choice(word_idxs)
        synonym = get_synonym(x.doc[idx].text, pos_allowed[x.doc[idx].pos_])
        # If there's a valid noun synonym, replace it.
        if synonym:
            x.text = replace_token(x.doc, idx, synonym)
            return x


@transformation_function(pre=[spacy])
def insert_word(x):
    pos_allowed = {'NOUN': 'n', 'ADJ': 'a', 'VERB': 'v', 'ADV': 'r'}
    # Get indices of tokens in sentence.
    word_idxs = [i for i, token in enumerate(x.doc) if token.pos_ in {'NOUN', 'ADJ', 'VERB'}]
    
    for i in range(len(word_idxs)):
        # Pick random noun idx to replace.
        idx = np.random.choice(word_idxs)
        synonym = get_synonym(x.doc[idx].text, pos_allowed[x.doc[idx].pos_])
        # If there's a valid synonym, insert it
        if synonym:
            position = np.random.choice(word_idxs)
            x.text = ' '.join([x.doc[:position].text, synonym, x.doc[position:].text])
            return x


@transformation_function(pre=[spacy])
def remove_word(x):
    pos_allowed = {'NOUN': 'n', 'ADJ': 'a', 'VERB': 'v', 'ADV': 'r'}
    # Get indices of tokens in sentence.
    word_idxs = [i for i, token in enumerate(x.doc) if token.pos_ in pos_allowed]
    if word_idxs:
        # Pick random word idx to replace.
        idx = np.random.choice(word_idxs)
        x.text = replace_token(x.doc, idx, '')
        return x


@transformation_function(pre=[spacy])
def swap_words(x):
    pos_allowed = {'NOUN': 'n', 'ADJ': 'a', 'VERB': 'v', 'ADV': 'r'}
    word_idxs = [i for i, token in enumerate(x.doc) if token.pos_ in pos_allowed]
    # Check that there are at least two adjectives to swap.
    if len(word_idxs) >= 2:
        idx1, idx2 = sorted(np.random.choice(word_idxs, 2, replace=False))
        # Swap tokens in positions idx1 and idx2.
        x.text = ' '.join(
            [
                x.doc[:idx1].text,
                x.doc[idx2].text,
                x.doc[1 + idx1 : idx2].text,
                x.doc[idx1].text,
                x.doc[1 + idx2 :].text,
            ]
        )
        return x 


def get_synonym(word, pos=None):
    '''Get synonym for word given its part-of-speech (pos).'''
    synsets = wn.synsets(word, pos=pos)
    # Return None if wordnet has no synsets (synonym sets) for this word and pos.
    if synsets:
        words = [lemma.name() for lemma in synsets[0].lemmas() if lemma.name() not in word]
        if len(words) > 0:
            # Multi word synonyms in wordnet use '_' as a separator e.g. reckon_with. Replace it with space.
            return words[0].replace('_', ' ')


def replace_token(spacy_doc, idx, replacement):
    '''Replace token in position idx with replacement.'''
    return ' '.join([spacy_doc[:idx].text, replacement, spacy_doc[1 + idx :].text])


def preview_tfs(df, tfs):
    transformed_examples = []
    for f in tfs:
        for i, row in df.iterrows():
            transformed_or_none = f(row)
            # If TF returned a transformed example, record it in dict and move to next TF.
            if transformed_or_none is not None:
                transformed_examples.append(
                    OrderedDict(
                        {
                            'TF Name': f.name,
                            'Original Text': row.text,
                            'Transformed Text': transformed_or_none.text,
                        }
                    )
                )
                break
    return pd.DataFrame(transformed_examples)


def augment_dataset(original_dataset, text_column, keep_original=True):
    initial_dataset = deepcopy(original_dataset)
    initial_dataset = initial_dataset.rename(columns={text_column: 'text'})
    
    tfs = [
        replace_word_with_synonym,
        remove_word,
        insert_word,
        swap_words
    ]

    random_policy = RandomPolicy(
        len(tfs), sequence_length=2, n_per_original=1, keep_original=True
    )

    mean_field_policy = MeanFieldPolicy(
        len(tfs),
        sequence_length=len(tfs) * 5,
        n_per_original=1,
        keep_original=True,
        p=[0.25, 0.25, 0.25, 0.25]
    )

    all_policy = ApplyAllPolicy(
        len(tfs),
        n_per_original=1,
        keep_original=keep_original
    )

    #tf_applier = PandasTFApplier(tfs, random_policy)
    #tf_applier = PandasTFApplier(tfs, mean_field_policy)
    tf_applier = PandasTFApplier(tfs, all_policy)
    augmented_dataset = tf_applier.apply(initial_dataset, progress_bar=False)
    augmented_dataset = augmented_dataset.rename(columns={'text': text_column})

    print('Original training set size:', len(initial_dataset))
    print('Augmented training set size:', len(augmented_dataset))
    
    return augmented_dataset
