# -*- coding: utf-8 -*-

import random
# from engine.data.data_manager import *
# from engine.db.mongo import *
import numpy as np
# import engine.sim.lexicon_gen.lexicon_generator as lg
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def bio_convert(len, name):
    if name is None:
        return ['O' for _ in range(len)]
    ret = [u'I-{}'.format(name) for _ in range(len)]
    ret[0] = u'B-{}'.format(name)
    return ret

def weighted_sample(weights_):
    weights = np.array(weights_, dtype = np.float32)
    weights /= weights.sum()
    rand_num, accu = random.random(), 0.0
    for i in range(len(weights_)):
        accu += weights[i]
        if accu >= rand_num:
            return i
    assert(False)

def lemmatize_a_word(word, pos = 'n'):
    """ For eg, if the input is "find experts at", the output is "find expert at"
    """
    exception2lemma = {'are': 'is'}
    lemmas = []
    for token in word.split():
        if token in exception2lemma:
            lemmas.append(exception2lemma[token])
        else:
            lemmas.append(lemmatizer.lemmatize(token, pos))
    lemma = ' '.join(lemmas)
    return lemma

class node:
    def __init__(self, content = None,
        entity = None, role = None, 
        exchangeable = False, pick_one = False, p_dropout = 0.0, p_cut = 0.0, p_word_cut = 0.0, lang = 'zh', gen_lemmas = False, lemma_pos = 'n'):

        """
        content: list or unicode.
            If content is not None, the current node is a leaf.
            If content is a list, it is the lexicon.
            If content is a unicode, it is the word to be generated.
        entity (str): entity for this node. None for non-entities.
        role (str): role for this node. None for the same name as entity or non-roles.
        exchangeable (bool): whether the order of children is exchangeable.
        pick_one (bool): pick only one child.
        p_dropout (float): the probability that the current node can be dropped during generation.
        p_cut (float): the probability that the word will get cut.
        p_word_cut (float): the probability each individual character will get cut. Only effective when p_cut > 0.
        lang (str): zh or en,
        gen_lemmas (bool): add in the lemmas of the content to the current content,
        lemma_pos (str): lemma part of speech tag. n or v.
        """

        self.children = []
        self.weights = []

        self.content = content
        self.entity = entity
        self.role = role
        self.exchangeable = exchangeable
        self.pick_one = pick_one
        self.p_dropout = p_dropout
        self.p_cut = p_cut
        self.p_word_cut = p_word_cut
        self.sep = ' '# lg.get_sep_by_lang(lang)
        if lang == 'en' and gen_lemmas and self.content is not None:
            if type(self.content) == list:
                lemmas = []
                for word in self.content:
                    lemmas.append(lemmatize_a_word(word, pos = lemma_pos))
            else:
                lemmas = [lemmatize_a_word(self.content, pos = lemma_pos)]
                self.content = list(self.content)
            self.content = list(set(self.content + lemmas))

        if not entity is None and role is None:
            self.role = self.entity

    def add_child(self, child, weight = 1.0):
        self.children.append(child)
        self.weights.append(weight)
        return self

    def generate(self):
        """
        p_simulator: the pointer to the simulator

        return (str_1, list_ent, list_role):
            str_1: the text generated,
            list_ent: the BIO entity representation in list,
            list_role: the BIO role representation in list
        """

        if random.random() < self.p_dropout:
            return u"", [], []
        if len(self.children) == 0:
            if type(self.content) == list:
                text = random.choice(self.content)
                return text, bio_convert(len(text), self.entity), bio_convert(len(text), self.role)
            else:
                text = self.content
            if random.random() < self.p_cut:
                n_text = u""
                for char in text:
                    if random.random() > self.p_word_cut:
                        n_text += char
                text = n_text
            return text, bio_convert(len(text), self.entity), bio_convert(len(text), self.role)
        else:
            if self.pick_one:
                i = weighted_sample(self.weights)
                return self.children[i].generate()
            texts, entities, roles = [], [], []
            for child in self.children:
                text, entity, role = child.generate()
                if len(text) == 0:  # filter our empty text
                    continue
                texts.append(text)
                entities.append(entity)
                roles.append(role)
            if self.exchangeable:
                index = range(len(texts))
                random.shuffle(list(index))
                texts = [texts[i] for i in index]
                entities = [entities[i] for i in index]
                roles = [roles[i] for i in index]



            r_text = self.sep.join(texts)

            r_entities = []
            for i, entity in enumerate(entities):
                if i > 0 and len(self.sep) > 0:
                    r_entities += ['O'] * len(self.sep)
                r_entities += entity

            r_roles = []
            for i, role in enumerate(roles):
                if i > 0 and len(self.sep) > 0:
                    r_roles += ['O'] * len(self.sep)
                r_roles += role

            return r_text, r_entities, r_roles

class simulator:

    def __init__(self, lang = 'zh'):
        random.seed(13)
        self.roots = []
        self.intents = []
        self.weights = []
        self.sep = ' ' # lg.get_sep_by_lang(lang)

    def add_root(self, root, intent, weight = 1.0):
        self.roots.append(root)
        self.intents.append(intent)
        self.weights.append(weight)
        return self


    def generate(self, num):
        annos = []
        self.unk_cnt = 0
        content = 'No,Category,Descript\n'
        for no in range(num):
            i = weighted_sample(self.weights)
            text, _, _ = self.roots[i].generate()
            if len(text) == 0:  # filter out empty simulation
                continue
            content += str(no) + ',' + self.intents[i] + ',\"' + text + '"\n'
        return content 
