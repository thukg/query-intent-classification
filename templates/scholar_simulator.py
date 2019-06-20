# -*- coding: utf-8 -*-

# import engine.utils as utils
# import engine.data.save_db_from_json as save_db_from_json
# from engine.data.data_manager import *
# from engine.db.mongo import *
from simulator import *
import codecs
import nltk
import sys
# nltk.download()
# import engine.sim.lexicon_gen.lexicon_generator as lg

config = {
        'root': 'aminer_lexicon/',
        'key': 'KEY.txt', 
        'insts': 'ORG.txt', 
        'names': 'NAME.txt',
        'years': 'DATE.txt', 
        'venues': 'CON.txt', 
        'locations': 'LOC.txt'
        }

class data_manager():
    def __init__(self, root):
        self.root = root
        pass

    def read_lexicon_set(self, filename, min_len):
        with open(self.root + filename) as file:
            res = eval(file.read().encode('utf8', 'replace'))
        return list(filter(lambda x: len(x) > min_len, res))

def simulate(debug = False):
    """
    debug (bool): True to print sentences to file, False to save it to db
    debug_file (str): the file name to save the debug output
    """
    file_name = sys.argv[1]
    print (file_name)
    dm = data_manager(config['root'])
    keywords = dm.read_lexicon_set(config['key'], min_len = 3)
    locations = dm.read_lexicon_set(config['locations'], min_len = 3)
    venues = dm.read_lexicon_set(config['venues'], min_len = 3)
    years = dm.read_lexicon_set(config['years'], min_len = 1)
    names = dm.read_lexicon_set(config['names'], min_len = 5)
    insts = dm.read_lexicon_set(config['insts'], min_len = 3)


    # keywords = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'aminer_keywords', th = 1000, min_len = 3))
    # insts = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'linkedin_inst', th = 1000, min_len = 3))
    # names = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'aminer_names', th = 10, min_len = 5))
    # years = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'year'))
    # venues = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'dblp_venues', th = 1000, min_len = 3))


    # keywords = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'aminer_keywords'))
    # insts = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'linkedin_inst'))
    # names = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'aminer_names'))
    # years = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'year'))
    # venues = list(dm.read_lexicon_set(keep_mention = False, from_file = True, filename = 'dblp_venues'))

    # f2e, _ = lg.load_mapping()
    f2e = {'aminer_names': 'name', 'linkedin_inst': 'inst', 'year':'year', 'dblp_venues':'dblop', 'aminer_keywords': 'key', 'cities': 'loc'}
    keyword_node = node(keywords, entity = f2e['aminer_keywords'])
    inst_node = node(insts, entity = f2e['linkedin_inst'])
    # the topics of a linkedin_inst, for example
    name_node = node(names, entity = f2e['aminer_names'])
    year_node = node(years, entity = f2e['year'])
    venue_node = node(venues, entity = f2e['dblp_venues'])
    loc_node = node(locations, entity = f2e['cities'])


    # begin intent search expert #
    front_keyword_node = node(p_dropout = 0.5).add_child(keyword_node)
    front_inst_node = node(p_dropout = 0.5).add_child(inst_node)
    expert_node = node([u"researchers", u"scientists", u"people", u"professors", u"experts"], p_dropout = 0.0, lang = 'en', gen_lemmas = True)

    front_cond_node = node(exchangeable = True, lang = 'en').add_child(front_keyword_node).add_child(front_inst_node)

    which_node = node([u"that", u"who"])
    rear_keyword_node_1 = node(lang = 'en').add_child(which_node).add_child(node([u"work on", u"are working on", u"are doing", u"do", u"are doing research on", u"are experts at", u"have conducted research on", u"have been working on"], lang = 'en', gen_lemmas = True)).add_child(keyword_node)
    rear_keyword_node_2 = node(lang = 'en').add_child(node([u"working on", u"doing", u"doing research on", u"conducting research on"])).add_child(keyword_node)
    rear_keyword_node_3 = node(lang = 'en').add_child(node([u"whose work", u"whose research", u"whose paper", u"whose works", u"whose papers", u"whose researches"])).add_child(node([u"focus on", u"are about", u"are in", u"are related to"], lang = 'en', gen_lemmas = True)).add_child(keyword_node)
    rear_keyword_node = node(pick_one = True, p_dropout = 0.5).add_child(rear_keyword_node_1, 0.2).add_child(rear_keyword_node_2, 0.6).add_child(rear_keyword_node_3, 0.2)
    rear_inst_node_1 = node(lang = 'en').add_child(which_node).add_child(node([u"are from", u"work at", u"are in", u"work in", u"are at"], lang = 'en', gen_lemmas = True)).add_child(inst_node)
    rear_inst_node_2 = node(lang = 'en').add_child(node([u"from", u"working at", u"in", u"at"])).add_child(inst_node)
    rear_inst_node = node(pick_one = True, p_dropout = 0.5).add_child(rear_inst_node_1, 0.2).add_child(rear_inst_node_2, 0.6)

    # add
    rear_loc_node_1 = node(lang = 'en').add_child(which_node).add_child(node([u"are from", u"work in", u"are in"], lang = 'en', gen_lemmas = True)).add_child(loc_node)
    rear_loc_node_2 = node(lang = 'en').add_child(node([u"from", u"working in", u"in"])).add_child(loc_node)
    rear_loc_node = node(pick_one = True, p_dropout = 0.5).add_child(rear_loc_node_1, 0.2).add_child(rear_loc_node_2, 0.6)

    rear_cond_node = node(exchangeable = True, lang = 'en').add_child(rear_inst_node).add_child(rear_keyword_node).add_child(rear_loc_node)
    conded_expert_node = node(lang = 'en').add_child(front_cond_node).add_child(expert_node).add_child(rear_cond_node)

    search_node_1 = node([u"are there any", u"are there", u"who are", u"what are"], lang = 'en', gen_lemmas = True)
    search_node_2 = node(lang = 'en').add_child(node([u"give me", u"want to find", u"wanna find", u"find", u"find for", u"search", u"search for", u"query", u"show", u"look up for"])).add_child(node([u"the", u"those", u"the group of", u"a group of", u"some", u"a number of", u"a list of", u"the list of"], p_dropout = 0.8))
    search_node = node(pick_one = True, p_dropout = 0.5).add_child(search_node_1).add_child(search_node_2)
    
    search_expert_node = node(lang = 'en').add_child(search_node).add_child(conded_expert_node)
    # end intent search expert #

    # begin intent search paper #
    front_keyword_node = node(p_dropout = 0.5).add_child(keyword_node)
    front_inst_node = node(p_dropout = 0.5).add_child(inst_node)
    front_year_node = node(p_dropout = 0.5).add_child(year_node)
    front_name_node = node(p_dropout = 0.5).add_child(name_node).add_child(node(u"'s", p_dropout = 0.5))
    front_venue_node = node(p_dropout = 0.5).add_child(venue_node)
    paper_node = node([u"papers", u"works", u"journals", u"publications", u"researches"], p_dropout = 0.0, lang = 'en', gen_lemmas = True)

    front_cond_node = node(exchangeable = True, lang = 'en').add_child(front_keyword_node).add_child(front_inst_node).add_child(front_year_node).add_child(front_name_node).add_child(front_venue_node)

    which_node = node([u"that", u"which"])
    rear_keyword_node_1 = node(lang = 'en').add_child(which_node).add_child(node([u"focus on", u"are about", u"are related to"], lang = 'en', gen_lemmas = True)).add_child(keyword_node)
    rear_keyword_node_2 = node(lang = 'en').add_child(node([u"focusing on", u"on", u"about", u"related to"])).add_child(keyword_node)
    rear_keyword_node = node(pick_one = True, p_dropout = 0.5).add_child(rear_keyword_node_1, 0.2).add_child(rear_keyword_node_2, 0.6)
    rear_inst_node_1 = node(lang = 'en').add_child(which_node).add_child(node([u"are from", u"are by", u"are written by", u"are made by", u"are done by", u"are published by"], lang = 'en', gen_lemmas = True)).add_child(inst_node)
    rear_inst_node_2 = node(lang = 'en').add_child(node([u"from", u"by", u"written by", u"made by", u"done by", u"published by"])).add_child(inst_node)
    rear_inst_node = node(pick_one = True, p_dropout = 0.5).add_child(rear_inst_node_1, 0.2).add_child(rear_inst_node_2, 0.6)
    rear_name_node_1 = node(lang = 'en').add_child(which_node).add_child(node([u"are from", u"are by", u"are written by", u"are made by", u"are done by", u"are published by"], lang = 'en', gen_lemmas = True)).add_child(name_node)
    rear_name_node_2 = node(lang = 'en').add_child(node([u"from", u"by", u"written by", u"made by", u"done by", u"published by"])).add_child(name_node)
    rear_name_node = node(pick_one = True, p_dropout = 0.5).add_child(rear_name_node_1, 0.2).add_child(rear_name_node_2, 0.6)
    rear_year_node_1 = node(lang = 'en').add_child(which_node).add_child(node([u"are in", u"are written in", u"are made in", u"are done in", u"are published in", u"are at", u"are written at", u"are made at", u"are done at", u"are published at"], lang = 'en', gen_lemmas = True)).add_child(year_node)
    rear_year_node_2 = node(lang = 'en').add_child(node([u"in", u"written in", u"made in", u"done in", u"published in", u"at", u"written at", u"made at", u"done at", u"published at"])).add_child(year_node)
    rear_year_node = node(pick_one = True, p_dropout = 0.5).add_child(rear_year_node_1, 0.2).add_child(rear_year_node_2, 0.6)
    rear_venue_node_1 = node(lang = 'en').add_child(which_node).add_child(node([u"appear in", u"appear on", u"are on", u"are published on", u"are received by", u"are accepted by"], lang = 'en', gen_lemmas = True)).add_child(venue_node)
    rear_venue_node_2 = node(lang = 'en').add_child(node([u"on", u"appearing in", u"appearing on", u"published on", u"received by", u"accepted by"])).add_child(venue_node)
    rear_venue_node = node(pick_one = True, p_dropout = 0.5).add_child(rear_venue_node_1, 0.2).add_child(rear_venue_node_2, 0.6)
    
    rear_cond_node = node(exchangeable = True, lang = 'en').add_child(rear_keyword_node).add_child(rear_inst_node).add_child(rear_name_node).add_child(rear_year_node).add_child(rear_venue_node)
    conded_paper_node = node(lang = 'en').add_child(front_cond_node).add_child(paper_node).add_child(rear_cond_node)

    search_node_1 = node([u"are there any", u"are there", u"what are", u"which are"], lang = 'en', gen_lemmas = True)
    search_node_2 = node(lang = 'en').add_child(node([u"give me", u"want to find", u"wanna find", u"find", u"find for", u"search", u"search for", u"query", u"show", u"look up for"])).add_child(node([u"the", u"those", u"the group of", u"a group of", u"some", u"a number of", u"a list of", u"the list of"], p_dropout = 0.8))
    search_node = node(pick_one = True, p_dropout = 0.5).add_child(search_node_1).add_child(search_node_2)
    
    search_paper_node = node(lang = 'en').add_child(search_node).add_child(conded_paper_node)
    # end intent search paper #

    # begin intent search venue #
    front_keyword_node = node(p_dropout = 0.5).add_child(keyword_node)
    venue_node = node([u"journals", u"conferences"], p_dropout = 0.0, lang = 'en', gen_lemmas = True)

    front_cond_node = node(exchangeable = True, lang = 'en').add_child(front_keyword_node)

    which_node = node([u"that", u"which"])
    rear_keyword_node_1 = node(lang = 'en').add_child(which_node).add_child(node([u"focus on", u"are about", u"are related to"], lang = 'en', gen_lemmas = True)).add_child(keyword_node)
    rear_keyword_node_2 = node(lang = 'en').add_child(node([u"focusing on", u"on", u"about", u"related to"])).add_child(keyword_node)
    rear_keyword_node_3 = node(lang = 'en').add_child(node([u"whose papers", u"of which the papers", u"on which the papers"], lang = 'en', gen_lemmas = True)).add_child(node([u"focus on", u"are about", u"are in", u"are related to"], lang = 'en', gen_lemmas = True)).add_child(keyword_node)
    rear_keyword_node = node(pick_one = True, p_dropout = 0.5).add_child(rear_keyword_node_1, 0.2).add_child(rear_keyword_node_2, 0.6).add_child(rear_keyword_node_3, 0.2)

    rear_loc_node_1 = node(lang = 'en').add_child(which_node).add_child(node([u"are held in", u"are held at", u"are located in"], lang = 'en', gen_lemmas = True)).add_child(loc_node)
    rear_loc_node_2 = node(lang = 'en').add_child(node([u"in", u"located in", u"held in", u"held at"])).add_child(loc_node)
    rear_loc_node = node(pick_one = True, p_dropout = 0.5).add_child(rear_loc_node_1, 0.2).add_child(rear_loc_node_2, 0.6)

    rear_cond_node = node(exchangeable = True, lang = 'en').add_child(rear_keyword_node).add_child(rear_loc_node)
    conded_venue_node = node(lang = 'en').add_child(front_cond_node).add_child(venue_node).add_child(rear_cond_node)

    search_node_1 = node([u"are there any", u"are there", u"which are", u"what are"], lang = 'en', gen_lemmas = True)
    search_node_2 = node(lang = 'en').add_child(node([u"give me", u"want to find", u"wanna find", u"find", u"find for", u"search", u"search for", u"query", u"show", u"look up for"])).add_child(node([u"the", u"those", u"the group of", u"a group of", u"some", u"a number of", u"a list of", u"the list of"], p_dropout = 0.8))
    search_node = node(pick_one = True, p_dropout = 0.5).add_child(search_node_1).add_child(search_node_2)
    
    search_venue_node = node(lang = 'en').add_child(search_node).add_child(conded_venue_node)
    # end intent search venue #

    # begin intent search topics #
    front_keyword_node = node(p_dropout = 0.5).add_child(keyword_node)
    # front_inst_node = node(p_dropout = 0.5).add_child(inst_node)
    topic_node = node([u"subtopics", u"subtopic", u'terms', u'term', u'subareas', u'subarea', u'subfield', u'subfields', u'trend', u'hot topics', u"topics", u"topic", u"areas", u"area", u'interests', u'interest'], 
        p_dropout = 0.0, lang = 'en', gen_lemmas = True)
    front_abj_node = node(pick_one = True, p_dropout=0.5).add_child(node([u'researching', u'studying', u'research']))
    com_topic_node = node().add_child(front_abj_node).add_child(topic_node)

    which_node = node([u"that", u"which"])
    rear_keyword_node_1 = node(lang = 'en').add_child(which_node).add_child(node([u"focus on", u"are about", u"are related to"], lang = 'en', gen_lemmas = True)).add_child(keyword_node)
    rear_keyword_node_2 = node(lang = 'en').add_child(node([u"of", u"focusing on", u"on", u"about", u"related to"])).add_child(keyword_node)
    rear_keyword_node = node(pick_one = True, p_dropout = 0.2).add_child(rear_keyword_node_1, 0.2).add_child(rear_keyword_node_2, 0.6)
    rear_inst_node_1 = node(lang = 'en').add_child(which_node).add_child(node([u"are studied at", u"are studied in", u"are researched at", u"are researched in"], lang = 'en', gen_lemmas = True)).add_child(inst_node)
    rear_inst_node_2 = node(lang = 'en').add_child(node([u"studied at", u"studied in", u"researched at", u"researched in"], lang = 'en', gen_lemmas = True)).add_child(inst_node)
    rear_inst_node_3 = node(lang = 'en').add_child(node([u"at", u"in"], lang = 'en', gen_lemmas = True)).add_child(inst_node)
    rear_inst_node = node(pick_one = True, p_dropout = 0.2).add_child(rear_inst_node_1, 0.3).add_child(rear_inst_node_2, 0.3).add_child(rear_inst_node_3, 0.3)

    rear_cond_node = node(exchangeable = True, lang = 'en').add_child(rear_inst_node).add_child(rear_keyword_node)
    conded_expert_node = node(lang = 'en').add_child(com_topic_node).add_child(rear_cond_node)

    search_node_1 = node([u"what are"], lang = 'en', gen_lemmas = False)
    search_node_2 = node(lang = 'en').add_child(node([u"give me", u"want to find", u"wanna find", u"find", u"find for", u"search", u"search for", u"query", u"show", u"look up for"])).add_child(node([u"the", u"those", u"some", u"a number of", u"a list of", u"the list of"], p_dropout = 0.8))
    search_node = node(pick_one = True, p_dropout = 0.5).add_child(search_node_1).add_child(search_node_2)
    
    
    search_topic_node = node(lang = 'en').add_child(search_node).add_child(conded_expert_node)
    # end intent search topics #

    # pattern2
    begin_node = node(p_dropout = 0.1).add_child(node([u"what does", "what is"]))
    topic = node(p_dropout = 0.2).add_child(topic_node)
    search_sub = node(pick_one = True, p_dropout=0.0).add_child(keyword_node).add_child(inst_node)
    end_node = node(['involes', 'studies', 'includes', 'researches in', 'contains', 'relates to', 'focus on', 'studies about'], p_dropout=0.0)

    search_pattern_2 = node().add_child(begin_node).add_child(topic).add_child(search_sub).add_child(end_node)

    s = simulator().add_root(search_pattern_2, u"topic", 0.05).add_root(search_topic_node, u'topic', 0.2).add_root(search_venue_node, u'venue', 0.25).add_root(search_paper_node, u'paper', 0.5).add_root(search_expert_node, u'expert', 0.5)
    text = s.generate(100000)
    fout = open(file_name, 'w')
    fout.write(text)

def test():
    simulate(debug = True)

if __name__ == '__main__':
    test()
