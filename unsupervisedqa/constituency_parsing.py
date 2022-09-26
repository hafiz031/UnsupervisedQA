# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
"""
Functionality to do constituency parsing, used for shortening cloze questions. We use AllenNLP and the
Parsing model from Stern et. al, 2018 "A Minimal Span-Based Neural Constituency Parser" arXiv:1705.03919
"""
import logging
import attr
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from tqdm import tqdm
from nltk import Tree
from .configs import CONSTITUENCY_MODEL, CONSTITUENCY_BATCH_SIZE, CONSTITUENCY_CUDA, CLOZE_SYNTACTIC_TYPES
from .generate_clozes import mask_answer
from .data_classes import Cloze
from allennlp_models.pretrained import load_predictor


def _load_constituency_parser():
    # return Predictor.from_path("constituency/elmo-constituency-parser-2018.03.14.tar.gz", 
    #         'constituency-parser',
    #         overrides={"dataset_reader.tokenizer.language": "en"}) # getting: allennlp.common.checks.ConfigurationError: key "token_embedders" is required at location "model.text_field_embedder."
    # return Predictor.from_path(CONSTITUENCY_MODEL) # perhaps a valid form
    # # archive = load_archive("constituency/elmo-constituency-parser-2018.03.14.tar.gz", cuda_device=CONSTITUENCY_CUDA)
    # # return Predictor.from_archive(archive, 'constituency-parser')

    # does it helps?: https://stackoverflow.com/q/66844202/6907424 (different model btw) or try DIDN'T WORK (couldn't install): "!pip install allennlp==1.0.0 allennlp-models==1.0.0"
    # archive = load_archive(CONSTITUENCY_MODEL, cuda_device=CONSTITUENCY_CUDA)
    # print(f"ARCHIVE: {archive}")
    # return Predictor.from_archive(archive, 'constituency-parser')
    # worked: https://paperswithcode.com/model/constituency-parser-with-elmo-embeddings
    return load_predictor("structured-prediction-constituency-parser") # download works


def get_constituency_parsed_clozes(clozes, predictor=None, verbose=True, desc='Running Constituency Parsing'):
    """
    A sample Cloze: (Note that here constituency_parse is None...it will be updated now)
    -----------------------------------------------------------------------------------
    Cloze(cloze_id='55622000e9486bfb321cdcabb8b3bf9e4163d36a', 
    paragraph=Paragraph(paragraph_id='ff7f0c796cc44b1dbeb821ccd8fda74c1fab7b66',
    text="CBS' broadcast of the game was the third most-watched program in American television history 
    with an average of 111.9 million viewers. The network charged an average of $5 million for a 30-second 
    commercial during the game.[12][13] It remains the highest-rated program in the history of CBS. The Super 
    Bowl 50 halftime show was headlined by Coldplay,[14] with special guest performers Beyoncé and Bruno Mars."), 
    source_text='The Super Bowl 50 halftime show was headlined by Coldplay,[14] with special guest performers 
    Beyoncé and Bruno Mars.', source_start=292, cloze_text='The Super Bowl 50 halftime show was headlined by Coldplay,
    [14] with special guest performers IDENTITYMASK and Bruno Mars.', answer_text='Beyoncé', answer_start=93, 
    constituency_parse=None, root_label=None, answer_type='NORP', question_text=None)
    """
    if predictor is None:
        predictor = _load_constituency_parser()
    jobs = range(0, len(clozes), CONSTITUENCY_BATCH_SIZE)
    for i in tqdm(jobs, desc=desc, ncols=80) if verbose else jobs:
        input_batch = clozes[i: i + CONSTITUENCY_BATCH_SIZE]
        output_batch = predictor.predict_batch_json([{'sentence': c.source_text} for c in input_batch])
        for c, t in zip(input_batch, output_batch):
            root = _get_root_type(t['trees'])
            if root in CLOZE_SYNTACTIC_TYPES:
                c_with_parse = attr.evolve(c, constituency_parse=t['trees'], root_label=root)
                yield c_with_parse


def _get_root_type(tree):
    try:
        t = Tree.fromstring(tree)
        label = t.label()
    except:
        label = 'FAIL'
    return label


def _get_sub_clauses(root, clause_labels):
    """Simplify a sentence by getting clauses:
    Sample Input:
    root = Tree('S', [Tree('NP', [Tree('DT', ['The']), Tree('NNP', ['Super']), Tree('NNP', ['Bowl']), Tree('CD', ['50']), Tree('NN', ['halftime']), Tree('NN', ['show'])]), Tree('VP', [Tree('VBD', ['was']), Tree('VP', [Tree('VBN', ['headlined']), Tree('PP', [Tree('IN', ['by']), Tree('NP', [Tree('NNP', ['Coldplay,[14'])])]), Tree('-RRB-', [']']), Tree('PP', [Tree('IN', ['with']), Tree('NP', [Tree('NP', [Tree('JJ', ['special']), Tree('NN', ['guest']), Tree('NNS', ['performers'])]), Tree('NNP', ['Beyoncé']), Tree('CC', ['and']), Tree('NNP', ['Bruno']), Tree('NNP', ['Mars'])])])])]), Tree('.', ['.'])])
    clause_labels = {S, }

    Sample Output:
    ['The Super Bowl 50 halftime show was headlined by Coldplay,[14 ] with special guest performers Beyoncé and Bruno Mars .']
    """
    subtexts = []
    for current in root.subtrees():
        if current.label() in clause_labels:
            subtexts.append(' '.join(current.leaves()))
    return subtexts


def _tokens2spans(sentence, tokens):
    """
    sentence: The Super Bowl 50 halftime show was headlined by Coldplay,[14] with special guest performers 
    Beyoncé and Bruno Mars.

    tokens (= root.leaves()):
    ['The',
    'Super',
    'Bowl',
    '50',
    'halftime',
    'show',
    'was',
    'headlined',
    'by',
    'Coldplay,[14',
    ']',
    'with',
    'special',
    'guest',
    'performers',
    'Beyoncé',
    'and',
    'Bruno',
    'Mars',
    '.']

    output:
    [(0, 3),
    (4, 9),
    (10, 14),
    (15, 17),
    (18, 26),
    (27, 31),
    (32, 35),
    (36, 45),
    (46, 48),
    (49, 61),
    (61, 62),
    (63, 67),
    (68, 75),
    (76, 81),
    (82, 92),
    (98, 105),
    (106, 109),
    (110, 115),
    (116, 120),
    (120, 121)]

    """
    off = 0
    spans = []
    for t in tokens:
        # index gives the starting char index of the first appearance of that token
        # searching on the substring to avoid calculating on the same portion (and potentially getting the same
        # previous appearance)...finally +off ensures the position is calculated with respect to the complete
        # string and not just the portion.
        span_start = sentence[off:].index(t) + off
        spans.append((span_start, span_start + len(t)))
        off = spans[-1][-1] # the ending position of the last tuple
    for t, (s, e) in zip(tokens, spans): # making sure if span collection is correct
        assert sentence[s:e] == t
    return spans # list of tuples


def _subseq2sentence(sentence, tokens, token_spans, subsequence):
    subsequence_tokens = subsequence.split(' ')
    for ind in (i for i, t in enumerate(tokens) if t == subsequence_tokens[0]):
        if tokens[ind: ind + len(subsequence_tokens)] == subsequence_tokens:
            return sentence[token_spans[ind][0]:token_spans[ind + len(subsequence_tokens) - 1][1]]
    raise Exception('Failed to repair sentence from token list')


def get_sub_clauses(sentence, tree):
    print("############def get_sub_clauses(sentence, tree):")
    clause_labels = CLOZE_SYNTACTIC_TYPES
    root = Tree.fromstring(tree) # Reading the "constituency_parse" attribute from Cloze()
    """
    root:
    Tree('S', [Tree('NP', [Tree('DT', ['The']), Tree('NNP', ['Super']), Tree('NNP', ['Bowl']), Tree('CD', ['50']), Tree('NN', ['halftime']), Tree('NN', ['show'])]), Tree('VP', [Tree('VBD', ['was']), Tree('VP', [Tree('VBN', ['headlined']), Tree('PP', [Tree('IN', ['by']), Tree('NP', [Tree('NNP', ['Coldplay,[14'])])]), Tree('-RRB-', [']']), Tree('PP', [Tree('IN', ['with']), Tree('NP', [Tree('NP', [Tree('JJ', ['special']), Tree('NN', ['guest']), Tree('NNS', ['performers'])]), Tree('NNP', ['Beyoncé']), Tree('CC', ['and']), Tree('NNP', ['Bruno']), Tree('NNP', ['Mars'])])])])]), Tree('.', ['.'])])
    """
    subs = _get_sub_clauses(root, clause_labels)
    tokens = root.leaves()
    print(f"TOKENS: {tokens}")
    """
    root.leaves(): # just returns the tokens
    ['The',
    'Super',
    'Bowl',
    '50',
    'halftime',
    'show',
    'was',
    'headlined',
    'by',
    'Coldplay,[14',
    ']',
    'with',
    'special',
    'guest',
    'performers',
    'Beyoncé',
    'and',
    'Bruno',
    'Mars',
    '.']
    """
    token_spans = _tokens2spans(sentence, tokens)
    print(f"TOKEN_SPANS: {token_spans}")
    print(f"INSIDE: get_sub_clauses: {token_spans}")
    return [_subseq2sentence(sentence, tokens, token_spans, sub) for sub in subs]


def shorten_cloze(cloze):
    print("###########SC")
    print(f"##shorten_cloze: {cloze}")
    """Return a list of shortened cloze questions from the original cloze question
    A sample cloze element:
    Cloze(cloze_id='bf9730746cdc38f899c8b8cf5583973bb2d98682', 
    paragraph=Paragraph(paragraph_id='ff7f0c796cc44b1dbeb821ccd8fda74c1fab7b66', 
    text="CBS' broadcast of the game was the third most-watched program in American television history with an 
    average of 111.9 million viewers. The network charged an average of $5 million for a 30-second commercial 
    during the game.[12][13] It remains the highest-rated program in the history of CBS. The Super Bowl 50 halftime 
    show was headlined by Coldplay,[14] with special guest performers Beyoncé and Bruno Mars."), 
    source_text='The Super Bowl 50 halftime show was headlined by Coldplay,[14] with special guest performers 
    Beyoncé and Bruno Mars.', source_start=292, cloze_text='The Super Bowl 50 halftime show was headlined by 
    Coldplay,[14] with special guest performers Beyoncé and IDENTITYMASK.', answer_text='Bruno Mars', 
    answer_start=105, constituency_parse='(S (NP (DT The) (NNP Super) (NNP Bowl) (CD 50) (NN halftime) (NN show)) 
    (VP (VBD was) (VP (VBN headlined) (PP (IN by) (NP (NNP Coldplay,[14))) (-RRB- ]) (PP (IN with) (NP (NP (JJ special) 
    (NN guest) (NNS performers)) (NNP Beyoncé) (CC and) (NNP Bruno) (NNP Mars))))) (. .))', 
    root_label='S', answer_type='PERSON', question_text=None)
    """
    simple_clozes = []
    try:
        subs = get_sub_clauses(cloze.source_text, cloze.constituency_parse)
        subs = sorted(subs)
        for sub in subs:
            if sub != cloze.source_text:
                sub_start_index = cloze.source_text.index(sub)
                sub_answer_start_index = cloze.answer_start - sub_start_index
                good_start = 0 <= sub_answer_start_index <= len(sub)
                good_end = 0 <= sub_answer_start_index + len(cloze.answer_text) <= len(sub)
                if good_start and good_end:
                    simple_clozes.append(
                        Cloze(
                            cloze_id=cloze.cloze_id + f'_{len(simple_clozes)}',
                            paragraph=cloze.paragraph,
                            source_text=sub,
                            source_start=cloze.source_start + sub_start_index,
                            cloze_text=mask_answer(sub, cloze.answer_text, sub_answer_start_index, cloze.answer_type),
                            answer_text=cloze.answer_text,
                            answer_start=sub_answer_start_index,
                            constituency_parse=None,
                            root_label=None,
                            answer_type=cloze.answer_type,
                            question_text=None
                        )
                    )
    except Exception as e:
        logging.exception(e)
        print(f'Failed to parse cloze: ID {cloze.cloze_id} Text: {cloze.source_text}')
    return simple_clozes
