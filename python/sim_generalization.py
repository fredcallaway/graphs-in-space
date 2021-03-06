"""Demonstrates generalization and composition.



"""
from collections import Counter
import itertools
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette('muted')
sns.set_context('poster')

from vectorgraph import VectorGraph
import utils

import pcfg

def main():

    # A pcfg that has free slots for a noun and determiner.
    toy_pcfg = '''
    S    -> NP VP    [1.0]
    VP   -> V NP     [0.5]
    VP   -> V        [0.5]
    NP   -> Det N    [0.5]
    NP   -> Name     [0.5]
    V    -> 'saw'    [0.5]
    V    -> 'ate'    [0.5]
    N    -> 'boy'    [0.5]
    N    -> '{NOUN}' [0.5]
    Name -> 'Jack'   [0.5]
    Name -> 'Bob'    [0.5]
    Det  -> 'the'    [0.5]
    Det  -> '{DET}'  [0.5]
    '''

    # Instantiate two versions of the above pcfg with (that, table)
    # and (my, bunny) as (DET, NOUN). As a result, neither pcfg can
    # generate "that bunny" nor "my table".
    that_table_pcfg = toy_pcfg.format(DET='that', NOUN='table')
    my_bunny_pcfg = toy_pcfg.format(DET='my', NOUN='bunny')

    corpus = (list(pcfg.random_sentences(that_table_pcfg, 100))
              + list(pcfg.random_sentences(my_bunny_pcfg, 100)))

    # Mix sentences from each corpus. This is essential for the dynamic
    # generalization algorithm to work.
    np.random.shuffle(corpus)

    # Check how many times each critical pair occured.
    bigrams = Counter(itertools.chain(*(list(utils.neighbors(utt.split(' '))) for utt in corpus)))
    for det in 'that', 'my':
        for noun in 'table', 'bunny':
            print(det, noun, ':', bigrams[(det, noun)])
    
    # Train the graph on the markov transitions in the combined corpus.
    graph = VectorGraph(DYNAMIC=True, COMPOSITION=1)
    train_bigram(graph, corpus)
    
    generalization(graph)
    composition(graph)


def train_bigram(graph, corpus):
    for sentence in corpus:
        bigrams = utils.neighbors(sentence.split(' '))
        for word1, word2 in bigrams:
            node1 = graph.get(word1, add=True)
            node2 = graph.get(word2, add=True)
            node1.bump_edge(node2)


def generalization(graph):
    """Figure 3"""
    data = [{'generalization' : str(gen and gen[1]),
             'det': det,
             'noun': noun,
             'edge weight': graph[det].edge_weight(graph[noun], generalize=gen)} 
            for det in ['my', 'that']
            for noun in ['boy', 'table', 'bunny', 'Jack', 'saw', 'the']
            for gen in [0, ('similarity', 0.5)]]
    df = pd.DataFrame(data)
    sns.factorplot('noun', 'edge weight', col='generalization', hue='det',
                   data=df, kind='bar', legend_out=True).despine(left=True).set_xticklabels(rotation=30)

    sns.plt.gcf().tight_layout()
    sns.plt.savefig('figs/generalization.pdf')
    print('created figs/generalization.pdf')


def composition(graph):
    """Figure 4"""
    dets = [graph[w] for w in ['that', 'my']]
    nouns = [graph[w] for w in [ 'table', 'bunny']]
    verbs = [graph[w] for w in ['saw', 'ate']]
    noun_phrases = [graph.bind(d, n) for d in dets for n in nouns]

    # Train (NP -> verb) pairs
    for NP in noun_phrases:
        graph.add(NP)
        for verb in verbs:
            NP.bump_edge(verb, factor=5)


    the, boy, saw, ate, jack = map(graph.get, ('the', 'boy', 'saw', 'ate', 'Jack'))
    that_table = graph.get('[that table]')

    data = [{'composition': str(composition),
             'noun phrase': str(NP),
             'verb': str(verb),
             'edge weight': NP.edge_weight(verb)} 
            for composition in (0, 0.5)
            for NP in [that_table, graph.bind(the, boy, composition=composition)]
            for verb in [saw, ate, the, boy]]  # include bad verbs (the, boy)

    df = pd.DataFrame(data)
    sns.factorplot('verb', 'edge weight', hue='noun phrase', col='composition',
                   data=df, kind='bar').despine(left=True)

    sns.plt.savefig('figs/composition.pdf')
    print('created figs/composition.pdf')


if __name__ == '__main__':
    main()