from plotting import mds

from numila import Numila
import pcfg
import utils


def main():
    toy_pcfg = '''
    S    -> NP VP    [1.0]
    VP   -> V NP     [1.0]
    NP   -> Det N_   [1.0]
    N_   -> N       [0.5]
    N_   -> Adj N_   [0.5]
    V    -> 'tricked' [1.0]
    N    -> 'dog'    [1.0]
    Adj  -> 'sly'    [1.0]
    Det  -> 'the'    [1.0]
    '''

    corpus = list(pcfg.random_sentences(toy_pcfg, 50, depth=10))
    print(*corpus[:10], sep='\n')

    model = Numila(COMPOSITION=1)
    model.fit(corpus)

    phrases = [
        'the',
        'sly',
        'dog',
        'tricked',
        '[sly dog]',
        '[sly sly]',
        '[[sly sly] dog]',
        '[sly [sly dog]]',
        '[[[sly sly] sly] dog]',
    ]

    nodes = list(filter(None, map(model.graph.get, phrases)))

    print(nodes)

    data = [[1-n1.similarity(n2) for n2 in nodes]
            for n1 in nodes]
    labels = [str(n) for n in nodes]
    import IPython; IPython.embed()
    mds(data, labels, name='composition')


if __name__ == '__main__':
    main()