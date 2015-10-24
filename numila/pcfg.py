import itertools

from nltk.grammar import PCFG
from nltk.probability import DictionaryProbDist


def generate(grammar, n, start=None, depth=5):
    """Yields n sentences from the distribution defined by a grammar.

    :param grammar: The Grammar used to generate sentences.
    :param start: The Nonterminal from which to start generate sentences.
    :param depth: The maximal depth of the generated tree.
    :param n: The maximum number of sentences to return.
    :return: An iterator of lists of terminal tokens.
    """
    class DepthExceededError(Exception): pass

    def generate_one(grammar, item, depth):
        if depth > 0:
            productions = {p.rhs(): p.prob()
                           for p in grammar.productions(item)}
            rhs = DictionaryProbDist(productions).generate()
            assert rhs is not None
            if len(rhs) is 1 and isinstance(rhs[0], str):
                # base case: a terminal
                return rhs[0]
            else:
                return [generate_one(grammar, elem, depth-1) for elem in rhs]
        else:
            raise DepthExceededError()
            raise 

    if not start:
        start = grammar.start()

    for _ in range(n):
        sent = False
        while not sent:
            try:
                sent = generate_one(grammar, start, depth)
            except DepthExceededError:  # exceeded max depth
                pass
        yield sent


def flatten(lst):
    """Returns a flat version of a list of (lists of (lists of...)) lists"""
    assert lst is not None
    def iterable(obj): 
        return hasattr(obj, '__iter__') and not isinstance(obj, str)
    if not any(iterable(elem) for elem in lst):
        return lst
    else:
        flat_lists = [flatten(elem) if iterable(elem) else [elem]
                      for elem in lst]
        return list(itertools.chain(*flat_lists))


def random_sentences(grammar_file, n):
    with open(grammar_file) as f:
        grammar = PCFG.fromstring(f.read())
    for tree in generate(grammar, n, depth=5):
        yield flatten(tree)


def draw_tree(tree_string):
    raise NotImplementedError()

    from nltk import Tree
    from nltk.draw.util import CanvasFrame
    from nltk.draw import TreeWidget

    cf = CanvasFrame()
    tree = Tree.fromstring(tree_string.replace('[','(').replace(']',')') )
    cf.add_widget(TreeWidget(cf.canvas(), tree), 10, 10)
    cf.print_to_file('tree.ps')
    cf.destroy

if __name__ == '__main__':
    print('\n\n')
    print(flatten([[[1,2], 3], [4,[5,6]]]))
    for s in random_sentences('tiny_pcfg.txt', 5):
        print(s)