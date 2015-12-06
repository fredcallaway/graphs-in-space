import utils

from nltk.grammar import PCFG
from nltk.probability import DictionaryProbDist

# modified function from NLTK
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


def random_sentences(grammar_string, n):
    grammar = PCFG.fromstring(grammar_string)
    for tree in generate(grammar, n, depth=5):
        yield ' '.join(utils.flatten(tree))


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
    with open('corpora/toy_pcfg2.txt') as f:
        grammar = f.read()
    with open('corpora/toy2.txt', 'w+') as f:
        for s in random_sentences(grammar, 1000):
            f.write(' '.join(s) + '\n')
