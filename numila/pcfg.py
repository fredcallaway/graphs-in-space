import utils

from nltk.grammar import PCFG
from nltk.probability import DictionaryProbDist

# modified function from NLTK
def generate(grammar, start=None, depth=5):
    """Returns 1 tree from the distribution defined by a grammar.

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

    if not start:
        start = grammar.start()


    for _ in range(1000):  # max tries
        try:
            return generate_one(grammar, start, depth)
        except DepthExceededError:
            pass  # try again


def random_sentences(grammar_string, n=None, depth=5):
    grammar = PCFG.fromstring(grammar_string)
    i = 0
    while True:
        if i == n: return
        tree = generate(grammar, depth=depth)
        yield ' '.join(utils.flatten(tree))
        i += 1

def toy2():
    with open('corpora/toy_pcfg2.txt') as f:
        grammar = f.read()
    return random_sentences(grammar)


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
    gen = toy2()
    for s in (next(gen) for _ in range(4)):
        print(s)
