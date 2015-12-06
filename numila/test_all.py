import pytest
from numila import *
from vectors import *


@pytest.fixture()
def numila():
    numila = Numila(DIM=1000, PERCENT_NON_ZERO=.01, CHUNK_THRESHOLD=0.5)
    with open('corpora/toy2.txt') as corpus:
                for i, s in enumerate(corpus.read().splitlines()):
                    numila.parse_utterance(s)
    return numila

@pytest.fixture()
def vector_model():
    return VectorModel(1000, .01, 'addition')


def test_parsing(numila):
    spanish = 'los gatos son grasos'
    parse = numila.parse_utterance(spanish)
    assert str(parse) == '(# | los | gatos | son | grasos | #)'

def test_speaking(numila):
    words = 'the hill ate my cookie'.split()
    utterance = str(numila.speak(words))
    assert all(w in utterance for w in words)
    no_brackets = utterance.replace('[', '').replace(']', '')
    assert len(no_brackets.split()) == len(words)
    print(utterance)

def test_vector_model(vector_model):
    vectors = [vector_model.sparse() for _ in range(5000)]
    assert(all(vec.shape == (vector_model.dim,) for vec in vectors))

    num_nonzero = vector_model.dim * vector_model.nonzero
    num_nonzeros = [len(np.nonzero(vec)[0]) for vec in vectors]
    assert(all(n == num_nonzero for n in num_nonzeros))

if __name__ == '__main__':
    pytest.main()