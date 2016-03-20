import pytest

from numila import Numila
import parse
import batch_parse
import utils

def test_inc_basic():
    model = Numila(PARSE='incremental', CHUNK_THRESHOLD=2)

    log = utils.log_parse(model, 'a b c')
    assert log.count('bump a -> b') is 2
    assert log.count('bump b -> c') is 2

    log = utils.log_parse(model, 'a b c d')
    assert log.count('bump a -> b') is 3
    assert log.count('bump b -> c') is 3
    assert log.count('bump c -> d') is 3


    log = utils.log_parse(model, 'a b c d e')
    assert log.count('bump a -> b') is 3
    assert log.count('bump b -> c') is 3
    assert log.count('bump c -> d') is 3
    assert log.count('bump d -> e') is 3


def test_batch_basic():
    model = Numila(PARSE='batch', CHUNK_THRESHOLD=2)

    log = utils.log_parse(model, 'a b c')
    assert log.count('bump a -> b') is 1
    assert log.count('bump b -> c') is 1

    log = utils.log_parse(model, 'a b c d')
    assert log.count('bump a -> b') is 1
    assert log.count('bump b -> c') is 1
    assert log.count('bump c -> d') is 1


def test_batch_score():
    model = Numila(PARSE='batch', CHUNK_THRESHOLD=2)
    for _ in range(5):
        model.parse('the dog ate the steak')
    

    assert model.score('the dog ate the steak') > model.score('the ate dog steak the')


if __name__ == '__main__':
    pytest.main(__file__)
    
    #test_incremental()
    #test_batch_basic()