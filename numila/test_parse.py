import numila
import parse

def test_parse():
    model = numila.Numila()
    parse.LOG.setLevel('DEBUG')
    model.parse('the dog went to the store')

if __name__ == '__main__':
    test_parse()