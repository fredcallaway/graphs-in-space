import re

def read_corpus(file, token_delim=' +', utt_delim='\n', num_utterances=None):
    """A list of lists of tokens in a corpus"""
    with open(file) as f:
        for idx, utterance in enumerate(re.split(utt_delim, f.read())):
            if idx == num_utterances:
                break
            if token_delim:
                tokens = re.split(token_delim, utterance.strip())
            else:
                tokens = list(utterance)  # split by character
            yield tokens

def cfg_corpus(n=None):
    corpus = read_corpus('corpora/toy2.txt', num_utterances=n)
    return corpus

def syl_corpus(n=None):
    corpus = read_corpus('../PhillipsPearl_Corpora/English/English-syl.txt',
                               token_delim=r'/| ', num_utterances=n)
    return corpus

def get_corpus(lang, kind):
    file = ('corpora/phillips-pearl/{lang}-syl.txt'
            .format(lang=lang))
    corpus = open(file).read()

    if kind == 'syl':
        token_delim = r'(?:/| )+'
    elif kind == 'word':
        corpus = corpus.replace('/', '')
        token_delim = r' +'
    elif kind == 'phone':
        corpus = corpus.replace('/', '')
        corpus = corpus.replace(' ', '')
        token_delim = None
    else:
        raise ValueError('invalid kind argument: {}'.format(kind))

    utt_delim = '\n'
    for utterance in re.split(utt_delim, corpus):
        utterance = utterance.strip()
        if token_delim:
            tokens = re.split(token_delim, utterance)
        else:
            tokens = list(utterance)  # split by character
        yield tokens

if __name__ == '__main__':
    corp = list(get_corpus('Hungarian', 'syl'))
    print(corp[:10])
