import numpy as np
from scipy.spatial import distance

DIM = 1000
NON_ZERO = 0.01

P1 = np.random.permutation(DIM)
INVERSE_P1 = np.argsort(P1)
P2 = np.random.permutation(DIM)
INVERSE_P2 = np.argsort(P2)

def sparse():
    vector = np.zeros(DIM)
    indices = (np.random.random(int(DIM * NON_ZERO)) * DIM).astype(int)
    for i in indices:
        vector[i] = np.random.choice((-1.0, 1.0))
    return vector

##cconv and ccor taken from https://github.com/mike-lawrence/wikiBEAGLE
def cconv(a, b):
    """Computes the circular convolution of the vectors a and b."""
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

def ccorr(a, b):
    """Computes the circular correlation (inverse convolution) of vectors a and b."""
    return cconv(np.roll(a[::-1], 1), b)

def cosine(v1, v2) -> float:
    return 1.0 - distance.cosine(v1, v2)

def bind(v1, v2) -> np.ndarray:
    return cconv(v1[P1], v2[P2])

if __name__ == '__main__':
    import IPython; IPython.embed()
    percent_non_zero = 0.2
    non_zeros = [len(np.nonzero(sparse(1000, percent_non_zero))[0])
                 for _ in range(1000)]
    avg_percent_non_zero = ((sum(non_zeros) / len(non_zeros)) / 1000)
    assert avg_percent_non_zero - percent_non_zero < .001