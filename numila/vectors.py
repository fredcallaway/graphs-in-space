import numpy as np
import itertools

class VectorModel(object):
    """Represents points in a high dimensional space."""
    def __init__(self, dim, nonzero, bind_op):
        super(VectorModel, self).__init__()
        self.dim = dim
        self.nonzero = nonzero
        
        if bind_op == 'addition':
            from operator import add
            self.bind_op = add
        elif bind_op == 'convolution':
            self.bind_op = cconv
        else:
            raise ValueError('Invalid bind_op: {bind_op}'.format_map(locals()))

        self.perm1 = np.random.permutation(self.dim)
        self.inverse_perm1 = np.argsort(self.perm1)
        self.perm2 = np.random.permutation(self.dim)
        self.inverse_perm2 = np.argsort(self.perm2)

        self.alternating_ints = itertools.cycle((1, -1))  # 1, -1 , 1, -1 ...
    
    #@profile
    def sparse(self) -> np.ndarray:
        """Returns a new sparse vector."""
        num_nonzero = int(np.ceil(self.dim * self.nonzero))
        
        indices = set()  # a set of num_nonzero unique indices between 0 and self.dim
        for _ in range(num_nonzero):
            idx = np.random.randint(self.dim)
            while idx in indices:
                # resample until we get a new index
                idx = np.random.randint(self.dim)
            indices.add(idx)

        vector = np.zeros(self.dim)
        for i in indices:
            vector[i] = next(self.alternating_ints)
        return vector

    def bind(self, v1, v2) -> np.ndarray:
        permuted_v1 = v1[self.perm1]
        permuted_v2 = v2[self.perm2]
        return self.bind_op(permuted_v1, permuted_v2)

    def permutation(self) -> np.ndarray:
        return np.random.permutation(self.dim)


# taken from https://github.com/mike-lawrence/wikiBEAGLE
def cconv(a, b):
    """Computes the circular convolution of the vectors a and b."""
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

def ccorr(a, b):
    """Computes the circular correlation (inverse convolution) of vectors a and b."""
    return cconv(np.roll(a[::-1], 1), b)

def cosine(a,b):
    """Computes the cosine of the angle between the vectors a and b."""
    sum_sq_a = np.sum(a**2.0)
    sum_sq_b = np.sum(b**2.0)
    result = np.dot(a,b) * (sum_sq_a * sum_sq_b) ** -0.5
    assert -1 <= result <= 1
    return result

def normalize(a):
    """Normalize a vector to length 1."""
    return a / np.sum(a**2.0)**0.5

if __name__ == '__main__':
    nonzero = 0.01
    dim = 1000

    vector_model = VectorModel(dim, nonzero)
    vectors = [vector_model.sparse() for _ in range(5000)]
    assert(all(vec.shape == (dim,) for vec in vectors))

    num_nonzero = dim * nonzero
    num_nonzeros = [len(np.nonzero(vec)[0]) for vec in vectors]
    assert(all(n == num_nonzero for n in num_nonzeros))
