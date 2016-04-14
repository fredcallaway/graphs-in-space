import numpy as np
import itertools
import utils

class VectorModel(object):
    """Represents points in a high dimensional space."""
    def __init__(self, dim, nonzero, bind_op):
        super(VectorModel, self).__init__()
        self.dim = dim
        self.nonzero = nonzero
        self.num_nonzero = int(np.ceil(dim * self.nonzero))

        
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

        # Vectors are normalized, with have alternating
        # positive and negative values.
        norm = self.num_nonzero ** 0.5
        self._vector_values = itertools.cycle((1/norm, -1/norm))
    
    #@profile
    def sparse(self, dim=None):
        """Returns a new sparse vector."""
        dim = dim or self.dim
        if not self.num_nonzero:
            raise ValueError('Too sparse!')
        
        indices = set()  # a set of num_nonzero unique indices between 0 and dim
        for _ in range(self.num_nonzero):
            idx = np.random.randint(dim)
            while idx in indices:
                # resample until we get a new index
                idx = np.random.randint(dim)
            indices.add(idx)

        assert len(indices) == self.num_nonzero
        vector = np.zeros(dim)
        for i in indices:
            vector[i] = next(self._vector_values)
        return vector

    def zeros(self):
        """Returns a new 0 vector."""
        return np.zeros(self.dim)

    def bind(self, v1, v2):
        permuted_v1 = v1[self.perm1]
        permuted_v2 = v2[self.perm2]
        return self.bind_op(permuted_v1, permuted_v2)

    def permutation(self):
        return np.random.permutation(self.dim)


# taken from https://github.com/mike-lawrence/wikiBEAGLE
def cconv(a, b):
    """Computes the circular convolution of the vectors a and b."""
    return np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)).real

def ccorr(a, b):
    """Computes the circular correlation (inverse convolution) of vectors a and b."""
    return cconv(np.roll(a[::-1], 1), b)

@utils.contract(lambda x: not np.isnan(x))
def cosine(a,b):
    """Computes the cosine of the angle between the vectors a and b."""
    assert len(np.nonzero(a)[0])
    assert len(np.nonzero(b)[0])
    sum_sq_a = np.sum(a**2.0)
    sum_sq_b = np.sum(b**2.0)
    result = np.dot(a,b) * (sum_sq_a * sum_sq_b) ** -0.5
    
    #if not -1.1 <= result <= 1.1:
    #    msg = 'sum(a) = {}, sum(b) = {}'.format(np.sum(a), np.sum(b))
    #    raise ValueError('cosine returned {}'.format(result) + '\n' + msg)

    return result

def normalize(a):
    """Normalize a vector to length 1."""
    return a / np.sum(a**2.0)**0.5

def _speed_test(n=100):
    nonzero = 0.01
    dim = 1000
    #for op in ('addition', 'convolution'):
    for op in ('addition',):
        vector_model = VectorModel(dim, nonzero, op)
        sparse1 = vector_model.sparse()
        sparse2 = vector_model.sparse()
        dense1 = np.random.rand(dim)
        dense2 = np.random.rand(dim)
        with utils.Timer(op + ' sparse'):
            for _ in range(n):
                vector_model.bind(sparse1, sparse2)
        with utils.Timer(op + ' dense'):
            for _ in range(n):
                vector_model.bind(dense1, dense2)


def _test():
    nonzero = 0.01
    dim = 1000
    for op in 'addition', 'convolution':
        print('\n' + op)

        vector_model = VectorModel(dim, nonzero, op)

        a = vector_model.sparse()
        assert abs(np.sqrt(np.sum((a ** 2))) - 1) < .001
        b = vector_model.sparse()
        c = vector_model.sparse()


        ab = vector_model.bind(a, b)
        ac = vector_model.bind(a, c)
        ba = vector_model.bind(b, a)
        print('(ab, ac)', cosine(ab, ac))
        print('(ab, ba)', cosine(ab, ba))

        a1 = a + 0.5 * vector_model.sparse()
        b1 = b + 0.5 * vector_model.sparse()

        a1b = vector_model.bind(a1, b)
        ab1 = vector_model.bind(a, b1)
        a1b1 = vector_model.bind(a1, b1)
        print('(ab, a1b)', cosine(ab, a1b))
        print('(ab, ab1)', cosine(ab, ab1))
        print('(ab, a1b1)', cosine(ab, a1b1))
    
    exit()




    vectors = [vector_model.sparse() for _ in range(5000)]
    assert(all(vec.shape == (dim,) for vec in vectors))

    num_nonzero = dim * nonzero
    num_nonzeros = [len(np.nonzero(vec)[0]) for vec in vectors]
    assert(all(n == num_nonzero for n in num_nonzeros))


if __name__ == '__main__':
    #_speed_test(10000)
    _test()