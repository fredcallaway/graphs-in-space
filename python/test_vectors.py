from vectors import VectorModel
import numpy as np


def test_vector_model():
    vector_model = VectorModel(1000, .01, 'addition')
    vectors = [vector_model.sparse() for _ in range(5000)]
    assert(all(vec.shape == (vector_model.dim,) for vec in vectors))

    num_nonzero = vector_model.dim * vector_model.nonzero
    num_nonzeros = [len(np.nonzero(vec)[0]) for vec in vectors]
    assert(all(n == num_nonzero for n in num_nonzeros))


if __name__ == '__main__':
    test_vector_model()