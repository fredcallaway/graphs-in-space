import numpy as np

def sparse_vector(size, non_zero):
    vector = np.zeros(size)
    indices = (np.random.random(int(size * non_zero)) * size).astype(int)
    for i in indices:
        vector[i] = np.random.choice((-1.0,1.0))
    return vector




if __name__ == '__main__':
    percent_non_zero = 0.2
    non_zeros = [len(np.nonzero(sparse_vector(1000, percent_non_zero))[0])
                 for _ in range(1000)]
    avg_percent_non_zero = ((sum(non_zeros) / len(non_zeros)) / 1000)
    assert avg_percent_non_zero - percent_non_zero < .001
