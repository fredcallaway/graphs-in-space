from production import *
from introspection import *

def main():
    quick_test(GRAPH='probgraph', CHUNK_THRESHOLD=0.1, LEARNING_RATE=1, train_len=100)

if __name__ == '__main__':
    main()