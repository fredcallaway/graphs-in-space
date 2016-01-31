from production import *
from introspection import *
import utils

LOG = utils.get_logger(__name__, stream='INFO', file='WARNING')

def main():
    quick_test(GRAPH='probgraph',
               CHUNK_THRESHOLD=0.3,
               LEARNING_RATE=1,
               BIND=False,
               train_len=5000)




if __name__ == '__main__':
    try:
        main()
    except:
        import traceback
        tb = traceback.format_exc()
        LOG.critical(tb)
        exit(1)

