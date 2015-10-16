class CounterMatrix(object):
    """A two dimensional sparse matrix of counts with default 0 values."""
    def __init__(self, tokens, smooth=False):
        super(CounterMatrix, self).__init__()
        self.smooth = smooth

        self._dict = defaultdict(Counter)
        for i in range(len(tokens) - 1):
            self._dict[tokens[i]][tokens[i+1]] += 1

    def __len__(self):
        return len(self._dict)

    @cached_property()
    def count_counts(self):
        """Value counts for each row in the matrix.

        value_counts['foo'][c] is the number of elements in the 'foo' row
        that are c. For example, if using the CounterMatrix to represent
        a co-occurence matrix for a bigram model, it would be the number of
        bigrams beginning with 'foo' that occurred two times."""
        count_counts = defaultdict(Counter)
        for token, followers in self._dict.items():
            for f, count in followers.items():
                count_counts[token][count] += 1
            count_counts[token][0] = len(self._dict) - sum(count_counts[token].values())
        return count_counts

    @cached_property()
    def good_turing_mapping(self, threshold=5) -> Dict[int, float]:
        """A dictionary mapping counts to good_turing smoothed counts."""
        total_count_counts = sum(self.count_counts.values(), Counter())
        # total_count_counts[2] is number of bigrams that occurred twice

        def good_turing(c): 
            return (c+1) * (total_count_counts[c+1]) / total_count_counts.get(c, 1)
        gtm = {c: good_turing(c) for c in range(threshold)}
        return {k: v for k, v in gtm.items() if v > 0}  # can't have 0 counts

    @cached_property()
    def unigram_distribution(self):
        """The probability of each token occurring irrespective of context."""
        counts = {token: sum(follower.values()) 
                  for token, follower in self._dict.items()}
        return Distribution(counts)

    @lru_cache(maxsize=100000)  # caches 100,000 most recent results
    def distribution(self, token):
        """Returns next-token probability distribution for the given token.

        distributions('the').sample() gives words likely to occur after 'the'"""
        if token not in self._dict:
            token = 'UNKNOWN_TOKEN'  # yes, yes, bad coupling I know...
        if self.smooth:
            smoothing_dict = self.good_turing_mapping
            return Distribution(self._dict[token], smoothing_dict,
                                self.count_counts[token])
        else:
            if self._dict[token]:
                return Distribution(self._dict[token])
            else:
                # no information -> use unigram
                return self.unigram_distribution


class Distribution(object):
    """A statistical distribution based on a dictionary of counts."""
    def __init__(self, counter, smoothing_dict={}, count_counts=None):
        assert counter
        self.counter = counter
        self.smoothing_dict = smoothing_dict

        # While finding the total, we also track each
        # intermediate total to make sampling faster.
        self._acc_totals = list(itertools.accumulate(counter.values()))
        self.total = self._acc_totals[-1]

        # Smoothing only applies to surprisal, not sampling so we maintain
        # a separate total that accounts for the smoothed counts
        if smoothing_dict:
            if not count_counts:
                raise ValueError('Must supply count_counts argument to use smoothing.')
            self.smooth_total = sum(smoothing_dict.get(count, count) * N_count 
                                    for count, N_count in count_counts.items())
        else:
            self.smooth_total = None

    def sample(self):
        """Returns an element from the distribution.

        Based on ideas from the following article:
        http://eli.thegreenplace.net/2010/01/22/weighted-random-generation-in-python"""
        rand = random.random() * self.total

        # Perform a binary search for index of highest number below rand.
        # index will thus be chosen with probability =
        # (self._acc_totals[i] - self._acc_totals[i-1]) / self.total
        index = bisect.bisect_right(self._acc_totals, rand)
        tokens = list(self.counter.keys())
        return tokens[index]

    def probability(self, item):
        """The probability of an item being sampled."""
        count = self.counter.get(item, 0)
        if self.smoothing_dict:
            smooth_count = self.smoothing_dict.get(count, count)
            assert smooth_count > 0
            return smooth_count / self.smooth_total
        else:
            return count / self.total
    
    def surprisal(self, item):
        """The negative log probability of an item being sampled."""
        return - math.log(self.probability(item))