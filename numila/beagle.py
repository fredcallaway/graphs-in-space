# Python code to implement holographic wordform representations, from: 
# Cox, G., Kachergis, G., Recchia, G., & Jones, M. N. (under review). Towards a scalable holographic word-form representation. Behavior Research Methods. 

import numpy

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '_', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
casedAlphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def hammingSim(a, b):
    '''
    Computes the normalized Hamming similarity between binary vectors a and b.
    '''
    h = float(numpy.sum((a > 0) * (b > 0))) / float(numpy.sum((a > 0) + (b > 0)))
    return h

def normalize(a):
    '''
    Normalize a vector to length 1.
    '''
    return a / numpy.sum(a**2.0)**0.5

def maj(args, p = .5):
    '''
    The majority-rule operation for binary vectors.
    '''
    if len(args) == 0:
        raise ArgumentError('Need something to work with!')
    if len(args) == 1:
        argSum = args[0]
    else:
        argSum = reduce(lambda a,b: a+b, args)
        argSum[argSum == -2*p + 1] = (numpy.roll(args[0], 1) * numpy.roll(args[len(args)-1], 1))[argSum == -2*p + 1]
    
    argSum[argSum < -2*p + 1] = -1.0
    argSum[argSum > -2*p + 1] = 1.0
    
    return argSum
    
def xor(a, b):
    '''
    The X-OR operation for binary (-1 or 1) vectors.
    '''
    return -(a * b)

def entropy(p):
    '''
    Compute the entropy of a vector p of non-negative numbers (normalized to sum
    to 1 and thereby be probabilities).
    '''
    p /= numpy.sum(p)
    return -numpy.dot(p[p > 0], numpy.log(p[p > 0]))

def cconv(a, b):
    '''
    Computes the circular convolution of the (real-valued) vectors a and b.
    '''
    return numpy.fft.ifft(numpy.fft.fft(a) * numpy.fft.fft(b)).real

def ccorr(a, b):
    '''
    Computes the circular correlation (inverse convolution) of the real-valued
    vector a with b.
    '''
    return cconv(numpy.roll(a[::-1], 1), b)

def convpow(a, p):
    '''
    Computes the convolutive power of the real-valued vector a, to the
    (real-valued) power p.
    '''
    return numpy.fft.ifft(numpy.fft.fft(a)**p).real

def cosine(a,b):
    '''
    Computes the cosine of the angle between the vectors a and b.
    '''
    sumSqA = numpy.sum(a**2.0)
    sumSqB = numpy.sum(b**2.0)
    if sumSqA == 0.0 or sumSqB == 0.0: return 0.0
    return numpy.dot(a,b) * (sumSqA * sumSqB)**-0.5

def euclidean_distance(a,b):
    '''
    Return the Euclidean distance between vectors a and b.
    '''
    return numpy.sum((a - b)**2.0)**0.5

def dl(str1, str2):
    '''
    Computes the Damerau-Levenshtein distance between the two given strings.
    '''
    f = numpy.zeros((len(str1) + 1,len(str2) + 1), dtype='int')
    cost = 0
    
    for i in range(1, f.shape[0]): f[i][0] = i
    for j in range(1, f.shape[1]): f[0][j] = j
    
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]: cost = 0
            else: cost = 1
            f[i][j] = min(f[i - 1][j - 1] + cost, f[i - 1][j] + 1, f[i][j - 1] + 1)
            if i > 1 and j > 1 and str1[i - 1] == str2[j - 2] and str1[i - 2] == str2[j - 1]:
                f[i][j] = min(f[i][j], f[i - 2, j - 2] + cost)
    
    return f[f.shape[0]-1][f.shape[1]-1]

def ld(str1, str2):
    '''
    Computes the Levenshtein distance between the two given strings.
    '''
    f = numpy.zeros((len(str1) + 1,len(str2) + 1), dtype='int')
    cost = 0
    
    for i in range(1, f.shape[0]): f[i][0] = i
    for j in range(1, f.shape[1]): f[0][j] = j
    
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]: cost = 0
            else: cost = 1
            f[i][j] = min(f[i - 1][j - 1] + cost, f[i - 1][j] + 1, f[i][j - 1] + 1)
            #if i > 1 and j > 1 and str1[i - 1] == str2[j - 2] and str1[i - 2] == str2[j - 1]:
            #    f[i][j] = min(f[i][j], f[i - 2, j - 2] + cost)
    
    return f[f.shape[0]-1][f.shape[1]-1]

def ordConv(a, b, p1, p2):
    '''
    Performs ordered (non-commutative) circular convolution on the vectors a and
    b by first permuting them according to the index vectors p1 and p2.
    '''
    return cconv(a[p1], b[p2])

def convBind(p1, p2, l):
    '''
    Given a list of vectors, iteratively convolves them into a single vector
    (i.e., "binds" them together).
    '''
    return reduce(lambda a,b: normalize(ordConv(a, b, p1, p2)), l)

def addBind(p1, p2, l):
    '''
    Given a list of vectors, binds them together by iteratively convolving them
    with place vectors and adding them up.
    '''
    return normalize(reduce(lambda a,b: cconv(a, p1) + cconv(b, p2), l))

def bscBind(p1, p2, l, p = .5):
    '''
    Given a list of binary (-1 or 1) vectors, binds them together by computing
    their XOR with place vectors and adding them up.  This is done in a pairwise
    fashion, i.e., ((a+b)+(c+d))+((e+f)+(g+h)), to balance the added noise across
    all components equally.
    '''
    tempElements = [elem for elem in l]
    
    while len(tempElements) > 1:
        newElements = []
        for i in xrange(0, len(tempElements)-1, 2):
            if i+2 == len(tempElements)-1:
                newElements.append(maj([xor(maj([xor(tempElements[i], p1), xor(tempElements[i+1], p2)], p), p1), xor(tempElements[i+2], p2)], p))
            else:
                newElements.append(maj([xor(tempElements[i], p1), xor(tempElements[i+1], p2)], p))
        tempElements = [elem for elem in newElements]
    
    return tempElements[0]

def getOpenNGrams(seg, scale, spaces):
    '''
    Returns a list of the open n-grams of the string "seg", with sizes specified
    by "scale", which should be a list of positive integers in ascending order.
    "Spaces" indicates whether a space character should be used to mark gaps in
    non-contiguous n-grams.
    '''
    ngrams = []
    
    for size in scale:
        if size > len(seg): break
        
        for i in xrange(len(seg)):
            if i+size > len(seg): break
            ngrams.append(seg[i:i+size])
            if i+size == len(seg): continue
            for b in xrange(1, size):
                for e in xrange(1, len(seg)-i-size+1):
                    ngrams.append(seg[i:i+b]+('_' if spaces else '')+seg[i+b+e:i+e+size])
    
    return ngrams

def getTRNGrams(seg, scale, spaces):
    '''
    Returns a list of n-grams from the string "seg", according to the "terminal
    relative" (TR) encoding procedure, i.e., any internal n-gram gets included
    both by itself and as part of a non-contiguous n-gram with n-grams at either
    end of "seg".  "Scale" is a list of ascending integers reflecting the sizes
    of the n-gram chunks.  "Spaces" indicates whether a space character should
    be used to mark gaps in non-contiguous n-grams.
    '''
    ngrams = []
    
    for size in scale:
        if size > len(seg): break
        
        for i in xrange(len(seg)-size+1):
            if '_' in seg[i:i+size]: continue
            
            ngrams.append(seg[i:i+size])
            if seg[0] != '_':
                for fsize in scale:
                    if '_' in seg[:fsize] or fsize > i or (i==fsize and fsize+size in scale): break
                    ngrams.append(seg[:fsize] + ('_' if spaces and i > fsize else '') + seg[i:i+size])
            if seg[-1] != '_':
                for fsize in scale:
                    if '_' in seg[-fsize:] or i + size > len(seg)-fsize or (i+size==len(seg)-fsize and fsize+size in scale): break
                    ngrams.append(seg[i:i+size] + ('_' if spaces and i+size < len(seg)-fsize else '') + seg[-fsize:])
    
    return ngrams

class HoloWordRep:
    '''
    Spawns objects capable of generating holographic word-form representations
    in both the visual and auditory modalities. Auditory representations depend
    on the given word being present in the CMU Pronouncing Dictionary.
    '''
    
    def __init__(self, d = 1024, ngramType = 'tr', vis_scale=[1,2], spaces = True, bindOp = 'convolution'):
        '''
        Creates a new holographic word-form representation generator.
        d - dimensionality of the representations (typically quite large)
        ngramType - how words are to be decomposed into n-grams; either 'tr'
            (terminal relative) or 'open'.
        vis_scale - a list of the size of successive chunks in visual word
            representations (in # of letters)
        spaces - whether or not to use a space character to mark gaps in non-
            contiguous n-grams
        bindOp - one of "convolution", "addition", or "bsc" (or just the initial
            letter of those) indicating the method by which n-grams should be
            bound together
        '''
        
        self.d = d
        self.vis_scale = sorted(vis_scale)
        self.spaces = spaces
        
        # Set the n-gram extraction method @FRED
        if ngramType.lower().startswith('t'):
            self.getNGrams = lambda s: getTRNGrams(s, self.vis_scale, self.spaces)
        elif ngramType.lower().startswith('o'):
            self.getNGrams = lambda s: getOpenNGrams(s, self.vis_scale, self.spaces)
        else:
            raise ArgumentError('Invalid n-gram type!')
        
        bindOp = bindOp.lower()
        
        if bindOp.startswith('c') or bindOp.startswith('a'):
            # Create random vectors representing individual letters, the "atoms"
            # of the word representations.
            self.letters = dict(zip(alphabet, [normalize(numpy.random.randn(d) * d**-0.5) for letter in alphabet]))
            
            self.chunk = lambda l: normalize(reduce(lambda a,b: a+b, l))    #Chunking is done by superposition (vector addition)
            
            if bindOp.startswith('c'):
                # Permutation operators that scramble the vectors in a
                # convolution operation; this makes the operation non-commutative and
                # thus allows it to encode order.
                self.place1 = numpy.random.permutation(d)
                self.place2 = numpy.random.permutation(d)
                
                self.invplace1 = numpy.zeros((d), dtype='int')
                self.invplace2 = numpy.zeros((d), dtype='int')
                
                for i in xrange(d):
                    self.invplace1[self.place1[i]] = i
                    self.invplace2[self.place2[i]] = i
                
                self.bind = lambda l: convBind(self.place1, self.place2, l)
            else:
                # Here, to make binding non-commutative, each operand is first
                # convolved with a place vector, and the two results are added.
                # (Results in "fuzzy position" coding)
                self.place1 = normalize(numpy.random.randn(d) * d**-0.5)
                self.place2 = normalize(numpy.random.randn(d) * d**-0.5)
                
                self.bind = lambda l: addBind(self.place1, self.place2, l)
        elif bindOp.startswith('b'):
            # Create random vectors representing individual letters, the "atoms"
            # of the word representations.  These are binary vectors taking the
            # values -1 or 1.
            self.letters = dict(zip(alphabet, [(numpy.random.rand(d) < 0.5) * 2.0 - 1.0 for letter in alphabet]))
            
            self.place1 = (numpy.random.rand(d) < 0.5) * 2.0 - 1.0
            self.place2 = (numpy.random.rand(d) < 0.5) * 2.0 - 1.0
            
            self.chunk = lambda l: maj(l, 0.5)
            self.bind = lambda l: bscBind(self.place1, self.place2, l, 0.5)
        else:
            raise ArgumentError('Invalid binding operator specification!')
    
    def make_rep(self, word):
        '''
        Returns a holographic representation of the given word. The word may be
        given as just a string, or as a list of lists, where each sublist is a
        way of segmenting the word, e.g., "homework" vs. [['homework'], ['home',
        'work']].
        '''
        
        if type(word) == type(''): word = [ [word] ]
        
        # Create a visual word form representation based on the letters present
        # in the word.
        formReps = []
        for form in word:
            segReps = []
            
            for seg in form:
                seg = seg.strip().lower()
                ngrams = self.getNGrams(seg)
                
                segRep = self.chunk([self.bind([self.letters[l] for l in ngram]) for ngram in ngrams])
                segReps.append(segRep)
            
            if len(segReps) > 1:
                segChunks = []
                for size in xrange(1,len(segReps)):
                    for i in xrange(len(segReps)-size+1):
                        segChunks.append(self.bind(segReps[i:(i+size)]))
            else:
                segChunks = [segReps[0]]
            
            formReps.append(self.chunk(segChunks))
        
        return self.chunk(formReps)

def EvalConstraints(n = 1, **repArgs):
    '''
    Evaluates the constraints from Hannagan, et al., (in press). These are
    based on the relative amount of facilitation from prime to target, which
    should correlate with greater similarity of the underlying word-form
    representations.  Similarities are averaged over "n" simulations.  Returns
    the estimated similarites, their standard deviations, and a vector of
    booleans reflecting whether or not that constraint was satisfied.
    "**repArgs" arguments get passed to the word-form generator.
    '''
    pairs = [('abcde', 'abcde'), ('abde', 'abcde'), ('abccde', 'abcde'), ('abcfde', 'abcde'), ('abfge', 'abcde'), ('afcde', 'abcde'), ('abgdef', 'abcdef'), ('abgdhf', 'abcdef'), ('fbcde', 'abcde'), ('abfde', 'abcde'), ('abcdf', 'abcde'), ('abdce', 'abcde'), ('badcfehg', 'abcdefgh'), ('abedcf', 'abcdef'), ('acfde', 'abcde'), ('abcde', 'abcdefg'), ('cdefg', 'abcdefg'), ('acdeg', 'abcdefg'), ('abcbef', 'abcbdef'), ('abcdef', 'abcbdef')]
    sim = numpy.zeros((len(pairs)))
    simSq = numpy.zeros((len(pairs)))
    
    for i in xrange(n):
        h = HoloWordRep(**repArgs)
        newData = numpy.array([cosine(h.make_rep(prime), h.make_rep(target)) for prime, target in pairs])
        sim += newData
        simSq += newData**2.0
    
    sim /= float(n)
    sd = ((simSq - n * sim**2.0) / float(n - 1))**0.5
    
    trends = [sim[0] == numpy.max(sim), sim[1] < sim[0], sim[2] < sim[0], sim[3] < sim[0], sim[4] < sim[0], sim[5] < sim[0], sim[6] < sim[0], sim[7] < sim[6], sim[8] < sim[9], sim[9] < sim[0], sim[10] < sim[9], sim[11] > sim[4], sim[12] == numpy.min(sim), sim[13] < sim[6] and sim[13] > sim[7], sim[14] < sim[5], sim[15] > numpy.min(sim), sim[16] > numpy.min(sim), sim[17] > numpy.min(sim), sim[18] > numpy.min(sim), numpy.abs(sim[19] - sim[18]) == numpy.min(numpy.abs(sim[19] - sim[:19]))]
    # (old last trend) == numpy.min(numpy.abs(numpy.tile(sim, [len(sim),1]) - numpy.transpose(numpy.tile(sim, [len(sim), 1]))) + 1000*numpy.eye(len(sim)))
    
    return sim, sd, trends

def Substitutions(length = 7, numsims = 1, **repArgs):
    '''
    For words of "length" unique letters, computes the similarity resulting
    from replacing a letter at each location.  Similarities (and their standard
    deviations) are computed over "numsims" simulations.  "**repArgs" get passed
    to the representation generator.
    '''
    target = ''.join(map(chr, range(97, 97+min(length, 25))))
    primes = [target[:i]+'z'+target[i+1:] for i in xrange(len(target))]
    sim = numpy.zeros((len(primes)))
    simSq = numpy.zeros((len(primes)))
    
    for n in xrange(numsims):
        h = HoloWordRep(**repArgs)
        
        targetRep = h.make_rep(target)
        newData = numpy.array([cosine(h.make_rep(prime), targetRep) for prime in primes])
        sim += newData
        simSq += newData**2.0
    
    sim /= float(numsims)
    sd = ((simSq - numsims * sim**2.0) / float(numsims - 1))**0.5
    
    return sim, sd

def MakeReps(words, segment = False, filename = None, **repArgs):
    '''
    Makes representations for the words in "words", which can be a list of strings
    or a string giving a filename.
    segment - Whether or not to find ways of segmenting each word into other,
        shorter words (e.g., "homework" = "home" + "work")
    filename - A filename to which the representations can be written in CSV format
    repArgs - named arguments to be passed to HoloWordRep
    '''
    if type(words) == type(''):
        word_list = []
        FIN = open(words, 'r')
        for line in FIN:
            line = line.strip().lower()
            line = line.split()
            word_list.append(line[0])
        FIN.close()
        words = word_list
        
    h = HoloWordRep(**repArgs)
    
    if segment:
        forms = SegmentWords(words)
        reps = numpy.array([h.make_rep(form) for form in forms])
    else:
        reps = numpy.array([h.make_rep(word) for word in words])            # Make representations for all words in the list
    
    if filename != None:
        FOUT = open(filename, 'w')
        for i, word in enumerate(words):
            FOUT.write(word + ',' + ','.join([str(x) for x in reps[i]]) + '\n')
        FOUT.close()
    
    return reps

def SimMatrix(words, numsims=1, numToCompare = None, segment = False, **repArgs):
    '''
    Returns a matrix containing cosine similarities between word-form representations
    of the words given in "words" (either a list of strings or a filename).  Similarities
    are averaged over "numsims" simulations.
    numToCompare - If specified, the matrix only contains the top "numToCompare"
        similarity values
    segment - Whether or not to find ways of segmenting each word into other,
        shorter words (e.g., "homework" = "home" + "work")
    repArgs - named arguments to be passed to HoloWordRep
    '''
    if type(words) == type(''):
        FIN = open(words, 'r')
        words = [line.strip() for line in FIN]
        FIN.close()
    
    if numToCompare == None:
        sim = numpy.zeros((len(words), len(words)))
    else:
        sim = numpy.zeros((len(words), numToCompare))

    for n in xrange(numsims):
        reps = MakeReps(words, segment, None, **repArgs)
        
        reps /= numpy.reshape(numpy.sum(reps**2.0, 1)**0.5, (len(reps), 1)) # Normalize the representations
        
        if numToCompare == None:
            sim += numpy.dot(reps, numpy.transpose(reps))    # Computes the dot product of all representations
        else:
            for i in xrange(len(words)):
                temp_sims = numpy.array([numpy.dot(reps[i], reps[j]) for j in xrange(len(words))])
                sim[i] += numpy.sort(temp_sims)[(-2):(-2-numToCompare):-1]
    
    return sim / float(numsims)

def strCombos(toAdd, start=''):
    '''
    Generates all combinations of the elements in the list "toAdd".
    '''
    if len(toAdd) == 1: return [start+toAdd[0]]
    combos = []
    for i, item in enumerate(toAdd):
        combos.append(start + item)
        combos.extend(strCombos(toAdd[:i]+toAdd[(i+1):], start+item))
    return combos

def SegmentWords(words):
    '''
    Returns segmented forms of each of the words in the list "words".
    '''
    forms = [[] for word in words]
    
    for w, word in enumerate(words):
        forms[w].extend(segmentWord(word, words[:w] + words[(w+1):]))
    
    return forms
        
def segmentWord(word, lexicon):
    '''
    Searches the lexicon (a list of strings) for words that form part of the given
    word, and returns a list of all such possible decompositions of the word.
    '''
    segs = [[word]]
    if len(word) < 3: return segs
    
    for i in xrange(len(word)-2):
        for j in xrange(2, len(word)-i):
            if word[i:i+j] in lexicon:
                suffixes = segmentWord(word[i+j:], lexicon)
                for suffix in suffixes:
                    if i > 0: toAdd = [word[:i], word[i:i+j]]
                    else: toAdd = [word[i:i+j]]
                    toAdd.extend(suffix)
                    segs.append(toAdd)
                #if i > 0: toAdd = [word[:i], word[i:i+j]]
                #else: toAdd = [word[i:i+j]]
                #toAdd.append(word[i+j:])
                #segs.append(toAdd)
    
    i = 0
    while i < len(segs):
        if segs[i] in segs[(i+1):]:
            del segs[i]
        else:
            i += 1
    
    return segs

def TopNSim(n = 100, words = 'elp_trimmed_words-freq.txt', numsims=1, filename=None, segment=False, thresholds = numpy.arange(0,1,.05), **repArgs):
    freq = []
    
    if type(words) == type(0):   # Then, words are the power set of the first 'words' letters
        words = strCombos([letter for letter in alphabet[:min(words, 26)]])
    if type(words) == type(''):
        word_list = []
        FIN = open(words, 'r')
        for line in FIN:
            line = line.strip().lower()
            line = line.split()
            word_list.append(line[0])
            if len(line) > 1:
                freq.append(float(line[1]))
            else:
                freq.append(1.0)
        FIN.close()
        if filename == None: filename = words+'sims.csv'
        words = word_list
    
    if len(freq) != len(words): freq = numpy.ones((len(words)))
    else: freq = numpy.array(freq)
    
    if n >= len(words): n = len(words) - 1
    
    aboveThreshold = numpy.zeros((len(words), len(thresholds)))
    numAboveThreshold = numpy.zeros((len(words), len(thresholds)))
    aboveThresholdVar = numpy.zeros((len(words), len(thresholds)))
    allMean = numpy.zeros((len(words)))
    allVar = numpy.zeros((len(words)))
    #allEntropy = numpy.zeros((len(words)))
    aboveThresholdF = numpy.zeros((len(words), len(thresholds)))
    allMeanF = numpy.zeros((len(words)))
    sorted_sim = numpy.zeros((len(words), n))
    sorted_freq = numpy.zeros((len(words), n))
    
    if len(words) < 1:
        closest_words = []
        sim = SimMatrix(words, numsims, segment=segment, **repArgs)
        sim -= 2.0*numpy.eye(len(sim))*sim
        for i in xrange(len(words)):
            notI = range(i)+range(i+1,len(words))
            for t, threshold in enumerate(thresholds):
                aboveThreshold[i][t] = numpy.mean(sim[i, sim[i] > threshold])
                numAboveThreshold[i][t] = len(sim[i][sim[i] > threshold])
                aboveThresholdVar[i][t] = numpy.var(sim[i, sim[i] > threshold])
                aboveThresholdF[i][t] = numpy.dot(sim[i, sim[i] > threshold], freq[sim[i] > threshold] / numpy.sum(freq[sim[i]>threshold]))
            allMean[i] = numpy.mean(sim[i][notI])
            allVar[i] = numpy.var(sim[i][notI])
            allMeanF[i] = numpy.dot(sim[i][notI], freq[notI]) / numpy.sum(freq[notI])
            topSims = numpy.argsort(sim[i])[(len(sim[i])-1):(len(sim[i])-1-n):-1]
            sorted_sim[i] = sim[i][topSims]
            sorted_freq[i] = freq[topSims]
            closest_words.append(words[topSims[0]])
    else:
        FOUT = open(filename, 'w')
        FOUT.write('Word,ClosestWord,' + ','.join(['sim'+str(i) for i in xrange(sorted_sim.shape[1])]) + ',' + ','.join(['freq'+str(i) for i in xrange(sorted_sim.shape[1])]) + ',AllMean,AllVar,AllMeanF,' + ','.join(['NumAbove'+str(t) for t in thresholds]) + ',' + ','.join(['Above'+str(t) for t in thresholds]) + ',' + ','.join(['Above'+str(t)+'Var' for t in thresholds]) + ',' + ','.join(['Above'+str(t)+'F' for t in thresholds]) + '\n')
        reps = MakeReps(words, segment, None, **repArgs)
        aboveThreshold = numpy.zeros((len(thresholds)))
        aboveThresholdVar = numpy.zeros((len(thresholds)))
        numAboveThreshold = numpy.zeros((len(thresholds)))
        aboveThresholdF = numpy.zeros((len(thresholds)))
        eBins = numpy.arange(-1.0, 1.05, 0.05)
        for i in xrange(len(words)):
            notI = range(i)+range(i+1,len(words))
            temp_sims = numpy.inner(reps[i], reps)
            temp_sims[i] = -1.0
            for t, threshold in enumerate(thresholds):
                aboveThreshold[t] = numpy.mean(temp_sims[temp_sims > threshold])
                numAboveThreshold[t] = len(temp_sims[temp_sims > threshold])
                aboveThresholdVar[t] = numpy.var(temp_sims[temp_sims > threshold])
                aboveThresholdF[t] = numpy.dot(temp_sims[temp_sims > threshold], freq[temp_sims > threshold]) / numpy.sum(freq[temp_sims > threshold])
            allMean = numpy.mean(temp_sims[notI])
            #allEntropy = entropy(temp_sims[notI]+numpy.min(temp_sims[notI]))
            #allEntropyF = entropy((temp_sims[notI]+numpy.min(temp_sims[notI])) * freq[notI])
            allVar = numpy.var(temp_sims[notI])
            allMeanF = numpy.dot(temp_sims[notI], freq[notI]) / numpy.sum(freq[notI])
            topSims = numpy.argsort(temp_sims)[(len(temp_sims)-1):(len(temp_sims)-1-n):-1]
            sorted_sim = temp_sims[topSims]
            sorted_freq = freq[topSims]
            
            FOUT.write(words[i]+','+words[topSims[0]]+','+ ','.join([str(s) for s in sorted_sim]) + ',' + ','.join([str(s) for s in sorted_freq])+','+str(allMean)+','+str(allVar)+','+str(allMeanF)+','+','.join([str(numAboveThreshold[t]) for t in xrange(len(thresholds))])+','+','.join([str(aboveThreshold[t]) for t in xrange(len(thresholds))])+','+','.join([str(aboveThresholdVar[t]) for t in xrange(len(thresholds))])+','+','.join([str(aboveThresholdF[t]) for t in xrange(len(thresholds))])+'\n')
            
            if i % 100 == 0: print i, words[i]
        FOUT.close()
        return
        
#        sorted_sim /= float(numsims)
#        sorted_freq /= float(numsims)
#        closest_words = ['NA']*len(sorted_sim)
#        aboveThreshold /= float(numsims)
#        allMean /= float(numsims)
#        aboveThresholdF /= float(numsims)
#        allMeanF /= float(numsims)
    
    if filename != None:
        FOUT = open(filename, 'w')
        FOUT.write('Word,ClosestWord,' + ','.join(['sim'+str(i) for i in xrange(sorted_sim.shape[1])]) + ',' + ','.join(['freq'+str(i) for i in xrange(sorted_sim.shape[1])]) + ',AllMean,AllMeanF,' + ','.join(['Above'+str(t) for t in thresholds]) + ',' + ','.join(['Above'+str(t)+'F' for t in thresholds]) + '\n')
        for i, word in enumerate(words):
            FOUT.write(word+','+closest_words[i]+','+ ','.join([str(s) for s in sorted_sim[i]]) + ',' + ','.join([str(s) for s in sorted_freq[i]])+','+str(allMean[i])+','+str(allMeanF[i])+','+','.join([str(aboveThreshold[i][t]) for t in xrange(len(thresholds))])+','+','.join([str(aboveThresholdF[i][t]) for t in xrange(len(thresholds))])+'\n')
        FOUT.close()
    
    return words, closest_words, sorted_sim, aboveThreshold, allMean, aboveThresholdF, allMeanF