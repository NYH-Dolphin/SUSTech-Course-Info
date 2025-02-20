# Adapted from https://github.com/Andras7/word2vec-pytorch/blob/master/word2vec/data_reader.py

import numpy as np

class CorpusReader:
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, inputFileName, min_count:int = 5, lang="zh") :
        self.negatives = []
        self.discards = []
        self.negpos = 0

        self.word2id = dict()
        self.id2word = dict()
        self.token_count = 0
        self.word_frequency = dict()

        self.lang = lang
        self.inputFileName = inputFileName
        self.read_words(min_count)
        # self.initTableNegatives()
        # self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="utf8"):
            if self.lang == "zh":
                words = list(line.strip())
            else:
                words = line.split()
            if len(words) > 0:
                for word in words:
                    self.token_count += 1
                    word_frequency[word] = word_frequency.get(word, 0) + 1
                    if self.token_count % 1000000 == 0:
                        print("Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in sorted(word_frequency.items(), key=lambda x: x[1], reverse=True):
            if c < min_count: # filter out low frequency words
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1
        print("Total vocabulary: " + str(len(self.word2id)))

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        pow_frequency = np.array(list(self.word_frequency.values())) ** (3/4)
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * CorpusReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size): 
        while True:
            response = self.negatives[self.negpos:self.negpos + size]
            self.negpos = (self.negpos + size) % len(self.negatives)
            if len(response) != size:
                response = np.concatenate((response, self.negatives[0:self.negpos]))
            if target in response: # prevent target word itself from being negative sample
                continue
            return response