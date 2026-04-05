

import torchdata.datapipes as dp
import random
import freerec
from freerec.data.tags import MATCHING, NEXTITEM, X, XX
from freerec.data.postprocessing import ValidSampler


@dp.functional_datapipe("valid_shuffled_sampling_")
class ValidShuffleSampler(ValidSampler):

    def __init__(self, source, ranking = 'full', num_negatives = ...):
        super().__init__(source, ranking, num_negatives)

        self.XSeq = self.ISeq.fork(X)
        self.XXSeq = self.ISeq.fork(XX)

    def _nextitem_from_pool(self):
        for row in self.source:
            user = row[self.User]
            seen = self.seenItems[user]
            for k, positive in enumerate(self.unseenItems[user]):
                seq = self.seenItems[user] + self.unseenItems[user][:k]
                unseen = (positive,) + self._sample_neg(user, k, positive, seen)
                pshuffled = list(seq[:-1])
                random.shuffle(pshuffled)
                pshuffled = tuple(pshuffled + [seq[-1]])
                shuffled = list(seq)
                random.shuffle(shuffled)
                yield {
                    self.User: user, self.ISeq: seq, 
                    self.XSeq: pshuffled, self.XXSeq: shuffled,
                    self.IUnseen: unseen, self.ISeen: seen
                }

    def _nextitem_from_full(self):
        for row in self.source:
            user = row[self.User]
            seen = self.seenItems[user]
            for k, positive in enumerate(self.unseenItems[user]):
                seq = self.seenItems[user] + self.unseenItems[user][:k]
                unseen = (positive,)
                pshuffled = list(seq[:-1])
                random.shuffle(pshuffled)
                pshuffled = tuple(pshuffled + [seq[-1]])
                shuffled = list(seq)
                random.shuffle(shuffled)
                yield {
                    self.User: user, self.ISeq: seq, 
                    self.XSeq: pshuffled, self.XXSeq: tuple(shuffled),
                    self.IUnseen: unseen, self.ISeen: seen
                }

    def __iter__(self):
        if self.dataset.TASK is MATCHING:
            if self.sampling_neg:
                yield from self._matching_from_pool()
            else:
                yield from self._matching_from_full()
        elif self.dataset.TASK is NEXTITEM:
            if self.sampling_neg:
                yield from self._nextitem_from_pool()
            else:
                yield from self._nextitem_from_full()


@dp.functional_datapipe("test_shuffled_sampling_")
class TestShuffleSampler(ValidShuffleSampler):

    @freerec.utils.timemeter
    def prepare(self):
        seenItems = [[] for _ in range(self.User.count)]
        unseenItems = [[] for _ in range(self.User.count)]

        self.listmap(
            lambda row: seenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.train().to_seqs()
        )

        self.listmap(
            lambda row: seenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.valid().to_seqs()
        )

        self.listmap(
            lambda row: unseenItems[row[self.User]].extend(row[self.ISeq]),
            self.dataset.test().to_seqs()
        )

        self.seenItems = seenItems
        self.unseenItems = unseenItems
        self.negItems = dict()
