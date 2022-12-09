############################################################
# CMPSC 442: Homework 6
############################################################

student_name = "Varun Bhatnagar"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
from collections import defaultdict
from math import log


############################################################
# Section 1: Hidden Markov Models
############################################################

def load_corpus(path):
    with open(path, "r") as f:
        s = [[tuple(y.split('=')) for y in x.strip().split(' ')] for x in f.readlines()]
    return s


class Tagger(object):

    def __init__(self, sentences):
        self.possible_states = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'DET', 'ADP', 'NUM', 'CONJ', 'PRT', '.', 'X']
        possible_states = self.possible_states
        tfreq = {S: {S2: 0 for S2 in possible_states} for S in possible_states}
        efreq = {S: defaultdict(int) for S in possible_states}
        ifreq = {S: 0 for S in possible_states}
        words = {}
        priorfreq = {S: 0 for S in possible_states}
        for sent in sentences:
            ifreq[sent[0][1]] += 1
            for i in range(0, len(sent) - 1):
                tfreq[sent[i][1]][sent[i + 1][1]] += 1
                efreq[sent[i][1]][sent[i][0]] += 1
                priorfreq[sent[i][1]] += 1
                words[sent[i][0]] = 1
            efreq[sent[-1][1]][sent[-1][0]] += 1
            priorfreq[sent[-1][1]] += 1
            words[sent[-1][0]] = 1
        self.tfreq = tfreq
        self.efreq = efreq
        self.ifreq = ifreq
        self.priorfreq = priorfreq
        self.words = words
        di = len(ifreq.keys())
        Ni = sum(ifreq.values())
        ed = len(words.keys())
        # edS = {S: ed for S in possible_states}
        tdS = {S: 0 for S in possible_states}
        eNS = {S: 0 for S in possible_states}
        tNS = {S: 0 for S in possible_states}
        for S in possible_states:
            # edS[S] = len(efreq[S].keys())
            eNS[S] = sum(efreq[S].values())
            tdS[S] = len(tfreq[S].keys())
            tNS[S] = sum(tfreq[S].values())
        # self.edS = edS
        self.tdS = tdS
        self.eNS = eNS
        self.tNS = tNS
        self.di = di
        self.Ni = Ni
        self.ed = ed

    def leprob(self, o, S):
        c = 0 if o not in self.efreq[S] else self.efreq[S][o]
        return log(c + .1, 2) - log(self.eNS[S] + self.ed * .1, 2)

    def ltprob(self, S1, S2):
        c = 0 if S1 not in self.tfreq[S2] else self.tfreq[S2][S1]
        return log(c + .1, 2) - log(self.tNS[S2] + self.tdS[S2] * .1, 2)

    def liprob(self, S):
        return log(self.ifreq[S] + .1, 2) - log(self.Ni + self.di * .1, 2)

    def argmax(self, d):
        return max(d, key=d.get)

    def most_probable_tags(self, tokens):
        return [self.argmax({S: self.leprob(token, S) for S in self.possible_states}) for token in tokens]

    def viterbi_tags(self, tokens):
        M = len(tokens)
        N = len(self.possible_states)
        trellis = [[0 for _ in range(N)] for _ in range(M)]
        backpointers = {(0, S): None for S in self.possible_states}
        trellis[0] = [self.liprob(S) + self.leprob(tokens[0], S) for S in self.possible_states]
        for t in range(1, M):
            token = tokens[t]
            for j in range(N):
                next_state = self.possible_states[j]
                prev_state = self.possible_states[0]
                best = (trellis[t - 1][0] + self.ltprob(next_state, prev_state), prev_state)
                for i in range(1,N):
                    prev_state = self.possible_states[i]
                    cur_prob = trellis[t - 1][i] + self.ltprob(next_state, prev_state)
                    if cur_prob > best[0]:
                        best = (cur_prob, prev_state)
                trellis[t][j] = best[0] + self.leprob(token, next_state)
                backpointers[t, next_state] = best[1]
        best = (trellis[M - 1][0], self.possible_states[0])
        for i in range(1,N):
            cur_prob = trellis[M - 1][i]
            if cur_prob > best[0]:
                best = (cur_prob, self.possible_states[i])
        best_state = best[1]
        optimal_path = [best_state]
        for t in range(M - 1, 0, -1):
            cur_state = backpointers[t, best_state]
            optimal_path.append(cur_state)
            best_state = cur_state
        optimal_path.reverse()
        return optimal_path

############################################################
# Section 2: Feedback
############################################################

feedback_question_1 = """
24 hours
"""

feedback_question_2 = """
I think their needs to be better instructions for the entire project. 
"""

feedback_question_3 = """
I would have provided more template code and better timings data.
"""
