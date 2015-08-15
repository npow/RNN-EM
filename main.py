import argparse
import numpy
import time
import sys
import subprocess
import os
import random

from rnn_em import model
from is13.data import load
from is13.metrics.accuracy import conlleval
from is13.utils.tools import shuffle, minibatch, contextwin

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=100, help='Hidden size')
    parser.add_argument('--memory_size', type=int, default=40, help='Memory size')
    parser.add_argument('--emb_size', type=int, default=100, help='Embedding size')
    parser.add_argument('--n_memory_slots', type=int, default=1, help='Memory slots')
    parser.add_argument('--n_epochs', type=int, default=50, help='Num epochs')
    parser.add_argument('--seed', type=int, default=345, help='Seed')
    parser.add_argument('--bs', type=int, default=9, help='Number of backprop through time steps')
    parser.add_argument('--win', type=int, default=7, help='Number of words in context window')
    parser.add_argument('--fold', type=int, default=4, help='Fold number, 0-4')
    parser.add_argument('--lr', type=float, default=0.0627142536696559, help='Learning rate')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose or not')
    parser.add_argument('--decay', type=int, default=0, help='Decay lr or not')
    s = parser.parse_args()

    print '*' * 80
    print s
    folder = os.path.basename(__file__).split('.')[0]
    if not os.path.exists(folder): os.mkdir(folder)

    # load the dataset
    train_set, valid_set, test_set, dic = load.atisfold(s.fold)
    idx2label = dict((k,v) for v,k in dic['labels2idx'].iteritems())
    idx2word  = dict((k,v) for v,k in dic['words2idx'].iteritems())

    train_lex, train_ne, train_y = train_set
    valid_lex, valid_ne, valid_y = valid_set
    test_lex,  test_ne,  test_y  = test_set

    vocsize = len(set(reduce(\
                       lambda x, y: list(x)+list(y),\
                       train_lex+valid_lex+test_lex)))

    nclasses = len(set(reduce(\
                       lambda x, y: list(x)+list(y),\
                       train_y+test_y+valid_y)))
    
    nsentences = len(train_lex)

    # instanciate the model
    numpy.random.seed(s.seed)
    random.seed(s.seed)
    rnn = model(    nh = s.hidden_size,
                    nc = nclasses,
                    ne = vocsize,
                    de = s.emb_size,
                    cs = s.win,
                    memory_size = s.memory_size,
                    n_memory_slots = s.n_memory_slots )

    # train with early stopping on validation set
    best_f1 = -numpy.inf
    s.clr = s.lr
    for e in xrange(s.n_epochs):
        # shuffle
        shuffle([train_lex, train_ne, train_y], s.seed)
        s.ce = e
        tic = time.time()
        for i in xrange(nsentences):
            cwords = contextwin(train_lex[i], s.win)
            words  = map(lambda x: numpy.asarray(x).astype('int32'),\
                         minibatch(cwords, s.bs))
            labels = train_y[i]
            for word_batch , label_last_word in zip(words, labels):
                rnn.train(word_batch, label_last_word, s.clr)
                rnn.normalize()
            if s.verbose:
                print '[learning] epoch %i >> %2.2f%%'%(e,(i+1)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-tic),
                sys.stdout.flush()
            
        # evaluation // back into the real world : idx -> words
        predictions_test = [ map(lambda x: idx2label[x], \
                             rnn.classify(numpy.asarray(contextwin(x, s.win)).astype('int32')))\
                             for x in test_lex ]
        groundtruth_test = [ map(lambda x: idx2label[x], y) for y in test_y ]
        words_test = [ map(lambda x: idx2word[x], w) for w in test_lex]

        predictions_valid = [ map(lambda x: idx2label[x], \
                             rnn.classify(numpy.asarray(contextwin(x, s.win)).astype('int32')))\
                             for x in valid_lex ]
        groundtruth_valid = [ map(lambda x: idx2label[x], y) for y in valid_y ]
        words_valid = [ map(lambda x: idx2word[x], w) for w in valid_lex]

        # evaluation // compute the accuracy using conlleval.pl
        res_test  = conlleval(predictions_test, groundtruth_test, words_test, folder + '/current.test.txt')
        res_valid = conlleval(predictions_valid, groundtruth_valid, words_valid, folder + '/current.valid.txt')

        if res_valid['f1'] > best_f1:
            rnn.save(folder)
            best_f1 = res_valid['f1']
            if s.verbose:
                print 'NEW BEST: epoch', e, 'valid F1', res_valid['f1'], 'best test F1', res_test['f1'], ' '*20
            s.vf1, s.vp, s.vr = res_valid['f1'], res_valid['p'], res_valid['r']
            s.tf1, s.tp, s.tr = res_test['f1'],  res_test['p'],  res_test['r']
            s.be = e
            subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
            subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
        else:
            print ''
        
        # learning rate decay if no improvement in 10 epochs
        if s.decay and abs(s.be-s.ce) >= 10: s.clr *= 0.5 
        if s.clr < 1e-5: break

    print 'BEST RESULT: epoch', e, 'valid F1', s.vf1, 'best test F1', s.tf1, 'with the model', folder

