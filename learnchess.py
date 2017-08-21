import chess, chess.uci
import re
import tensorflow as tf
import numpy as np
import random
import time
import multiprocessing as mp
import sys

STOCKFISH_PATH = "/Users/thdy/Downloads/stockfish-8-mac/Mac/stockfish-8-bmi2"

def setup_stockfish(pv):
    engine = chess.uci.popen_engine(STOCKFISH_PATH)
    engine.uci()
    engine.setoption({'Threads':8, 'MultiPv':pv})
    engine.isready()
    engine.ucinewgame()
    infos = chess.uci.InfoHandler()
    engine.info_handlers.append(infos)
    return engine, infos

def get_features(board):
    ''' Returns a feature vector representation of board.
        Note: We expect the board is from the perspective of white. '''
    white = board.occupied_co[chess.WHITE]
    black = board.occupied_co[chess.BLACK]
    positions = []
    counts = []
    for piece in (board.pawns, board.knights, board.bishops,
                  board.rooks, board.queens, board.kings):
        positions.append([
              ((piece & white) >> (63-i) & 1)
            - ((piece & black) >> i & 1) for i in range(64)])
        counts.append(
                bin(piece & white).count('1')
                -bin(piece & black).count('1'))
    return positions, counts

def next_move(engine, infos, pv, movetime):
    ''' Runs the engine for 'movetime' milliseconds,
        then reads the score of the best pv
        and chooses a move from the 'pv' best options. '''
    engine.go(movetime=movetime)
    pvs, scores = infos.info['pv'], infos.info['score']
    # If the game is finished, or data is bad, return None
    if not pvs or not scores or 1 not in scores \
            or scores[1].mate is not None or scores[1].cp is None:
        return
    # Choose random move
    next_move = []
    for i in range(1,pv+1):
        if i in pvs and pvs[i]:
            next_move.append(pvs[i][0])
    return scores[1].cp, random.choice(next_move)

def calc_batch(movetime=1, pv=6, moves=60):
    ''' Starts stockfish and runs it for 'moves' steps.
        Returns a pair of feature vectors and score vectors from the seen positions.'''
    engine, infos = setup_stockfish(pv)
    board = chess.Board()
    res = []
    for i in range(moves):
        # White to move
        engine.position(board)
        score_move = next_move(engine, infos, pv, movetime)
        if not score_move:
            break
        res.append(get_features(board) + (score_move[0],))
        board.push(score_move[1])
        # Black to move
        engine.position(board)
        score_move = next_move(engine, infos, pv, movetime)
        if not score_move:
            break
        board.push(score_move[1])
    engine.quit()
    return res

def setup_model():
    sess = tf.Session()
    xp = tf.placeholder(tf.float32, shape=[None, 6, 64])
    xpm = tf.reshape(xp, (-1, 64*6))
    xt = tf.placeholder(tf.float32, shape=[None, 6])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    P = tf.Variable(tf.zeros([6,64,1]))
    Pm = tf.reshape(P, (64*6, 1))
    T = tf.Variable(tf.zeros([6,1]))
    b = tf.Variable(tf.zeros([1]))
    # Optimize with relu loss on negative weights.
    # The symetry of the model makes negatives redundant.
    regular = tf.abs(tf.reduce_mean(P)) * 10
    l1loss = tf.reduce_mean(tf.abs(xpm@Pm + xt@T + b - y))
    loss = l1loss + regular
    # For Visualization
    for i in range(6):
        tf.summary.image('positions {}'.format(i), tf.reshape(P[i],(-1,8,8,1)))
        #tf.summary.histogram('positions'+str(i), P[i])
    for i in range(6):
        tf.summary.scalar('types {}'.format(i), T[i][0])
    tf.summary.scalar('bias', tf.reshape(b,()))
    tf.summary.scalar('loss', l1loss)
    tf.summary.scalar('regu', regular)
    summ = tf.summary.merge_all()
    # We use the Adam Optimizer, since our weights have very different target values and variances. In particular this seems to make the queen and forward pawns converge much faster.
    #train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    train = tf.train.AdamOptimizer(.1).minimize(loss)
    sess.run(tf.global_variables_initializer())
    return sess, train, summ, xp, xt, y

def fish_process(q):
    while True:
        q.put(calc_batch())

data_queue = mp.Queue()
fishes = []
for _ in range(2):
    fish = mp.Process(target=fish_process, args=(data_queue,))
    fish.start()
    fishes.append(fish)

sess, train, summ, xp, xt, y = setup_model()
writer = tf.summary.FileWriter('/tmp/learnchess/6')
writer.add_graph(sess.graph)

data = []
gen_time, train_time, test_time = 0, 0, 0
for i in range(10**6):
    t = time.time()
    if not data_queue.empty() or not data:
        data += data_queue.get()
    batch = random.sample(data, min(len(data), 100))
    xp_train, xt_train, y_train = list(zip(*batch))
    y_train = np.reshape(y_train, (-1, 1))
    gen_time += time.time()-t

    # We use the same batch for testing and training, but we
    # test before we train.
    if i % 20 == 0:
        t = time.time()
        s = sess.run(summ, {xp: xp_train, xt: xt_train, y: y_train})
        writer.add_summary(s, i)
        test_time += time.time()-t

    t = time.time()
    sess.run(train, {xp: xp_train, xt: xt_train, y: y_train})
    train_time += time.time()-t

