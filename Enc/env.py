import numpy as np
from utils import make_systematic, decimal_in_binary_array, pcm2gm
from ber_bp import test_ber
from ber_gnn import test_ber_gnn


# md怎么也有问题啊，大一点的mat为什么会出现负值
def count_cyc(pcm, girth=4):
    m, n = np.shape(pcm)
    dv = np.sum(pcm, axis = 0)
    num_edges = int(np.sum(pcm))
    edges_list = np.stack(np.where(np.transpose(pcm)), axis=1) # vn = edges_list[i, 0], cn = edges_list[i, 1]

    counter = np.zeros(girth-1, dtype=np.int16)
    local_counter = np.zeros(girth-1, dtype=np.int16)

    for i in range(n):
        if dv[i] != 0:
            # initialization
            mes_v2c = np.zeros([num_edges, dv[i]], dtype=np.int16)
            pos = np.where(edges_list[:, 0] == i)
            mes_v2c[pos] = np.eye(dv[i], dtype=int)
            mes_c2v = np.zeros([num_edges, dv[i]], dtype=np.int16)

            for k in range(girth-1):
                # message passing from cn to vn
                for j in range(m):
                    for edge in np.where(edges_list[:, 1] == j)[0]:
                        neigbors = np.where((edges_list[:, 1] == j) & (edges_list[:, 0] != edges_list[edge, 0]) & (edges_list[:, 0] >= i))[0]
                        mes_c2v[edge] = np.sum(mes_v2c[neigbors], axis = 0)

                # counting cycles
                mes_c2v_tem = mes_c2v[pos]
                di = np.diag_indices(dv[i])
                mes_c2v_tem[di] = 0
                local_counter[k] = np.sum(mes_c2v_tem)

                # message passing from vn to cn
                for ii in range(i+1, n):
                    for edge in np.where(edges_list[:, 0] == ii)[0]:
                        neigbors = np.where((edges_list[:, 0] == ii) & (edges_list[:, 1] != edges_list[edge, 1]))[0]
                        mes_v2c[edge] = np.sum(mes_c2v[neigbors], axis = 0)

            counter = counter + local_counter/2

    return counter


def check_proto_matrix(pcm):
    try:
        # print(proto_m)
        pcm_sys, col_swap = make_systematic(pcm, is_pcm=True)
    except:
        return False
    else:
        return True


def ham_distance(gnm):
    n = gnm.shape[1]
    k = gnm.shape[1] - gnm.shape[0]
    ham_d = []
    min_d = n
    for i in range(1, 2 ** (k)):
        b = decimal_in_binary_array(i, k)
        c = np.dot([b], gnm) % 2
        d = sum(sum(c))
        ham_d.append(d)
        if d < min_d:
            min_d = d
    return min_d, np.mean(ham_d), np.var(ham_d)


class PCMEnv():
    def __init__(self, m, n, initial='ldpc', reward='cyc', flip=0.3,  model_fp = None):
        self.m = m
        self.n = n
        self.mat_init = initial
        self.r_func = reward
        self.model_fp = model_fp
        self.flip = flip

    def reset_state(self):
        if self.mat_init == 'ldpc':
            k = self.n - self.m
            file_h = format('./Matrix/LDPC_%d_%d_chk.txt' % (self.n, k))
            initial_matrix = np.loadtxt(file_h, dtype=np.int32)
        if self.mat_init == 'random':
            initial_matrix = np.random.randint(2, size=(self.m, self.n), dtype=np.int32)
        else:
            initial_matrix = np.loadtxt(self.mat_init, dtype=np.int32)
        return initial_matrix

    def reward(self, state, action):
        # state [0, 1] ndarray
        # action in (0, 1) ndarray
        # action_sample = ((np.random.random_sample(size=action.shape)) < action).astype(int)
        action_sample = (np.where(action < self.flip, 0, 1)).astype(int)
        next_state = (state + action_sample) % 2
        # pcm = next_state.astype(int)

        if check_proto_matrix(next_state):
            reward = 1
            if self.r_func == 'cyc':
                cycs = count_cyc(next_state)
                # girth = 4
                if cycs[1] > 0:
                    reward += 500 / (500 + cycs[1])
                # girth = 6
                elif cycs[1] == 0 and cycs[2] > 0:
                    reward += 1
                else:
                    print('Error: wrong number of cycles, ', cycs)
            if self.r_func == 'dis':
                # next state is PCM!!!
                gnm = pcm2gm(next_state)
                min_d, mean_d, var_d = ham_distance(gnm)
                reward += min_d
            if self.r_func == 'cyc_and_dist':
                gnm = pcm2gm(next_state)
                min_d, mean_d, var_d = ham_distance(gnm)
                cycs = count_cyc(next_state)
                reward += min_d / 8 + 500 / (500 + cycs[1])
            if self.r_func == 'bp_ber2':
                ber, bler = test_ber(next_state, bp_inter=8, snr_db=2.0)
                reward += abs(np.log(ber))
            if self.r_func == 'bp_ber4':
                ber, bler = test_ber(next_state, bp_inter=8, snr_db=4.0)
                reward += abs(np.log(ber))
            if self.r_func == 'bp_ber6':
                ber, bler = test_ber(next_state, bp_inter=8, snr_db=6.0)
                reward += abs(np.log(ber))
            if self.r_func == 'bp_ber8':
                ber, bler = test_ber(next_state, bp_inter=8, snr_db=8.0)
                reward += abs(np.log(ber))
            if self.r_func == 'gnn_ber1':
                ber, bler = test_ber_gnn(next_state, bp_inter=8, snr_db=1.0, model_fp = self.model_fp)
                reward += abs(np.log(ber))
            if self.r_func == 'gnn_ber2':
                ber, bler = test_ber_gnn(next_state, bp_inter=8, snr_db=2.0, model_fp = self.model_fp)
                reward += abs(np.log(ber))
            if self.r_func == 'gnn_ber4':
                ber, bler = test_ber_gnn(next_state, bp_inter=8, snr_db=4.0, model_fp = self.model_fp)
                reward += abs(np.log(ber))
            if self.r_func == 'gnn_ber6':
                ber, bler = test_ber_gnn(next_state, bp_inter=8, snr_db=6.0, model_fp = self.model_fp)
                reward += abs(np.log(ber))
        else:
            reward = 0
        return next_state, reward


