from __future__ import division, print_function
import argparse
import math
import os.path
import timeit
from multiprocessing import JoinableQueue, Queue, Process
import time
import random
import numpy as np
import tensorflow as tf
import copy


class ConMask:
    @property
    def n_entity(self):
        return self.__n_entity
    #################################################################
    @property
    def n_vocab(self):
        return self.__n_vocab
    @property
    def embed_dim(self):
        return self.__embed_dim
    @property
    def n_close_ent(self):
        return self.__n_close_ent
    @property
    def entity_title_wordid(self):
        return self.__entity_title_wordid
    @property
    def entity_des_wordid(self):
        return self.__entity_des_wordid

    @property
    def word_title_entityid(self):
        return self.__word_title_entityid
    @property
    def word_des_entityid(self):
        return self.__word_des_entityid

    @property
    def word_emb(self):
        return self.__word_emb
    @property
    def r_t(self):
        return self.__r_t

    @property
    def r_h(self):
        return self.__r_h
    #################################################################

    @property
    def n_train(self):
        return self.__train_triple.shape[0]

    @property
    def trainable_variables(self):
        return self.__trainable

    @property
    def hr_t(self):
        return self.__hr_t

    @property
    def tr_h(self):
        return self.__tr_h

    @property
    def train_hr_t(self):
        return self.__train_hr_t

    @property
    def train_tr_h(self):
        return self.__train_tr_h

    @property
    def rel_embedding(self):
        return self.__rel_embedding

    @property
    def word_embedding(self):
        return self.__word_embedding
    
    def get_all_group(self, all_entity, group_size=200):
        all_group = []
        n_triple = len(all_entity)
        rand_idx = np.random.permutation(n_triple)
        start = 0
        while start < n_triple:
            end = min(start + group_size, n_triple)
            all_group.append(all_entity[rand_idx[start:end]])
            start = end
        # all_group[-1] = np.pad(all_group[-1], (0, group_size - len(all_group[-1])), 'constant')
        return all_group

    def raw_training_data(self, batch_size=100):
        n_triple = len(self.__train_triple)
        rand_idx = np.random.permutation(n_triple)

        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__train_triple[rand_idx[start:end]]
            start = end

    def testing_data(self, batch_size=100):
        n_triple = len(self.__test_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__test_triple[start:end, :]
            start = end
    def testing_data_open_head_test_tail(self, batch_size=100):
        n_triple = len(self.__open_head_test_tail)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__open_head_test_tail[start:end, :]
            start = end
    def testing_data_open_tail_test_head(self, batch_size=100):
        n_triple = len(self.__open_tail_test_head)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__open_tail_test_head[start:end, :]
            start = end

    def validation_data(self, batch_size=100):
        n_triple = len(self.__valid_triple)
        start = 0
        while start < n_triple:
            end = min(start + batch_size, n_triple)
            yield self.__valid_triple[start:end, :]
            start = end

    def __init__(self, data_dir, embed_dim=100, dropout=0.5,
                 entity_word_title_len = 10, entity_word_des_len = 256,
                 word_entity_title_len = 8, word_entity_des_len = 128,
                 neg_num = 4,
                 scale_1 = 10, scale_2 = 10):
        self.__embed_dim = embed_dim
        self.__neg_num = neg_num
        self.__initialized = False
        self.__word_embed_dim = 200

        self.__trainable = list()
        self.__dropout = dropout
        self.__scale_1 = scale_1
        self.__scale_2 = scale_2


        with open(os.path.join(data_dir, 'entity2id.txt'), 'r', encoding='utf8') as f:
            self.__n_entity = len(f.readlines())

        with open(os.path.join(data_dir, 'entity2id.txt'), 'r', encoding='utf8') as f:
            self.__entity_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
            self.__id_entity_map = {v: k for k, v in self.__entity_id_map.items()}

        print("N_ENTITY: %d" % self.__n_entity)
#########################################################################
        with open(os.path.join(data_dir, 'label.txt'), 'r', encoding='utf8') as f:
            self.__n_close_ent = len(f.readlines())
        with open(os.path.join(data_dir, 'label.txt'), 'r', encoding='utf8') as f:
            self.__close_entity_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
            self.__close_id_entity_map = {v: k for k, v in self.__close_entity_id_map.items()}
#########################################################################
        with open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf8') as f:
            self.__n_relation = len(f.readlines())

        with open(os.path.join(data_dir, 'relation2id.txt'), 'r', encoding='utf8') as f:
            self.__relation_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
            self.__id_relation_map = {v: k for k, v in self.__relation_id_map.items()}

        print("N_RELATION: %d" % self.__n_relation)
# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        with open(os.path.join(data_dir, 'vocab2id.txt'), 'r', encoding='utf8') as f:
            self.__n_vocab = len(f.readlines())

        with open(os.path.join(data_dir, 'vocab2id.txt'), 'r', encoding='utf8') as f:
            self.__vocab_id_map = {x.strip().split('\t')[0]: int(x.strip().split('\t')[1]) for x in f.readlines()}
            self.__id_vocab_map = {v: k for k, v in self.__vocab_id_map.items()}

        print("N_VOCAB: %d" % self.__n_vocab)

        self.__entity_title_wordid = []
        self.__entity_title_wordid_len = []
        with open(os.path.join(data_dir, 'entity_names.txt'), 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().split('\t')
                word_id = []
                if len(line) > 2:
                    for word in line[2].split():
                        word_id.append(self.__vocab_id_map[word] if word in self.__vocab_id_map else 0)
                self.__entity_title_wordid_len.append(max(len(word_id), 1e-10))
                while len(word_id) < entity_word_title_len:
                    word_id.append(0)
                word_id = word_id[0:entity_word_title_len]
                self.__entity_title_wordid.append(word_id)
        self.__entity_title_wordid = np.concatenate([np.zeros([1, entity_word_title_len], dtype = np.int32), np.asarray(self.__entity_title_wordid, dtype = np.int32)], axis = 0)
        self.__entity_title_wordid_len = np.concatenate([np.zeros([1, 1], dtype = np.float32) - 1e-10, np.reshape(np.asarray(self.__entity_title_wordid_len, dtype = np.float32), [self.__n_entity, 1])], axis = 0)
        # print (np.shape(self.__entity_title_wordid_len))
        print ('Dict entity to word title len:', np.shape(self.__entity_title_wordid))
        # print (self.__entity_title_wordid)

        self.__word_title_entityid = []
        self.__word_title_entityid_len = []
        with open(os.path.join(data_dir, 'entity_names_prim.txt'), 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().split('\t')
                entity_id = []
                if len(line) > 2:
                    ent_len = min(word_entity_title_len, int(line[1]))
                    for entity in line[2].split()[:ent_len]:
                        entity_id.append(self.__entity_id_map[entity])
                self.__word_title_entityid_len.append(max(len(entity_id), 1e-10))
                while len(entity_id) < word_entity_title_len:
                    entity_id.append(0)
                self.__word_title_entityid.append(entity_id)
        self.__word_title_entityid = np.concatenate([np.zeros([1, word_entity_title_len], dtype = np.int32), np.asarray(self.__word_title_entityid, dtype = np.int32)], axis = 0)
        self.__word_title_entityid_len = np.concatenate([np.zeros([1, 1], dtype = np.float32) - 1e-10, np.reshape(np.asarray(self.__word_title_entityid_len, dtype = np.float32), [self.__n_vocab, 1])], axis = 0)
        print ('Dict word to entity title len:', np.shape(self.__word_title_entityid))
        # print (self.__word_title_entityid)

        self.__entity_des_wordid = []
        self.__entity_des_wordid_len = []
        with open(os.path.join(data_dir, 'descriptions.txt'), 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().split('\t')
                word_id = []
                if len(line) > 2:
                    word_len = min(entity_word_des_len, int(line[1]))
                    for word in line[2].split()[0:word_len]:
                        word_id.append(self.__vocab_id_map[word] if word in self.__vocab_id_map else 0)
                self.__entity_des_wordid_len.append(max(len(word_id), 1e-10))
                while len(word_id) < entity_word_des_len:
                    word_id.append(0)
                self.__entity_des_wordid.append(word_id)
        self.__entity_des_wordid = np.concatenate([np.zeros([1, entity_word_des_len], dtype = np.int32), np.asarray(self.__entity_des_wordid, dtype = np.int32)], axis = 0)
        self.__entity_des_wordid_len = np.concatenate([np.zeros([1, 1], dtype = np.float32) - 1e-10, np.reshape(np.asarray(self.__entity_des_wordid_len, dtype = np.float32), [self.__n_entity, 1])], axis = 0)
        print ('Dict entity to word des len:', np.shape(self.__entity_des_wordid))
        # print (self.__entity_des_wordid)

        self.__word_des_entityid = []
        self.__word_des_entityid_len = []
        with open(os.path.join(data_dir, 'descriptions_prim.txt'), 'r', encoding='utf8') as f:
            for line in f:
                line = line.strip().split('\t')
                entity_id = []
                if len(line) > 2:
                    ent_len = min(word_entity_des_len, int(line[1]))
                    for entity in line[2].split()[:ent_len]:
                        entity_id.append(self.__entity_id_map[entity])
                self.__word_des_entityid_len.append(max(len(entity_id), 1e-10))
                while len(entity_id) < word_entity_des_len:
                    entity_id.append(0)
                self.__word_des_entityid.append(entity_id)
        self.__word_des_entityid = np.concatenate([np.zeros([1, word_entity_des_len], dtype = np.int32), np.asarray(self.__word_des_entityid, dtype = np.int32)], axis = 0)
        self.__word_des_entityid_len = np.concatenate([np.zeros([1, 1], dtype = np.float32) - 1e-10, np.reshape(np.asarray(self.__word_des_entityid_len, dtype = np.float32), [self.__n_vocab, 1])], axis = 0)
        print ('Dict word to entity des len:', np.shape(self.__word_des_entityid))
        # print (self.__word_des_entityid)


        self.__word_emb = np.ones((self.__n_vocab, self.__word_embed_dim), dtype=np.float32)
        with open(os.path.join(data_dir, 'embed.txt'), 'r', encoding='UTF-8') as f:
            index = 0
            for line in f:
                line = line.strip().split()
                if line[0] in self.__vocab_id_map:
                    word_id = self.__vocab_id_map[line[0]]
                    emb = line[1:]
                    self.__word_emb[index] = np.asarray(emb)
                    index += 1

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        def load_triple(file_path):
            with open(file_path, 'r', encoding='utf8') as f_triple:
                return np.asarray([[self.__entity_id_map[x.strip().split('\t')[0]],
                                    self.__entity_id_map[x.strip().split('\t')[1]],
                                    self.__relation_id_map[x.strip().split('\t')[2]]] for x in f_triple.readlines()],
                                  dtype=np.int32)

        def gen_hr_t(triple_data):
            hr_t = dict()
            for h, t, r in triple_data:
                if h not in hr_t:
                    hr_t[h] = dict()
                if r not in hr_t[h]:
                    hr_t[h][r] = set()
                hr_t[h][r].add(t)

            return hr_t

        def gen_tr_h(triple_data):
            tr_h = dict()
            for h, t, r in triple_data:
                if t not in tr_h:
                    tr_h[t] = dict()
                if r not in tr_h[t]:
                    tr_h[t][r] = set()
                tr_h[t][r].add(h)
            return tr_h

        self.__train_triple = load_triple(os.path.join(data_dir, 'train.txt'))
        print("N_TRAIN_TRIPLES: %d" % self.__train_triple.shape[0])

        self.__test_triple = load_triple(os.path.join(data_dir, 'test.txt'))
        print("N_TEST_TRIPLES: %d" % self.__test_triple.shape[0])

        self.__open_head_test_tail = []
        self.__open_tail_test_head = []
        for i in range(self.__test_triple.shape[0]):
            if self.__test_triple[i][0] not in self.__close_id_entity_map and self.__test_triple[i][1] in self.__close_id_entity_map:
                self.__open_head_test_tail.append(self.__test_triple[i])
            if self.__test_triple[i][0] in self.__close_id_entity_map and self.__test_triple[i][1] not in self.__close_id_entity_map:
                self.__open_tail_test_head.append(self.__test_triple[i])
        self.__open_head_test_tail = np.asarray(self.__open_head_test_tail)
        self.__open_tail_test_head = np.asarray(self.__open_tail_test_head)
        print ('open_head_test_tail: ', len(self.__open_head_test_tail))
        print ('open_tail_test_head: ', len(self.__open_tail_test_head))


        self.__valid_triple = load_triple(os.path.join(data_dir, 'valid.txt'))
        print("N_VALID_TRIPLES: %d" % self.__valid_triple.shape[0])

        def gen_r_h_and_t(triple_data):
            r_h = dict()
            r_t = dict()
            for h, t, r in triple_data:
                if r not in r_h:
                    r_h[r] = set()
                    r_t[r] = set()
                r_h[r].add(h)
                r_t[r].add(t)
            for r in self.__id_relation_map.keys():
                if r not in r_h:
                    r_h[r] = set()
                    r_t[r] = set()
            # max_r_h_set_len = 0
            # all_r_h_set_len = 0
            # for r, h_set in r_h.items():
            #     if max_r_h_set_len < len(h_set):
            #         max_r_h_set_len = len(h_set)
            #     all_r_h_set_len += len(h_set)
            # avg_r_h_set_len = float(all_r_h_set_len) / len(r_h)

            # max_r_t_set_len = 0
            # all_r_t_set_len = 0
            # for r, t_set in r_t.items():
            #     if max_r_t_set_len < len(t_set):
            #         max_r_t_set_len = len(t_set)
            #     all_r_t_set_len += len(t_set)
            # avg_r_t_set_len = float(all_r_t_set_len) / len(r_t)
            return r_h, r_t
        self.__r_h, self.__r_t = gen_r_h_and_t(self.__train_triple)
        # print ('max_r_h_set_len: ', max_r_h_set_len)
        # print ('avg_r_h_set_len: ', avg_r_h_set_len)
        # print ('max_r_t_set_len: ', max_r_t_set_len)
        # print ('avg_r_t_set_len: ', avg_r_t_set_len)
        # self.__valid_r_h, self.__valid_r_t = gen_r_h(self.__valid_triple)

        self.__train_hr_t = gen_hr_t(self.__train_triple)
        self.__train_tr_h = gen_tr_h(self.__train_triple)
        self.__test_hr_t = gen_hr_t(self.__test_triple)
        self.__test_tr_h = gen_tr_h(self.__test_triple)

        self.__hr_t = gen_hr_t(np.concatenate([self.__train_triple, self.__test_triple, self.__valid_triple], axis=0))
        self.__tr_h = gen_tr_h(np.concatenate([self.__train_triple, self.__test_triple, self.__valid_triple], axis=0))

        self.__weight_dim = 200
        self.__layer = 3

        bound = 6 / math.sqrt(embed_dim)

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        self.__word_embedding = tf.get_variable("word_embedding",
                                               initializer=self.__word_emb)
        # self.__trainable.append(self.__word_embedding)
        self.__word_embedding = tf.concat([tf.constant(0., shape=[1, self.__word_embed_dim]), self.__word_embedding], axis = 0)



        self.__predict_weight = tf.get_variable("predict_weight",initializer=tf.ones([15]))
        self.__trainable.append(self.__predict_weight)

        self.__relation_embedding = tf.get_variable("relation_embedding",
                                               [self.__n_relation, self.__embed_dim],
                                               dtype=tf.float32,
                                               initializer=tf.random_uniform_initializer(minval=-bound,maxval=bound,seed = 345))
        # self.__trainable.append(self.__relation_embedding)

        # [2, layer, 2, vocab_num, weight_dim]   [title or des, layer, send or recieve, vocab_num, weight_dim]
        self.__word_weight = tf.get_variable("word_weight",
                                              initializer=tf.ones([2, self.__layer, 2, self.__n_vocab, self.__weight_dim]))
        self.__trainable.append(self.__word_weight)
        self.__word_weight = tf.concat([tf.constant(0., shape=[2, self.__layer, 2, 1, self.__weight_dim]), self.__word_weight], axis = -2)


        self.__ent_weight = tf.get_variable("ent_weight",
                                              initializer=tf.ones([2, self.__layer, 2, self.__n_close_ent, self.__weight_dim]))
        self.__trainable.append(self.__ent_weight)
        self.__ent_weight = tf.concat([tf.constant(0., shape=[2, self.__layer, 2, 1, self.__weight_dim]), self.__ent_weight], axis = -2)



        self.__project_matrix_layer_one = tf.get_variable("project_matrix_layer_one",
                                                         [2, 100, 200],
                                                         dtype=tf.float32,
                                                         initializer=tf.random_uniform_initializer(minval=-bound,maxval=bound,seed=354))
        # self.__trainable.append(self.__project_matrix_layer_one)

        # self.__project_matrix_layer_two = tf.get_variable("project_matrix_layer_two",
        #                                                  [2, 100, 200],
        #                                                  dtype=tf.float32,
        #                                                  initializer=tf.random_uniform_initializer(minval=-bound,maxval=bound,seed=356))
        # # self.__trainable.append(self.__project_matrix_layer_two)


        # self.__word_title_entityid_matrix = tf.constant(self.__word_title_entityid, dtype = tf.int32)
        # self.__word_des_entityid_matrix = tf.constant(self.__word_des_entityid, dtype = tf.int32)
        # self.__entity_title_wordid_matrix = tf.constant(self.__entity_title_wordid[0:self.__n_close_ent], dtype = tf.int32)
        # self.__entity_des_wordid_matrix = tf.constant(self.__entity_des_wordid[0:self.__n_close_ent], dtype = tf.int32)

# ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    @staticmethod
    def __l1_normalize(x, dim, epsilon=1e-12, name=None):
        square_sum = tf.reduce_sum(tf.abs(x), [dim], keep_dims=True)
        x_inv_norm = tf.rsqrt(tf.maximum(square_sum, epsilon))
        return tf.mul(x, x_inv_norm, name=name)

    @staticmethod
    def sampled_softmax(tensor, weights):
        max_val = tf.reduce_max(tensor * tf.abs(weights), 1, keep_dims=True)
        tensor_rescaled = tensor - max_val
        tensor_exp = tf.exp(tensor_rescaled)
        tensor_sum = tf.reduce_sum(tensor_exp * tf.abs(weights), 1, keep_dims=True)

        return (tensor_exp / tensor_sum) * tf.abs(weights)  # all ignored elements will have a prob of 0.
    def normalized_embedding(self, embedding):
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), -1, keep_dims=True), name='norm') + 1e-10
        norm_embed = embedding / norm
        return norm_embed

    def get_avg(self, embedding, content, content_len):
        return tf.reduce_sum(tf.nn.embedding_lookup(embedding, content), axis = -2) / content_len
    
    # def transform_word_to_word_embedding(self, word_embedding):
        # one_ent_title_embedding = self.get_avg(word_embedding, self.__entity_title_wordid[0:self.__n_close_ent+1], self.__entity_title_wordid_len[0:self.__n_close_ent+1])
        # one_ent_des_embedding = self.get_avg(word_embedding, self.__entity_des_wordid[0:self.__n_close_ent+1], self.__entity_des_wordid_len[0:self.__n_close_ent+1])

        # one_word_title_embedding = self.get_avg(one_ent_title_embedding, self.__word_title_entityid, self.__word_title_entityid_len)
        # one_word_des_embedding = self.get_avg(one_ent_des_embedding, self.__word_des_entityid, self.__word_des_entityid_len)

        # return one_word_title_embedding, one_word_des_embedding

    def transform_word_to_word_embedding(self, word_title_embedding, word_des_embedding, layer):
        # with tf.device('/gpu:0'):
        word_title_embedding = tf.nn.embedding_lookup(word_title_embedding * self.__word_weight[0, layer, 0, :, :], 
                                                      self.__entity_title_wordid[0:self.__n_close_ent+1])
        ent_title_embedding = tf.reduce_sum(word_title_embedding, axis = -2) * self.__ent_weight[0, layer, 1, :, :] \
                                             / self.__entity_title_wordid_len[0:self.__n_close_ent+1]
        ent_title_embedding = tf.nn.tanh(ent_title_embedding)
        ent_title_embedding = tf.nn.embedding_lookup(ent_title_embedding * self.__ent_weight[0, layer, 0, :, :],
                                                     self.__word_title_entityid)
        word_title_embedding = tf.reduce_sum(ent_title_embedding, axis = -2) * self.__word_weight[0, layer, 1, :, :] \
                                              / self.__word_title_entityid_len
        word_title_embedding = tf.nn.tanh(word_title_embedding)
        # with tf.device('/gpu:1'):
        word_des_embedding = tf.nn.embedding_lookup(word_des_embedding * self.__word_weight[1, layer, 0, :, :], 
                                                    self.__entity_des_wordid[0:self.__n_close_ent+1])
        ent_des_embedding = tf.reduce_sum(word_des_embedding, axis = -2) * self.__ent_weight[1, layer, 1, :, :] \
                                           / self.__entity_des_wordid_len[0:self.__n_close_ent+1]
        ent_des_embedding = tf.nn.tanh(ent_des_embedding)
        # with tf.device('/gpu:1'):
        ent_des_embedding = tf.nn.embedding_lookup(ent_des_embedding * self.__ent_weight[1, layer, 0, :, :],
                                                    self.__word_des_entityid)
        word_des_embedding = tf.reduce_sum(ent_des_embedding, axis = -2) * self.__word_weight[1, layer, 1, :, :] \
                                            / self.__word_des_entityid_len
        word_des_embedding = tf.nn.tanh(word_des_embedding)
        return word_title_embedding, word_des_embedding



    def predict(self, ent_title_embedding, ent_des_embedding, h, r, t):
        h_title_embedding = self.normalized_embedding(tf.nn.embedding_lookup(ent_title_embedding, h))
        h_des_embedding = self.normalized_embedding(tf.nn.embedding_lookup(ent_des_embedding, h))
        r_embedding = self.normalized_embedding(tf.nn.embedding_lookup(self.__relation_embedding, r))
        t_title_embedding = self.normalized_embedding(tf.nn.embedding_lookup(ent_title_embedding, t))
        t_des_embedding = self.normalized_embedding(tf.nn.embedding_lookup(ent_des_embedding, t))
        
        # predict_head_score = tf.reduce_sum(predict_head_h * predict_head_r * predict_head_t, axis = -1)
        predict_score = self.FM2(h_title_embedding, h_des_embedding, r_embedding, t_title_embedding, t_des_embedding)
        return predict_score
    def FM2(self, h_title_embedding, h_des_embedding, r_embedding, t_title_embedding, t_des_embedding):
        self.__predict_weight = self.__scale_1 * tf.nn.softmax(self.__predict_weight)
        # with tf.device('/gpu:0'):
        [fm1, fm2, fm3, fm4, fm5, fm6, fm7, fm8, fm9, fm10, fm11, fm12, fm13, fm14, fm15] = \
                       [h_title_embedding * h_title_embedding,
                        h_title_embedding * h_des_embedding,
                        h_title_embedding * r_embedding,
                        h_title_embedding * t_title_embedding,
                        h_title_embedding * t_des_embedding,
                        h_des_embedding * h_des_embedding,
                        h_des_embedding * r_embedding,
                        h_des_embedding * t_title_embedding,
                        h_des_embedding * t_des_embedding,
                        r_embedding * r_embedding,
                        r_embedding * t_title_embedding,
                        r_embedding * t_des_embedding,
                        t_title_embedding * t_title_embedding,
                        t_title_embedding * t_des_embedding,
                        t_des_embedding * t_des_embedding]

        candidate_size = tf.shape(h_title_embedding * t_title_embedding)[1]
        one = tf.ones(1,dtype = tf.int32)[0]
        
        fm1 = tf.cond(tf.equal(tf.shape(fm1)[1], one), lambda: tf.tile(fm1, [1, candidate_size, 1]), lambda: fm1)
        fm2 = tf.cond(tf.equal(tf.shape(fm2)[1], one), lambda: tf.tile(fm2, [1, candidate_size, 1]), lambda: fm2)
        fm3 = tf.cond(tf.equal(tf.shape(fm3)[1], one), lambda: tf.tile(fm3, [1, candidate_size, 1]), lambda: fm3)
        fm4 = tf.cond(tf.equal(tf.shape(fm4)[1], one), lambda: tf.tile(fm4, [1, candidate_size, 1]), lambda: fm4)
        fm5 = tf.cond(tf.equal(tf.shape(fm5)[1], one), lambda: tf.tile(fm5, [1, candidate_size, 1]), lambda: fm5)
        fm6 = tf.cond(tf.equal(tf.shape(fm6)[1], one), lambda: tf.tile(fm6, [1, candidate_size, 1]), lambda: fm6)
        fm7 = tf.cond(tf.equal(tf.shape(fm7)[1], one), lambda: tf.tile(fm7, [1, candidate_size, 1]), lambda: fm7)
        fm8 = tf.cond(tf.equal(tf.shape(fm8)[1], one), lambda: tf.tile(fm8, [1, candidate_size, 1]), lambda: fm8)
        fm9 = tf.cond(tf.equal(tf.shape(fm9)[1], one), lambda: tf.tile(fm9, [1, candidate_size, 1]), lambda: fm9)
        fm10 = tf.cond(tf.equal(tf.shape(fm10)[1], one), lambda: tf.tile(fm10, [1, candidate_size, 1]), lambda: fm10)
        fm11 = tf.cond(tf.equal(tf.shape(fm11)[1], one), lambda: tf.tile(fm11, [1, candidate_size, 1]), lambda: fm11)
        fm12 = tf.cond(tf.equal(tf.shape(fm12)[1], one), lambda: tf.tile(fm12, [1, candidate_size, 1]), lambda: fm12)
        fm13 = tf.cond(tf.equal(tf.shape(fm13)[1], one), lambda: tf.tile(fm13, [1, candidate_size, 1]), lambda: fm13)
        fm14 = tf.cond(tf.equal(tf.shape(fm14)[1], one), lambda: tf.tile(fm14, [1, candidate_size, 1]), lambda: fm14)
        fm15 = tf.cond(tf.equal(tf.shape(fm15)[1], one), lambda: tf.tile(fm15, [1, candidate_size, 1]), lambda: fm15)
        
        fm = tf.stack([fm1, fm2, fm3, fm4, fm5, fm6, fm7, fm8, fm9, fm10, fm11, fm12, fm13, fm14, fm15], axis = -2)
        r_embedding = tf.tile(tf.expand_dims(r_embedding, axis = -2), [1, 1, 15, 1])
        attention_weight = self.__scale_2 * tf.nn.softmax(tf.reduce_sum(fm * r_embedding, axis = -1))
        fm_attention = tf.reduce_sum(fm * tf.expand_dims(attention_weight, axis = -1), axis = -1)
        score = tf.reduce_sum(fm_attention * self.__predict_weight, axis = -1)
        

        return score

    def score_6sim(self, h_title_embedding, h_des_embedding, r_embedding, t_title_embedding, t_des_embedding):
        score = tf.reduce_sum(h_title_embedding * r_embedding, axis = -1) * self.__predict_weight[0] + \
                tf.reduce_sum(t_title_embedding * r_embedding, axis = -1) * self.__predict_weight[1] + \
                tf.reduce_sum(h_title_embedding * t_title_embedding, axis = -1) * self.__predict_weight[2] + \
                tf.reduce_sum(h_title_embedding * t_des_embedding, axis = -1) * self.__predict_weight[3] + \
                tf.reduce_sum(h_des_embedding * t_des_embedding, axis = -1) * self.__predict_weight[4] + \
                tf.reduce_sum(h_des_embedding * t_title_embedding, axis = -1) * self.__predict_weight[5]
        return score

    def train(self, inputs, regularizer_weight=1., scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            if self.__initialized:
                scp.reuse_variables()

            predict_head_h_input, predict_head_r_input, predict_head_t_input, \
            predict_tail_h_input, predict_tail_r_input, predict_tail_t_input = inputs

            predict_head_r_input = tf.tile(predict_head_r_input, [1, tf.shape(predict_head_h_input)[1]])
            predict_head_t_input = tf.tile(predict_head_t_input, [1, tf.shape(predict_head_h_input)[1]])
            predict_tail_h_input = tf.tile(predict_tail_h_input, [1, tf.shape(predict_tail_t_input)[1]])
            predict_tail_r_input = tf.tile(predict_tail_r_input, [1, tf.shape(predict_tail_t_input)[1]])

            word_title_embedding, word_des_embedding = self.transform_word_to_word_embedding(self.__word_embedding, self.__word_embedding, 0)
            # word_title_embedding = tf.matmul(word_title_embedding, self.__project_matrix_layer_one[0, :])
            # word_des_embedding = tf.matmul(word_des_embedding, self.__project_matrix_layer_one[1, :])

            # word_title_embedding, word_des_embedding = self.transform_word_to_word_embedding(word_title_embedding, word_des_embedding, 1)
            # word_title_embedding = tf.matmul(word_title_embedding, self.__project_matrix_layer_two[0, :])
            # word_des_embedding = tf.matmul(word_des_embedding, self.__project_matrix_layer_two[1, :])
            

            ent_title_embedding = self.get_avg(word_title_embedding, self.__entity_title_wordid[0:self.__n_close_ent+1], self.__entity_title_wordid_len[0:self.__n_close_ent+1])
            ent_des_embedding = self.get_avg(word_des_embedding, self.__entity_des_wordid[0:self.__n_close_ent+1], self.__entity_des_wordid_len[0:self.__n_close_ent+1])

            predict_head_score = self.predict(ent_title_embedding, ent_des_embedding, predict_head_h_input, predict_head_r_input, predict_head_t_input)

            # predict_head_labels = tf.reshape(tf.constant([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32, name='labels'), [1, 11])

            _labels = tf.reshape(tf.constant([[1.0] * 1 + [0.0] * self.__neg_num], dtype=tf.float32, name='labels'), [1, self.__neg_num + 1])

            predict_head_labels = tf.tile(_labels, [tf.shape(predict_head_h_input)[0], 1])
            predict_head_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=predict_head_labels,logits=predict_head_score))

            predict_tail_score = self.predict(ent_title_embedding, ent_des_embedding, predict_tail_h_input, predict_tail_r_input, predict_tail_t_input)

            # predict_tail_labels = tf.reshape(tf.constant([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32, name='labels'), [1, 11])
            predict_tail_labels = tf.tile(_labels, [tf.shape(predict_tail_h_input)[0], 1])
            predict_tail_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=predict_tail_labels,logits=predict_tail_score))

            return predict_head_loss + predict_tail_loss
    def predict_test(self, word_title_embedding, word_des_embedding, h, r, t):
        h_title_wordid = tf.nn.embedding_lookup(self.__entity_title_wordid, h)
        h_des_wordid = tf.nn.embedding_lookup(self.__entity_des_wordid, h)
        t_title_wordid = tf.nn.embedding_lookup(self.__entity_title_wordid, t)
        t_des_wordid = tf.nn.embedding_lookup(self.__entity_des_wordid, t)

        h_title_wordid_len = tf.nn.embedding_lookup(self.__entity_title_wordid_len, h)
        h_des_wordid_len = tf.nn.embedding_lookup(self.__entity_des_wordid_len, h)
        t_title_wordid_len = tf.nn.embedding_lookup(self.__entity_title_wordid_len, t)
        t_des_wordid_len = tf.nn.embedding_lookup(self.__entity_des_wordid_len, t)

        h_title_embedding = self.get_avg(word_title_embedding, h_title_wordid, h_title_wordid_len)
        h_des_embedding = self.get_avg(word_des_embedding, h_des_wordid, h_des_wordid_len)
        t_title_embedding = self.get_avg(word_title_embedding, t_title_wordid, t_title_wordid_len)
        t_des_embedding = self.get_avg(word_des_embedding, t_des_wordid, t_des_wordid_len)

        h_title_embedding = self.normalized_embedding(h_title_embedding)
        h_des_embedding = self.normalized_embedding(h_des_embedding)
        r_embedding = self.normalized_embedding(tf.nn.embedding_lookup(self.__relation_embedding, r))
        t_title_embedding = self.normalized_embedding(t_title_embedding)
        t_des_embedding = self.normalized_embedding(t_des_embedding)
        
        predict_score = self.FM2(h_title_embedding, h_des_embedding, r_embedding, t_title_embedding, t_des_embedding)
        return predict_score


    def test(self, inputs, scope=None):
        with tf.variable_scope(scope or type(self).__name__) as scp:
            scp.reuse_variables()
            self.predict_weight_output = self.__predict_weight

            word_title_embedding, word_des_embedding = self.transform_word_to_word_embedding(self.__word_embedding, self.__word_embedding, 0)
            self.ent_title_embedding = self.get_avg(word_title_embedding, self.__entity_title_wordid, self.__entity_title_wordid_len)
            self.ent_des_embedding = self.get_avg(word_des_embedding, self.__entity_des_wordid, self.__entity_des_wordid_len)
            # self.word_title_embedding = tf.matmul(word_title_embedding, self.__project_matrix_layer_one[0, :])
            # self.word_des_embedding = tf.matmul(word_des_embedding, self.__project_matrix_layer_one[1, :])

            # word_title_embedding, word_des_embedding = self.transform_word_to_word_embedding(word_title_embedding, word_des_embedding, 1)
            # self.word_title_embedding = tf.matmul(word_title_embedding, self.__project_matrix_layer_two[0, :])
            # self.word_des_embedding = tf.matmul(word_des_embedding, self.__project_matrix_layer_two[1, :])

            test_head_h_input, test_head_r_input, test_head_t_input, \
            test_tail_h_input, test_tail_r_input, test_tail_t_input, \
            ent_title_embedding_input, ent_des_embedding_input = inputs

            test_head_r_input = tf.tile(test_head_r_input, [1, tf.shape(test_head_h_input)[1]])
            test_head_t_input = tf.tile(test_head_t_input, [1, tf.shape(test_head_h_input)[1]])
            test_tail_h_input = tf.tile(test_tail_h_input, [1, tf.shape(test_tail_t_input)[1]])
            test_tail_r_input = tf.tile(test_tail_r_input, [1, tf.shape(test_tail_t_input)[1]])

            # ent_title_embedding = self.get_avg(word_title_embedding_input, self.__entity_title_wordid, self.__entity_title_wordid_len)
            # ent_des_embedding = self.get_avg(word_des_embedding_input, self.__entity_des_wordid, self.__entity_des_wordid_len)
            
            # ent_title_embedding = self.get_avg(word_embedding, self.__entity_title_wordid, self.__entity_title_wordid_len)
            # ent_des_embedding = self.get_avg(word_embedding, self.__entity_des_wordid, self.__entity_des_wordid_len)

            # test_head_score = self.predict(ent_title_embedding, ent_des_embedding, test_head_h_input, test_head_r_input, test_head_t_input)
            # test_tail_score = self.predict(ent_title_embedding, ent_des_embedding, test_tail_h_input, test_tail_r_input, test_tail_t_input)
            test_head_score = self.predict(ent_title_embedding_input, ent_des_embedding_input, test_head_h_input, test_head_r_input, test_head_t_input)
            test_tail_score = self.predict(ent_title_embedding_input, ent_des_embedding_input, test_tail_h_input, test_tail_r_input, test_tail_t_input)

            return test_head_score, test_tail_score



def train_ops(model, learning_rate=0.1, optimizer_str='gradient', regularizer_weight=1.0, neg_num = 5):
    with tf.device('/gpu:0'):

        predict_head_h_input = tf.placeholder(tf.int32, [None, neg_num + 1])
        predict_head_r_input = tf.placeholder(tf.int32, [None, 1])
        predict_head_t_input = tf.placeholder(tf.int32, [None, 1])

        predict_tail_h_input = tf.placeholder(tf.int32, [None, 1])
        predict_tail_r_input = tf.placeholder(tf.int32, [None, 1])
        predict_tail_t_input = tf.placeholder(tf.int32, [None, neg_num + 1])


        loss = model.train([predict_head_h_input, predict_head_r_input, predict_head_t_input,
                            predict_tail_h_input, predict_tail_r_input, predict_tail_t_input],
                           regularizer_weight=regularizer_weight)
        if optimizer_str == 'gradient':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optimizer_str == 'rms':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        elif optimizer_str == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            raise NotImplementedError("Does not support %s optimizer" % optimizer_str)
        grads = optimizer.compute_gradients(loss, model.trainable_variables)

        # grads = [(grad, var) if 'predict_weight' not in var.name else (tf.clip_by_value(grad, -1., 1.), var) for
        #              grad, var in grads]
        # grads = [(tf.clip_by_value(grad, -1., 1.), var) if 
        #                                                    'word_title_send_weight' in var.name 
        #                                                 or 'word_title_recieve_weight' in var.name 
        #                                                 or 'word_des_send_weight' in var.name
        #                                                 or 'word_des_recieve_weight' in var.name
        #                                                 or 'ent_title_send_weight' in var.name
        #                                                 or 'ent_title_recieve_weight' in var.name
        #                                                 or 'ent_des_send_weight' in var.name
        #                                                 or 'ent_des_recieve_weight' in var.name
        #                                                 or 'predict_weight' in var.name
        #                                                 #    'project_matrix_title_layer_one' in var.name
        #                                                 # or 'project_matrix_des_layer_one' in var.name
        #                                                 # or 'project_matrix_title_layer_two' in var.name
        #                                                 # or 'project_matrix_des_layer_two' in var.name
        #                             else (grad, var) for grad, var in grads]

        op_train = optimizer.apply_gradients(grads)

        return predict_head_h_input, predict_head_r_input, predict_head_t_input, \
               predict_tail_h_input, predict_tail_r_input, predict_tail_t_input, \
               loss, op_train


def test_ops(model):
    with tf.device('/gpu:1'):

        test_head_h_input = tf.placeholder(tf.int32, [None, None])
        test_head_r_input = tf.placeholder(tf.int32, [None, 1])
        test_head_t_input = tf.placeholder(tf.int32, [None, 1])

        test_tail_h_input = tf.placeholder(tf.int32, [None, 1])
        test_tail_r_input = tf.placeholder(tf.int32, [None, 1])
        test_tail_t_input = tf.placeholder(tf.int32, [None, None])

        ent_title_embedding_input = tf.placeholder(tf.float32, [model.n_entity + 1, model.embed_dim])
        ent_des_embedding_input = tf.placeholder(tf.float32, [model.n_entity + 1, model.embed_dim])

        predict_head, predict_tail = model.test([
                      test_head_h_input, test_head_r_input, test_head_t_input,
                      test_tail_h_input, test_tail_r_input, test_tail_t_input,
                      ent_title_embedding_input, ent_des_embedding_input])

    return test_head_h_input, test_head_r_input, test_head_t_input, \
           test_tail_h_input, test_tail_r_input, test_tail_t_input, \
           ent_title_embedding_input, ent_des_embedding_input, \
           predict_head, predict_tail


def worker_func(in_queue, out_queue, r_t, r_h, hr_t, tr_h):
    while True:
        dat = in_queue.get()
        if dat is None:
            in_queue.task_done()
            continue
        testing_data, score, pos, Tail_Or_Head = dat
        out_queue.put(test_evaluation(testing_data, score, pos, Tail_Or_Head, r_t, r_h, hr_t, tr_h))
        in_queue.task_done()


def data_generator_func(in_queue, out_queue, r_h, r_t, tr_h, hr_t, n_entity, neg_num, 
                  n_close_ent, entity_title_wordid, entity_des_wordid, word_emb,
                  word_title_entityid, word_des_entityid):
    while True:
        dat = in_queue.get()
        if dat is None:
            break
        htr = dat

        predict_head_h = list()
        predict_head_r = list()
        predict_head_t = list()
        predict_tail_h = list()
        predict_tail_r = list()
        predict_tail_t = list()

        for idx in range(htr.shape[0]):
            neg_entities = []
            if np.random.uniform(-1, 1) > 0:  # t r predict h
                neg_entities.append(htr[idx, 0])
                while len(neg_entities) <= neg_num:
                    neg_entity = np.random.randint(0, n_close_ent - 1)
                    if neg_entity not in neg_entities and neg_entity not in tr_h[htr[idx, 1]][htr[idx, 2]]:
                        neg_entities.append(neg_entity)

                predict_head_h.append(neg_entities)
                predict_head_r.append(htr[idx, 2])
                predict_head_t.append(htr[idx, 1])

            else:
                neg_entities.append(htr[idx, 1])
                while len(neg_entities) <= neg_num:
                    neg_entity = np.random.randint(0, n_close_ent - 1)
                    if neg_entity not in neg_entities and neg_entity not in hr_t[htr[idx, 0]][htr[idx, 2]]:
                        neg_entities.append(neg_entity)

                predict_tail_h.append(htr[idx, 0])
                predict_tail_r.append(htr[idx, 2])
                predict_tail_t.append(neg_entities)

        out_queue.put((
                       dat,
                       np.asarray(predict_head_h), np.expand_dims(np.asarray(predict_head_r), axis = -1), np.expand_dims(np.asarray(predict_head_t), axis = -1),
                       np.expand_dims(np.asarray(predict_tail_h), axis = -1), np.expand_dims(np.asarray(predict_tail_r), axis = -1), np.asarray(predict_tail_t),
                       ))


def test_evaluation(testing_data, scores, pos, Tail_Or_Head, r_t, r_h, hr_t, tr_h):
    mean_rank_h = list()
    mean_rank_t = list()
    filtered_mean_rank_h = list()
    filtered_mean_rank_t = list()
    if Tail_Or_Head:
        for i in range(len(scores)):
            score = scores[i]
            # t = testing_data[i][1]
            pos_score = score[pos[i]]
            rank_pos = 1
            for j in range(len(score)):
                if j > pos[i]:
                    continue
                # if j in hr_t[testing_data[i,0]][testing_data[i,2]]:
                #     continue
                # if j not in r_t[testing_data[i,2]]:
                #     continue
                if j == pos[i]:
                    continue
                if score[j] <= pos_score:
                    continue
                rank_pos += 1
            filtered_mean_rank_t.append(rank_pos)
    else:
        for i in range(len(scores)):
            score = scores[i]
            # h = testing_data[i][0]
            pos_score = score[pos[i]]
            rank_pos = 1
            for j in range(len(score)):
                if j > pos[i]:
                    continue
                # if j in tr_h[testing_data[i,1]][testing_data[i,2]]:
                #     continue
                # if j not in r_h[testing_data[i,2]]:
                #     continue
                if j == pos[i]:
                    continue
                if score[j] <= pos_score:
                    continue
                rank_pos += 1
            filtered_mean_rank_h.append(rank_pos)
        # filter_rule = r_h
    # print (filtered_mean_rank_t)

    return (mean_rank_h, filtered_mean_rank_h), (mean_rank_t, filtered_mean_rank_t)


def main(_):
    parser = argparse.ArgumentParser(description='ConMask.')
    parser.add_argument('--data', dest='data_dir', type=str, help="Data folder", default='./data/FB15k/')
    parser.add_argument('--lr', dest='lr', type=float, help="Learning rate", default=0.01)
    parser.add_argument("--dim", dest='dim', type=int, help="Embedding dimension", default=200)
    parser.add_argument("--batch", dest='batch', type=int, help="Batch size", default=100)
    parser.add_argument("--worker", dest='n_worker', type=int, help="Evaluation worker", default=3)
    parser.add_argument("--generator", dest='n_generator', type=int, help="Data generator", default=10)
    parser.add_argument("--eval_batch", dest="eval_batch", type=int, help="Evaluation batch size", default=500)
    parser.add_argument("--save_dir", dest='save_dir', type=str, help="Model path", default='GCN+FM/')
    parser.add_argument("--load_model", dest='load_model', type=str, help="Model file", default="")
    parser.add_argument("--save_per", dest='save_per', type=int, help="Save per x iteration", default=10)
    parser.add_argument("--eval_per", dest='eval_per', type=int, help="Evaluate every x iteration", default=1)
    parser.add_argument("--max_iter", dest='max_iter', type=int, help="Max iteration", default=100)
    parser.add_argument("--summary_dir", dest='summary_dir', type=str, help="summary directory",
                        default='./ConMask_summary/')
    parser.add_argument("--keep", dest='drop_out', type=float, help="Keep prob (1.0 keep all, 0. drop all)",
                        default=0.5)
    parser.add_argument("--optimizer", dest='optimizer', type=str, help="Optimizer", default='adam')
    parser.add_argument("--prefix", dest='prefix', type=str, help="model_prefix", default='DEFAULT')
    parser.add_argument("--loss_weight", dest='loss_weight', type=float, help="Weight on parameter loss", default=1e-5)
    # parser.add_argument("--neg_weight", dest='neg_weight', type=float, help="Sampling weight on negative examples",
    #                     default=0.5)
    parser.add_argument("--neg_num", dest='neg_num', type=int, help="",
                        default=1)
    parser.add_argument("--entity_word_title_len", dest='entity_word_title_len', type=int, help="entity_word_title_len",default=16)
    parser.add_argument("--entity_word_des_len", dest='entity_word_des_len', type=int, help="entity_word_des_len",default=128)
    parser.add_argument("--word_entity_title_len", dest='word_entity_title_len', type=int, help="word_entity_title_len",default=8)
    parser.add_argument("--word_entity_des_len", dest='word_entity_des_len', type=int, help="word_entity_des_len",default=32)
    parser.add_argument("--seed", dest='seed', type=int, help="seed",default=1345)
    parser.add_argument("--scale_1", dest='scale_1', type=int, help="scale_1",default=10)
    parser.add_argument("--scale_2", dest='scale_2', type=int, help="scale_2",default=10)

    args = parser.parse_args()

    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    model = ConMask(args.data_dir, embed_dim=args.dim,
                  dropout=args.drop_out,
                  entity_word_title_len = args.entity_word_title_len, 
                  entity_word_des_len = args.entity_word_des_len,
                  word_entity_title_len = args.word_entity_title_len, 
                  word_entity_des_len = args.word_entity_des_len,
                  neg_num = args.neg_num,
                  scale_1 = args.scale_1,
                  scale_2 = args.scale_2)

    # train_hrt_input, train_hrt_weight, train_trh_input, train_trh_weight, \
    predict_head_h_input, predict_head_r_input, predict_head_t_input, \
    predict_tail_h_input, predict_tail_r_input, predict_tail_t_input, \
    train_loss, train_op = train_ops(model, learning_rate=args.lr,
                                     optimizer_str=args.optimizer,
                                     regularizer_weight=args.loss_weight, 
                                     neg_num = args.neg_num)
    # test_input, \
    # test_head_title_avg_input, test_head_des_avg_input, test_tail_title_avg_input, test_tail_des_avg_input, test_rel_title_avg_input,\
    test_head_h_input, test_head_r_input, test_head_t_input, \
    test_tail_h_input, test_tail_r_input, test_tail_t_input, \
    ent_title_embedding_input, ent_des_embedding_input, \
    predict_head_score, predict_tail_score = test_ops(model)
    
    config = tf.ConfigProto(allow_soft_placement=True, device_count={"CPU":24})
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver()
        iter_offset = 0
        if args.load_model is not None and os.path.exists(args.load_model):
        # if args.load_model is not None:
            saver.restore(session, args.load_model)
            iter_offset = int(args.load_model.split('.')[-2].split('_')[-1]) + 1
            print("Load model from %s, iteration %d restored." % (args.load_model, iter_offset))

        total_inst = model.n_train
        # training data generator
        raw_training_data_queue = Queue()
        training_data_queue = Queue()
        data_generators = list()
        for i in range(args.n_generator):
            data_generators.append(Process(target=data_generator_func, args=(
                raw_training_data_queue, training_data_queue, model.r_h, model.r_t, model.train_tr_h, model.train_hr_t, model.n_entity, args.neg_num, 
                model.n_close_ent, model.entity_title_wordid, model.entity_des_wordid, model.word_emb,
                model.word_title_entityid, model.word_des_entityid)))
            data_generators[-1].start()


        data_evaluators = list()
        evaluation_queue = JoinableQueue()
        result_queue = Queue()
        for i in range(args.n_worker):
            data_evaluators.append(Process(target=worker_func, args=(evaluation_queue, result_queue, model.r_t, model.r_h, model.hr_t, model.tr_h)))
            # worker.start()
            data_evaluators[-1].start()
        ######################################################################################

        ######################################################################################
        # print (session.run(model.word_embedding[129156]))
        for n_iter in range(iter_offset, args.max_iter):
            start_time = timeit.default_timer()

            # sess.run()

            accu_loss = 0.
            # accu_loss = 0.
            ninst = 0

            print("initializing raw training data...")
            nbatches_count = 0

            for dat in model.raw_training_data(batch_size=args.batch):
                raw_training_data_queue.put(dat)
                nbatches_count += 1
            print("raw training data initialized.")

            while nbatches_count > 0:
                nbatches_count -= 1

                # hr_tlist, hr_tweight, tr_hlist, tr_hweight, \
                train_data, \
                predict_head_h, predict_head_r, predict_head_t, \
                predict_tail_h, predict_tail_r, predict_tail_t = training_data_queue.get()
                # print ('==============================')
                # print (np.shape(predict_head_h))
                # print (np.shape(predict_tail_h))

                l, _ = session.run(
                    [train_loss, train_op], {
                                             predict_head_h_input: predict_head_h,
                                             predict_head_r_input: predict_head_r,
                                             predict_head_t_input: predict_head_t,
                                             predict_tail_h_input: predict_tail_h,
                                             predict_tail_r_input: predict_tail_r,
                                             predict_tail_t_input: predict_tail_t
                                             })
                # print (all_entity_group)

                accu_loss += l

                ninst += len(predict_head_r) + len(predict_tail_r)

                # print (np.shape(predict_head_h_title))
                # print (np.shape(predict_tail_h_title))
                if ninst % (5000) is not None:
                    print(
                        '[%d sec](%d/%d) : %.2f -- loss: %.5f ' % (
                            timeit.default_timer() - start_time, ninst, total_inst, float(ninst) / total_inst,
                            l / (np.shape(predict_head_r)[0] + np.shape(predict_tail_r)[0])))
            print("")
            print("iter %d avg loss %.5f, time %.3f" % (n_iter, 
                accu_loss / ninst, timeit.default_timer() - start_time))

            if n_iter % args.save_per == 0 or n_iter == args.max_iter - 1:
                save_path = saver.save(session,
                                       os.path.join(args.save_dir,
                                                    "ConMask_" + str(args.prefix) + "_" + str(n_iter) + ".ckpt"))
                print("Model saved at %s" % save_path)

            if n_iter % args.eval_per == 0 or n_iter == args.max_iter - 1:

                for data_func, test_type in zip([model.testing_data_open_head_test_tail, model.testing_data_open_tail_test_head], ['tail', 'head']):

                    accu_mean_rank_h = list()
                    accu_mean_rank_t = list()
                    accu_filtered_mean_rank_h = list()
                    accu_filtered_mean_rank_t = list()

                    evaluation_count = 0

                    ent_title_embedding, ent_des_embedding, predict_weight = session.run([model.ent_title_embedding, model.ent_des_embedding, model.predict_weight_output])
                    print (predict_weight)
                    for testing_data in data_func(batch_size=args.eval_batch):
                        h = testing_data[:,0].reshape((np.shape(testing_data)[0], 1))
                        t = testing_data[:,1].reshape((np.shape(testing_data)[0], 1))
                        r = testing_data[:,2].reshape((np.shape(testing_data)[0], 1))
                        # if r not in model.r_h:
                        #     continue
                        # print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                        if test_type == 'tail':
                            # test_tail_close_target = np.asarray(range(model.n_close_ent), dtype = np.int32)
                            # test_tail_close_target = np.tile(test_tail_close_target, (np.shape(testing_data)[0], 1))
                            test_tail = []
                            test_tail_target = []
                            test_tail_target_pos = []
                            max_len = 0
                            for index in range(len(r)):
                                temp_candidate = list(model.r_t[r[index][0]] - model.hr_t[h[index][0]][r[index][0]])
                                temp_candidate.append(t[index][0])
                                if max_len < len(temp_candidate):
                                    max_len = len(temp_candidate)
                                test_tail_target_pos.append(len(temp_candidate) - 1)
                                test_tail.append(temp_candidate)

                            for value in test_tail:
                                value = np.pad(value, (0, max_len - np.shape(value)[0]), 'constant')
                                test_tail_target.append(value)

                            test_tail_target = np.asarray(test_tail_target, dtype = np.int32)

                            tail_score = session.run(predict_tail_score,
                                                           {
                                                            test_tail_h_input: h,
                                                            test_tail_r_input: r,
                                                            test_tail_t_input: test_tail_target,
                                                            ent_title_embedding_input: ent_title_embedding,
                                                            ent_des_embedding_input: ent_des_embedding
                                                           })
                            # print (test_tail_close_target_pos)
                            evaluation_queue.put((testing_data, tail_score, test_tail_target_pos, True))
                            evaluation_count += 1
                        # print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

                        if test_type == 'head':
                            # test_head_close_target = np.asarray(range(model.n_close_ent), dtype = np.int32)
                            # test_head_close_target = np.tile(test_head_close_target, (np.shape(testing_data)[0], 1))

                            test_head = []
                            test_head_target = []
                            test_head_target_pos = []
                            max_len = 0
                            for index in range(len(r)):
                                temp_candidate = list(model.r_h[r[index][0]] - model.tr_h[t[index][0]][r[index][0]])
                                temp_candidate.append(h[index][0])
                                if max_len < len(temp_candidate):
                                    max_len = len(temp_candidate)
                                test_head_target_pos.append(len(temp_candidate) - 1)
                                test_head.append(temp_candidate)
                            for value in test_head:
                                value = np.pad(value, (0, max_len - np.shape(value)[0]), 'constant')
                                test_head_target.append(value)
                            test_head_target = np.asarray(test_head_target, dtype = np.int32)
                            head_score = session.run(predict_head_score,
                                                           {
                                                            test_head_h_input: test_head_target,
                                                            test_head_r_input: r,
                                                            test_head_t_input: t,
                                                            ent_title_embedding_input: ent_title_embedding,
                                                            ent_des_embedding_input: ent_des_embedding
                                                           })
                            evaluation_queue.put((testing_data, head_score, test_head_target_pos, False))
                            evaluation_count += 1

                        
                    for i in range(args.n_worker):
                        evaluation_queue.put(None)
                    print("waiting for worker finishes their work")
                    evaluation_queue.join()
                    print("all worker stopped.")
                    while evaluation_count > 0:
                        evaluation_count -= 1

                        (mrh, fmrh), (mrt, fmrt) = result_queue.get()
                        accu_mean_rank_h += mrh
                        accu_mean_rank_t += mrt
                        accu_filtered_mean_rank_h += fmrh
                        accu_filtered_mean_rank_t += fmrt

                    # print (accu_filtered_mean_rank_h)
                    if test_type == 'tail':
                        print ('\ntail:')
                        print ('Filter Mean Rank:',np.mean(np.asarray(accu_filtered_mean_rank_t)))
                        print ('MRR:',np.mean(np.reciprocal(np.asarray(accu_filtered_mean_rank_t),dtype=np.float32)))
                        print ('filter hit @1:',np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 2))
                        print ('filter hit @3:',np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 4))
                        print ('filter hit @10:',np.mean(np.asarray(accu_filtered_mean_rank_t, dtype=np.int32) < 11))
                    else:
                        print ('\nhead')
                        print ('Filter Mean Rank:',np.mean(np.asarray(accu_filtered_mean_rank_h)))
                        print ('MRR:',np.mean(np.reciprocal(np.asarray(accu_filtered_mean_rank_h),dtype=np.float32)))
                        print ('filter hit @1:',np.mean(np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 2))
                        print ('filter hit @3:',np.mean(np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 4))
                        print ('filter hit @10:',np.mean(np.asarray(accu_filtered_mean_rank_h, dtype=np.int32) < 11))
        for p in data_evaluators:
            p.terminate()
        for p in data_generators:
            p.terminate()


if __name__ == '__main__':
    tf.app.run()
    exit(0)
