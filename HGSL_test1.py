# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:30:16 2021

@author: Ling Sun
"""

import math
#import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
#from layer import HGATLayer
import torch.nn.init as init
import Constants
from torch.nn.parameter import Parameter
from TransformerBlock import TransformerBlock




class CosineSimilarityCalculator(nn.Module):
    def __init__(self):
        super(CosineSimilarityCalculator, self).__init__()

    def forward(self, vector, matrix, method='average'):
        vector = vector.unsqueeze(0)  # 扩展维度以便广播
        dot_product = torch.matmul(matrix, vector.t()).squeeze()
        norm_vector = torch.norm(vector)
        norm_matrix = torch.norm(matrix, dim=1)
        cosine_similarity = dot_product / (norm_vector * norm_matrix)

        return torch.mean(cosine_similarity).item()

class Gated_fusion(nn.Module):
    def __init__(self, input_size, out_size=1, dropout=0.2):
        super(Gated_fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out_size)
        self.linear3 = nn.Linear(input_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)
        init.xavier_normal_(self.linear3.weight)

    def forward(self, X1, X2, TextVector):
        
        TextVector = X2.unsqueeze(dim=0)
        emb = torch.cat([X1.unsqueeze(dim=0), TextVector], dim=0)
        emb = torch.cat([X1.unsqueeze(dim=0), X2.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out

class HGSL(nn.Module):
    def __init__(self, opt):
        super(HGSL, self).__init__()

        self.hidden_size = opt.d_model
        self.n_node = opt.user_size
        self.dropout = nn.Dropout(opt.dropout)
        self.initial_feature = opt.initialFeatureSize
        self.TextVector_size = opt.textVector_size
        
        self.hgnn = HGNN(self.initial_feature, self.hidden_size, self.TextVector_size, dropout = opt.dropout)
        
        

        self.user_embedding = nn.Embedding(self.n_node, self.initial_feature)

        
        self.stru_attention = TransformerBlock(self.hidden_size, n_heads=8)
        self.temp_attention = TransformerBlock(self.hidden_size, n_heads=8)

        self.global_cen_embedding = nn.Embedding(600, self.hidden_size)

        # !@添加Vector_embedding
        self.vectorlen = opt.textVector_size
        self.vectoremb = nn.Embedding(self.vectorlen, self.hidden_size)
        
        self.local_time_embedding = nn.Embedding(5000, self.hidden_size)
        self.cas_pos_embedding = nn.Embedding(50, self.hidden_size)
        self.local_inf_embedding = nn.Embedding(200, self.hidden_size)

        self.weight = Parameter(torch.Tensor(self.hidden_size+2, self.hidden_size+2))
        self.weight2 = Parameter(torch.Tensor(self.hidden_size+2, self.hidden_size+2))
        self.fus = Gated_fusion(self.hidden_size+2)
        self.linear = nn.Linear((self.hidden_size+2), 2)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


    def forward(self, data_idx, hypergraph_list, id2vector):
        id2vectoritem = []
        news_centered_graph, user_centered_graph, spread_status = (item for item in hypergraph_list)
        # print(news_centered_graph)
        seq, timestamps, user_level = (item for item in news_centered_graph)
        useq, user_inf, user_cen = (item for item in user_centered_graph)
        
        # 节点：用文本向量替换随机向量
        # for item in id2vector:
            # print(id2vector[item])
        #     id2vectoritem.append(id2vector[item])
            
        # id2vectoritem = id2vector[2804]
        # print(id2vectoritem.shape)
        # id2vectorembeding = self.vectoremb(id2vectoritem)

        # id2vectoritem = (id2vector[item] for item in id2vector)
        # print(id2vectoritem)
        

        #Global learning
        id2vectorembeding = self.dropout(self.user_embedding.weight)
        
        user_cen = self.global_cen_embedding(user_cen)
        # tweet_hidden = hidden + user_cen
        # hidden = self.dropout(self.user_embedding.weight)
        
        tweet_hidden = id2vectorembeding + user_cen
        user_hgnn_out = self.hgnn(tweet_hidden, seq, useq, self.TextVector_size)
        #print(user_hgnn_out.device)

        #Normalize
        zero_vec1 = -9e15 * torch.ones_like(seq[data_idx])
        one_vec = torch.ones_like(seq[data_idx], dtype=torch.float)
        nor_input = torch.where(seq[data_idx] > 0, one_vec, zero_vec1)
        nor_input = F.softmax(nor_input, 1)
        att_mask = (seq[data_idx] == Constants.PAD)
        adj_with_fea = F.embedding(seq[data_idx], user_hgnn_out)
        #print(seq[data_idx].size(), user_hgnn_out.size())

        #Local temporal learning
        global_time = self.local_time_embedding(timestamps[data_idx])
        att_hidden = adj_with_fea + global_time

        att_out = self.temp_attention(att_hidden, att_hidden, att_hidden, mask = att_mask )
        news_out = torch.einsum("abc,ab->ac", (att_out, nor_input))

        #Concatenate temporal propagation status
        news_out = torch.cat([news_out, spread_status[data_idx][:, 2:]/3600/24], dim=-1)
        news_out = news_out.matmul(self.weight)

        #Local structural learning
        local_inf = self.local_inf_embedding(user_inf[data_idx])
        cas_pos = self.cas_pos_embedding(user_level[data_idx])
        att_hidden_str = adj_with_fea + local_inf + cas_pos

        att_out_str = self.stru_attention(att_hidden_str, att_hidden_str, att_hidden_str, mask=att_mask, pos = False)
        news_out_str = torch.einsum("abc,ab->ac", (att_out_str, nor_input))

        # Concatenate structural propagation status
        news_out_str = torch.cat([news_out_str, spread_status[data_idx][:,:2]], dim=-1)
        news_out_str = news_out_str.matmul(self.weight2)

        #Gated fusion
        news_out = self.fus(news_out, news_out_str, id2vector)
        output = self.linear(news_out)
        output = F.log_softmax(output, dim=1)
        #print(output)

        return output

'''Learn hypergraphs'''
class HGNN_layer(nn.Module):
    def __init__(self, input_size, output_size, TextVector_size, dropout=0.5):
        super(HGNN_layer, self).__init__()
        self.dropout = dropout
        self.in_features = input_size
        self.out_features = output_size
        self.TextVector_size = TextVector_size
        self.weight1 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.Tensor(self.out_features, self.out_features))

        self.weight3 = Parameter(torch.Tensor(self.TextVector_size, self.TextVector_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        
        self.weight3.data.uniform_(-stdv, stdv)

    def forward(self, x, seq, useq, TextVector):

        
        zero_vec1 = -9e15 * torch.ones_like(seq)
        TextVector = self.weight3[0]
        one_vec = torch.ones_like(seq, dtype=torch.float)
        nor_input = torch.where(seq > 0, one_vec, zero_vec1)
        nor_input = F.softmax(nor_input, 1)
        
        # !@Calculate weights
        inter_nw_set = []
        dot_product = torch.matmul(self.weight3, TextVector).squeeze()
        norm_vector = torch.norm(TextVector)
        norm_matrix = torch.norm(self.weight3, dim=1)
        cosine_similarity = dot_product / (norm_vector * norm_matrix)
        inter_nw = torch.mean(cosine_similarity).item()
        
        inter_nw_set.append(inter_nw)
        
        # print(inter_w)
        x = x.matmul(self.weight1)
        x = x * inter_nw
        adj_with_fea = F.embedding(seq, x)
        

        edge = torch.einsum("abc,ab->ac", (adj_with_fea, nor_input))
        edge = F.dropout(edge, self.dropout, training=self.training)
        edge = F.relu(edge, inplace=False)
        e1 = edge.matmul(self.weight2)
        edge_adj_with_fea = F.embedding(useq, e1)

        
        
        
        zero_vec1 = -9e15 * torch.ones_like(useq)
        one_vec = torch.ones_like(useq, dtype=torch.float)
        u_nor_input = torch.where(useq > 0, one_vec, zero_vec1)
        u_nor_input = F.softmax(u_nor_input, 1)
        node = torch.einsum("abc,ab->ac", (edge_adj_with_fea, u_nor_input))

        
        node = F.dropout(node, self.dropout, training=self.training)

        return node

class HGNN(nn.Module):
    def __init__(self, input_size, output_size, TextVector_size, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.gnn1 = HGNN_layer(input_size, output_size, TextVector_size, dropout=self.dropout)


    def forward(self, x, seq, TextVector, useq):
        node = self.gnn1(x, seq, TextVector, useq)
        return node
