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

class Gated_fusion(nn.Module):
    def __init__(self, input_size, out_size=1, dropout=0.2):
        super(Gated_fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, X1, X2):
        emb = torch.cat([X1.unsqueeze(dim=0), X2.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out

class interseted_Matrix(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.inter_matrix = interseted_Matrix()




class HGSL(nn.Module):
    def __init__(self, opt):
        super(HGSL, self).__init__()

        self.hidden_size = opt.d_model
        self.n_node = opt.user_size
        self.dropout = nn.Dropout(opt.dropout)
        self.initial_feature = opt.initialFeatureSize

        self.emb_size = opt.emb_size
        self.hgnn = HGNN(self.initial_feature, self.hidden_size, self.emb_size, dropout = opt.dropout)
        self.text2emb_linear = nn.Linear(self.emb_size, 200)
        self.text2emb_embedding = nn.Embedding(1000, self.hidden_size)
        

        self.user_embedding = nn.Embedding(self.n_node, self.initial_feature)
        self.stru_attention = TransformerBlock(self.hidden_size, n_heads=8)
        self.temp_attention = TransformerBlock(self.hidden_size, n_heads=8)
        

        self.global_cen_embedding = nn.Embedding(600, self.hidden_size)
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


    def forward(self, data_idx, hypergraph_list):
        # print(self.hidden_size)
        # 解包超图数据
        news_centered_graph, user_centered_graph, spread_status = (item for item in hypergraph_list)
        seq, timestamps, user_level, text2emb= (item for item in news_centered_graph)
        # print("timestamps.size():")
        # print(timestamps.size())
        # print(text2emb.float().size())

        useq, user_inf, user_cen = (item for item in user_centered_graph)

        #Global learning全局学习
        hidden = self.dropout(self.user_embedding.weight)
        user_cen = self.global_cen_embedding(user_cen)
        tweet_hidden = hidden + user_cen
        user_hgnn_out = self.hgnn(tweet_hidden, seq, text2emb, useq, data_idx)
        # print("user_hgnn_out_size:")
        # print(user_hgnn_out.size())
        #print(user_hgnn_out.device)

        #Normalize归一化处理
        zero_vec1 = -9e15 * torch.ones_like(seq[data_idx])
        one_vec = torch.ones_like(seq[data_idx], dtype=torch.float)
        nor_input = torch.where(seq[data_idx] > 0, one_vec, zero_vec1)
        nor_input = F.softmax(nor_input, 1)
        att_mask = (seq[data_idx] == Constants.PAD)
        adj_with_fea = F.embedding(seq[data_idx], user_hgnn_out)
        
        
        '''
        print(data_idx.size())
        print("---"*3)
        print("seq[data_idx]_size:")
        print(seq[data_idx].size())
        #print(seq[data_idx].size(), user_hgnn_out.size())

        #Local temporal learning局部时间学习
        print("timestamps[data_idx]:")
        print(timestamps[data_idx])
        print(timestamps.size())
        print("---"*3)
        print(timestamps[0])
        print(timestamps[1])
        print(timestamps[2])
        print("timestamps[data_idx].size():")
        print(timestamps[data_idx].size())
        '''
        global_time = self.local_time_embedding(timestamps[data_idx])
        '''
        print("global_time:")
        print(global_time)
        print(global_time[0])
        print(global_time[1])
        print(global_time[2])
        print("global_time.size():")
        print(global_time.size())
        print("emb_size:")
        print(self.emb_size)
        print("text2emb.float().size():")
        print(text2emb.float().size())
        print(text2emb[data_idx])
        print(text2emb[data_idx].size())
        print(text2emb[data_idx].max())
        
        
        '''
        text2emb = self.text2emb_linear(text2emb.float())
        text2emb = (text2emb - text2emb.min())/(text2emb.max()-text2emb.min())*999
        text2emb = text2emb.int()
        '''print("@@@text2emb.float().size():")
        print(text2emb.float().size())
        print(text2emb[data_idx])
        print(text2emb[data_idx].size())
        print(text2emb[data_idx].max())'''
        # x = text2emb[data_idx]
        # print("x.size():")
        # print(x.size())
        # print(x.max())
        # text2emb_fea = self.text2emb_embedding(x)
        
        

        att_hidden = adj_with_fea + global_time
        # att_hidden = adj_with_fea + global_time+text2emb_fea
        '''
        print("att_hidden.size():")
        print(att_hidden.size())
        print("adj_with_fea.size():")
        print(adj_with_fea.size())
        '''
        
        att_out = self.temp_attention(att_hidden, att_hidden, att_hidden, mask = att_mask )
        

        news_out = torch.einsum("abc,ab->ac", (att_out, nor_input))

        #Concatenate temporal propagation status拼接时间传播状态
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
        news_out = self.fus(news_out, news_out_str)
        output = self.linear(news_out)
        output = F.log_softmax(output, dim=1)
        #print(output)

        return output

'''Learn hypergraphs'''
class HGNN_layer(nn.Module):
    def __init__(self, input_size, output_size, emb_size, dropout=0.5):
        super(HGNN_layer, self).__init__()
        self.dropout = dropout
        self.in_features = input_size
        self.out_features = output_size
        self.weight1 = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.Tensor(self.out_features, self.out_features))
        
        self.weight3 = Parameter(torch.Tensor(100, emb_size))
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_features)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        self.weight1.data.uniform_(-stdv, stdv)
        self.weight2.data.uniform_(-stdv, stdv)
        
        self.weight3.data.uniform_(-stdv, stdv)

    def forward(self, x, seq, text2emb, useq, data_idx):
        
        # print(len(data_idx))
        # print(text2emb[data_idx].size())
        # print(self.weight3.size())
        
        large_matrix_norm = F.normalize(self.weight3, p=2, dim=1)
        # small_matrix_norm = F.normalize(text2emb[data_idx].float(), p=2, dim=1)
        small_matrix_norm = large_matrix_norm 
        
        cosine_similarities = torch.matmul(small_matrix_norm, large_matrix_norm.T)

        average_cosine_similarities = cosine_similarities.mean(dim=1)

        overall_average_similarity = average_cosine_similarities.mean()

        # print("Overall Average Cosine Similarity:", overall_average_similarity.item())
        text_weight = overall_average_similarity.item()
        # temp = torch.mean(torch.sum( text2emb[data_idx] * self.weight3 ))
        
        # print("temp:")
        # print(temp)
        
        x = x.matmul(self.weight1)
        # x = x * text_weight
        adj_with_fea = F.embedding(seq, x)
        zero_vec1 = -9e15 * torch.ones_like(seq)
        one_vec = torch.ones_like(seq, dtype=torch.float)
        nor_input = torch.where(seq > 0, one_vec, zero_vec1)
        nor_input = F.softmax(nor_input, 1)
        
        
        
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
        # node = node * text_weight
        node = F.dropout(node, self.dropout, training=self.training)

        return node

class HGNN(nn.Module):
    def __init__(self, input_size, output_size, emb_size, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.gnn1 = HGNN_layer(input_size, output_size, emb_size, dropout=self.dropout)


    def forward(self, x, seq, text2emb, useq, data_idx):
        node = self.gnn1(x, seq, text2emb, useq, data_idx)
        return node
