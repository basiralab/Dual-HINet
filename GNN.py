import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import NNConv
import torch_geometric.utils as utils

import config
import helper

from torch.nn import init


# GCN basic operation

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cpu())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cpu())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)

        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)

        return y


class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, num_layers, view_dim,
                 concat=True, bn=True, args=None):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        self.bias = True
        if args is not None:
            self.bias = args.bias
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim, hidden_dim, embedding_dim, view_dim, num_layers)

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, view_dim, num_layers):
        # For only 3 layers:
        seq_nn = Sequential(Linear(view_dim, input_dim * hidden_dim), ReLU())
        conv_first = NNConv(input_dim, hidden_dim, seq_nn, aggr='mean')
        seq_nn = Sequential(Linear(view_dim, hidden_dim * hidden_dim), ReLU())
        conv_block = nn.ModuleList([NNConv(hidden_dim, hidden_dim, seq_nn, aggr='mean') for i in range(num_layers - 2)])
        seq_nn = Sequential(Linear(view_dim, hidden_dim * embedding_dim), ReLU())
        conv_last = NNConv(hidden_dim, embedding_dim, seq_nn, aggr='mean')
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
                pred_layers.append(nn.Linear(pred_dim, label_dim))  # Put into loop? FSD 02.06.22
            pred_model = nn.Sequential(*pred_layers)

        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        ''' For each num_nodes in batch_num_nodes, the first num_nodes entries of the 
        corresponding column are 1's, and the rest are 0's (to be masked out).
        Dimension of mask: [batch_size x max_nodes x 1]
        '''
        # masks
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, :int(batch_num_nodes[i])] = mask  # (there was no int(...)) FSD 02.01.22
        return out_tensor.unsqueeze(2).cpu()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cpu()
        return bn_module(x)

    def gcn_forward(self, x, edge_index, edge_attr, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, edge_index, edge_attr)

        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]

        for i in range(len(conv_block)):
            x = conv_block[i](x, edge_index, edge_attr)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)

        x = conv_last(x, edge_index, edge_attr)
        x_all.append(x)

        x_tensor = torch.cat(x_all, dim=1)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask

        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        # conv
        x = self.conv_first(x, adj)

        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x, adj)

        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)

        return ypred


class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, num_layers,
                 assign_hidden_dim, view_dim, not_ablated=True, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1,
                 concat=True, bn=True, assign_input_dim=-1, args=None):  # FSD 02.06.22
        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, num_layers,
                                                     view_dim, concat=concat, bn=bn, args=args)
        self.num_pooling = num_pooling
        self.assign_ent = True
        self.not_ablated = not_ablated

        # GC
        self.conv_first_after_pool = nn.ModuleList()
        self.conv_block_after_pool = nn.ModuleList()
        self.conv_last_after_pool = nn.ModuleList()
        for i in range(num_pooling):
            # use self to register the modules in self.modules()
            conv_first2, conv_block2, conv_last2 = self.build_conv_layers(
                self.pred_input_dim, hidden_dim, embedding_dim, view_dim, num_layers)
            self.conv_first_after_pool.append(conv_first2)
            self.conv_block_after_pool.append(conv_block2)
            self.conv_last_after_pool.append(conv_last2)

        # assignment
        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = nn.ModuleList()
        self.assign_conv_block_modules = nn.ModuleList()
        self.assign_conv_last_modules = nn.ModuleList()
        self.assign_pred_modules = nn.ModuleList()
        assign_dim = round(max_num_nodes * assign_ratio)

        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            assign_conv_first, assign_conv_block, assign_conv_last = self.build_conv_layers(
                assign_input_dim, assign_hidden_dim, assign_dim, view_dim, assign_num_layers)
            assign_pred_input_dim = assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim
            assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            # next pooling layer
            assign_input_dim = self.pred_input_dim
            assign_dim = round(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(assign_conv_first)
            self.assign_conv_block_modules.append(assign_conv_block)
            self.assign_conv_last_modules.append(assign_conv_last)
            self.assign_pred_modules.append(assign_pred)

        n_Encoded_Embeddings = (num_pooling + 1 * not_ablated) * (
                    hidden_dim * (num_layers - 1) + embedding_dim) if concat else (
                    hidden_dim * (num_layers - 1) + embedding_dim)
        self.decode = nn.Linear(n_Encoded_Embeddings, config.N_Nodes)  # n_Encoded_Embeddings -> n_ROIs. FSD 01.31.22

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, data, batch_num_nodes=None):
        x, edge_attr, edge_index, adj = data.x, data.edge_attr, data.edge_index, data.con_mat
        x_a = x  # (For the mean time) FSD 02.06.22

        # mask
        max_num_nodes = config.N_Nodes  # adj.size()[1] FSD 02.06.22
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []

        # GCN(X, A) = Z
        embedding_tensor = self.gcn_forward(x, edge_index, edge_attr,
                                            self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        if (self.not_ablated):
            out, _ = torch.max(embedding_tensor, dim=0)
            if self.num_aggs == 2:
                out = torch.sum(embedding_tensor, dim=0)
            out_all.append(out)

        S_list = []

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            # softmax(GCN(X, A)) = S
            self.assign_tensor = self.gcn_forward(x_a, edge_index, edge_attr,
                                                  self.assign_conv_first_modules[i], self.assign_conv_block_modules[i],
                                                  self.assign_conv_last_modules[i],
                                                  embedding_mask)
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred_modules[i](self.assign_tensor))

            S_list.append(self.assign_tensor)

            # update pooled features and adj matrix
            # S * Z = X
            x = torch.matmul(torch.transpose(self.assign_tensor, 0, 1), embedding_tensor)
            # S.T * A * S = A
            if i > 0:
                adj = utils.to_dense_adj(edge_index, edge_attr=edge_attr)
                adj = torch.squeeze(adj)
            repeated_S = self.assign_tensor.repeat(adj.size(2), 1, 1)
            adj = torch.permute(adj, (2, 0, 1))
            adj = torch.matmul(torch.transpose(repeated_S, 1, 2), adj) @ repeated_S

            for j in range(adj.size(0)):
                adj[j, :, :].fill_diagonal_(0)

            adj = torch.permute(adj, (1, 2, 0))
            edge_index, edge_attr = helper.dense_to_sparse(adj)
            x_a = x  # FSD 02.06.22

            embedding_tensor = self.gcn_forward(x, edge_index, edge_attr,
                                                self.conv_first_after_pool[i], self.conv_block_after_pool[i],
                                                self.conv_last_after_pool[i])

            out, _ = torch.max(embedding_tensor, dim=0)

            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(embedding_tensor, dim=0)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=-1)
        else:
            output = out

        D = self.decode(output)  # N_encoded_embeddings -> N_ROIs

        repeated_D = D.repeat(max_num_nodes, 1)
        diff = torch.abs(repeated_D - torch.transpose(repeated_D, 0, 1))
        cbt = diff

        return cbt, S_list
