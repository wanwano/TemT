import torch.nn as nn
import torch.nn.functional as F
import dgl

from utils_ct import *

class MeanAggregator(nn.Module):
    def __init__(self, h_dim, embd_rank, dropout, seq_len, args, gcn=False):
        super(MeanAggregator, self).__init__()
        self.h_dim = h_dim
        self.embd_rank = embd_rank
        self.dropout = nn.Dropout(dropout)
        # args.max_hist_len
        self.seq_len = seq_len
        self.gcn = gcn
        if gcn:
            self.gcn_layer = nn.Linear(h_dim, h_dim)

    def forward(self, s_hist, s, r, o, t, raw, ent_embeds, rel_embeds, s_hist_dt):

        s_idx, s_len_non_zero, s_tem, r_tem, embeds_stack, len_s, embeds_split, s_hist_dt_sorted_truncated = \
            get_sorted_s_r_embed(s_hist, s, r, t, ent_embeds, s_hist_dt)

        # To get mean vector at each time
        curr = 0
        rows = []
        cols = []

        # lens stores the number of neighbors of each timestamp for all subjects
        for i, leng in enumerate(len_s):
            rows.extend([i] * leng)
            cols.extend(list(range(curr, curr + leng)))
            curr += leng

        rows = to_device(torch.LongTensor(rows))

        cols = to_device(torch.LongTensor(cols))

        idxes = torch.stack([rows, cols], dim=0)

        mask_tensor = to_device(torch.sparse.FloatTensor(idxes, torch.ones(len(rows), device=idxes.device)))

        embeds_sum = torch.sparse.mm(mask_tensor, embeds_stack)

        embeds_mean = embeds_sum / to_device(torch.Tensor(len_s)).view(-1, 1)

        if self.gcn:
            embeds_mean = self.gcn_layer(embeds_mean)
            embeds_mean = F.relu(embeds_mean)

        # split embds_mean to each subjects with non_zero history tuple
        embeds_split = torch.split(embeds_mean, s_len_non_zero.tolist())

        # cat aggregation, subject embedding, relation embedding together.
        s_embed_seq_tensor = to_device(
            torch.zeros(len(s_len_non_zero), self.seq_len, self.h_dim + 2 * self.embd_rank))

        s_hist_dt_seq_tensor = to_device(torch.zeros(len(s_len_non_zero), self.seq_len))

        for i, dts in enumerate(s_hist_dt_sorted_truncated):
            s_hist_dt_seq_tensor[i, torch.arange(len(dts))] = to_device(
                torch.tensor(dts, dtype=s_hist_dt_seq_tensor.dtype))

        for i, embeds in enumerate(embeds_split):
            s_embed_seq_tensor[i, torch.arange(len(embeds)), :] = torch.cat(
                (embeds, ent_embeds(s_tem[i]).repeat(len(embeds), 1),
                 rel_embeds[r_tem[i]].repeat(len(embeds), 1)),
                  dim=1)

        s_embed_seq_tensor = self.dropout(s_embed_seq_tensor)
        return s_embed_seq_tensor, s_hist_dt_seq_tensor, s_idx, len(s_len_non_zero)

class RAW(nn.Module):
    def __init__(self, args, graph_dict):
        super(RAW, self).__init__()
        self.graph_dict = graph_dict
        self.h_dim = args.n_hidden
        self.path_num = args.path_num
        self.gru_num_layers = 1
        self.gruDFS_layer = nn.GRU(input_size=self.h_dim, hidden_size=self.h_dim, num_layers=self.gru_num_layers, batch_first=True)
        self.gruBFS_layer = nn.GRU(input_size=self.h_dim, hidden_size=self.h_dim, num_layers=self.gru_num_layers, batch_first=True)

        self.att_heads_num = 2
        self.att_heads = nn.Parameter(torch.zeros(self.att_heads_num, self.path_num, self.h_dim))
        nn.init.xavier_uniform_(self.att_heads)

        self.LRelu = nn.LeakyReLU(0.1)
        self.softMax = nn.Softmax(-2)
        self.relu = nn.ReLU()

        self.linear = nn.Linear(2*self.att_heads_num*self.h_dim, self.h_dim)


    def getIndex(self, data):
        """
            Get the node to index mapping.

            Returns:
                dict{node:index}
        """
        dict = {}
        for i in range(data.shape[0]):
            dict[data[i][0].item()] = i
        return dict

    def init_gru_hidden(self, batch_size):
        """
        Initialize the hidden state.

        Returns:
            hidden
        """
        h0 = to_device(torch.zeros(self.gru_num_layers, batch_size, self.h_dim))
        return h0

    def forward(self, v, t, ent_embeds, l=3):
        nodes_seq = []
        for i in range(v.shape[0]):
            # Get the node to index mapping
            dict = self.graph_dict[int(t[i])].ids
            node_seq = []
            for j in range(self.path_num*2):
                if j < self.path_num:
                    # DFS
                    p, q = 10, 0.1
                else:
                    # BFS
                    p, q = 0.1, 10
                seq = dgl.sampling.node2vec_random_walk(self.graph_dict[int(t[i])], torch.tensor(dict[int(v[i])]), p, q, l)
                # Get the index to node mapping
                for k in range(seq.shape[1]):
                    seq[0][k] = self.graph_dict[int(t[i])].nodes[seq[0][k]].data['id']
                node_seq.append(seq)
            nodes_seq.append(torch.stack(node_seq).squeeze(dim=1))

        input = torch.stack(nodes_seq).permute(1, 0, 2)

        input_dfs = input[:self.path_num, :, :]
        input_bfs = input[self.path_num:, :, :]

        input_dfs = ent_embeds.weight[input_dfs]
        input_bfs = ent_embeds.weight[input_bfs]

        input_dfs = input_dfs.reshape(input_dfs.shape[0] * input_dfs.shape[1], input_dfs.shape[2], input_dfs.shape[3])
        input_bfs = input_bfs.reshape(input_bfs.shape[0] * input_bfs.shape[1], input_bfs.shape[2], input_bfs.shape[3])

        h0_dfs = self.init_gru_hidden(v.shape[0] * self.path_num)
        output_dfs, _ = self.gruDFS_layer(input_dfs, h0_dfs)
        h0_bfs = self.init_gru_hidden(v.shape[0] * self.path_num)
        output_bfs, _ = self.gruBFS_layer(input_bfs, h0_bfs)

        output_dfs = output_dfs[:, -1, :]
        output_bfs = output_bfs[:, -1, :]

        output_dfs = output_dfs.reshape(-1, self.path_num, self.h_dim)
        output_bfs = output_bfs.reshape(-1, self.path_num, self.h_dim)

        att_dfs = []
        att_bfs = []

        for i in range(self.att_heads_num):
            e_dfs = self.LRelu(self.att_heads[i]*output_dfs)
            alpha_dfs = self.softMax(e_dfs)

            h_dfs = self.relu(torch.sum(alpha_dfs * output_dfs, dim=1))
            att_dfs.append(h_dfs)

            e_bfs = self.LRelu(self.att_heads[i] * output_bfs)
            alpha_bfs = self.softMax(e_bfs)
            h_bfs = self.relu(torch.sum(alpha_bfs * output_bfs, dim=1))
            att_bfs.append(h_bfs)

        h_att_dfs = torch.concat(att_dfs, dim=-1)
        h_att_bfs = torch.concat(att_bfs, dim=-1)
        res = torch.concat([h_att_dfs, h_att_bfs], dim=-1)
        res = nn.functional.leaky_relu(res)
        res = self.linear(res)

        return res


