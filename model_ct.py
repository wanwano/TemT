import torch.nn as nn
import torch
import math

from Aggregator_ct import MeanAggregator,RAW
from utils_ct import *
from settings import settings

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(nn.functional.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        output, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn

class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = nn.functional.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x

class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """
    def __init__(
            self,
            num_types, d_model, d_rnn, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model*3) for i in range(d_model*3)],
            device=torch.device('cuda'))

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model*3, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

        self.output_g = nn.Linear(d_model*3, d_model)

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(self.d_model*3, d_rnn)

    def temporal_enc(self, time, non_pad_mask):
        """
            temporal encoding
            return real_batch * his_len * d_model
        """

        result = time.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask

    def get_attn_key_pad_mask(self, seq_k, seq_q):
        """
            For masking out the padding part of key sequence.
            return real_batch*his_len*his_len
        """

        # expand to fit the shape of key query attention matrix
        len_q = seq_q.size(1)
        padding_mask = seq_k.eq(0)
        padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
        return padding_mask

    def get_subsequent_mask(self, seq):
        """
            For masking out the subsequent info, i.e., masked self-attention.
            return triu: real_batch*his_len*his_len
        """

        sz_b, len_s = seq.size()
        subsequent_mask = torch.triu(
            torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
        return subsequent_mask

    def forward(self, event_type, event_time, non_pad_mask, sort_idx):
        """ Encode event sequences via masked self-attention. """

        non_pad_mask_temp = non_pad_mask.reshape(non_pad_mask.shape[0], -1)
        event_time = event_time.cumsum(1)
        event_time = event_time*non_pad_mask_temp

        # prepare attention masks
        # slf_attn_mask is where we cannot look, i.e., the future and the padding
        slf_attn_mask_subseq = self.get_subsequent_mask(event_time)
        slf_attn_mask_keypad = self.get_attn_key_pad_mask(seq_k=event_time, seq_q=event_time)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        tem_enc = self.temporal_enc(event_time, non_pad_mask)

        enc_output = event_type
        for enc_layer in self.layer_stack:

            enc_output += tem_enc
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        #optional
        enc_output = self.rnn(enc_output, non_pad_mask)

        enc_output = enc_output[:, 0, :]
        enc_output = self.output_g(enc_output)

        _, ori_idx = sort_idx.sort()

        enc_output = torch.cat((enc_output, to_device(torch.zeros(len(sort_idx) - enc_output.shape[0],
                                                                            self.d_model))), dim=0)[ori_idx]
        return enc_output

class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out

class TemT(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self, args,
            num_e, num_r):
        super().__init__()

        self.h_dim = args.n_hidden

        self.d_rnn = args.d_rnn

        self.d_inner = args.d_inner

        self.n_layers = args.n_layers

        self.n_head = args.n_head

        self.d_k = args.d_k

        self.d_v = args.d_v

        self.args = args

        self.encoder_s = Encoder(
            num_types=num_e,
            d_model=self.h_dim,
            d_rnn=self.d_rnn,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            n_head=self.n_head,
            d_k=self.d_k,
            d_v=self.d_v,
            dropout=args.dropout
        )

        self.encoder_o = Encoder(
            num_types=num_e,
            d_model=self.h_dim,
            d_rnn=self.d_rnn,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            n_head=self.n_head,
            d_k=self.d_k,
            d_v=self.d_v,
            dropout=args.dropout
        )

        self.gelu = GELU()

        self.num_e = num_e

        self.num_rels = num_r

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(self.h_dim, num_e)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # event type embedding
        self.ent_embeds = nn.Embedding(num_e + 1, args.embd_rank, padding_idx=0)

        # relation type embedding
        self.rel_embeds = nn.Embedding(num_r * 2, args.embd_rank, padding_idx=0)

        self.aggregator_s = MeanAggregator(self.h_dim, args.embd_rank, args.dropout, args.max_hist_len, args,gcn=False)
        self.aggregator_o = self.aggregator_s

        self.alpha_t = nn.Parameter(torch.zeros(self.num_e + 1, 1))

        nn.init.xavier_uniform_(self.alpha_t)

        self.base_t = nn.Parameter(torch.zeros(self.num_e + 1, 1))
        nn.init.xavier_uniform_(self.base_t)

        self.linear_h = nn.Linear(self.h_dim, args.embd_rank, bias=False)

        self.linear_inten_layer = nn.Linear(self.h_dim + 2 * args.embd_rank, args.embd_rank, bias=False)

        self.Softplus = nn.Softplus(beta=args.softrelu_scale)

        self.criterion_time = nn.CrossEntropyLoss()
        self.criterion_link = nn.CrossEntropyLoss()

        self.dropout = nn.Dropout(args.dropout)

        self.graph_dict = None

        self.raw_s_encoder = None
        self.raw_o_encoder = None

        self.start_layer = nn.Sequential(
            nn.Linear(self.h_dim + 3 * args.embd_rank, args.embd_rank, bias=True),
            self.gelu
        )

        self.converge_layer = nn.Sequential(
            nn.Linear(self.h_dim + 3 * args.embd_rank, args.embd_rank, bias=True),
            self.gelu
        )

        self.decay_layer = nn.Sequential(
            nn.Linear(self.h_dim + 3 * args.embd_rank, args.embd_rank, bias=True)
            , nn.Softplus(beta=10.0)
        )

        self.intensity_layer = nn.Sequential(
            nn.Linear(self.h_dim + 3 * args.embd_rank, args.embd_rank, bias=True)
            , nn.Softplus(beta=1.)
        )

        self.t_start_layer = nn.Sequential(
            nn.Linear(self.h_dim + 2 * args.embd_rank, args.embd_rank, bias=True),
            self.gelu
        )

        self.t_converge_layer = nn.Sequential(
            nn.Linear(self.h_dim + 2 * args.embd_rank, args.embd_rank, bias=True),
            self.gelu
        )

        self.t_decay_layer = nn.Sequential(
            nn.Linear(self.h_dim + 2 * args.embd_rank, args.embd_rank, bias=True)
            , nn.Softplus(beta=10.0)
        )

        self.t_intensity_layer = nn.Sequential(
            nn.Linear(self.h_dim + 2 * args.embd_rank, args.embd_rank, bias=True)
            , nn.Softplus(beta=1.)
        )


    def get_non_pad_mask(self, seq):
        """
            Get the non-padding positions.
            return real_batch*his_len*1
        """

        assert seq.dim() == 2
        return seq.ne(0).type(torch.float).unsqueeze(-1)

    def forward(self, input, mode_tp, mode_lk, graph_dict):

        if mode_lk == 'Training':
            quadruples, s_history_event_tp, s_history_event_lk, o_history_event_tp, o_history_event_lk, \
            s_history_dt_tp, s_history_dt_lk, o_history_dt_tp, o_history_dt_lk, dur_last_tp, sub_synchro_dt_tp, obj_synchro_dt_tp = input
            self.graph_dict = graph_dict
        elif mode_lk in ['Valid', 'Test']:
            quadruples, s_history_event_tp, s_history_event_lk, o_history_event_tp, o_history_event_lk, \
            s_history_dt_tp, s_history_dt_lk, o_history_dt_tp, o_history_dt_lk, dur_last_tp, sub_synchro_dt_tp, obj_synchro_dt_tp,\
            val_subcentric_fils_lk, val_objcentric_fils_lk= input
            self.graph_dict = graph_dict
        else:
            raise ValueError('Not implemented')

        #prepare model input
        s = quadruples[:, 0]
        r = quadruples[:, 1]
        o = quadruples[:, 2]
        t = quadruples[:, 3]

        self.raw_s_encoder = RAW(self.args, graph_dict).to('cuda')
        s_raw = self.raw_s_encoder(s, t, self.ent_embeds)

        self.raw_o_encoder = RAW(self.args, graph_dict).to('cuda')
        o_raw = self.raw_o_encoder(o, t, self.ent_embeds)

        if isListEmpty(s_history_event_tp) or isListEmpty(o_history_event_tp):
            error_tp, density_tp, dt_tp, mae_tp, dur_last_nonzero_tp, den1_tp, den2_tp, tpred, abs_error = [None] * 9
        else:
            # Aggregating concurrent events
            s_packed_input_tp, s_packed_dt_tp, s_idx_tp, s_nonzero_tp = \
                self.aggregator_s(s_history_event_tp, s, r, o, t, s_raw,
                                  self.ent_embeds, self.rel_embeds(to_device(torch.arange(0, self.num_rels, 1))),
                                  s_history_dt_tp)

            o_packed_input_tp, o_packed_dt_tp, o_idx_tp, o_nonzero_tp = \
                self.aggregator_o(o_history_event_tp, o, r, s, t, o_raw, self.ent_embeds,
                                  self.rel_embeds(to_device(torch.arange(self.num_rels, 2 * self.num_rels, 1))),
                                  o_history_dt_tp)

            # compute hidden state
            sub_non_pad_mask = self.get_non_pad_mask(s_packed_dt_tp)
            obj_non_pad_mask = self.get_non_pad_mask(o_packed_dt_tp)

            sub_hidden_tp = self.encoder_s(s_packed_input_tp, s_packed_dt_tp, sub_non_pad_mask, s_idx_tp)
            obj_hidden_tp = self.encoder_o(o_packed_input_tp, o_packed_dt_tp, obj_non_pad_mask, o_idx_tp)

            dur_last_tp = to_device(torch.tensor(dur_last_tp))

            dur_non_zero_idx_tp = (dur_last_tp > 0).nonzero().squeeze()

            dur_last_nonzero_tp = dur_last_tp[dur_non_zero_idx_tp]

            # add synchro_dt_tp to synchronize the concatenated intensity from subject centric and object centeric
            sub_synchro_dt_tp = to_device(torch.tensor(sub_synchro_dt_tp, dtype=torch.float))
            sub_synchro_non_zero_idx_tp = (sub_synchro_dt_tp >= 0).nonzero().squeeze()

            sub_synchro_dt_nonzero_tp = sub_synchro_dt_tp[sub_synchro_non_zero_idx_tp]
            assert (torch.all(torch.eq(sub_synchro_non_zero_idx_tp, dur_non_zero_idx_tp)))

            obj_synchro_dt_tp = to_device(torch.tensor(obj_synchro_dt_tp, dtype=torch.float))
            obj_synchro_non_zero_idx_tp = (obj_synchro_dt_tp >= 0).nonzero().squeeze()
            obj_synchro_dt_nonzero_tp = obj_synchro_dt_tp[obj_synchro_non_zero_idx_tp]
            assert (torch.all(torch.eq(obj_synchro_non_zero_idx_tp, dur_non_zero_idx_tp)))

            if mode_tp == 'MSE':
                dur_last_nonzero_tp = dur_last_nonzero_tp.type(torch.float)
                sub_inten_tp = self.compute_inten_t(dur_non_zero_idx_tp, sub_synchro_dt_nonzero_tp,
                                                      t, sub_hidden_tp, s, o, r, self.rel_embeds.weight[:self.num_rels], s_raw)
                obj_inten_tp = self.compute_inten_t(dur_non_zero_idx_tp, obj_synchro_dt_nonzero_tp,
                                                      t, obj_hidden_tp, s, o, r, self.rel_embeds.weight[self.num_rels:], o_raw)
                dt_tp, error_tp, density_tp, mae_tp, den1_tp, den2_tp, tpred, abs_error = self.predict_t(sub_inten_tp,
                                                                                                         obj_inten_tp,
                                                                                                         dur_last_nonzero_tp)

            else:
                raise ValueError('Not implemented')

        if isListEmpty(s_history_event_lk) or isListEmpty(o_history_event_lk):
            sub_rank, obj_rank, cro_entr_lk = [None] * 3
            if mode_lk == 'Training':
                return cro_entr_lk, error_tp, density_tp, dt_tp, mae_tp, dur_last_nonzero_tp, den1_tp, den2_tp, tpred, abs_error
            elif mode_lk in ['Valid', 'Test']:
                return sub_rank, obj_rank, cro_entr_lk, error_tp, density_tp, dt_tp, mae_tp, dur_last_nonzero_tp, den1_tp, den2_tp, tpred, abs_error
            else:
                raise ValueError('Not implemented')
        else:
            #Aggregating concurrent events
            s_packed_input_lk, s_packed_dt_lk, s_idx_lk, s_nonzero_lk = \
                self.aggregator_s(s_history_event_lk, s, r, o, t, s_raw,
                                  self.ent_embeds, self.rel_embeds(to_device(torch.arange(0, self.num_rels, 1))),
                                  s_history_dt_lk)

            o_packed_input_lk, o_packed_dt_lk, o_idx_lk, o_nonzero_lk = \
                self.aggregator_o(o_history_event_lk, o, r, s, t, o_raw, self.ent_embeds,
                                  self.rel_embeds(to_device(torch.arange(self.num_rels, 2 * self.num_rels, 1))),
                                  o_history_dt_lk)

            # compute hidden state
            sub_non_pad_mask = self.get_non_pad_mask(s_packed_dt_lk)
            obj_non_pad_mask = self.get_non_pad_mask(o_packed_dt_lk)

            sub_hidden_lk = self.encoder_s(s_packed_input_lk, s_packed_dt_lk, sub_non_pad_mask, s_idx_lk)
            obj_hidden_lk = self.encoder_o(o_packed_input_lk, o_packed_dt_lk, obj_non_pad_mask, o_idx_lk)

            # compute intensity
            if mode_lk == 'Training':
                sub_cro_entr_loss = self.predict_link(sub_hidden_lk, s, o, r, self.rel_embeds(to_device(torch.arange(0,self.num_rels, 1))), mode_lk, s_raw)
                obj_cro_entr_loss = self.predict_link(obj_hidden_lk, o, s, r, self.rel_embeds(to_device(torch.arange(self.num_rels, 2*self.num_rels, 1))), mode_lk, o_raw)
                cro_entr_lk = (sub_cro_entr_loss + obj_cro_entr_loss) / 2
                return cro_entr_lk, error_tp, density_tp, dt_tp, mae_tp, dur_last_nonzero_tp, den1_tp, den2_tp, tpred, abs_error

            elif mode_lk in ['Valid', 'Test']:
                sub_cro_entr_loss, sub_rank = self.predict_link(sub_hidden_lk, s, o, r, self.rel_embeds(to_device(torch.arange(0,self.num_rels, 1))), mode_lk, s_raw,
                                                        val_fils =  val_subcentric_fils_lk)
                obj_cro_entr_loss, obj_rank = self.predict_link(obj_hidden_lk, o, s, r, self.rel_embeds(to_device(torch.arange(self.num_rels, 2*self.num_rels, 1))), mode_lk, o_raw,
                                                       val_fils = val_objcentric_fils_lk)
                cro_entr_lk = (sub_cro_entr_loss + obj_cro_entr_loss) / 2
                return sub_rank, obj_rank, cro_entr_lk, error_tp, density_tp, dt_tp, mae_tp, dur_last_nonzero_tp, den1_tp, den2_tp, tpred, abs_error

            else:
                raise ValueError('Not implemented')

    def predict_link(self, hiddens_ti, actor1, actor2, r, rel_embeds, mode_lk, actor_raw, val_fils=None):
        start_point = self.start_layer(torch.cat((self.ent_embeds.weight[actor1.long()], actor_raw,
                                                                    self.linear_h(hiddens_ti),
                                                                    rel_embeds[r.long()]), dim=1))
        converge_point = self.converge_layer(torch.cat((self.ent_embeds.weight[actor1.long()], actor_raw,
                                                                    self.linear_h(hiddens_ti),
                                                                    rel_embeds[r.long()]), dim=1))
        omega = self.decay_layer(torch.cat((self.ent_embeds.weight[actor1.long()], actor_raw,
                                                                    self.linear_h(hiddens_ti),
                                                                    rel_embeds[r.long()]), dim=1))

        inten_raw = torch.tanh(converge_point + (start_point - converge_point) * torch.exp(- omega))
        intens = self.Softplus(inten_raw.mm(self.ent_embeds.weight.transpose(0, 1)))  # shape of pred_intens: num_batch*num_e

        cro_entr_loss = self.criterion_link(intens, actor2.to(torch.int64))
        ranks = []
        if mode_lk == 'Training':
            return cro_entr_loss
        elif mode_lk in ['Valid', 'Test']:
            ground = intens.gather(1, actor2.to(torch.int64).view(-1, 1))
            assert (len(val_fils) == intens.shape[0])
            for i in range(len(val_fils)):
                if self.args.filtering:
                    intens[i, :][val_fils[i]] = 0
                intens[i, actor2[i]] = ground[i]
                pred_comp1 = (intens[i, 1:] > ground[i]).sum().item() + 1
                ranks.append(pred_comp1)
            return cro_entr_loss, ranks
        else:
            raise ValueError('Not implemented')

    def compute_inten_t(self, non_zero_idx, synchro_dt_nonzero_tp, t, hidden_tp, actors, another_actors, r, rel_embeds, actor_raw):
        hmax = settings['time_horizon']
        timestep = settings['CI']
        n_samples = int(hmax / timestep) + 1
        dt = to_device(torch.linspace(0, hmax, n_samples).repeat(non_zero_idx.shape[0], 1)
                       .transpose(0, 1)) + synchro_dt_nonzero_tp[None, :]

        start_point = self.t_start_layer(torch.cat((self.ent_embeds.weight[actors[non_zero_idx].long()],
                                    hidden_tp[non_zero_idx],
                                    rel_embeds[r[non_zero_idx].long()]),dim=1))
        converge_point = self.t_converge_layer(torch.cat((self.ent_embeds.weight[actors[non_zero_idx].long()],
                                    hidden_tp[non_zero_idx],
                                    rel_embeds[r[non_zero_idx].long()]),dim=1))
        omega = self.t_decay_layer(torch.cat((self.ent_embeds.weight[actors[non_zero_idx].long()],
                                    hidden_tp[non_zero_idx],
                                    rel_embeds[r[non_zero_idx].long()]),dim=1))

        inten_raw = torch.tanh(converge_point + (start_point - converge_point) * torch.exp(- omega[None, :, :] * dt[:, :, None]))
        o = self.ent_embeds.weight[another_actors[non_zero_idx].long()].repeat(n_samples, 1, 1)

        intens = self.Softplus((inten_raw * o).sum(dim=2))

        return intens

    def predict_t(self, sub_inten_t, obj_inten_t, gt_t):
        timestep = settings['CI']
        hmax = settings['time_horizon']
        n_samples = int(hmax / timestep) + 1
        dt = to_device(torch.linspace(0, hmax, n_samples).repeat(gt_t.shape[0], 1).transpose(0, 1))
        intens = (sub_inten_t + obj_inten_t) / 2
        integral_ = torch.cumsum(timestep * intens, dim=0)
        density = (intens * torch.exp(-integral_))
        t_pit = dt * density
        estimate_dt = (timestep * 0.5 * (t_pit[1:] + t_pit[:-1])).sum(dim=0)
        mse = nn.MSELoss()
        error_dt = mse(estimate_dt, gt_t)

        with torch.no_grad():
            abs_error = (estimate_dt - gt_t).abs()
            mae = abs_error.mean()
        return dt, error_dt, density, mae, intens, torch.exp(-integral_), estimate_dt.detach(), abs_error








