import numpy as np
import os
import pickle
import dgl
import torch

def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)

    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)

    times = list(times)
    times.sort()
    return np.asarray(quadrupleList), np.asarray(times)

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def reset_hist_lists(num_e = None, num_r = None, num_dataset = None):
    if num_e and num_r:
        return [[[] for _ in range(num_e)] for _ in range(num_r)]
    if num_dataset:
        return [[] for _ in range(num_dataset)]

def get_history(dataset, dataset_name, latest_t):
    s_history_entities = reset_hist_lists(num_dataset=len(dataset))
    s_history_times = reset_hist_lists(num_dataset=len(dataset))
    o_history_entities = reset_hist_lists(num_dataset=len(dataset))
    o_history_times = reset_hist_lists(num_dataset=len(dataset))
    s_concurrent_objects = [[] for _ in range(len(dataset))]
    o_concurrent_subjects = [[] for _ in range(len(dataset))]

    e_idx_last_t = []
    for i, data in enumerate(dataset):
        if i % 10000==0:
            print(dataset_name ,i, len(dataset))
        t = data[3]
        s = data[0]
        r = data[1]
        o = data[2]
        if latest_t != t:
            for idx in e_idx_last_t:
                dev_event = dataset[idx]
                e_s = dev_event[0]
                e_r = dev_event[1]
                e_o = dev_event[2]
                s_concurrent_objects[idx] = s_his_cache[e_r][e_s].copy()
                o_concurrent_subjects[idx] = o_his_cache[e_r][e_o].copy()
            e_idx_last_t = []

            for rr in range(num_r):
                for ee in range(num_e):
                    if len(s_his_cache[rr][ee]) != 0:
                        if len(s_his[rr][ee]) >= history_len:
                            s_his[rr][ee].pop(0)
                            s_his_dt[rr][ee].pop(0)
                        s_his[rr][ee].append(s_his_cache[rr][ee].copy())
                        assert(all(x == s_his_t_cache[rr][ee][0] for x in s_his_t_cache[rr][ee]))
                        s_his_dt[rr][ee].append(s_his_t_cache[rr][ee][0])
                        s_his_cache[rr][ee]= []
                        s_his_t_cache[rr][ee] = []
                    if len(o_his_cache[rr][ee]) != 0:
                        if len(o_his[rr][ee]) >=history_len:
                            o_his[rr][ee].pop(0)
                            o_his_dt[rr][ee].pop(0)
                        o_his[rr][ee].append(o_his_cache[rr][ee].copy())
                        assert(all(x == o_his_t_cache[rr][ee][0] for x in o_his_t_cache[rr][ee]))
                        o_his_dt[rr][ee].append(o_his_t_cache[rr][ee][0])
                        o_his_cache[rr][ee]=[]
                        o_his_t_cache[rr][ee] = []
            latest_t = t
        s_history_entities[i] = s_his[r][s].copy()
        o_history_entities[i] = o_his[r][o].copy()
        s_his_cache[r][s].append(o)
        o_his_cache[r][o].append(s)
        s_his_t_cache[r][s].append(t)
        o_his_t_cache[r][o].append(t)
        if len(s_his_dt[r][s]) > 1:
            s_history_times[i] = list(np.asarray(s_his_dt[r][s][1:]) - np.asarray(s_his_dt[r][s][:-1]))
            s_history_times[i].append(t - s_his_dt[r][s][-1])
        elif len(s_his_dt[r][s]) == 1:
            s_history_times[i].append(t - s_his_dt[r][s][-1])

        if len(o_his_dt[r][o])>1:
            o_history_times[i] = list(np.asarray(o_his_dt[r][o][1:]) - np.asarray(o_his_dt[r][o][:-1]))
            o_history_times[i].append(t - o_his_dt[r][o][-1])
        elif len(o_his_dt[r][o]) == 1:
            o_history_times[i].append(t - o_his_dt[r][o][-1])
        if e_idx_last_t is not None:
            e_idx_last_t.append(i)


    with open(dataset_name + '_history_sub1.txt', 'wb') as fp:
        pickle.dump(s_history_entities, fp)
    with open(dataset_name + '_history_ob1.txt', 'wb') as fp:
        pickle.dump(o_history_entities, fp)
    with open(dataset_name + '_history_sub_dt1.txt', 'wb') as fp:
        pickle.dump(s_history_times, fp)
    with open(dataset_name + '_history_ob_dt1.txt', 'wb') as fp:
        pickle.dump(o_history_times, fp)

    for idx in e_idx_last_t:
        event = dataset[idx]
        e_s = event[0]
        e_r = event[1]
        e_o = event[2]
        s_concurrent_objects[idx] = s_his_cache[e_r][e_s].copy()
        o_concurrent_subjects[idx] = o_his_cache[e_r][e_o].copy()
    np.savetxt(dataset_name + '_subcentric_filtering_objs.txt', np.asarray(s_concurrent_objects), fmt='%s')
    np.savetxt(dataset_name + '_objcentric_filtering_subs.txt', np.asarray(o_concurrent_subjects), fmt='%s')

    return latest_t

def get_data_with_t(data, tim):
    x = data[np.where(data[:,3] == tim)].copy()
    x = np.delete(x, 3, 1)
    return x

def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm

def get_big_graph(data, num_rels):
    src, rel, dst = data.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    g = dgl.DGLGraph()
    g.add_nodes(len(uniq_v))
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel_o = np.concatenate((rel + num_rels, rel))
    rel_s = np.concatenate((rel, rel + num_rels))
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    g.ndata.update({'id': torch.from_numpy(uniq_v).long().view(-1, 1), 'norm': norm.view(-1, 1)})
    g.edata['type_s'] = torch.LongTensor(rel_s)
    g.edata['type_o'] = torch.LongTensor(rel_o)
    g.ids = {}
    idx = 0
    for id in uniq_v:
        g.ids[id] = idx
        idx += 1
    return g


if __name__ == '__main__':
    train_data, train_times = load_quadruples('', 'train.txt')
    test_data, test_times = load_quadruples('', 'test.txt')
    num_e, num_r = get_total_number('', 'stat.txt')

    num_e = num_e + 1

    s_his = reset_hist_lists(num_e=num_e, num_r=num_r)
    s_his_dt = reset_hist_lists(num_e=num_e, num_r=num_r)
    o_his = reset_hist_lists(num_e=num_e, num_r=num_r)
    o_his_dt = reset_hist_lists(num_e=num_e, num_r=num_r)

    s_his_cache = reset_hist_lists(num_e=num_e, num_r=num_r)
    s_his_t_cache = reset_hist_lists(num_e=num_e, num_r=num_r)
    o_his_cache = reset_hist_lists(num_e=num_e, num_r=num_r)
    o_his_t_cache = reset_hist_lists(num_e=num_e, num_r=num_r)
    history_len = 8
    latest_t = 0
    latest_t = get_history(train_data, 'train', latest_t)

    graph_dict_train = {}
    graph_dict_test = {}

    for tim in train_times:
        print(str(tim) + '\t' + str(max(train_times)))
        data = get_data_with_t(train_data, tim)
        graph_dict_train[tim] = get_big_graph(data, num_r)

    for tim in test_times:
        print(str(tim) + '\t' + str(max(test_times)))
        data = get_data_with_t(test_data, tim)
        graph_dict_test[tim] = get_big_graph(data, num_r)

    with open('train_graphs.txt', 'wb') as fp:
        pickle.dump(graph_dict_train, fp)

    with open('test_graphs.txt', 'wb') as fp:
        pickle.dump(graph_dict_test, fp)

    _ = get_history(test_data, 'test', latest_t)


