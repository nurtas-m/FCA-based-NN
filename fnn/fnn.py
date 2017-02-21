
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class FNN():
    def __init__(self, taked_fc):
        self.taked_fc = taked_fc
        self.neuron_inputs, self.curr_index_list, self.index_list = {}, {}, {}

        for hyp_type in self.taked_fc:
            self.neuron_inputs[hyp_type], self.curr_index_list[hyp_type], \
                self.index_list[hyp_type] = {}, {}, {}
            lst = [list(x.keys())[0] for x in self.taked_fc[hyp_type]
                   if len(x) > 0]
            max_len = 0
            for l in lst:
                if max_len < len(l.split(',')):
                    max_len = len(l.split(','))
            for i in range(1, max_len+1):
                self.index_list[hyp_type][i] = {}
            for l in lst:
                self.__split_list(l.split(','), 0, hyp_type)

        for hyp_type in self.taked_fc:
            self.neuron_inputs[hyp_type] = self.__fnn_completion(
                self.neuron_inputs[hyp_type])

        self.weights = self.init_weights()
        self.bias = self.init_bias()
        #self.results = self.init_results()
        #self.deltas = self.init_deltas()


    def __split_list(self, in_list, hidden_layer, hyp_type):
        # Инициализация входов нейронов (фактически, задание архитектуры сети)
        if (len(in_list) > 1):
            hidden_layer += 1
            inp_node_num = len(in_list)
            out_node_num = len(in_list)//2
            out_list = []
            for i in range(out_node_num):
                if hidden_layer in self.neuron_inputs[hyp_type]:
                    if hidden_layer == 1:
                        self.curr_index_list[hyp_type][hidden_layer+1] =\
                            self.curr_index_list[hyp_type][hidden_layer+1] + 1
                        self.neuron_inputs[hyp_type][hidden_layer].append(
                            in_list[2*i] + ',' + in_list[2*i+1])
                        self.index_list[hyp_type][hidden_layer][
                            in_list[2*i] + ',' + in_list[2*i+1]] =\
                            str(self.curr_index_list[hyp_type][hidden_layer+1])
                    else:
                        self.curr_index_list[hyp_type][hidden_layer+1] =\
                            self.curr_index_list[hyp_type][hidden_layer+1] + 1
                        self.neuron_inputs[hyp_type][hidden_layer].append(
                            in_list[2*i] + ',' + in_list[2*i+1])
                        self.index_list[hyp_type][hidden_layer][
                            in_list[2*i] + ',' + in_list[2*i+1]] =\
                            str(self.curr_index_list[hyp_type][hidden_layer+1])
                else:
                    if hidden_layer+1 not in self.curr_index_list[hyp_type]:
                        self.curr_index_list[hyp_type][hidden_layer+1] = 1
                    else:
                        self.curr_index_list[hyp_type][hidden_layer+1] =\
                            self.curr_index_list[hyp_type][hidden_layer+1] + 1
                    if hidden_layer == 1:
                        self.neuron_inputs[hyp_type][hidden_layer] =\
                            [in_list[2*i] + ',' + in_list[2*i+1]]
                        self.index_list[hyp_type][hidden_layer][
                            in_list[2*i] + ',' + in_list[2*i+1]] =\
                            str(self.curr_index_list[hyp_type][hidden_layer+1])
                    else:
                        self.neuron_inputs[hyp_type][hidden_layer] =\
                            [in_list[2*i] + ',' + in_list[2*i+1]]
                        self.index_list[hyp_type][hidden_layer][
                            in_list[2*i] + ',' + in_list[2*i+1]] =\
                            str(self.curr_index_list[hyp_type][hidden_layer+1])
                out_list.append(self.index_list[hyp_type][hidden_layer][
                    in_list[2*i] + ',' + in_list[2*i+1]])
            if (inp_node_num % 2) != 0:
                if hidden_layer == 1:
                    self.curr_index_list[hyp_type][hidden_layer+1] =\
                        self.curr_index_list[hyp_type][hidden_layer+1] + 1
                    self.neuron_inputs[hyp_type][hidden_layer].append(
                        in_list[inp_node_num-1])
                    self.index_list[hyp_type][hidden_layer][
                        in_list[inp_node_num-1]] =\
                        str(self.curr_index_list[hyp_type][hidden_layer+1])
                else:
                    self.curr_index_list[hyp_type][hidden_layer+1] =\
                        self.curr_index_list[hyp_type][hidden_layer+1] + 1
                    self.neuron_inputs[hyp_type][hidden_layer].append(
                        str(self.curr_index_list[hyp_type][hidden_layer]))
                    self.index_list[hyp_type][hidden_layer][
                        str(self.curr_index_list[hyp_type][hidden_layer])] =\
                        str(self.curr_index_list[hyp_type][hidden_layer+1])
                out_list.append(self.index_list[hyp_type][hidden_layer][
                    in_list[inp_node_num-1]])
            out_list, hidden_layer = self.__split_list(out_list, hidden_layer,
                                                       hyp_type)
            return out_list, hidden_layer
        else:
            if hidden_layer == 0:
                hidden_layer += 1
                if 1 in self.neuron_inputs[hyp_type]:
                    self.curr_index_list[hyp_type][hidden_layer+1] =\
                        self.curr_index_list[hyp_type][hidden_layer+1] + 1
                    self.neuron_inputs[hyp_type][1].append(in_list[0])
                    self.index_list[hyp_type][hidden_layer][in_list[0]] =\
                        str(self.curr_index_list[hyp_type][hidden_layer+1])
                else:
                    self.curr_index_list[hyp_type][hidden_layer+1] = 1
                    self.neuron_inputs[hyp_type][1] = [in_list[0]]
                    self.index_list[hyp_type][hidden_layer][in_list[0]] =\
                        str(self.curr_index_list[hyp_type][hidden_layer+1])
            return in_list, hidden_layer


    def __fnn_completion(self, hidden_layer_inputs):
        num_hid_layer = len(hidden_layer_inputs)
        for layer in hidden_layer_inputs:
            if layer < len(hidden_layer_inputs):
                lst = ','.join(hidden_layer_inputs[layer+1]).split(',')
                print("Layer: ", layer)
                print(lst)
                for j in range(len(hidden_layer_inputs[layer])):
                    print("j: ", j)
                    if str(j+1) not in lst:
                        hidden_layer_inputs[layer+1].append(str(j+1))
        return hidden_layer_inputs


    def init_weights(self):
        weights = {}
        for output_neuron in self.neuron_inputs.keys():
            if self.neuron_inputs[output_neuron]:
                layer_po = len(self.neuron_inputs[output_neuron])
                num_neurons_lpo = len(self.neuron_inputs[output_neuron][layer_po])
                weights[output_neuron] = {'i':{}, layer_po:{}}
                for hidden_layer, neurons in \
                    self.neuron_inputs[output_neuron].items():
                    if hidden_layer > 1:
                        if hidden_layer-1 not in weights[output_neuron]:
                            weights[output_neuron][hidden_layer-1] = {}
                        for curr_index, prev_indexes in enumerate(neurons):
                            lst = prev_indexes.split(',')
                            for prev_index in lst:
                                weights[output_neuron][hidden_layer-1][
                                    prev_index + '-' + str(curr_index+1)] =\
                                    np.random.random()/10.
                    else:
                        for curr_index, prev_indexes in enumerate(neurons):
                            lst = prev_indexes.split(',')
                            for prev_index in lst:
                                weights[output_neuron]['i'][
                                    prev_index + '-' + str(curr_index+1)] =\
                                    np.random.random()/10.
                for from_neuron_index in range(num_neurons_lpo):
                    weights[output_neuron][layer_po][
                        str(from_neuron_index+1) + '-' + str(output_neuron)] =\
                        np.random.random()/10.
        return weights

    def init_bias(self):
        bias ={}
        for hyp_type in self.neuron_inputs:
            bias[hyp_type] = {'o' : np.random.random()/10.}
            #bias[hyp_type] = {}
            for layer in self.neuron_inputs[hyp_type]:
                bias[hyp_type][layer] = []
                for neuron_index in self.neuron_inputs[hyp_type][layer]:
                    bias[hyp_type][layer].append(np.random.random()/10.)
        return bias

    def init_results(self):
        #results={'r':{output_neuron:0.0
        #              for output_neuron in self.neuron_inputs}}
        results = {}
        for output_neuron in self.neuron_inputs:
            results[output_neuron] = {'o':0.0}
            for hidden_layer in self.neuron_inputs[output_neuron]:
                results[output_neuron][hidden_layer] = {}
                for neuron_index in range(
                    len(self.neuron_inputs[output_neuron][hidden_layer])):
                    results[output_neuron][hidden_layer][
                        str(neuron_index+1)] = 0.0
        return results

    def init_deltas(self):
        #deltas={'r':{output_neuron:0.0
        #             for output_neuron in self.neuron_inputs}}
        deltas = {'r':0.0}
        for output_neuron in self.neuron_inputs:
            deltas[output_neuron] = {'o':0.0}
            for hidden_layer in self.neuron_inputs[output_neuron]:
                deltas[output_neuron][hidden_layer] = {}
                for neuron_index in range(
                    len(self.neuron_inputs[output_neuron][hidden_layer])):
                    deltas[output_neuron][hidden_layer][str(neuron_index+1)] =\
                                                                             0.0
        return deltas


