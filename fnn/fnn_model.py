

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numbers
import copy
import random
import numpy as np

class FNN_Model():
    def __init__(self, NN, upper_lat, X_train, y_train, X_test, y_test):
        self.NN = NN
        self.upper_lat = upper_lat
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.lst_y = self.upper_lat.y_classes_num
        #self.lst_y = self.upper_lat.get_y_classes(y_train)


    def __summator(self, X_obj, hyp_type, layer, to_neuron, results):
        if layer == 1:
            activations = self.NN.bias[hyp_type][layer][int(to_neuron)-1]
            from_neurons = self.NN.neuron_inputs[hyp_type][layer][
                int(to_neuron)-1].split(',')
            for from_neuron in from_neurons:
                activations += (X_obj[int(from_neuron)-1] *
                      self.NN.weights[hyp_type]['i'][from_neuron+'-'+to_neuron])
        elif isinstance(layer, numbers.Number):
            activations = self.NN.bias[hyp_type][layer][int(to_neuron)-1]
            from_neurons = self.NN.neuron_inputs[hyp_type][layer][
                int(to_neuron)-1].split(',')
            for from_neuron in from_neurons:
                activations += (results[hyp_type][layer-1][from_neuron] *
                      self.NN.weights[hyp_type][layer-1][from_neuron+'-'+
                                                         to_neuron])
        elif layer == 'o':
            # The meaning of "po" is a "prior output"
            layer_po = len(self.NN.neuron_inputs[hyp_type])
            num_neurons_lpo = len(self.NN.neuron_inputs[hyp_type][layer_po])
            activations = self.NN.bias[hyp_type]['o']
            for from_neuron_index in range(num_neurons_lpo):
                activations +=\
                    (results[hyp_type][layer_po][str(from_neuron_index+1)] *
                     self.NN.weights[hyp_type][layer_po][
                         str(from_neuron_index+1)+'-'+str(hyp_type)])
            #num_neurons_lpo = len(self.NN.neuron_inputs[hyp_type][layer_po])
            #for from_neuron_index in range(num_neurons_lpo):
            #    activations += (results[hyp_type][layer_po][str(from_neuron_index+1)] *
            #          self.NN.weights[hyp_type][layer_po][
            #              str(from_neuron_index+1)+'-'+str(hyp_type)])
        else:
            raise ValueError('Not correct type of layer: %s' % layer)
        #activations = min(max(activations, -100), 10000)
        #unit_output = 1 / (1 + math.exp(-activations))
        return activations

    def __activation_func(self, activations, act_type = 'sigmoid',
                          activations_znam = None):
        activations_working = activations
        if act_type == 'sigmoid':
            activations_working = min(max(activations_working, -100), 10000)
            unit_output = 1 / (1 + math.exp(-activations_working))
            return unit_output
        elif act_type == 'softmax':
            softmax_chisl = math.exp(activations_working)
            softmax_znam = 0
            for hyp_type in self.NN.neuron_inputs:
                softmax_znam += math.exp(activations_znam[hyp_type])
            unit_output = softmax_chisl / softmax_znam
            return unit_output
        else:
            raise ValueError('Activation function %s not found' % act_type)


    def output_calc (self, X_obj):
        results = self.NN.init_results()
        for hyp_type in self.NN.neuron_inputs:
            layers = range(1, len(self.NN.neuron_inputs[hyp_type])+1)
            for layer in layers:
                for to_neuron_index in range(len(
                    self.NN.neuron_inputs[hyp_type][layer])):
                    activations = self.__summator(X_obj, hyp_type, layer,
                                                  str(to_neuron_index+1),
                                                  results)
                    results[hyp_type][layer][str(to_neuron_index+1)] =\
                        self.__activation_func(activations,
                                               act_type = 'sigmoid')
        all_activations = {}
        for hyp_type in self.NN.neuron_inputs:
            all_activations[hyp_type] =\
                self.__summator(X_obj, hyp_type, 'o', '1', results)
        for hyp_type in self.NN.neuron_inputs:
            results[hyp_type]['o'] =\
                self.__activation_func(all_activations[hyp_type],
                                       act_type = 'softmax',
                                       activations_znam = all_activations)
        #softmax_znam = 0
        #for from_hyp_type in self.NN.neuron_inputs:
        #    softmax_znam += math.exp(results[from_hyp_type]['o'])
        #for hyp_type in self.NN.neuron_inputs:
        #    print('hyp_type: ', hyp_type, results[0]['o'],
        #          results[1]['o'])
        #    softmax_chisl = math.exp(results[hyp_type]['o'])
        #    results['r'][hyp_type] = softmax_chisl / softmax_znam
        return results



    def __sum_vars(self, dic1, dic2):
        sum_dic = copy.deepcopy(dic1)
        for key_depth1, value_depth1 in sum_dic.items():
            if (isinstance(value_depth1, dict)):
                for key_depth2, value_depth2 in value_depth1.items():
                    if (isinstance(value_depth2, dict)):
                        for key_depth3 in value_depth2:
                            sum_dic[key_depth1][key_depth2][key_depth3] +=\
                                dic2[key_depth1][key_depth2][key_depth3]
                    elif (isinstance(value_depth2, list)):
                        for list_depth3, value_depth3 in enumerate(value_depth2):
                            sum_dic[key_depth1][key_depth2][list_depth3] +=\
                                dic2[key_depth1][key_depth2][list_depth3]
                    else:
                        sum_dic[key_depth1][key_depth2] +=\
                            dic2[key_depth1][key_depth2]
            elif (isinstance(value_depth1, list)):
                for list_depth2, value_depth2 in enumerate(value_depth1):
                    if (isinstance(value_depth2, dict)):
                        for key_depth3 in value_depth2:
                            sum_dic[key_depth1][list_depth2][key_depth3] +=\
                                dic2[key_depth1][list_depth2][key_depth3]
                    if (isinstance(value_depth2, list)):
                        for list_depth3, value_depth in enumerate(value_depth2):
                            sum_dic[key_depth1][list_depth2][list_depth3] +=\
                                dic2[key_depth1][list_depth2][list_depth3]
            else:
                sum_dic[key_depth1] += dic2[key_depth1]
        return sum_dic


    def __div_vars_by_num(self, dic, division):
        dic_working = copy.deepcopy(dic)
        for key_depth1, value_depth1 in dic_working.items():
            if (isinstance(value_depth1, dict)):
                for key_depth2, value_depth2 in value_depth1.items():
                    if (isinstance(value_depth2, dict)):
                        for key_depth3 in value_depth2:
                            dic_working[key_depth1][key_depth2][key_depth3]/=\
                                division
                    elif (isinstance(value_depth2, list)):
                        for list_depth3, value_depth3 in enumerate(value_depth2):
                            dic_working[key_depth1][key_depth2][list_depth3]/=\
                                division
                    else:
                        dic_working[key_depth1][key_depth2] /= division
            elif (isinstance(value_depth1, list)):
                for list_depth2, value_depth2 in enumerate(value_depth1):
                    if (isinstance(value_depth2, dict)):
                        for key_depth3 in value_depth2:
                            dic_working[key_depth1][list_depth2][key_depth3]/=\
                                division
                    else:
                        dic_working[key_depth1][list_depth2] /= division
            else:
                dic_working[key_depth1] /= division
        return dic_working


    def __backprop_delta(self, target, y, results):
        deltas = copy.deepcopy(self.NN.init_deltas())
        cross_entropy_error = 0
        for hyp_type in self.NN.neuron_inputs:
            cross_entropy_error -= target[hyp_type]*math.log(y[hyp_type])
            print(target[hyp_type], math.log(y[hyp_type]), cross_entropy_error)
        deltas['r'] = cross_entropy_error
        kronecker_delta = {}
        for output_unit in self.NN.neuron_inputs:
            kronecker_delta[output_unit] = {}
            for output_unit_iter in self.NN.neuron_inputs:
                if output_unit == output_unit_iter:
                    kronecker_delta[output_unit][output_unit_iter] = 1.
                else:
                    kronecker_delta[output_unit][output_unit_iter] = 0.
        for hyp_type in self.NN.neuron_inputs:
            #print(z[hyp_type], y[hyp_type])
            # crossв_entropy_error
            #deltas['r'][hyp_type] = -(z[hyp_type] - y[hyp_type])
            #deltas[hyp_type]['o'] = (deltas['r'][hyp_type] *
            #                         self.NN.weights[hyp_type]['o'])
            for hyp_type_other in self.NN.neuron_inputs:
                deltas[hyp_type]['o'] += (target[hyp_type_other] *
                    (y[hyp_type] - kronecker_delta[hyp_type][hyp_type_other]))
        for hyp_type in self.NN.neuron_inputs:
            layer_po = len(self.NN.neuron_inputs[hyp_type])
            num_neurons_lpo = len(self.NN.neuron_inputs[hyp_type][layer_po])
            for neuron_index in range(num_neurons_lpo):
                deltas[hyp_type][layer_po][str(neuron_index+1)] =\
                    (results[hyp_type][layer_po][str(neuron_index+1)] *
                     (1. - results[hyp_type][layer_po][str(neuron_index+1)]) *
                     (deltas[hyp_type]['o'] *
                      self.NN.weights[hyp_type][layer_po][str(neuron_index+1)+
                                                          '-'+str(hyp_type)]))
        for hyp_type in self.NN.neuron_inputs:
            layer_po = len(self.NN.neuron_inputs[hyp_type])
            hidden_layers_desc = sorted([layer for layer in range(1, layer_po)],
                                        reverse=True)
            for layer in hidden_layers_desc:
                from_to_mapping = {}
                for to_neuron_index, from_neurons in enumerate(
                    self.NN.neuron_inputs[hyp_type][layer+1]):
                    for from_neuron in from_neurons.split(','):
                        deltas[hyp_type][layer][from_neuron] +=\
                            (deltas[hyp_type][layer+1][str(to_neuron_index+1)] *
                             self.NN.weights[hyp_type][layer][
                                 from_neuron+'-'+str(to_neuron_index+1)])
                        if from_neuron in from_to_mapping:
                            from_to_mapping[from_neuron] =\
                                from_to_mapping[from_neuron] + ',' +\
                                str(to_neuron_index+1)
                        else:
                            from_to_mapping[from_neuron] =\
                                str(to_neuron_index+1)
                for from_neuron, to_neurons in from_to_mapping.items():
                    to_neurons_split = to_neurons.split(',')
                    for to_neuron in to_neurons_split:
                        deltas[hyp_type][layer][from_neuron] *=\
                            (results[hyp_type][layer][from_neuron] *
                             (1. - results[hyp_type][layer][from_neuron]))
        return deltas


    def __update_weights(self, deltas_upd, results_upd, X_avg, learning_rate):
        weights_prime = copy.deepcopy(self.NN.weights)
        bias_prime = copy.deepcopy(self.NN.bias)
        sum_weights = 0
        min_weight = 0
        for hyp_type in self.NN.neuron_inputs:
            layer_po = len(self.NN.neuron_inputs[hyp_type])
            num_neurons_lpo = len(self.NN.neuron_inputs[hyp_type][layer_po])
            hidden_layers_sort = sorted([hidden_layer for hidden_layer in
                                         range(1, layer_po+1)])
            for layer in hidden_layers_sort:
                if layer == 1:
                    for to_neuron_index, from_neurons in enumerate(
                        self.NN.neuron_inputs[hyp_type][layer]):
                        from_neurons_split = from_neurons.split(',')
                        for from_neuron in from_neurons_split:
                            weights_prime[hyp_type]['i'][from_neuron+'-'+
                                str(to_neuron_index+1)] =\
                                (self.NN.weights[hyp_type]['i'][from_neuron+'-'+
                                    str(to_neuron_index+1)] -
                                 (learning_rate *
                                  deltas_upd[hyp_type][layer][
                                      str(to_neuron_index+1)] *
                                  #results_upd[hyp_type][layer][
                                  #    str(to_neuron_index+1)] *
                                  #(1. - results_upd[hyp_type][layer][
                                  #    str(to_neuron_index+1)]) *
                                  X_avg[int(from_neuron)-1]))
                        bias_prime[hyp_type][layer][to_neuron_index] =\
                            (self.NN.bias[hyp_type][layer][to_neuron_index] -
                             (learning_rate *
                              deltas_upd[hyp_type][layer][
                                  str(to_neuron_index+1)]))
                else:
                    for to_neuron_index, from_neurons in enumerate(
                        self.NN.neuron_inputs[hyp_type][layer]):
                        from_neurons_split = from_neurons.split(',')
                        for from_neuron in from_neurons_split:
                            weights_prime[hyp_type][layer-1][from_neuron+'-'+
                                str(to_neuron_index+1)] =\
                                (self.NN.weights[hyp_type][layer-1][
                                    from_neuron+'-'+str(to_neuron_index+1)] -
                                 (learning_rate *
                                  deltas_upd[hyp_type][layer][
                                      str(to_neuron_index+1)] *
                                  #results_upd[hyp_type][layer][
                                  #    str(to_neuron_index+1)] *
                                  #(1. -
                                  # results_upd[hyp_type][layer][
                                  #     str(to_neuron_index+1)]) *
                                  results_upd[hyp_type][layer-1][from_neuron]))
                        bias_prime[hyp_type][layer][to_neuron_index] =\
                            (self.NN.bias[hyp_type][layer][to_neuron_index] -
                             (learning_rate *
                              deltas_upd[hyp_type][layer][
                                  str(to_neuron_index+1)]))
            for from_neuron_index in range(num_neurons_lpo):
                weights_prime[hyp_type][layer_po][
                    str(from_neuron_index+1)+'-'+str(hyp_type)] =\
                    (self.NN.weights[hyp_type][layer_po][
                        str(from_neuron_index+1)+'-'+str(hyp_type)] -
                     (learning_rate * deltas_upd[hyp_type]['o'] *
                      #results_upd[hyp_type]['o'] *
                      #(1. - results_upd[hyp_type]['o']) *
                      results_upd[hyp_type][layer_po][
                          str(from_neuron_index+1)]))
            #weights_prime[hyp_type]['o'] = (
            #    self.NN.weights[hyp_type]['o'] +
            #    (learning_rate * deltas_upd['r'] *
            #     results_upd[hyp_type]['o']))
            #new deletion
            bias_prime[hyp_type]['o'] =\
                (self.NN.bias[hyp_type]['o'] -
                 (learning_rate * deltas_upd[hyp_type]['o']))
        return weights_prime, bias_prime


    def step (self, X_feed, y_feed, learning_rate):
        self.cum_deltas = copy.deepcopy(self.NN.init_deltas())
        self.cum_results = copy.deepcopy(self.NN.init_results())
        loss = 0
        for ind in range(y_feed.shape[0]):
            self.results = self.output_calc(X_feed[ind])
            #y_pred = self.results['r']
            target, y_pred = {}, {}
            for cl in self.lst_y:
                y_pred[cl] = self.results[cl]['o']
                if y_feed[ind] != cl:
                    target[cl] = 0
                else:
                    target[cl] = 1
            print(target, y_pred)
            self.deltas = self.__backprop_delta(target, y_pred, self.results)
            loss += self.deltas['r']
            self.cum_deltas = self.__sum_vars(self.cum_deltas, self.deltas)
            self.cum_results = self.__sum_vars(self.cum_results, self.results)
        self.deltas_upd = self.__div_vars_by_num(self.cum_deltas,
                                                 y_feed.shape[0])
        self.results_upd = self.__div_vars_by_num(self.cum_results,
                                                  y_feed.shape[0])
        #X_transpose = np.transpose(X_feed)
        #X_avg = np.average(X_transpose, axis=1)
        weights_upd, bias_upd = self.__update_weights(self.deltas_upd,
            self.results_upd, X_feed[0], learning_rate)
        self.NN.weights, self.NN.bias =\
            copy.deepcopy(weights_upd), copy.deepcopy(bias_upd)
        return loss
