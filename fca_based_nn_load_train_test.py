
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
sys.path.insert(1, './fca')
sys.path.insert(1, './fnn')

import copy
import math
import numpy as np
import numbers
from fca.fca_lattice import FCA_Lattice
from fca.fca_model import FCA_Model
from fca.fca_sort_util import FCA_Sort
from fnn.fnn import FNN
from fnn.fnn_model import FNN_Model
from fnn.data_gen import Data_Gen
import re
import json
import time
import random

prog_start_time = time.time()
data_set = 'breast_cancer'
sample_inx = '0'
shuffled_dir = '/home/nurtas/Documents/UCIMLR/shuffled_samples/'
train_path = os.path.join(shuffled_dir+data_set, data_set+'_train_'+sample_inx)
test_path = os.path.join(shuffled_dir+data_set, data_set+'_test_'+sample_inx)

src_train = open(train_path).readlines()
src_test = open(test_path).readlines()

# Из расчета 1000 дополнительных объекта или (глубина+1)
train_len_to_depth_ratio = 1

depth = 2

depth = int(max(depth - train_len_to_depth_ratio, 1))
conf_share=0.8

log_params_file_path = os.path.join(
    '/home/nurtas/Documents/aspirantura/logs_and_architectures/ismis2017',
    data_set, 'depth-'+str(depth), 'selection-'+sample_inx, 'params-1.log')

if not os.path.exists(os.path.dirname(log_params_file_path)):
    os.makedirs(os.path.dirname(log_params_file_path))

max_counter = 0
for file_in_dir in os.listdir(os.path.dirname(log_params_file_path)):
    if file_in_dir.endswith('.log'):
        file_in_dir_wo_ext = os.path.splitext(file_in_dir)[0]
        if int(file_in_dir_wo_ext.split('-')[-1]) > max_counter:
            max_counter = int(file_in_dir_wo_ext.split('-')[-1])

new_log_params_file_name = 'params-' + str(max_counter+1) + '.log'
log_params_file_path = log_params_file_path.replace('params-1.log',
                                                    new_log_params_file_name)

if not os.path.exists(log_params_file_path):
    with open(log_params_file_path, 'w') as log_params_file:
        log_params_file.write('Params for %s.\n' % data_set)
        log_params_file.write('Train file path: %s\n' % train_path)
        log_params_file.write('Test file path: %s\n' % test_path)

log_params_file = open(log_params_file_path, 'a')

print('Cover upper lattice with depth = ', depth)
log_params_file.write('Cover upper lattice with depth: %d\n' %
                      depth)

log_params_file.write('==============================================')
log_params_file.write('Started at: %s\n' %
                      time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = int(sec_elapsed % 60)
    return "{}:{:>02}:{:>02}".format(h, m, s)

src_ref_train = []
for s in src_train:
    s_ref = s
    src_ref_train.append(s_ref.replace('\n',''))


y_map = {'0':0, '1':1, '0.':0, '1.':1, 'B':0, 'M':1, '2':2, '3':3, '4':4, '5':5,
         'e':0, 'p':1, 'unacc':0, 'acc':1, 'good':1, 'vgood':1}

print('Upper Lattice creation started.')
upper_lat = FCA_Lattice(src_ref_train, y_map)

if data_set in ('heart_disease', 'mammographic_mass_data',
                'seismic_bumps', 'credit_card_default',
                'car_evaluation'):
    X_cols = [i for i in range(0, upper_lat.var_num-1)]#HD, MM, SB, CAR, CC
    y_col = upper_lat.var_num-1#HD, MM, SB, CAR, CC
elif data_set == 'mushrooms':
    X_cols = [i for i in range(1, upper_lat.var_num-1)] #Mush
    y_col = 0 #Mush
elif data_set == 'breast_cancer':
    X_cols = [i for i in range(2, upper_lat.var_num-1)]#BC
    y_col = 1#BC

print('Split to train, validation and test sets started.')
X_train, y_train, X_valid, y_valid, _, _ =\
    upper_lat.split_dataset(X_cols, y_col,
                            dataset_parts = [0.7, 0.3, 0.0])


src_ref_test = []
for s in src_test:
    s_ref = s
    src_ref_test.append(s_ref.replace('\n',''))

upper_lat_test = FCA_Lattice(src_ref_test, y_map)

_, _, _, _, X_test, y_test = upper_lat_test.split_dataset(X_cols, y_col,
    dataset_parts = [0.0, 0.0, 1.0])


log_params_file.write('Num of objects to train: %d, valid: %d, test: %d\n'
                      % (len(X_train), len(X_valid),
                         len(X_test)))

X_not_scaled = np.vstack([X_train, X_valid])
y_not_scaled = np.vstack([y_train, y_valid])
X_not_scaled = np.vstack([X_not_scaled, X_test])
y_not_scaled = np.vstack([y_not_scaled, y_test])

print('Start mapping variables to attributes.')
upper_lat.map_var_attr(X_not_scaled, y_not_scaled)
print('Variables: %d, Attributes: %d' % (upper_lat.var_num,
                                         len(upper_lat.attr_var_map)))
log_params_file.write('Variables num: %d, Attributes num: %d\n'
                      % (upper_lat.var_num, len(upper_lat.attr_var_map)))
print('Move symbols to numbers.')
X_train, y_train = upper_lat.sym2num(X_train, y_train)
X_valid, y_valid = upper_lat.sym2num(X_valid, y_valid)
X_test, y_test = upper_lat.sym2num(X_test, y_test)

lst_y = upper_lat.y_classes_num

y_classes_sym_num_map = upper_lat.y_classes_sym_num_map
classes_for_print = '['
for sym_cl, num_cl in y_classes_sym_num_map.items():
    classes_for_print = classes_for_print + str(sym_cl) + ':' +\
                        str(num_cl) + ', '
classes_for_print = classes_for_print[:-2] + ']'

print('Classes: %s' % classes_for_print)
log_params_file.write('Classes: %s\n' % classes_for_print)

print('Start creation of binary context.')
X_train_bin = upper_lat.scaling(X_train, y_train)
X_valid_bin = upper_lat.scaling(X_valid, y_valid)
X_test_bin = upper_lat.scaling(X_test, y_test)


train_len = len(X_train_bin)
print('Initial FCA train set length: ', train_len)
log_params_file.write('Initial FCA train set length: %d\n' % train_len)

cut_train_len = min(round(1000 * train_len_to_depth_ratio), train_len)
print('Cut FCA train set length: %d' % cut_train_len)
log_params_file.write('Cut FCA train set length: %d\n' % cut_train_len)

X_train_cut, y_train_cut = copy.deepcopy(X_train_bin), copy.deepcopy(y_train)

if cut_train_len <= train_len * 0.5:
    X_train_cut = X_train_cut[:cut_train_len]
    y_train_cut = y_cut[:cut_train_len]
    log_params_file.write('Hypothesis generated for cutted selection with size %d',
                          cut_train_len)    
    log_params_file.flush()

fca_model = FCA_Model(X_train_cut, y_train_cut, lst_y)
print('Hypothesis generation started.')
gen_start_time = time.time()
log_params_file.flush()
fca_model.generate_hypothesis(lattice_layers_num=depth)
log_params_file.write('Generated hypothesis number: %d\n' %
                      len(fca_model.fc_list))
print('Time for hypothesis generation: %s'
      % hms_string(time.time() - gen_start_time))
log_params_file.write('Time for hypothesis generation: %s\n' %
    hms_string(time.time() - gen_start_time))


print('Calculate performance of generated FC on validation set.')
log_params_file.flush()
gen_perf_start_time = time.time()
fca_model.calc_performance(X_valid_bin, y_valid)#(X_train_bin, y_train)
print('Time for performance calculation for generated FC: %s'
      % hms_string(time.time() - gen_perf_start_time))
log_params_file.write('Time for performance calculation for generated FC: %s\n'
      % hms_string(time.time() - gen_perf_start_time))


print('Sort hypothesis.')
log_params_file.flush()
sort_start_time = time.time()
fca_sorted = FCA_Sort(fca_model.fc_list, X_train_bin, y_train, lst_y)
log_params_file.write('Sorting based on F-measure.\n')
#log_params_file.write('Sorting based on confidence-support ratio =%f' %
#                      conf_share)
fca_sorted.sort_based_f_measure()#sort_based_conf_supp(conf_share=conf_share)
print('Time for sorting hypothesis: %s'
      % hms_string(time.time() - sort_start_time))
log_params_file.write('Time for sorting hypothesis: %s\n'
      % hms_string(time.time() - sort_start_time))


print('Select best hypothesis.')
log_params_file.flush()
select_start_time = time.time()
fca_sorted.select_hyp_based_on_true_more_false()
log_params_file.write('Selected hypothesis number:\n')
for cl in lst_y:
    if cl in fca_sorted.taked_fc:
        log_params_file.write('class %s: %d\n'
                              % (str(cl), len(fca_sorted.taked_fc[cl])))
    else:
        log_params_file.write('class %s: 0\n' % str(cl))
print('Time for selecting best hypothesis: ',
      hms_string(time.time() - select_start_time))
log_params_file.write('Time for selecting best hypothesis: %s\n' %
      hms_string(time.time() - select_start_time))


def save_fc(taked_fc, file_path):
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(file_path, 'w') as file:
        json.dump(taked_fc, file)


fc_path = os.path.join(os.path.dirname(log_params_file_path),
                       'selected-fc_' + str(max_counter+1) + '.json')

save_fc(fca_sorted.taked_fc, file_path=fc_path)

fnn = FNN(fca_sorted.taked_fc)
fnn_model = FNN_Model(fnn, upper_lat, X_train_bin, y_train, X_valid, y_valid)

class Params(object):
    def __init__(self):
        self.learning_rate = 1.00
        self.num_epochs = 1
        self.batch_size = 1


def train(fnn_model, params, X_cut, y_cut):
    s_gen = Data_Gen(X_cut, y_cut, params.batch_size)
    previous_losses = []
    num_steps = 0
    for epoch in range(params.num_epochs):
        for batch in s_gen.gen():
            start_time = time.time()
            X_feed = batch['X_feed']
            y_feed = batch['y_feed']
            step_loss = fnn_model.step(X_feed, y_feed,
                                       params.learning_rate)
            curr_step_time = (time.time() - start_time)
            if (num_steps % 10) == 0:
                print('step_time: %f, step perplexity: %f' % (curr_step_time,
                                                              step_loss))
            if (len(previous_losses) > 10
                and step_loss >= min(previous_losses[-10:])):
                params.learning_rate *= 0.99
                print(params.learning_rate)
            previous_losses.append(step_loss)
            if (len(previous_losses) > 300 #and y_feed == 1
                and step_loss >= min(previous_losses[-300:])):
                return
            num_steps += 1
    log_params_file.write('Num of epochs = %d, num of steps = %d\n' %
                          (params.num_epochs, num_steps))
            

params = Params()


X_train_nparray, y_train_nparray = np.array(X_train_bin), np.array(y_train)

log_params_file.flush()
nn_start_time = time.time()
train(fnn_model, params, X_train_nparray, y_train_nparray)
print('Time for NN training: %s'
      % hms_string(time.time() - nn_start_time))
log_params_file.write('Time for NN training: %s'
      % hms_string(time.time() - nn_start_time))


def statistics(fnn_model, X_test, y_test, log_params_file, r1=0.0):
    log_params_file.write('---------------------------------\n')
    log_params_file.write('adjust rseult to = %f\n' % r1)
    fnn_working_model = copy.deepcopy(fnn_model)
    tp, tn, fp, fn, test_1_num, test_0_num = 0, 0, 0, 0, 0, 0
    for i in range(len(X_test)):
        results = fnn_working_model.output_calc(X_test[i])
        y1 = {}
        for cl in lst_y:
            y1[cl] = results[cl]['o']
        r = 0
        for cl in fnn_working_model.NN.neuron_inputs:
            r += cl * y1[cl]
        if y_test[i] == round(r+r1) and y_test[i] != 0: tp += 1
        elif y_test[i] == round(r+r1) and y_test[i] == 0: tn += 1
        elif y_test[i] != round(r+r1) and y_test[i] == 0: fp += 1
        else: fn += 1
        if y_test[i] > 0: test_1_num += 1
        else: test_0_num += 1
    print('Total Objects = ', len(X_test))
    print('Total Positive = ', test_1_num)
    print('Total Negative = ', test_0_num)
    print('Truly predicted Positive = ', tp)
    print('Truly predicted Negative = ', tn)
    print('False predicted Positive = ', fp)
    print('False predicted Negative = ', fn)
    log_params_file.write('Performance on test set:\n')
    log_params_file.write('Test Objects = %d\n' % len(X_test))
    log_params_file.write('Positives = %d\n' % test_1_num)
    log_params_file.write('Negatives = %d\n' % test_0_num)
    log_params_file.write('Truly predicted Positive = %d\n' % tp)
    log_params_file.write('Truly predicted Negative = %d\n' % tn)
    log_params_file.write('False predicted Positive = %d\n' % fp)
    log_params_file.write('False predicted Negative = %d\n' % fn)
    acc = ((tp + tn)/len(X_test))
    print('Accuracy = ', acc)
    log_params_file.write('Accuracy = %s\n' % str(acc))
    if (tp + fp) > 0:
        prec_nn = ((tp)/(tp + fp))
    else:
        prec_nn = 0
    print('Precision = ', prec_nn)
    log_params_file.write('Precision = %s\n' % str(prec_nn))
    if (tp + fn) > 0:
        recall_nn = ((tp)/(tp + fn))
    else:
        recall_nn = 0
    print('Recall = ', recall_nn)
    log_params_file.write('Recall = %s\n' % str(recall_nn))
    if (prec_nn + recall_nn) > 0:
        f_score = (2*prec_nn*recall_nn/(prec_nn + recall_nn))
    else:
        f_score = 0
    print('F-score = ', f_score)
    log_params_file.write('F-score = %s\n' % str(f_score))
    log_params_file.flush()


X_test_nparray = np.array(X_test_bin)
log_params_file.flush()
print('Testing NN started.')
statistics(fnn_model, X_test_nparray, y_test, log_params_file, r1=0.5)


log_params_file.write('Total passed time: %s\n' %
                      (hms_string(time.time() - prog_start_time)))

log_params_file.write('Ended at: %s' %
                      time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
log_params_file.flush()
#log_params_file.close()
