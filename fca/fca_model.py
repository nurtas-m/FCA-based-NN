#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time


class FormalConcept():
    def __init__(self, intent, hyp_type, bin_col,
                 tp_num = 0, tn_num = 0, fn_num = 0, fp_num = 0):
        self.intent = intent
        self.hyp_type = hyp_type
        self.bin_col = bin_col
        self.tp_num = tp_num
        self.tn_num = tn_num
        self.fn_num = fn_num
        self.fp_num = fp_num


class FCA_Model():

    def __init__(self, X_bin, y, lst_y):
        self.X_bin = X_bin
        self.y = y
        self.lst_y = lst_y
        self.fc_attr_list = self.__calc_stats()
        self.fc_list = copy.deepcopy(self.fc_attr_list)
        self.fc_list_sorted = copy.deepcopy(self.fc_list)
        self.num_pos, self.num_neg = 0, 0


    def __get_hyp_type(self, y_classes):
        key_max, max_value = list(y_classes.items())[0]
        max_tp = max_value['True-Pos']
        for key, value in y_classes.items():
            if value['True-Pos'] > max_tp:
                key_max = key
                max_tp = value['True-Pos']
        return key_max


    def __calc_stats(self):
        attr_num = len(self.X_bin[0])
        self.num_pos, self.num_neg = 0, 0
        for i in range(len(self.y)):
            if self.y[i] == 0:
                self.num_neg += 1
            else:
                self.num_pos += 1
        fc_list = []
        for j in range(attr_num):
            bin_col = ''
            fc_with_pos_num, fc_with_neg_num = 0, 0
            for i in range(len(self.y)):
                if self.X_bin[i][j] == 1:
                    bin_col = bin_col + '1'
                    if self.y[i] == 1:
                        fc_with_pos_num += 1
                    elif self.y[i] == 0:
                        fc_with_neg_num += 1
                elif self.X_bin[i][j] == 0:
                    bin_col = bin_col + '0'
            hyp_type = 1
            if fc_with_pos_num < fc_with_neg_num:
                hyp_type = 0
            fc = FormalConcept(str(j+1), hyp_type, bin_col)
            fc_list.append(copy.deepcopy(fc))
        return fc_list


    def generate_hypothesis(self, lattice_layers_num=2,
                            freq_of_fc_in_class=0.02):
        attr_num = len(self.X_bin[0])
        heap = []
        num_fc_on_prev_layer = 0
        num_fc_on_curr_layer = attr_num
        start_gen_time = time.time()
        for i in range(lattice_layers_num):
            # Проходим по всем новым образованным формальным понятиям
            curr_gen_time = 0
            for j in range(num_fc_on_prev_layer, num_fc_on_curr_layer):
                start_curr_gen_time = time.time()
                # Проверяем, встречается ли данное понятие среди объектов
                if '1' in self.fc_list[j].bin_col:
                    # Проходим по всем атрибутам. Добавляем атрибут к ФП и образуем новое ФП, если
                    # существует хотя бы один объект с новым ФП
                    for k in range(attr_num):
                        # Проверяем, не существует ли в Формальном понятии
                        # уже атрибут X_pos[k].
                        # Если атрибут уже присутствует в ФП, тогда его
                        # добавлять мы не будем.
                        if (self.fc_list[k].intent not in
                            self.fc_list[j].intent.split(',')
                            and '1' in self.fc_list[k].bin_col):
                            new_col = copy.deepcopy(self.fc_list[j].bin_col)
                            y_classes ={x:{'True-Pos':0,'False-Pos':0}
                                        for x in self.lst_y}
                            pos_progn_num = 0
                            neg_progn_num = 0
                            for l in range(len(self.y)):
                                # Смотрим, есть ли у объекта данное Формальное
                                # понятие
                                if (self.fc_list[j].bin_col[l]
                                    == self.fc_list[k].bin_col[l]
                                    == '1'):
                                    # Формируем бинарную цепочку совпадения
                                    # прогноза исходного
                                    # формального понятия X_pos[j] и
                                    # добавляемого атрибута X_pos[k]
                                    # среди объектов
                                    y_classes[self.y[l][0]]['True-Pos'] =\
                                        y_classes[self.y[l][0]]['True-Pos'] + 1
                                    pos_progn_num += 1
                                else:
                                    if l < len(self.y) - 1:
                                        new_col = new_col[:l] + '0' +\
                                                  new_col[l+1:]
                                    else:
                                        new_col = new_col[:l] + '0'
                                    y_classes[self.y[l][0]]['False-Pos'] += 1
                                    neg_progn_num += 1
                            # Проверяем, был ли такой же паттерн бинарной
                            # цепочки у предыдущих
                            # Формальных понятий
                            if (new_col not in heap and '1' in new_col
                                and ((pos_progn_num >
                                     (self.num_pos*freq_of_fc_in_class))
                                     or (neg_progn_num >
                                         (self.num_neg*freq_of_fc_in_class)))):
                                hyp_type = self.__get_hyp_type(y_classes)
                                # Добавляем новое формальное понятие вместе с
                                # бинарной цепочкой
                                # и статистиками
                                intent = self.fc_list[j].intent + ',' +\
                                         self.fc_list[k].intent
                                fc = FormalConcept(intent, hyp_type, new_col)
                                self.fc_list.append(fc)
                                heap.append(new_col)
                curr_gen_time += time.time() - start_curr_gen_time
                if (j % 10) == 0:
                    print('Depth %d, current hypothesis %d out of %d.' %
                          (i, j, num_fc_on_curr_layer))
                    avg_gen_time = curr_gen_time / 10
                    expect_time = avg_gen_time * (num_fc_on_curr_layer - j)
                    curr_gen_time = 0
                    print('Current fc gen time: %f, Expected time to complete layer: %s'
                          % (avg_gen_time, hms_string(expect_time)))
                    print('Time passed: %s' %
                          hms_string(time.time() - start_gen_time))
            num_fc_on_prev_layer = num_fc_on_curr_layer
            num_fc_on_curr_layer = len(self.fc_list)
            print(i, num_fc_on_curr_layer)


    def calc_performance(self, X_valid_bin, y_valid_bin):
        for i_fc in range(len(self.fc_list)):
            for i_valid_obj in range(len(y_valid_bin)):
                fc_exist_in_obj = True
                for i_attr in self.fc_list[i_fc].intent.split(','):
                    if (X_valid_bin[i_valid_obj][int(i_attr)-1] > 0):
                        fc_exist_in_obj = False
                if fc_exist_in_obj:
                    if (self.fc_attr_list[int(i_attr)-1].hyp_type ==
                        y_valid_bin[i_valid_obj]):
                        self.fc_list[i_fc].tp_num += 1
                    else:
                        # False Alarm
                        self.fc_list[i_fc].fp_num += 1
                else:
                    if (self.fc_attr_list[int(i_attr)-1].hyp_type != 
                        y_valid_bin[i_valid_obj]):
                        self.fc_list[i_fc].tn_num += 1
                    else:
                        # Here FC expected to worked out, but didn't
                        self.fc_list[i_fc].fn_num += 1


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = int(sec_elapsed % 60)
    return "{}:{:>02}:{:>02}".format(h, m, s)
