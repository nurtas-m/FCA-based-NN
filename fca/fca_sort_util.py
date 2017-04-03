
"""Utilities for sorting fca."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

class FCA_Sort():
    """Class with methods of sorting FCA.
    """
    def __init__(self, fc_list, X, y, y_classes):
        self.y_classes = y_classes
        self.X = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        self.taked_fc = {}
        self.sorted_index = 0
        self.sort_dataset_part = 1.0
        self.fc_classes = self.__reshape_fc_list(fc_list)
        self.total_fc_num = len(fc_list)
        self.classes_obj_num_dic = self.__calc_obj_num_in_classes()

    def __reshape_fc_list(self, fc_list):
        fc_classes = {cl : [] for cl in self.y_classes}
        for fc in fc_list:
            fc_classes[fc.hyp_type].append(fc)
        return fc_classes

    def __calc_obj_num_in_classes(self):
        classes_obj_num_dic = {cl : 0 for cl in self.y_classes}
        for obj_target in self.y:
            classes_obj_num_dic[obj_target[0]] += 1
        return classes_obj_num_dic

    def __calc_f_value(self, fc):
        f_value = 0
        if ((fc.tp_num + fc.fp_num) > 0 and (fc.tp_num + fc.fn_num) > 0):
            prec = (fc.tp_num / (fc.tp_num + fc.fp_num))
            recall = (fc.tp_num / (fc.tp_num + fc.fn_num))
            if (prec + recall) > 0:
                f_value = (2 * prec * recall / (prec + recall))
        return f_value


    def sort_based_f_measure(self, sort_dataset_part = 0.2):
        #num_consider_fc = int(len(self.fc_list) * sort_dataset_part)
        #self.sort_dataset_part = sort_dataset_part
        self.sorted_index = 0
        for cl in self.fc_classes:
            for i_fc in range(len(self.fc_classes[cl]) - 1):
                #print('Sorted %d out of %d. Only %.2f part will be sorted.' %
                #      (i, num_consider_fc, sort_dataset_part))
                if (i_fc % 10 == 0):
                    print('Sorted %d out of %d.' % (self.sorted_index,
                                                    self.total_fc_num))
                # Сортировка гипотез
                prior_f_value = self.__calc_f_value(self.fc_classes[cl][i_fc])
                if prior_f_value == 0:
                    tmp_fc = copy.deepcopy(self.fc_classes[cl][i_fc])
                    self.fc_classes[cl].pop(i_fc)
                    self.fc_classes[cl].append(tmp_fc)
                    continue
                prior_ind = i_fc
                for j in range(i_fc + 1, len(self.fc_classes[cl])):
                    # Отбор лучшей гипотезы из оставшегося массива
                    curr_f_value = self.__calc_f_value(self.fc_classes[cl][j])
                    if curr_f_value == 0:
                        tmp_fc = copy.deepcopy(self.fc_classes[cl][j])
                        self.fc_classes[cl].pop(j)
                        self.fc_classes[cl].append(tmp_fc)
                        continue
                    if curr_f_value > prior_f_value:
                        prior_f_value = curr_f_value
                        prior_ind = j
                if i_fc != prior_ind:
                    tmp_fc = copy.deepcopy(self.fc_classes[cl][i_fc])
                    self.fc_classes[cl][i_fc] =\
                        copy.deepcopy(self.fc_classes[cl][prior_ind])
                    self.fc_classes[cl][prior_ind] = copy.deepcopy(tmp_fc)
                self.sorted_index += 1
                #if self.sorted_index > round(num_consider_fc * sort_dataset_part):
                #    return


    def sort_based_conf_supp(self, conf_share=0.7, sort_dataset_part = 1.0):
        #num_consider_fc = int(len(self.fc_list) * sort_dataset_part)
        self.sort_dataset_part = sort_dataset_part
        self.sorted_index = 0
        for cl in self.fc_classes:
            for i_fc in range(len(self.fc_classes[cl]) - 1):
                if (i_fc % 10 == 0):
                    print('Sorted %d out of %d.' % (self.sorted_index,
                                                    self.total_fc_num))
                # Сортировка гипотез
                if (self.fc_classes[cl][i_fc].tp_num +
                    self.fc_classes[cl][i_fc].fn_num) > 0:
                    prior_conf = (self.fc_classes[cl][i_fc].tp_num /
                                  (self.fc_classes[cl][i_fc].tp_num +
                                   self.fc_classes[cl][i_fc].fn_num))
                else:
                    prior_conf = 0
                prior_supp = (self.fc_classes[cl][i_fc].tp_num /
                              len(self.fc_classes[cl]))
                prior_conf_supp = (prior_conf*conf_share +
                                   prior_supp*(1 - conf_share))
                prior_ind = i_fc
                for j in range(i_fc + 1, len(self.fc_classes[cl])):
                    # Отбор лучшей гипотезы из оставшегося массива
                    if (self.fc_classes[cl][j].tp_num +
                        self.fc_classes[cl][j].fn_num) > 0:
                        curr_conf = (self.fc_classes[cl][j].tp_num /
                                     (self.fc_classes[cl][j].tp_num +
                                      self.fc_classes[cl][j].fn_num))
                    else:
                        curr_conf = 0
                    curr_supp = (self.fc_classes[cl][j].tp_num /
                                 len(self.fc_classes[cl]))
                    curr_conf_supp = (curr_conf*conf_share +
                                      curr_supp*(1 - conf_share))
                    if curr_conf_supp > prior_conf_supp:
                        prior_conf = curr_conf
                        prior_supp = curr_supp
                        prior_conf_supp = curr_conf_supp
                        prior_ind = j
                if i_fc != prior_ind:
                    tmp_fc = copy.deepcopy(self.fc_classes[cl][i_fc])
                    self.fc_classes[cl][i_fc] =\
                        copy.deepcopy(self.fc_classes[cl][prior_ind])
                    self.fc_classes[cl][prior_ind] = copy.deepcopy(tmp_fc)
                self.sorted_index = i_fc
                #if self.sorted_index > round(num_consider_fc * sort_dataset_part):
                #    return


    def select_hyp_based_on_added(self):
        self.taked_fc = {}
        heap_obj = []
        #num_consider_fc = int(len(self.fc_list) * self.sort_dataset_part)
        for i in range(num_consider_fc):
            init_heap_size = len(heap_obj)
            added_true_num, added_false_num, true_num, false_num = 0, 0, 0, 0

            if (self.fc_list[i].tp_num > self.fc_list[i].fn_num
                or self.fc_list[i].tn_num > self.fc_list[i].fp_num):
                for k in range(len(self.y)):
                    if k not in heap_obj:
                        for i_attr in self.fc_list[i].intent.split(','):
                            fc_exist_in_obj = True
                            if (self.X[k][int(i_attr)-1] == 0):
                                fc_exist_in_obj = False
                            if fc_exist_in_obj:
                                if (self.y[k] == self.fc_list[i].hyp_type):
                                    added_true_num += 1
                                    heap_obj.append(k)
                                else:
                                    added_false_num += 1

            if len(heap_obj) > init_heap_size:
                if (self.fc_list[i].hyp_type in self.taked_fc):
                    self.taked_fc[self.fc_list[i].hyp_type].append(
                        {self.fc_list[i].intent:
                         {'Added True' : added_true_num,
                          'Added False' : added_false_num}})
                else:
                    self.taked_fc[self.fc_list[i].hyp_type] =\
                    [{self.fc_list[i].intent :
                      {'Added True' : added_true_num,
                       'Added False' : added_false_num}}]


    def select_hyp_based_on_true_more_false(self,
                                            part_of_class_with_obj_thresh=0.1,
                                            true_false_ratio_thresh=4,
                                            fc_attr_num_ratio_thresh=2):
        self.taked_fc = {cl : [] for cl in self.y_classes}
        #num_consider_fc = int(len(self.fc_list) * self.sort_dataset_part)
        i = 0
        tp_num, tn_num, fp_num, fn_num = 0, 0, 0, 0
        attr_num = len(self.X[0])
        uncondit_include_num = int(attr_num / len(self.y_classes))
        for cl in self.fc_classes:
            for i_fc in range(len(self.fc_classes[cl])):
                if i_fc <= uncondit_include_num:
                    self.taked_fc[cl].append(
                        {self.fc_classes[cl][i_fc].intent:
                         {'True-Pos' : self.fc_classes[cl][i_fc].tp_num,
                          'True-Neg' : self.fc_classes[cl][i_fc].tn_num,
                          'False-Pos' : self.fc_classes[cl][i_fc].fp_num,
                          'False-Neg' : self.fc_classes[cl][i_fc].fn_num}})
                elif (# Check if FC's predicted tp number true_false_ratio_thresh
                      # times more than fp
                      (self.fc_classes[cl][i_fc].tp_num >=
                       (self.fc_classes[cl][i_fc].fp_num *
                        true_false_ratio_thresh))
                      # Check if FC cover more than part_of_class_with_obj_thresh
                      # part of all objects that belong to this class
                      and (self.fc_classes[cl][i_fc].tp_num >=
                           (self.classes_obj_num_dic[cl] *
                               part_of_class_with_obj_thresh))
                      # Check if selected FC already too much compared to
                      # number of attributes
                      and (len(self.taked_fc[cl]) <=
                           (attr_num * fc_attr_num_ratio_thresh))):
                    self.taked_fc[cl].append(
                        {self.fc_classes[cl][i_fc].intent:
                         {'True-Pos' : self.fc_classes[cl][i_fc].tp_num,
                          'True-Neg' : self.fc_classes[cl][i_fc].tn_num,
                          'False-Pos' : self.fc_classes[cl][i_fc].fp_num,
                          'False-Neg' : self.fc_classes[cl][i_fc].fn_num}})
