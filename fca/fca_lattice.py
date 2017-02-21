
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import numpy as np
import numbers
import re
import random


class FCA_Lattice():

    def __init__(self, src, y_map):
        self.src = src
        self.var_num = len(re.split(';|,|\t', self.src[0]))
        self.y_map = y_map
        #self.X, self.y, self.X_test, self.y_test, self.var_num = __train_test_split(self.src)

    def get_dataset(self, X_cols, y_col, raw_train_set, raw_test_set):
        row3 = re.split(';|,|\t', self.src[2])
        y_type_str = True
        if row3[y_col].replace('.','').isdigit():
            y_type_str = False
        total_obj_num = len(self.src)
        self.var_num = len(X_cols) + 1
        self.y_classes_sym_num_map, self.y_classes_num_sym_map,\
            self.y_classes_sym, self.y_classes_num =\
            self.__get_y_classes_original(y_col)
        X_train, y_train, X_test, y_test =\
           np.empty((0, self.var_num-1), float), np.empty((0, 1), float),\
           np.empty((0, self.var_num-1), float), np.empty((0, 1), float)
        for s in raw_train_set:
            lst=re.split(';|,|\t', s)
            lst_select = []
            for var_ind in X_cols:
                lst_select.append(lst[var_ind].strip())
            X_train = np.vstack([X_train, np.array(lst_select)])
            if not y_type_str:
                y_train = np.vstack([y_train,
                    np.array(float(lst[y_col].strip()))])
            else:
                y_train = np.vstack([y_train,
                    np.array(
                        self.y_classes_sym_num_map[lst[y_col].strip()])])
        for s in raw_test_set:
            lst=re.split(';|,|\t', s)
            lst_select = []
            for var_ind in X_cols:
                lst_select.append(lst[var_ind].strip())
            X_test = np.vstack([X_test, np.array(lst_select)])
            if not y_type_str:
                y_test = np.vstack([y_test,
                    np.array(float(lst[y_col].strip()))])
            else:
                y_test = np.vstack([y_test, np.array(
                    self.y_classes_sym_num_map[lst[y_col].strip()])])
        return X_train, y_train, X_test, y_test


    def split_dataset(self, X_cols, y_col, dataset_parts =
                      [0.45, 0.45, 0.1], shuffle_data=False):
        data_started_row = 0
        row1 = re.split(';|,|\t', self.src[0])
        row2 = re.split(';|,|\t', self.src[1])
        row3 = re.split(';|,|\t', self.src[2])
        y_type_str = True
        if row3[y_col].replace('.','').isdigit():
            y_type_str = False
        for var_ind in range(len(row1)):
            if (not row1[var_ind].isdigit() and
                row2[var_ind].replace('.','').isdigit()):
                data_started_row = 2
                break
            elif (not row2[var_ind].isdigit() and
                  row3[var_ind].replace('.','').isdigit()):
                data_started_row = 3
                break
        self.src = self.src[data_started_row:]
        total_obj_num = len(self.src)
        self.var_num = len(X_cols) + 1
        self.y_classes_sym_num_map, self.y_classes_num_sym_map,\
            self.y_classes_sym, self.y_classes_num =\
            self.__get_y_classes_original(y_col)
        X_fca, y_fca, X_nn, y_nn, X_test, y_test =\
           np.empty((0, self.var_num-1), float), np.empty((0, 1), float),\
           np.empty((0, self.var_num-1), float), np.empty((0, 1), float),\
           np.empty((0, self.var_num-1), float), np.empty((0, 1), float)
        if shuffle_data:
            index_list = [i for i in range(len(self.src))]
            random.shuffle(index_list)
            src_shuf = []
            for index in index_list:
                src_shuf.append(copy.deepcopy(self.src[index]))
            self.src = src_shuf
        i = 0
        fca_part = int(dataset_parts[0] * 100)
        nn_part = int((dataset_parts[0] + dataset_parts[1]) * 100)
        for s in self.src:
            lst=re.split(';|,|\t', s)
            lst_select = []
            for var_ind in X_cols:
                lst_select.append(lst[var_ind].strip())
            if i >= 0 and i < total_obj_num-1:
                if i % 100 < fca_part:
                #if i % 20 == 16 or i % 20 == 17:
                    X_fca = np.vstack([X_fca, np.array(lst_select)])
                    if not y_type_str:
                        y_fca = np.vstack([y_fca,
                            np.array(float(lst[y_col].strip()))])
                    else:
                        y_fca = np.vstack([y_fca,
                            np.array(
                                self.y_classes_sym_num_map[lst[y_col].strip()])])
                            #np.array(self.y_map[lst[y_col].strip()])])
                elif i % 100 < nn_part:
                #elif i % 20 == 18 or i % 20 == 19:
                    X_nn = np.vstack([X_nn, np.array(lst_select)])
                    if not y_type_str:
                        y_nn = np.vstack([y_nn,
                            np.array(float(lst[y_col].strip()))])
                    else:
                        y_nn = np.vstack([y_nn, np.array(
                            self.y_classes_sym_num_map[lst[y_col].strip()])])
                            #np.array(y_map[lst[y_col].strip()])])
                else:
                    X_test = np.vstack([X_test, np.array(lst_select)])
                    if not y_type_str:
                        y_test = np.vstack([y_test,
                            np.array(float(lst[y_col].strip()))])
                    else:
                        y_test = np.vstack([y_test, np.array(
                            self.y_classes_sym_num_map[lst[y_col].strip()])])
                            #np.array(y_map[lst[y_col].strip()])])
            i += 1
        return X_fca, y_fca, X_nn, y_nn, X_test, y_test


    def prepare_input_matrices(self, src, X_cols, y_col, y_map=None):
        data_started_row = 0
        row1 = re.split(';|,|\t', src[0])
        row2 = re.split(';|,|\t', src[1])
        row3 = re.split(';|,|\t', src[2])
        y_type_str = True
        if row3[y_col].replace('.','').isdigit():
            y_type_str = False
        for var_ind in range(len(row1)):
            if not row1[var_ind].isdigit() and row2[var_ind].replace('.','').isdigit():
                data_started_row = 2
                break
            elif not row2[var_ind].isdigit() and row3[var_ind].replace('.','').isdigit():
                data_started_row = 3
                break
        src = src[data_started_row:]
        total_obj_num = len(src)
        self.var_num = len(X_cols) + 1
        X, y = np.empty((0, self.var_num-1), float), np.empty((0, 1), float)
        i = 0
        for s in src:
            lst=re.split(';|,|\t', s)
            lst_select = []
            for var_ind in X_cols:
                lst_select.append(lst[var_ind].strip())
            if i >= 0 and i < total_obj_num-1:
                X = np.vstack([X, np.array(lst_select)])
                if not y_type_str:
                    y = np.vstack([y, np.array(float(lst[y_col].strip()))])
                else:
                    y = np.vstack([y, np.array(y_map[lst[y_col].strip()])])
            i += 1
        return X, y

    #default_num = 0
    #nondefault_num = 0
    #for i in range(len(y)):
    #    if y[i] == 1: default_num += 1
    #    else: nondefault_num += 1

    #def get_y_classes(self, y):
    #    lst_y = []
    #    for i in range(len(y)):
    #        if y[i][0] not in lst_y:
    #            lst_y.append(y[i][0])
    #    return sorted(lst_y)

    def __get_y_classes_original(self, y_col):
        y_classes_sym_num_map = {}
        y_classes_num_sym_map = {}
        y_classes_sym = []
        y_classes_num = []
        for line in self.src:
            line_split=re.split(';|,|\t', line)
            y_sym = line_split[y_col]
            y_num = self.y_map[y_sym]
            if y_sym not in y_classes_sym_num_map:
                y_classes_sym_num_map[y_sym] = y_num
                y_classes_num_sym_map[y_num] = y_sym
                y_classes_sym.append(y_sym)
                y_classes_num.append(y_num)
        return y_classes_sym_num_map, y_classes_num_sym_map,\
               y_classes_sym, y_classes_num


    def __define_var_type(self, X):
        self.var_type = {}
        self.var_unique_values = {}
        for j in range(self.var_num-1):
            counter = 0
            self.var_unique_values[str(j+1)] = []
            str_flag = False
            unique_int_counter = 0
            for i in range(X.shape[0]):
                #Check if only one value is not digit and this value not occur
                #before
                if not X[i][j].replace('.','',1).replace('-','').isdigit():
                    #and X[i][j] not in self.var_unique_values[str(j+1)]:
                    str_flag = True
                    #if X[i][j] not in self.var_unique_values[str(j+1)]:
                    #    counter += 1
                    #    self.var_unique_values[str(j+1)].append(X[i][j])
                #Check if value not occur before
                #elif str_flag and X[i][j] not in self.var_unique_values[str(j+1)]:
                #    counter += 1
                #    self.var_unique_values[str(j+1)].append(X[i][j])
                if X[i][j] not in self.var_unique_values[str(j+1)]:
                    counter += 1
                    self.var_unique_values[str(j+1)].append(X[i][j])
                    if X[i][j].replace('.','',1).replace('-','').isdigit():
                        unique_int_counter += 1
                #elif counter > 9:
                #    self.var_type[str(j+1)] = 'int'
                #    self.var_unique_values[str(j+1)] = []
                #    break
            #else:
            if (not str_flag and len(self.var_unique_values[str(j+1)]) > 9):
                self.var_type[str(j+1)] = 'int'
                self.var_unique_values[str(j+1)] = []
            elif (str_flag and unique_int_counter > 9):
                self.var_type[str(j+1)] = 'int-bl'
                upd_unique_values = []
                for uniq_val in self.var_unique_values[str(j+1)]:
                    if not uniq_val.replace('.', '', 1).replace('-', '', 1).isdigit():
                        upd_unique_values.append(uniq_val)
                self.var_unique_values[str(j+1)] = copy.deepcopy(upd_unique_values)
            else:
                self.var_type[str(j+1)] = 'str'
            #elif counter <= 9:
            #    self.var_type[str(j+1)] = 'enum'

    def __transpose(self, X):
        X_t = []
        for j in range(len(X[0])):
            lst = []
            for i in range(len(X)):
                lst.append(X[i][j])
            X_t.append(lst)
        return X_t


    def __calc_avg(self, x):
        s = 0
        n = 0
        average = None
        max_value = None
        min_value = None
        if x[0].replace('.','',1).replace('-','').isdigit():
            max_value = float(x[0])
            min_value = float(x[0])
        for i in range(len(x)):
            if (x[i] != ''
                and x[i].replace('.', '', 1).replace('-', '').isdigit()):
                if max_value == None:
                    max_value = float(x[i])
                    min_value = float(x[i])
                s += float(x[i])
                if float(x[i]) > max_value:
                    max_value = float(x[i])
                if float(x[i]) < min_value:
                    min_value = float(x[i])
                n += 1
        if n > 0:
            average = s/n
        return average, max_value, min_value


    def map_var_attr(self, X, y):
        #global i, j, cnt
        self.__define_var_type(X)
        self.counter = 0
        self.var_attr_map ={}
        self.attr_var_map = {}
        X_t = self.__transpose(X)
        for i in range(self.var_num-1):
            if self.var_type[str(i+1)] == 'int':
                avg_value, max_value, min_value = self.__calc_avg(X_t[i])
                self.var_attr_map[str(i+1)] = \
                    {'int': {str(self.counter+1) : '> '+str(avg_value),
                             str(self.counter+2) : '<= '+str(avg_value)}}
                self.attr_var_map[str(self.counter+1)] = \
                    {str(i+1) : {'Type': 'int', 'Y_0': avg_value,
                                 'sigma': max_value-min_value}}
                self.attr_var_map[str(self.counter+2)] = \
                    {str(i+1) : {'Type': 'int', 'Y_0': avg_value,
                                 'sigma': max_value-min_value}}
                self.counter += 2
            elif self.var_type[str(i+1)] == 'enum':
                self.var_attr_map[str(i+1)]={ 'enum': {} }
                for j in self.var_unique_values[str(i+1)]:
                    self.counter += 1
                    self.var_attr_map[str(i+1)]['enum'][str(self.counter)] = j
                    self.attr_var_map[str(self.counter)] =\
                        {str(i+1) : {'Type': 'enum', 'Value': j}}
            elif self.var_type[str(i+1)] == 'int-bl':
                avg_value, max_value, min_value = self.__calc_avg(X_t[i])
                self.var_attr_map[str(i+1)] = \
                    {'int-bl': {str(self.counter+1) : '> ' + str(avg_value),
                                str(self.counter+2) : '<= ' + str(avg_value)}}
                self.attr_var_map[str(self.counter+1)] = \
                    {str(i+1) : {'Type' : 'int-bl', 'Y_0' : avg_value,
                                 'sigma' : max_value-min_value}}
                self.attr_var_map[str(self.counter+2)] = \
                    {str(i+1) : {'Type' : 'int-bl', 'Y_0' : avg_value,
                                 'sigma' : max_value-min_value}}
                self.counter += 2
                for val in self.var_unique_values[str(i+1)]:
                    self.counter += 1
                    self.var_attr_map[str(i+1)]['int-bl'][str(self.counter)] =\
                        {'Value' : val, 'Mapped Index' : max_value + 1}
                    self.attr_var_map[str(self.counter)] = \
                        {str(i+1) : {'Type' : 'int-bl', 'Value' : val}}
            else:
                self.var_attr_map[str(i+1)]={'str': {}}
                #ind = 0
                #min_val = min([float(s) for s in self.var_unique_values[str(i+1)]
                #               if s.replace('.','',1).replace('-','',1).isdigit()])
                arr = [float(s) for s in self.var_unique_values[str(i+1)]
                       if s.replace('.','',1).replace('-','',1).isdigit()]
                max_val = 0
                if len(arr) > 0:
                    max_val = round(max(arr))
                for val in self.var_unique_values[str(i+1)]:
                    self.counter += 1
                    if val.replace('.','',1).replace('-','',1).isdigit():
                        self.var_attr_map[str(i+1)]['str'][str(self.counter)] =\
                            {'Value' : val, 'Mapped Index' : float(val)}
                    else:
                        self.var_attr_map[str(i+1)]['str'][str(self.counter)] =\
                            {'Value' : val, 'Mapped Index' : max_val + 1}
                        #self.var_attr_map[str(i+1)]['str'][str(self.counter)] =\
                        #    {'Value' : val, 'Mapped Index' : float(len(self.var_unique_values[str(i+1)])-ind)}
                    self.attr_var_map[str(self.counter)] = \
                        {str(i+1) : {'Type' : 'str', 'Value' : val}}
                    #ind += 1
                    max_val += 1


    def sym2num(self, X, y):
        #global i, j, cnt
        #self.map_var_attr(X, y)
        X_num = np.empty((0, self.var_num-1), float)
        for i in range(len(y)):
            lst = []
            for var in range(self.var_num-1):
                if list(self.var_attr_map[str(var+1)].keys())[0] == 'str':
                    for val_map in self.var_attr_map[str(var+1)]['str'].values():
                        if X[i][var] == val_map['Value']:
                            lst.append(val_map['Mapped Index'])
                elif list(self.var_attr_map[str(var+1)].keys())[0] == 'int-bl':
                    for val_map in self.var_attr_map[str(var+1)]['int-bl'].values():
                        if isinstance(val_map, dict):
                            if X[i][var] == val_map['Value']:
                                lst.append(val_map['Mapped Index'])
                            else:
                                lst.append(float(X[i][var]))
                else:
                    lst.append(float(X[i][var]))
            X_num = np.vstack([X_num, np.array(lst)])
            y[i] = int(y[i])
        return X_num, y


    def scaling(self, X, y):
        #global i, j, cnt
        #self.map_var_attr(X, y)
        X_bin = []
        for i in range(len(y)):
            lst = [0 for any_ind in range(self.counter)]
            #cnt = -1
            for j in range(self.var_num-1):
                if list(self.var_attr_map[str(j+1)].keys())[0] == 'int':
                    for i_attr in self.var_attr_map[str(j+1)]['int']:
                        if self.var_attr_map[str(j+1)]['int'][i_attr].split(' ')[0] == '<=' \
                           and float(X[i][j]) <= float(self.var_attr_map[str(j+1)]['int'][i_attr].split(' ')[1]):
                            lst[int(i_attr)-1] = 1
                        elif self.var_attr_map[str(j+1)]['int'][i_attr].split(' ')[0] == '>' \
                             and float(X[i][j]) > float(self.var_attr_map[str(j+1)]['int'][i_attr].split(' ')[1]):
                            lst[int(i_attr)-1] = 1
                    #cnt += 1
                    #if float(X[i][j]) > float(self.var_attr_map[str(j+1)]['int'][str(cnt+1)].split(' ')[1]):
                    #    lst[]
                    #    lst[cnt] = 1
                    #else:
                    #    lst[cnt+1] = 1
                    #cnt += 1
                elif list(self.var_attr_map[str(j+1)].keys())[0] == 'int-bl':
                    for i_attr in self.var_attr_map[str(j+1)]['int-bl']:
                        if isinstance(self.var_attr_map[str(j+1)]['int-bl']
                                      [i_attr], dict):
                            if (X[i][j] ==
                                self.var_attr_map[str(j+1)]['int-bl'][i_attr]
                                ['Mapped Index']):
                                lst[int(i_attr)-1] = 1
                        else:
                            if ((self.var_attr_map[str(j+1)]['int-bl']\
                                [i_attr].split(' ')[0] == '<=')
                                and (float(X[i][j]) <=
                                     float(self.var_attr_map[str(j+1)]['int-bl']
                                           [i_attr].split(' ')[1]))):
                                lst[int(i_attr)-1] = 1
                            elif ((self.var_attr_map[str(j+1)]['int-bl']\
                                   [i_attr].split(' ')[0] == '>')
                                  and (float(X[i][j]) >
                                       float(self.var_attr_map[str(j+1)]['int-bl']
                                             [i_attr].split(' ')[1]))):
                                lst[int(i_attr)-1] = 1
                elif list(self.var_attr_map[str(j+1)].keys())[0] == 'str':
                    #cnt += len(self.var_attr_map[str(j+1)]['str'])
                    for k in self.var_attr_map[str(j+1)]['str'].keys():
                        if X[i][j] == self.var_attr_map[str(j+1)]['str'][k]['Mapped Index']:
                            lst[int(k)-1] = 1
                else:
                    #cnt += len(self.var_attr_map[str(j+1)]['enum'])
                    for k in self.var_attr_map[str(j+1)]['enum'].keys():
                        if X[i][j] == self.var_attr_map[str(j+1)]['enum'][k]['Mapped Index']:
                            lst[int(k)-1] = 1
            lst_copy = copy.deepcopy(lst)
            X_bin.append(lst_copy)
        return X_bin
