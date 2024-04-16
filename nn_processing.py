import project_variables
import json
from project_variables import *
import h5py
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
import random
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.applications import ResNet50
import tensorflow as tf
import numpy as np
import pickle
import gc
import shutil
import keras

def arch_simple_autoencoder(input_dim, output_dim='default', middle_activation='linear', final_activation='linear'):
    if output_dim == 'default':
        output_dim = input_dim
    input_layer = Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation=middle_activation)(input_layer)
    encoded = layers.Dense(32, activation=middle_activation)(encoded)
    decoded = layers.Dense(64, activation=middle_activation)(encoded)
    decoded = layers.Dense(output_dim, activation=final_activation)(decoded)
    autoencoder = Model(input_layer, decoded)
    return autoencoder


def arch_deep_autoencoder(input_dim, output_dim='default', middle_activation='linear', final_activation='linear'):
    if output_dim == 'default':
        output_dim = input_dim
    input_layer = Input(shape=(input_dim,))
    encoded = layers.Dense(256, activation=middle_activation)(input_layer)
    encoded = layers.Dense(256, activation=middle_activation)(encoded)
    encoded = layers.Dense(128, activation=middle_activation)(encoded)
    encoded = layers.Dense(128, activation=middle_activation)(encoded)
    encoded = layers.Dense(64, activation=middle_activation)(encoded)
    encoded = layers.Dense(64, activation=middle_activation)(encoded)
    # encoded = layers.Dense(32, activation=middle_activation)(encoded)

    encoded = layers.Dense(32,
                           # kernel_initializer=keras.initializers.Ones(),
                           kernel_constraint=keras.constraints.MinMaxNorm(min_value=0.0,
                                                                          max_value=1.0,
                                                                          rate=1,
                                                                          axis=0)
                           # activation=middle_activation,
                           )(encoded)

    decoded = layers.Dense(64, activation=middle_activation)(encoded)
    decoded = layers.Dense(64, activation=middle_activation)(decoded)
    decoded = layers.Dense(128, activation=middle_activation)(decoded)
    decoded = layers.Dense(128, activation=middle_activation)(decoded)
    decoded = layers.Dense(256, activation=middle_activation)(decoded)
    decoded = layers.Dense(256, activation=middle_activation)(decoded)
    decoded = layers.Dense(output_dim, activation=final_activation)(decoded)
    autoencoder = Model(input_layer, decoded)
    return autoencoder


def test_tensorflow(epochs=5, count=5):
    from time import time
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    list_of_durations, list_of_accuracies = [], []
    for i in range(count):
        start_time = int(time())
        print(f'\n Test number {i}')
        model = ResNet50(include_top=True, weights=None, input_shape=(32, 32, 3), classes=100, )
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=epochs, batch_size=2**10)
        duration = time() - start_time
        list_of_durations.append(duration)
        # Evaluate the model
        _, accuracy = model.evaluate(x_test, y_test)
        list_of_accuracies.append(accuracy)
        print(f'Cycle {i} time duration: {duration} sec. Test accuracy: {accuracy * 100:.2f}%')
    print(f'Mean cycle time: {sum(list_of_durations) / len(list_of_durations)} sec. '
          f'Mean test accuracy: {sum(list_of_accuracies) / len(list_of_accuracies) * 100:.2f}%')


def plot_history_graphs():

    from matplotlib import pyplot as plt

    history_list_paths = sorted([x for x in os.listdir(path_to_nn_models_dir) if x.endswith('history.pkl')])

    for hist_path in history_list_paths:
        with open(os.path.join(path_to_nn_models_dir, hist_path), 'rb') as f:
            hist = pickle.load(f)
            # remove file?
        data = hist['val_loss']
        title = f'{hist_path[:-12]}, min: {min(data):.4f}'
        plt.title(title)
        plt.plot(hist['val_loss'], label='val_loss')
        plt.plot(hist['loss'], label='loss')
        # plt.plot([x * 100 for x in hist['lr']], label='lr*100')
        plt.yscale('log')
        plt.ylim((0.01, 1))
        plt.grid(axis='y', linestyle='-', which='both')
        plt.legend()
        # ticks = [str(x) for x in np.arange(0.01, 0.1, 0.01)]
        # plt.yticks(ticks, minor=True)
        filename = os.path.join(path_to_nn_models_dir, f'{title.replace(": ", "_").replace(", ", "_")}.png')
        plt.savefig(filename, dpi=300)
        plt.clf()


def build_and_fit_nns(model_types_list=None):

    if model_types_list is None:
        model_types_list = ['arch_deep_autoencoder']

    with open(project_variables.path_to_analytics_of_dataset2_parameters, 'r') as file:
        d_dataset_description = json.load(file)
    with h5py.File(project_variables.path_to_third_step_dataset, 'r') as h5:
        # columns = h5.attrs['columns'].split('\t')
        array = h5['array']
        indices = h5['indices'][:]
        output_columns_indices = h5['indices_for_output_columns'][:]

        result_indexes = indices > (max(indices) - 3600 * 24)  # индексы за последние сутки
        other_indexes = result_indexes != 1

        print(f'Распределение выборок: тренировочная, валидационная, тестовая (данные за последние сутки)')
        result_data = array[result_indexes, :]
        train_data, val_data = train_test_split(array[other_indexes, :], test_size=0.05, random_state=42)

    print(train_data.shape, val_data.shape, result_data.shape)
    print('Создание цикла обучения моделей')

    # очистка директории с моделями
    shutil.rmtree(project_variables.path_to_nn_models_dir)

    for model_type in model_types_list:
        for i in range(project_variables.nn_calc_depth):
            print(f'Start calculate {i+1} of {project_variables.nn_calc_depth} models')
            model_name = f'{model_type}_{random.randint(100000, 999999)}'
            model_path = os.path.join(project_variables.path_to_nn_models_dir, model_name)

            model = None
            if model_type == 'simple_autoencoder':
                model = arch_simple_autoencoder(train_data.shape[1], output_columns_indices.shape[0])
            elif model_type == 'arch_deep_autoencoder':
                model = arch_deep_autoencoder(train_data.shape[1], output_columns_indices.shape[0])

            checkpoint_callback = callbacks.ModelCheckpoint(filepath=model_path,
                                                            save_best_only=True,
                                                            # verbose=True,
                                                            # save_weights_only=True,
                                                            monitor='loss',
                                                            )
            loss = tf.keras.losses.MeanAbsoluteError()
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
            early_stopping_callback = callbacks.EarlyStopping(monitor='loss',
                                                              patience=20,
                                                              mode='min',
                                                              min_delta=0.00001,
                                                              start_from_epoch=int(project_variables.nn_calc_depth/10))
            reduce_lr_callback = callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                             factor=0.2,
                                                             patience=project_variables.nn_calc_depth,
                                                             min_delta=0.0002,
                                                             min_lr=0.000001,
                                                             mode='min')
            model.compile(optimizer=optimizer, loss=loss)
            history = model.fit(train_data, train_data[:,output_columns_indices], epochs=int(project_variables.nn_calc_depth)*100,
                                batch_size=2**14,
                                validation_data=(val_data, val_data[:,output_columns_indices]),
                                callbacks=[
                                    checkpoint_callback,
                                    # reduce_lr_callback,
                                    early_stopping_callback
                                ],
                                # verbose=2,
                                )

            with open(os.path.join(model_path+'_history.pkl'), 'wb') as file_pi:
                pickle.dump(history.history, file_pi)
    plot_history_graphs()


def create_fourth_step_dataset():
    """
    Выбирается наилучшая модель и по ней строится датасет для интерпретации
    :return:
    """
    # chose and enable best nn model
    min_score, best_model_path = 99999999999, 'none_path'
    history_paths = [os.path.join(path_to_nn_models_dir, x) for x in os.listdir(path_to_nn_models_dir)
                     if x.endswith('history.pkl')]
    for model_name_history in history_paths:
        with open(model_name_history, 'rb') as f:
            hist = pickle.load(f)
        if min(hist['val_loss']) < min_score:
            best_model_path = model_name_history[:-12]
            min_score = hist['val_loss'][-1]
    print(f'min_score: {min_score}, model: {os.path.basename(best_model_path)}')

    # download dataset from third step
    with h5py.File(project_variables.path_to_third_step_dataset, 'r') as h5:
        array = h5['array'][:]

    # create new array with prediction of model
    model = tf.keras.models.load_model(best_model_path)
    new_array = model.predict(array)
    with h5py.File(project_variables.path_to_fourth_step_dataset, 'w') as hdf5_file:
        hdf5_file.create_dataset(name='array', data=new_array)


def create_big_fourth_step_dataset():
    """
    Выбирается наилучшая модель и по ней строится датасет для интерпретации
    :return:
    """
    # download dataset from third step
    with h5py.File(project_variables.path_to_third_step_dataset, 'r') as h5:
        array = h5['array'][:]
        indices = h5['indices'][:]
    with h5py.File(project_variables.path_to_fourth_step_dataset, 'w'):
        pass

    history_paths = [os.path.join(path_to_nn_models_dir, x) for x in os.listdir(path_to_nn_models_dir)
                     if x.endswith('history.pkl')]
    for i, model_history_path in enumerate(history_paths):
        print(f'{i+1}/{len(history_paths)} dataset calculation')
        model_path = model_history_path[:-12]
        # create new array with prediction of model
        model = tf.keras.models.load_model(model_path)
        new_array = model.predict(array)
        # print(new_array.shape)

        with h5py.File(project_variables.path_to_fourth_step_dataset, 'a') as hdf5_file:
            model_name = os.path.basename(model_path)
            group = hdf5_file.create_group(model_name)
            # hdf5_file[modeL_name]['array'] = new_array
            group.create_dataset(name='array', data=new_array,
                                 # chunks=(new_array.shape[0], 1)
                                 )
            group.create_dataset(name='indices', data=indices)

        del new_array


def lr_education(lr=0.001):
    """
    Функция дообучает все модели с сохраненным history с шагом обучения равным lr
    :param lr:
    :return:
    """
    d_history = {}
    for hist_file in os.listdir(path_to_nn_models_dir):
        if hist_file.endswith('.pkl'):
            with open(os.path.join(path_to_nn_models_dir, hist_file), 'rb') as f:
                history = pickle.load(f)
            d_history[min(history['val_loss'])] = hist_file

    # add arrays
    with h5py.File(project_variables.path_to_third_step_dataset, 'r') as h5:
        # columns = h5.attrs['columns'].split('\t')
        array = h5['array']
        indices = h5['indices'][:]
        output_columns_indices = h5['indices_for_output_columns'][:]
        result_indexes = indices > (max(indices) - 3600 * 24)  # индексы за последние сутки
        other_indexes = result_indexes != 1
        print(f'Распределение выборок: тренировочная, валидационная, тестовая (данные за последние сутки)')
        result_data = array[result_indexes, :]
        train_data, val_data = train_test_split(array[other_indexes, :], test_size=0.05, random_state=42)
    print(train_data.shape, val_data.shape, result_data.shape)

    for i, key in enumerate(sorted(d_history.keys())):
        print('______________________________________')
        print(f'Model number {i}, name {d_history[key][:-12]}, current_loss {key}')
        model_path = os.path.join(path_to_nn_models_dir, d_history[key][:-12])
        model = tf.keras.models.load_model(model_path)

        checkpoint_callback = callbacks.ModelCheckpoint(filepath=model_path,
                                                        save_best_only=True,
                                                        # verbose=True,
                                                        # save_weights_only=True,
                                                        monitor='loss',
                                                        )
        early_stopping_callback = callbacks.EarlyStopping(monitor='loss',
                                                          patience=5,
                                                          mode='min',
                                                          min_delta=0.00001,
                                                          start_from_epoch=int(project_variables.nn_calc_depth / 10))
        loss = tf.keras.losses.MeanAbsoluteError()
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
        model.compile(optimizer=optimizer, loss=loss)
        history = model.fit(train_data, train_data[:,output_columns_indices], epochs=int(project_variables.nn_calc_depth),
                            batch_size=2 ** 12,
                            validation_data=(val_data, val_data[:,output_columns_indices]),
                            callbacks=[
                                checkpoint_callback,
                                # reduce_lr_callback,
                                early_stopping_callback
                            ],
                            verbose=1,
                            )

        with open(os.path.join(model_path + '_history.pkl'), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        del model, history
        gc.collect()
        print('______________________________________')
        print('______________________________________')
    plot_history_graphs()



def main():
    print('# 4.0 nn_processing')
    # test_tensorflow()
    build_and_fit_nns()
    lr_education(lr=0.001)
    # lr_education(lr=0.0001)
    # plot_history_graphs()
    create_fourth_step_dataset()
    create_big_fourth_step_dataset()
    print('# 4.0 finish')
    pass

# def arch_simple_conv(input_dim):
    # from tensorflow.keras.layers import Conv1D, Conv1DTranspose
    # input_data = Input(shape=(*input_dim,))
    # encoded = Conv1D(32, input_dim[0], activation='linear')(input_data)
    # encoded = Conv1D(64, 1, activation='linear')(encoded)
    # decoded = Conv1DTranspose(1, input_dim[0], activation='linear')(encoded)
    # # decoded = Conv1DTranspose(1, 18, activation='linear')(decoded)
    # autoencoder = Model(input_data, decoded)
    # # print(autoencoder.summary())
    # return autoencoder


# class nn_bam_ns:
#     import tensorflow as tf
#     from struct import unpack
#     def run_nn(pack, test_mode = False):
#         # from protocol_MPSUiD_PSD import mpsuid_protocol_1
#         # from analytic_code import prepare_data_to_nn
#         # from nn_processing import create_autoencoder
#         import struct
#         import json
#
#         # Initialize status, index for answer, and car number variables
#         status = 'initiate status' # temp status
#         ind_temp = 0  # temp index for answer
#         car_number = 0  # temp car number
#         ind = None
#
#         # Define a function to create and return status information
#         def status_list():
#             # answer = 2
#             if status == 0 or status == 1:
#                 answer = status
#             # if test_mode:
#             #     answer = f"{ind} | {car_number} | {status}"
#             else:
#                 # if type(status) == str:
#                 answer = 2
#             return answer
#
#         # Load data dictionary from a pickle file. This file is constantly overwritten
#         with open(path_to_dict_BAM_NS_file, 'rb') as f:
#             data_dict = pickle.load(f)
#
#         header = struct.unpack("<BBBq", pack[:11])  # Unpack the header from the incoming packet
#         # If incoming message type is 1, process the data from MPSUiD to PSD system
#         type_of_messages = header[2]
#         if type_of_messages == 1:
#             bs = pack[10:10 + 19]  # заголовок данных от МПСУиД к системе ПСД
#             car_number = bs[3] & 0b00001111
#             car_type = bs[4] & 0b00001111
#             # Only process if car number and type are valid
#             if car_number != 0 and car_type != 0:
#
#                 ind = header[3]/1000
#                 ind_temp = 0  # temp
#
#                 # Update the latest package for this car
#                 data_dict['last_index'][car_number] = ind
#                 # Process the incoming packet with MPSUiD protocol
#                 d = mpsuid_protocol_1(pack)
#                 # Update the dictionary with the latest data
#                 for k in d.keys():
#                     data_dict[ind_temp][k + f'_{car_number}'] = d[k]
#                 status = 'added mpsud string'
#             else:
#                 status = 'no verify data, reset data'
#                 return status_list()
#         else:
#             status = 'no verify data, invalid package type'
#             return status_list()
#
#
#         # Validate data, specifically for train speed
#         if f"Скорость вращения 1-й оси_{car_number}" in data_dict[ind_temp].keys():
#             status = 'no valid data, by initialization time of the last packet'
#         else:
#             if len(data_dict[ind_temp].keys()) < EXPECTED_NUMBER_OF_PARAMETERS_FROM_THE_BIN_MPSUiD_DATA:
#                 status = 'no verify data, by dictionary size'
#
#         # Additional check if something is forgotten
#         if 'no' in status:
#             return status_list()
#         #############
#
#         # Transform file into an array for the neural network
#         del data_dict['last_index']
#         df_temp = pd.DataFrame.from_dict(data_dict, orient='index')
#         df_temp['Unnamed: 0'] = ind # bug
#         dataset, indexes, dataset_description = prepare_data_to_nn(path_to_analytic_raw_dataset_columns, df_temp)
#
#         # Initialize and run the neural network
#         model = create_autoencoder(input_dim=dataset.shape[1])
#         model.load_weights(path_to_nn_model)
#         dataset_loss = np.abs(model.predict(dataset, verbose=0) - dataset)
#
#         # Data interpretation
#         with open(path_to_nn_status, 'r') as file:
#             nn_status = json.load(file)
#
#         if nn_status['max_train_loss_by_ind'] > np.sum(dataset_loss):
#             status = 0
#         else:
#             status = 1
#         return status_list()
#     from analytic_code import prepare_data_to_nn
#     from nn_processing import create_autoencoder
#     from protocol_MPSUiD_PSD import mpsuid_protocol_1
#     data_dict = {0: {},
#                  'last_index': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}}
#     time_range_for_valid_data = 1  # sec
#     status = 'initiate status'
#     def __init__(self,
#                  path_to_analytic_raw_dataset_columns_init=path_to_analytic_raw_dataset_columns,
#                  path_to_nn_model_init=path_to_nn_model,
#                  path_to_nn_status=path_to_nn_status):
#         # dictionary for preparing data for a neural network
#         self.analytic_dict_for_data = self.download_analytic_data(path_to_analytic_raw_dataset_columns_init)
#         self.nn_model = self.init_nn(path_to_nn_model_init)
#         self.nn_sch_statuses = self.download_nn_statuses(path_to_nn_status)
#
#     def download_nn_statuses(self, path_to_nn_status):
#         with open(path_to_nn_status, 'r') as file:
#             nn_status = json.load(file)
#         return nn_status
#
#     def download_analytic_data(self, path):
#         f = open(path)
#         anal_dict = json.load(f)
#         return anal_dict
#
#     def init_nn(self, path_to_nn_model_init):
#         ### calculate model shape
#         shape_nn = 0
#         for key in self.analytic_dict_for_data.keys():
#             if self.analytic_dict_for_data[key]['nn_data_type'] == 'categotical':
#                 shape_nn += len(self.analytic_dict_for_data[key]['unique_values'])
#             elif self.analytic_dict_for_data[key]['nn_data_type'] == 'numerical':
#                 shape_nn += 1
#         ###
#         nn = create_autoencoder(shape_nn)
#         nn.load_weights(path_to_nn_model_init)
#         return nn
#
#     def process_pack(self, pack, mpsuid_protocol_1=mpsuid_protocol_1, prepare_data_to_nn=prepare_data_to_nn):
#         ind_main = 0  # основной индекс словаря
#         try:
#             header = self.unpack("<BBBq", pack[:11])  # Unpack the header from the incoming packet
#             # If incoming message type is 1, process the data from MPSUiD to PSD system
#             type_of_messages = header[2]
#             if type_of_messages == 1:
#                 bs = pack[10:10 + 19]  # заголовок данных от МПСУиД к системе ПСД
#                 car_number = bs[3] & 0b00001111
#                 car_type = bs[4] & 0b00001111
#                 # Only process if car number and type are valid
#                 if car_number != 0 and car_type != 0:
#                     ind = header[3] / 1000
#                     # Update the latest package for this car
#                     self.data_dict['last_index'][car_number] = ind
#                     # Process the incoming packet with MPSUiD protocol
#                     d = mpsuid_protocol_1(pack)
#                     # Validate data, specifically for train speed
#                     if d["Скорость вращения 1-й оси"] != 0:
#                         # Update the dictionary with the latest data
#                         for k in d.keys():
#                             self.data_dict[ind_main][k + f'_{car_number}'] = d[k]
#                         self.status = 'added mpsuid string'
#
#                         ############# Input data validation block
#                         time_range_for_valid_data = 1
#                         if sum(np.abs(ind - np.array(list(self.data_dict['last_index'].values())))
#                                >= time_range_for_valid_data) >= 1:
#                             self.status = 'no valid data, by initialization time of the last packet'
#                         else:
#                             if (len(self.data_dict[ind_main].keys())
#                                     < EXPECTED_NUMBER_OF_PARAMETERS_FROM_THE_BIN_MPSUiD_DATA):
#                                 self.status = 'no verify data, by dictionary size'
#                     else:
#                         self.status = 'no valid, the train is stationary'
#                 else:
#                     self.status = 'no verify data, reset data'
#             else:
#                 self.status = 'no verify data, invalid package type'
#
#             # return not valid data
#             if 'no' in str(self.status):
#                 return 2
#
#             # продолжаем разговор
#             temp_dict = {ind_main: self.data_dict[ind_main]}
#             df_temp = pd.DataFrame.from_dict(temp_dict, orient='index')
#             df_temp['Unnamed: 0'] = ind_main  # bug
#             dataset, indexes, dataset_description = prepare_data_to_nn(path_to_analytic_raw_dataset_columns, df_temp)
#
#             # run the neural network
#             # model = create_autoencoder(input_dim=dataset.shape[1])
#             # model.load_weights(path_to_nn_model)
#             prediction = self.nn_model.predict(dataset, verbose=0)
#             dataset_loss = np.abs(prediction - dataset)
#
#             # Data interpretation
#
#
#             if self.nn_sch_statuses['max_train_loss_by_ind'] > np.sum(dataset_loss):
#                 status = 0
#             else:
#                 status = 1
#             return status
#         except Exception as exc:
#             # на случай неверной байтовой строки, я полагаю
#             print('SOMETHING WRONG WITH NN_SCH, EXCEPTION:', exc)
#             return 2


# arch_simple_conv([1202,1])

# def data_downloader():
#     dataset = np.load(project_variables.path_to_nn_dataset)
#     indexes = np.load(project_variables.path_to_nn_dataset_indexes)
#     with open(project_variables.path_dataset_description, 'r') as file:
#         dataset_description = json.load(file)
#
#     cheking_string = 'dimension matches, dataset: {}, indexes: {}, description params {}'.format(
#         dataset.shape,
#         indexes.shape,
#         len(dataset_description.keys()),
#     )
#     if dataset.shape[0] == indexes.shape[0]:
#         print(cheking_string)
#     else:
#         print('ERROR: dataset dimension matches')
#         print(cheking_string)
#         return 0

    # return dataset, indexes, dataset_description


# def dataset_markup(dataset, indexes, dataset_description):
#     # получаем индексы с наличием скорости
#     for index, col in dataset_description.items():
#         if col == 'Скорость вращения 1-й оси_1':
#             velocity_col_number = int(index)
#             break
#     indexes_with_velocity = dataset[:, velocity_col_number] != 0
#     indexes_without_velocity = indexes_with_velocity == False
#
#     # получаем метки времени с данными за последние сутки и остальные
#     # примечание: по максимальному значению времени пока использовать нельзя
#     last_day_time_range = 60 * 60 * 24  # last day
#     indexes_last_cycle = indexes > (indexes[-1] - last_day_time_range)
#     indexes_not_last_cycle = indexes_last_cycle == False
#
#     # получаем метки времени с данными за последний месяц
#     # примечание: по максимальному значению времени пока использовать нельзя
#     last_mounth_time_range = 60 * 60 * 24 * 30  # last month
#     indexes_last_mounth = (indexes > (indexes[-1] - last_mounth_time_range)) & (
#                 (indexes[-1] - last_day_time_range) > indexes)
#     # indexes_not_last_cycle = indexes_last_cycle == False
#
#     # final_indexes
#     train_indexes = indexes_with_velocity * indexes_not_last_cycle
#     train_2_indexes = indexes_with_velocity * indexes_last_mounth
#     final_indexes = indexes_with_velocity * indexes_last_cycle
#
#     print('presence/absence of speed:', sum(indexes_with_velocity) / sum(indexes_without_velocity))
#     print('train_indexes: {:.2f}%, train_2_indexes: {:.2f}%, final_indexes {:.2f}%'.format(
#         100*sum(train_indexes) / sum(indexes_with_velocity),
#         100*sum(train_2_indexes) / sum(indexes_with_velocity),
#         100*sum(final_indexes) / sum(indexes_with_velocity)))
#     print('final_indexes/train_indexes:', sum(final_indexes) / sum(train_indexes))
#     # print('Размер датасета:', dataset.shape)
#
#     indexes_dict = {
#         'train': train_indexes,
#         'train_2': train_2_indexes,
#         'final': final_indexes,
#         'all_calculated_indexes': train_indexes | final_indexes,
#     }
#     return indexes_dict

#
# def calculating_the_output(dataset, model, indexes_dict):
#     dataset_loss = np.zeros_like(dataset)
#     dataset_loss_vel = np.abs(model.predict(dataset[indexes_dict['all_calculated_indexes']]) -
#                               dataset[indexes_dict['all_calculated_indexes']])
#
#     i2 = 0
#     for i, if_vel in enumerate(indexes_dict['all_calculated_indexes']):
#         if if_vel:
#             dataset_loss[i] = dataset_loss_vel[i2]
#             i2 += 1
#     return dataset_loss
#
# def creation_of_a_server_neural_network():
#     print('# 4.0 neural network manipulations')
#
#     # download dataset
#     dataset, indexes, dataset_description = data_downloader()
#
#     # dataset markup
#     indexes_dict = dataset_markup(dataset,
#                                   indexes,
#                                   dataset_description)
#
#     ### general topic
#
#     # base education callbacks and params
#     best_loss = np.inf
#     best_model = None
#     checkpoint_callback = callbacks.ModelCheckpoint(filepath=project_variables.path_to_nn_model, save_best_only=True,
#                                                     monitor='loss')
#     reduce_lr_callback = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, min_lr=0.000001,
#                                                      mode='min')
#     loss = tf.keras.losses.MeanAbsoluteError()
#     # cycle of model generating
#     model = create_autoencoder(input_dim=dataset.shape[1])
#     if project_variables.load_nn_model:
#         model.load_weights(project_variables.path_to_nn_model)
#     else:
#         for i in range(project_variables.nn_calc_depth):
#             print(f'Model number {i}')
#             # Создание модели автокодировщика
#             model = create_autoencoder(input_dim=dataset.shape[1])
#             # Оптимизатор и функция потерь
#             optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
#             # Коллбэки
#             early_stopping_callback = callbacks.EarlyStopping(monitor='loss', patience=8, mode='min')
#             # Компиляция модели
#             model.compile(optimizer=optimizer, loss=loss)
#
#             # Обучение модели
#
#             history = model.fit(dataset[indexes_dict['train']], dataset[indexes_dict['train']],
#                                 epochs=project_variables.nn_calc_depth*3,
#                                 batch_size=2 ** 10,
#                                 # validation_split=0.2,
#                                 callbacks=[checkpoint_callback, reduce_lr_callback, early_stopping_callback])
#
#         # Использование лучшей модели
#         model.load_weights(project_variables.path_to_nn_model)
#
#         # дообучение на малом шаге, на данных за последний месяц, необязательный шаг
#         optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
#         model.compile(optimizer=optimizer, loss=loss)
#         model.fit(dataset[indexes_dict['train_2']], dataset[indexes_dict['train_2']],
#                   epochs=project_variables.nn_calc_depth,
#                   batch_size=512,
#                   # validation_split=0.1,
#                   # shuffle=False,
#                   callbacks=[checkpoint_callback, reduce_lr_callback])
#
#     # calculating the output of a neural network over the entire dataset
#     dataset_loss = calculating_the_output(dataset, model, indexes_dict)
#     np.save(project_variables.path_to_output_loss_dataset, dataset_loss)
#     with open(project_variables.path_to_loss_dataset_indexes, 'wb') as file:
#         pickle.dump(indexes_dict, file)
#
#         if data_dict[ind_temp][f"Скорость вращения 1-й оси_{car_number}"] == 0:
#             status = 'no valid, the train is stationary'
#             return status_list()
#     # Save updated data_dict back to the pickle file
#     with open(path_to_dict_BAM_NS_file, 'wb') as f:
#         pickle.dump(data_dict, f)
#
#     # ############# Input data validation block
#     # time_range_for_valid_data = 1
#     # if sum(np.abs(ind - np.array(list(data_dict['last_index'].values()))) >= time_range_for_valid_data) >= 1:
#


if __name__ == '__main__':
    main()
    # n®n_b = nn_bam_ns()
    # print(nn_b.nn_model.summary())
    pass
