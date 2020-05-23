import tensorflow as tf
import datetime
from data_descriptor import DataDescriptor
import data_loader
import math
import os


class BaseNet:

    _tensorboard_dir_base_path = "./tensorboard/fit/"
    _saved_model_base_path = "./saved_models/"
    _name = ""
    _params_file_name = "params.txt"
    _unique_file_id = None

    def __init__(self, name, data_descriptor: DataDescriptor):
        self._name = name
        self.data = data_descriptor
        self.encoder_weights = 'imagenet'
        self.input_shape = (224, 224, 3)
        self.epochs = 20
        self.train_batch_size = 8
        self.test_batch_size = 1
        self.optimizer = "Adam"

    def configure(self,
                  epochs=20,
                  train_batch_size=8,
                  test_batch_size=1,
                  input_shape=(224, 224, 3),
                  encoder_weights="imagenet",
                  optimizer="Adam"):

        self.epochs = epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.input_shape = input_shape
        self.encoder_weights = encoder_weights
        self.optimizer = optimizer
        self.__save_model_params(self._unique_file_id)

    def get_name(self):
        return self._name

    def build_file_name(self, shape, epochs, backbone=None):
        shape_str = str(shape[0]) + "_" + str(shape[1])
        file_prefix = '{}_ep{}_sh{}'.format(self.get_name(), epochs, shape_str)
        if backbone is not None:
            file_prefix = file_prefix + "_" + backbone
        date_time_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        return file_prefix + "_" + date_time_name

    def build_tensorboard_callback(self, unique_file_id):
        log_dir = self._tensorboard_dir_base_path + unique_file_id + "/"
        return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq="batch", profile_batch=5)

    def build_saved_model_folder(self, unique_file_id):
        saved_model_folder = self._saved_model_base_path + unique_file_id
        if not os.path.exists(saved_model_folder):
            os.makedirs(saved_model_folder)
        return saved_model_folder + "/"

    def build_tf_lite_file_path(self, unique_file_id):
        saved_model_folder = self.build_saved_model_folder(unique_file_id)
        return saved_model_folder + unique_file_id + ".tflite"

    def build_saved_model_file_path(self, unique_file_id):
        saved_model_folder = self.build_saved_model_folder(unique_file_id)
        return saved_model_folder + unique_file_id + ".h5"

    def build_params_file_path(self, unique_file_id):
        saved_model_folder = self.build_saved_model_folder(unique_file_id)
        return saved_model_folder + self._params_file_name

    def build_saved_model_callback(self, unique_file_id):
        saved_model_path = self.build_saved_model_file_path(unique_file_id)
        return tf.keras.callbacks.ModelCheckpoint(saved_model_path, save_best_only=True, monitor="val_loss", mode='min')

    def build_model_callbacks(self, unique_file_id):
        tensorboard_callback = self.build_tensorboard_callback(unique_file_id)
        saved_models_callback = self.build_saved_model_callback(unique_file_id)
        return [tensorboard_callback, saved_models_callback]

    def calculate_steps_per_epoch(self):
        count_files = data_loader.get_count_files_dir(self.data.get_train_rgb_path())
        return math.floor(count_files / self.train_batch_size)

    def print_train_params_summary(self, train_image_path, train_batch_size, steps_per_epoch_param):
        count_files = data_loader.get_count_files_dir(train_image_path)
        info = 'Epochs = {}, batch_size = {}, count_files = {}, steps_per_epoch = {}, n_classes = {}'.format(self.epochs,
                                                                                                             train_batch_size,
                                                                                                             count_files,
                                                                                                             steps_per_epoch_param,
                                                                                                             self.data.get_n_classes())
        print(info)

    def get_validation_steps(self):
        return data_loader.get_count_files_dir(self.data.get_test_rgb_path())

    def get_steps_per_epoch(self):
        return self.calculate_steps_per_epoch()

    def train(self):
        steps_per_epoch = self.get_steps_per_epoch()
        self.print_train_params_summary(self.data.get_train_rgb_path(), self.train_batch_size, steps_per_epoch)

    def __save_model_params(self, unique_file_id):
        params_file_path = self.build_params_file_path(unique_file_id)
        with open(params_file_path, "w") as f:
            f.write("NAME = " + self._name + "\n")
            f.write("DATASET = " + self.data.get_name() + "\n")
            f.write("PRETRAIN_ENCODER_WEIGHTS = " + self.encoder_weights + "\n")
            f.write("INPUT SHAPE = " + str(self.input_shape) + "\n")
            f.write("EPOCHS = " + str(self.epochs) + "\n")
            f.write("TRAIN BATCH SIZE = " + str(self.train_batch_size) + "\n")
            f.write("TEST BATCH SIZE = " + str(self.test_batch_size) + "\n")
            f.write("OPTIMIZER = " + self.optimizer + "\n")

