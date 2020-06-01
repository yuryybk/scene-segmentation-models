import datetime
from data_descriptor import DataDescriptor
import data_loader
import math
import os
import numpy as np
import tensorflow as tf
import profiler


class BaseNet:

    _tensorboard_dir_base_path = "./tensorboard/fit/"
    _saved_model_base_path = "./saved_models/"
    _params_file_name = "params.txt"
    _model = None

    def __init__(self,
                 name,
                 data_descriptor: DataDescriptor,
                 epochs=None,
                 train_batch_size=None,
                 test_batch_size=None,
                 input_shape=None,
                 optimizer=None,
                 loss=None,
                 metrics=None,
                 steps_per_epoch=None,
                 steps_validation=None,
                 unique_file_id=None):

        self._unique_file_id = unique_file_id
        self._name = name
        self._data = data_descriptor
        self._epochs = epochs
        self._train_batch_size = train_batch_size
        self._test_batch_size = test_batch_size
        self._input_shape = input_shape
        self._optimizer = optimizer
        self._model_callbacks = self.build_model_callbacks(unique_file_id)
        self._loss = loss
        self._metrics = metrics

        if steps_per_epoch is not None:
            self._steps_per_epoch = steps_per_epoch
        else:
            self._steps_per_epoch = self.get_steps_per_epoch()

        if steps_validation is not None:
            self._steps_validation = steps_validation
        else:
            self._steps_validation = self.get_validation_steps()

        self.save_model_params(unique_file_id)

    def get_name(self):
        return self._name

    @staticmethod
    def build_unique_file_id(model_name, shape, epochs, data: DataDescriptor, run_for_check=False, backbone=None):
        shape_str = str(shape[0]) + "_" + str(shape[1])
        is_check = ""
        if run_for_check:
            is_check = "check_"
        file_prefix = '{}{}_ep{}_sh{}_{}'.format(is_check, model_name, epochs, shape_str, data.get_name())
        if backbone is not None:
            file_prefix = file_prefix + "_" + backbone
        date_time_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f")
        return file_prefix + "_" + date_time_name

    def build_tensorboard_callback(self, unique_file_id):
        log_dir = self._tensorboard_dir_base_path + unique_file_id + "/"
        return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq="batch", profile_batch=5)

    def build_saved_model_folder(self):
        saved_model_folder = self._saved_model_base_path + self._unique_file_id
        if not os.path.exists(saved_model_folder):
            os.makedirs(saved_model_folder)
        return saved_model_folder + "/"

    def build_tf_lite_file_path(self, unique_file_id):
        saved_model_folder = self.build_saved_model_folder()
        return saved_model_folder + unique_file_id + ".tflite"

    def build_saved_model_file_path(self, unique_file_id):
        saved_model_folder = self.build_saved_model_folder()
        return saved_model_folder + unique_file_id + ".h5"

    def get_saved_model_pb_file_path(self):
        saved_model_folder = self.build_saved_model_folder()
        return saved_model_folder + self._unique_file_id + "/saved_model.pb"

    def get_saved_model_file_path(self):
        return self.build_saved_model_file_path(self._unique_file_id)

    def build_params_file_path(self, unique_file_id):
        saved_model_folder = self.build_saved_model_folder()
        return saved_model_folder + self._params_file_name

    def build_saved_model_callback(self, unique_file_id):
        saved_model_path = self.build_saved_model_file_path(unique_file_id)
        return tf.keras.callbacks.ModelCheckpoint(saved_model_path, save_best_only=True, monitor="val_loss", mode='min')

    def build_model_callbacks(self, unique_file_id):
        tensorboard_callback = self.build_tensorboard_callback(unique_file_id)
        saved_models_callback = self.build_saved_model_callback(unique_file_id)
        return [tensorboard_callback, saved_models_callback]

    def build_frozen_graph_file_name(self):
        return self._unique_file_id + ".frozen"

    def build_frozen_graph_file_path(self):
        frozen_file_name = self.build_frozen_graph_file_name()
        saved_model_folder = self.build_saved_model_folder()
        return saved_model_folder + frozen_file_name

    def calculate_steps_per_epoch(self):
        count_files = data_loader.get_count_files_dir(self._data.get_train_rgb_path())
        return math.floor(count_files / self._train_batch_size)

    def print_train_params_summary(self):
        train_image_path = self._data.get_train_rgb_path()
        count_files = data_loader.get_count_files_dir(train_image_path)
        info = 'Epochs = {}, batch_size = {}, count_files = {}, steps_per_epoch = {}, n_classes = {}, steps_validation = {}'.format(self._epochs,
                                                                                                                                    self._train_batch_size,
                                                                                                                                    count_files,
                                                                                                                                    self._steps_per_epoch,
                                                                                                                                    self._data.get_n_classes(),
                                                                                                                                    self._steps_validation)
        print(info)

    def get_input_shape(self):
        return self._input_shape

    def get_validation_steps(self):
        return data_loader.get_count_files_dir(self._data.get_test_rgb_path())

    def get_steps_per_epoch(self):
        return self.calculate_steps_per_epoch()

    def get_model(self):
        return self._model

    def train(self):
        self.print_train_params_summary()

    def load_saved_model(self):
        saved_model_file_path = self.build_saved_model_file_path(self._unique_file_id)
        return tf.keras.models.load_model(saved_model_file_path, compile=False)

    def save_frozen_graph_tf1(self):
        session = tf.compat.v1.keras.backend.get_session()
        graph = session.graph
        with session.as_default():
            with graph.as_default():
                tf.compat.v1.keras.backend.set_learning_phase(0)
                model = tf.keras.models.load_model(self.get_saved_model_file_path())
                output_names = [out.op.name for out in model.outputs]
                frozen_graph = profiler.freeze_session(session, output_names=output_names)

                saved_model_folder = self.build_saved_model_folder()
                frozen_graph_file_name = self.build_frozen_graph_file_name()
                tf.compat.v1.train.write_graph(frozen_graph, saved_model_folder, frozen_graph_file_name, as_text=False)

    def save_frozen_graph_tf2(self):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

        # Convert Keras model to ConcreteFunction
        full_model = tf.function(self._model).get_concrete_function(tf.TensorSpec(self._model.inputs[0].shape, self._model.inputs[0].dtype))

        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 50)
        print("Frozen model layers: ")
        for layer in layers:
            print(layer)

        print("-" * 50)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)

        # Save frozen graph from frozen ConcreteFunction to hard drive
        saved_model_folder = self.build_saved_model_folder()
        frozen_graph_file_name = self.build_frozen_graph_file_name()
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                          logdir=saved_model_folder,
                          name=frozen_graph_file_name,
                          as_text=False)

    def save_tf_lite(self):

        # Load model with best saved params
        model = self.load_saved_model()

        # Convert the model.
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.dump_graphviz_dir = self.build_saved_model_folder()
        converter.debug_info = True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        tf_lite_model = converter.convert()
        tf_lite_model_path = self.build_tf_lite_file_path(self._unique_file_id)
        with open(tf_lite_model_path, "wb") as f:
            f.write(tf_lite_model)

    def validate_tf_lite_model(self):

        # Load TFLite model and allocate tensors.
        tf_lite_model_path = self.build_tf_lite_file_path(self._unique_file_id)
        interpreter = tf.lite.Interpreter(model_path=tf_lite_model_path)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test the TensorFlow Lite model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        tflite_results = interpreter.get_tensor(output_details[0]['index'])

        # Test the TensorFlow model on random input data.
        model = self.load_saved_model()
        self.compile_model(model)
        tf_results = model(tf.constant(input_data))

        # Compare the result.
        for tf_result, tflite_result in zip(tf_results, tflite_results):
            np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)

    def compile_model(self, model):
        pass

    def save_model_params(self, unique_file_id):
        params_file_path = self.build_params_file_path(unique_file_id)
        with open(params_file_path, "w") as f:
            f.write("NAME = " + self._name + "\n")
            f.write("DATASET = " + self._data.get_name() + "\n")
            f.write("INPUT SHAPE = " + str(self._input_shape) + "\n")
            f.write("EPOCHS = " + str(self._epochs) + "\n")
            f.write("TRAIN BATCH SIZE = " + str(self._train_batch_size) + "\n")
            f.write("TEST BATCH SIZE = " + str(self._test_batch_size) + "\n")
            f.write("OPTIMIZER = " + self._optimizer + "\n")
            f.write("STEPS_PER_EPOCH =" + str(self._steps_per_epoch) + "\n")
            f.write("STEPS_PER_VALIDATION = " + str(self._steps_per_epoch) + "\n")
