import segmentation_models as sm
import generator
import tensorflow as tf
from base_net import BaseNet
from data_descriptor import DataDescriptor
import numpy as np


class SMUNet(BaseNet):

    def __init__(self, data_descriptor: DataDescriptor):
        super().__init__("sm_unet", data_descriptor)
        self.loss = sm.losses.bce_jaccard_loss
        self.backbone = "mobilenetv2"
        self._unique_file_id = self.build_file_name(self.input_shape, self.epochs, self.backbone)
        self._model_callbacks = self.build_model_callbacks(self._unique_file_id)
        self._model = None

    def configure(self,
                  epochs=20,
                  train_batch_size=8,
                  test_batch_size=1,
                  input_shape=(224, 224, 3),
                  encoder_weights="imagenet",
                  backbone="mobilenetv2",
                  optimizer="Adam",
                  loss=sm.losses.bce_jaccard_loss):

        super().configure(epochs, train_batch_size, test_batch_size, input_shape, encoder_weights, optimizer)
        self.backbone = backbone
        self.loss = loss
        self.save_model_params(self._unique_file_id)

    def save_model_params(self, unique_file_id):
        params_file_path = self.build_params_file_path(unique_file_id)
        with open(params_file_path, "a") as f:
            f.write("BACKBONE = " + self.backbone + "\n")
            f.write("LOSS = " + str(self.loss.name) + "\n")

    def train(self):
        super().train()

        # Train model
        self._model = sm.Unet(self.backbone, classes=self.data.get_n_classes(), encoder_weights=self.encoder_weights, input_shape=self.input_shape)
        self._model.compile(self.optimizer, loss=self.loss, metrics=[sm.metrics.iou_score])
        train_gen = generator.segmentation_generator(images_path=self.data.get_train_rgb_path(),
                                                     mask_path=self.data.get_train_mask_path(),
                                                     batch_size=self.train_batch_size,
                                                     n_classes=self.data.get_n_classes(),
                                                     input_width=self.input_shape[0],
                                                     input_height=self.input_shape[1],
                                                     output_width=self.input_shape[0],
                                                     output_height=self.input_shape[1])

        val_gen = generator.segmentation_generator(images_path=self.data.get_test_rgb_path(),
                                                   mask_path=self.data.get_test_mask_path(),
                                                   batch_size=self.test_batch_size,
                                                   n_classes=self.data.get_n_classes(),
                                                   input_width=self.input_shape[0],
                                                   input_height=self.input_shape[1],
                                                   output_width=self.input_shape[0],
                                                   output_height=self.input_shape[1])

        self._model.fit_generator(train_gen,
                                  steps_per_epoch=self.get_steps_per_epoch(),
                                  epochs=self.epochs,
                                  validation_data=val_gen,
                                  validation_steps=self.get_validation_steps(),
                                  callbacks=[self._model_callbacks])

    def save_tf_lite(self):

        # Load model with best saved params
        saved_model_file_path = self.build_saved_model_file_path(self._unique_file_id)
        self._model.load_weights(saved_model_file_path)

        # Convert the model.
        converter = tf.lite.TFLiteConverter.from_keras_model(self._model)
        converter.dump_graphviz_dir = self.build_saved_model_folder(self._unique_file_id)
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
        tf_results = self._model(tf.constant(input_data))

        # Compare the result.
        for tf_result, tflite_result in zip(tf_results, tflite_results):
            np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)




