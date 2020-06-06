import segmentation_models as sm
import generator
from base_net import BaseNet
from data_descriptor import DataDescriptor


class SMUNet(BaseNet):

    _model_name = "sm_unet"

    def __init__(self,
                 data_descriptor: DataDescriptor,
                 epochs=20,
                 train_batch_size=8,
                 test_batch_size=1,
                 input_shape=(224, 224, 3),
                 encoder_weights="imagenet",
                 backbone="mobilenetv2",
                 optimizer="Adam",
                 loss=sm.losses.dice_loss,
                 metrics=[sm.metrics.iou_score],
                 steps_per_epoch=None,
                 steps_validation=None,
                 run_for_check=False):

        unique_file_id = self.build_unique_file_id(self._model_name, input_shape, epochs, data_descriptor, run_for_check, backbone)
        super().__init__(self._model_name,
                         data_descriptor,
                         epochs,
                         train_batch_size,
                         test_batch_size,
                         input_shape,
                         optimizer,
                         loss,
                         metrics,
                         steps_per_epoch,
                         steps_validation,
                         unique_file_id)

        self._encoder_weights = encoder_weights
        self._backbone = backbone

        self.save_additional_model_params(unique_file_id)

    def save_additional_model_params(self, unique_file_id):
        params_file_path = self.build_params_file_path()
        with open(params_file_path, "a") as f:
            f.write("PRETRAIN_ENCODER_WEIGHTS = " + self._encoder_weights + "\n")
            f.write("BACKBONE = " + self._backbone + "\n")
            f.write("LOSS = " + str(self._loss.name) + "\n")

    def train(self):
        super().train()

        # Train model
        self._model = sm.Unet(self._backbone, classes=self._data.get_n_classes(), encoder_weights=self._encoder_weights, input_shape=self._input_shape)
        self.compile_model(self._model)
        train_gen = generator.segmentation_generator(images_path=self._data.get_train_rgb_path(),
                                                     mask_path=self._data.get_train_mask_path(),
                                                     batch_size=self._train_batch_size,
                                                     n_classes=self._data.get_n_classes(),
                                                     input_width=self._input_shape[0],
                                                     input_height=self._input_shape[1],
                                                     output_width=self._input_shape[0],
                                                     output_height=self._input_shape[1],
                                                     do_augment=True)

        val_gen = generator.segmentation_generator(images_path=self._data.get_test_rgb_path(),
                                                   mask_path=self._data.get_test_mask_path(),
                                                   batch_size=self._test_batch_size,
                                                   n_classes=self._data.get_n_classes(),
                                                   input_width=self._input_shape[0],
                                                   input_height=self._input_shape[1],
                                                   output_width=self._input_shape[0],
                                                   output_height=self._input_shape[1],
                                                   do_augment=False)

        self._model.fit_generator(train_gen,
                                  steps_per_epoch=self._steps_per_epoch,
                                  epochs=self._epochs,
                                  validation_data=val_gen,
                                  validation_steps=self._steps_validation,
                                  callbacks=[self._model_callbacks])

    def compile_model(self, model):
        model.compile(self._optimizer, loss=self._loss, metrics=self._metrics)
