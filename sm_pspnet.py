import segmentation_models as sm
from base_net import BaseNet
from data_descriptor import DataDescriptor


class SMPSPNet(BaseNet):

    _model_name = "sm_pspnet"

    def __init__(self,
                 data_descriptor: DataDescriptor,
                 epochs=20,
                 train_batch_size=8,
                 test_batch_size=1,
                 input_shape=(288, 288, 3),
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

        self.save_additional_model_params()
        self._model = self.create_model()

    def save_additional_model_params(self):
        params_file_path = self.build_params_file_path()
        with open(params_file_path, "a") as f:
            f.write("PRETRAIN_ENCODER_WEIGHTS = " + self._encoder_weights + "\n")
            f.write("BACKBONE = " + self._backbone + "\n")
            f.write("LOSS = " + str(self._loss.name) + "\n")

    def create_model(self):
        return sm.PSPNet(self._backbone,
                         activation="softmax",
                         downsample_factor=16,
                         classes=self._data.get_n_classes(),
                         encoder_weights=self._encoder_weights,
                         input_shape=self._input_shape)

    def train(self):
        # Create model
        super().train()
        self.compile_model(self._model)
        self.fit_generator(self._model)
