import segmentation_models as sm
from sm_unet import SMUNet
from sm_pspnet import SMPSPNet
from nyu_v2_descriptor import NYU2Data
import profiler
import tensorflow as tf

sm.set_framework('tf.keras')


def train_ny2_sm_unet_data_1():
    data_set = NYU2Data()
    sm_u_net = SMUNet(data_set,
                      epochs=1,
                      train_batch_size=4,
                      input_shape=(224, 224, 3),
                      steps_per_epoch=1,
                      steps_validation=1,
                      run_for_check=True,
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.MeanIoU(num_classes=13)])
    sm_u_net.train()
    sm_u_net.run_inference()

    # --------------------------

    sm_u_net.save_tf_lite()
    sm_u_net.validate_tf_lite_model()

    # --------------------------

    # profiler.calculate_flops_with_session_meta(sm_u_net.get_saved_model_file_path(),
    #                                            sm_u_net.build_saved_model_folder(),
    #                                            sm_u_net.get_input_shape())

    # --------------------------

    # sm_u_net.save_frozen_graph_tf2()
    #
    # profiler.calculate_flops_from_frozen_graph(sm_u_net.build_frozen_graph_file_path(),
    #                                            sm_u_net.build_saved_model_folder(),
    #                                            sm_u_net.get_input_shape())

    # --------------------------

    # profiler.calculate_flops_with_session_meta(sm_u_net.get_saved_model_file_path(),
    #                                            sm_u_net.get_input_shape())

    # --------------------------

    # profiler.calculate_flops_flopco(sm_u_net.get_model())


def train_ny2_sm_pspnet_data_1():
    data_set = NYU2Data()
    sm_psp_net = SMPSPNet(data_set,
                          epochs=1,
                          train_batch_size=4,
                          input_shape=(240, 240, 3),
                          steps_per_epoch=1,
                          steps_validation=1,
                          run_for_check=True,
                          backbone="resnet152",
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=[tf.keras.metrics.MeanIoU(num_classes=13)])
    sm_psp_net.train()
    sm_psp_net.run_inference()
    sm_psp_net.save_tf_lite()
    sm_psp_net.save_frozen_graph_tf2()
    profiler.calculate_flops_from_frozen_graph(sm_psp_net.build_frozen_graph_file_path(),
                                               sm_psp_net.build_saved_model_folder(),
                                               sm_psp_net.get_input_shape())


train_ny2_sm_unet_data_1()
# train_ny2_sm_pspnet_data_1()


