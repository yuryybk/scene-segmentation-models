import segmentation_models as sm
from sm_unet import SMUNet
from sm_pspnet import SMPSPNet
from sm_fpnnet import SMFPNNet
from sm_linknet import SMLinkNet
from mobile_net_v2 import MobileNetV2
from nyu_v2_descriptor import NYU2Data
import profiler
import tensorflow as tf

sm.set_framework('tf.keras')


def check_ny2_sm_unet_data_1(input_shape=(224, 224, 3)):
    data_set = NYU2Data()
    sm_u_net = SMUNet(data_set,
                      epochs=1,
                      train_batch_size=4,
                      input_shape=input_shape,
                      steps_per_epoch=1,
                      steps_validation=1,
                      run_for_check=True,
                      loss=tf.keras.losses.CategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.MeanIoU(num_classes=13)])
    sm_u_net.train()
    # sm_u_net.run_inference()
    # sm_u_net.save_onnx()

    # --------------------------

    sm_u_net.save_tf_lite()
    sm_u_net.validate_tf_lite_model()

    # --------------------------

    profiler.calculate_flops_with_session_meta(sm_u_net.get_h5_saved_model_file_path(),
                                               sm_u_net.build_saved_model_folder(),
                                               sm_u_net.get_input_shape())

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


def check_ny2_mobile_net_v2(input_shape=(224, 224, 3)):
    data_set = NYU2Data()
    mobile_net_v2 = MobileNetV2(data_set,
                                epochs=1,
                                train_batch_size=4,
                                input_shape=input_shape,
                                steps_per_epoch=1,
                                steps_validation=1,
                                run_for_check=True,
                                loss=tf.keras.losses.CategoricalCrossentropy(),
                                metrics=[tf.keras.metrics.MeanIoU(num_classes=13)])
    mobile_net_v2.train()
    mobile_net_v2.run_inference()
    mobile_net_v2.save_tf_lite(best_saved=False)
    mobile_net_v2.validate_tf_lite_model(best_saved=False)
    mobile_net_v2.save_frozen_graph_tf2()
    profiler.calculate_flops_from_frozen_graph(mobile_net_v2.build_frozen_graph_file_path(),
                                               mobile_net_v2.build_saved_model_folder(),
                                               mobile_net_v2.get_input_shape())


def check_ny2_sm_pspnet_data_1(input_shape=(288, 288, 3)):
    data_set = NYU2Data()
    sm_psp_net = SMPSPNet(data_set,
                          epochs=1,
                          train_batch_size=4,
                          input_shape=input_shape,
                          steps_per_epoch=1,
                          steps_validation=1,
                          run_for_check=True,
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=[tf.keras.metrics.MeanIoU(num_classes=13)])
    sm_psp_net.train()
    sm_psp_net.run_inference()
    sm_psp_net.save_tf_lite()
    sm_psp_net.save_frozen_graph_tf2()

    profiler.calculate_flops_with_session_meta(sm_psp_net.get_h5_saved_model_file_path(),
                                               sm_psp_net.build_saved_model_folder(),
                                               sm_psp_net.get_input_shape())

    # profiler.calculate_flops_from_frozen_graph(sm_psp_net.build_frozen_graph_file_path(),
    #                                            sm_psp_net.build_saved_model_folder(),
    #                                            sm_psp_net.get_input_shape())


def check_ny2_sm_fpnnet_data_1(input_shape=(224, 224, 3)):
    data_set = NYU2Data()
    sm_psp_net = SMFPNNet(data_set,
                          epochs=1,
                          train_batch_size=4,
                          input_shape=input_shape,
                          steps_per_epoch=1,
                          steps_validation=1,
                          run_for_check=True,
                          backbone="mobilenetv2",
                          loss=tf.keras.losses.CategoricalCrossentropy(),
                          metrics=[tf.keras.metrics.MeanIoU(num_classes=13)])
    sm_psp_net.train()
    sm_psp_net.run_inference()
    sm_psp_net.save_tf_lite()
    sm_psp_net.save_frozen_graph_tf2()

    profiler.calculate_flops_with_session_meta(sm_psp_net.get_h5_saved_model_file_path(),
                                               sm_psp_net.build_saved_model_folder(),
                                               sm_psp_net.get_input_shape())


def check_ny2_sm_linknet_data_1(input_shape=(224, 224, 3)):
    data_set = NYU2Data()
    sm_psp_net = SMLinkNet(data_set,
                           epochs=1,
                           train_batch_size=4,
                           input_shape=input_shape,
                           steps_per_epoch=1,
                           steps_validation=1,
                           run_for_check=True,
                           backbone="mobilenetv2",
                           loss=tf.keras.losses.CategoricalCrossentropy(),
                           metrics=[tf.keras.metrics.MeanIoU(num_classes=13)])

    sm_psp_net.train()
    sm_psp_net.run_inference()
    sm_psp_net.save_tf_lite()
    # sm_psp_net.save_frozen_graph_tf2()

    profiler.calculate_flops_with_session_meta(sm_psp_net.get_h5_saved_model_file_path(),
                                               sm_psp_net.build_saved_model_folder(),
                                               sm_psp_net.get_input_shape())

# =========================================================================================================


def train_ny2_sm_unet_data_1(input_shape=(224, 224, 3)):
    data_set = NYU2Data()
    sm_u_net = SMUNet(data_set,
                      epochs=40,
                      train_batch_size=8,
                      input_shape=input_shape)
    sm_u_net.train()


def train_ny2_sm_linknet_data_1(input_shape=(224, 224, 3)):
    data_set = NYU2Data()
    sm_link_net = SMLinkNet(data_set,
                            epochs=40,
                            train_batch_size=8,
                            input_shape=input_shape)
    sm_link_net.train()


def train_ny2_sm_pspnet_data_1(input_shape=(224, 224, 3)):
    data_set = NYU2Data()
    sm_psp_net = SMPSPNet(data_set,
                          epochs=40,
                          train_batch_size=8,
                          input_shape=input_shape)
    sm_psp_net.train()


def train_ny2_sm_fpn_data_1(input_shape=(224, 224, 3)):
    data_set = NYU2Data()
    sm_fpn_net = SMFPNNet(data_set,
                          epochs=40,
                          train_batch_size=8,
                          input_shape=input_shape)
    sm_fpn_net.train()


# ======================================== 224 x 224 x 3 =====================================================================

# check_ny2_mobile_net_v2((224, 224, 3))
# check_ny2_sm_unet_data_1((224, 224, 3))
# check_ny2_sm_pspnet_data_1((288, 288, 3))
# check_ny2_sm_fpnnet_data_1((224, 224, 3))
# check_ny2_sm_linknet_data_1((224, 224, 3))

# train_ny2_sm_linknet_data_1((224, 224, 3))
# train_ny2_sm_pspnet_data_1((288, 288, 3))
# train_ny2_sm_fpn_data_1((224, 224, 3))
# train_ny2_sm_unet_data_1((224, 224, 3))

# ======================================== 448 x 448 x 3 =====================================================================

# check_ny2_mobile_net_v2((448, 448, 3))
# check_ny2_sm_unet_data_1((448, 448, 3))
# check_ny2_sm_pspnet_data_1((576, 576, 3))
# check_ny2_sm_fpnnet_data_1((448, 448, 3))
# check_ny2_sm_linknet_data_1((448, 448, 3))

# ======================================== 96 x 96 x 3======================================================================

# check_ny2_mobile_net_v2((96, 96, 3))
# check_ny2_sm_unet_data_1((96, 96, 3))
# check_ny2_sm_pspnet_data_1((96, 96, 3))
# check_ny2_sm_fpnnet_data_1((96, 96, 3))
# check_ny2_sm_linknet_data_1((96, 96, 3))

# ======================================== 336 x 336 x 3======================================================================

check_ny2_mobile_net_v2((336, 336, 3))
# check_ny2_sm_unet_data_1((336, 336, 3))
# check_ny2_sm_pspnet_data_1((384, 384, 3))
# check_ny2_sm_fpnnet_data_1((336, 336, 3))
# check_ny2_sm_linknet_data_1((336, 336, 3))