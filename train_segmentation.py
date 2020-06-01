import segmentation_models as sm
from sm_unet import SMUNet
from nyu_v2_descriptor import NYU2Data
import profiler
import tensorflow as tf

sm.set_framework('tf.keras')


def train_ny2_data_1():
    data_set = NYU2Data()
    sm_u_net = SMUNet(data_set,
                      epochs=1,
                      train_batch_size=4,
                      input_shape=(224, 224, 3),
                      steps_per_epoch=1,
                      steps_validation=1,
                      run_for_check=True,
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=[tf.keras.metrics.MeanIoU(num_classes=13)])
    sm_u_net.train()

    # sm_u_net.save_tf_lite()

    # sm_u_net.save_frozen_graph_tf2()

    # profiler.calculate_flops_from_frozen_graph(sm_u_net.build_frozen_graph_file_path(),
    #                                            sm_u_net.get_input_shape())

    profiler.calculate_flops_with_session_meta(sm_u_net.get_saved_model_file_path(),
                                               sm_u_net.get_input_shape())


train_ny2_data_1()


