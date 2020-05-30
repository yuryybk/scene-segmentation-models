import segmentation_models as sm
from sm_unet import SMUNet
from nyu_v2_descriptor import NYU2Data

sm.set_framework('tf.keras')


def train_ny2_data_1():
    data_set = NYU2Data()
    sm_u_net = SMUNet(data_set, epochs=1, train_batch_size=4, input_shape=(224, 224, 3), steps_per_epoch=1, steps_validation=1, run_for_check=True)
    sm_u_net.train()
    sm_u_net.save_tf_lite()


train_ny2_data_1()

