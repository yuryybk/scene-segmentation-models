import segmentation_models as sm
from sm_unet import SMUNet
from nyu_v2_descriptor import NYU2Data

sm.set_framework('tf.keras')


def train_ny2_data_1():
    data_set = NYU2Data()
    sm_u_net = SMUNet(data_set)
    sm_u_net.configure(epochs=2, train_batch_size=4, input_shape=(224, 224, 3), encoder_weights="imagenet")
    sm_u_net.train()
    sm_u_net.save_tf_lite()


train_ny2_data_1()

