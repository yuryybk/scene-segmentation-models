from data_descriptor import DataDescriptor


class NYU2Data(DataDescriptor):

    _folder = "nuy2"

    def __init__(self):
        super().__init__()

    def get_train_rgb_path(self):
        return "data/" + self._folder + "/train_rgb"

    def get_train_mask_path(self):
        return "data/" + self._folder + "/train_mask"

    def get_test_rgb_path(self):
        return "data/" + self._folder + "/test_rgb"

    def get_test_mask_path(self):
        return "data/" + self._folder + "/test_mask"

    def get_n_classes(self):
        return 13

    def get_name(self):
        return self._folder
