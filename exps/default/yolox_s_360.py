import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.num_classes = 1
        self.max_epoch = 200
        self.data_num_workers = 4
        self.eval_interval = 1
        self.test_size = (640, 640)
        self.input_size = (640, 640)
        self.multiscale_range = 1
        self.augmentation_preset = "360"

        self.data_dir = r"N:\_temp\SoS2.6_NDS_extended2"
        self.train_ann = r"N:\_temp\SoS2.6_NDS_extended2\annotations\instances_train2017.json"
        self.val_ann = r"N:\_temp\SoS2.6_NDS_extended2\annotations\instances_val2017.json"