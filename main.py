import tensorflow as tf

from trainer import train
from generator import Generator
from discriminator import Discriminator64, Discriminator128, Discriminator256
from dataloader import DatasetGenerator

from config import *

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental_run_functions_eagerly(True)

    Gnet = Generator()
    Dnets = [ Discriminator64(), Discriminator128() ]
    # Dnets = [ Discriminator64(), Discriminator128(), Discriminator256() ]

    train_dataset = DatasetGenerator().generate()

    train( Gnet, Dnets, train_dataset )
