import tensorflow as tf
import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

from generator import Generator

from config import *

def testing_dataset_generator():
    captions = pd.read_pickle( cfg.TEST.CAPTION_PATH )
    test_id = pd.read_pickle( cfg.TEST.ID_PATH )
    test_id = test_id['ID'].values

    import numpy as np
    captions = np.reshape( captions, (-1, cfg.TEXT.DIMENSION ) ) 
    print( np.shape( captions ) )

    dataset = tf.data.Dataset.from_tensor_slices((captions, test_id))
    dataset = dataset.batch( cfg.TEST.BATCH_SIZE, drop_remainder=False )

    return dataset

def test( model ):
    testing_dataset = testing_dataset_generator()
    noise = tf.random.normal( (cfg.TEST.BATCH_SIZE, cfg.GAN.Z_DIM) )

    start = time.time()
    for inputs in testing_dataset:
        text_embedding, index = inputs
        # noise = tf.random.normal( (cfg.TEST.BATCH_SIZE, cfg.GAN.Z_DIM) )

        fake_imgs, _, _ = model( noise, text_embedding )
        img64, img128 = fake_imgs 
        for idx, fake_img in zip( index, img128 ):
            fake_img = tf.image.resize( fake_img, (64, 64) )
            #plt.imsave( os.path.join( cfg.TEST.INFERENCE_PATH, 'inference64_{:04d}.jpg'.format(idx) ), fake_img[0].numpy() * 0.5 + 0.5 )
            plt.imsave( os.path.join( cfg.TEST.INFERENCE_PATH, 'inference_{:04d}.jpg'.format(idx) ), fake_img.numpy() * 0.5 + 0.5 )
            #plt.imsave( os.path.join( cfg.TEST.INFERENCE_PATH, 'inference256_{:04d}.jpg'.format(idx) ), fake_img[2].numpy() * 0.5 + 0.5 )

    print( 'Time for inference is {:.4f} sec'.format(time.time()-start) )

if __name__ == '__main__':
    Gnet = Generator()
    Gnet.load_weights( sys.argv[1] )
    
    test( Gnet )

