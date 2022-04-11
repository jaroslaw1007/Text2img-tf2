import tensorflow as tf
import numpy as np
import random
import pandas as pd

import matplotlib.pyplot as plt

from config import *

class DatasetGenerator:
    def __init__( self ):
        df_image = pd.read_pickle( cfg.DATASET.IMAGE_PATH )
        self.image_idx = df_image.index.values
        self.image_path = df_image['ImagePath'].values
        self.wrong_path = np.load( cfg.DATASET.WRONG_PATH )
        self.lookup_table = np.load(cfg.DATASET.LOOK_UP)

        list_captions = pd.read_pickle( cfg.DATASET.CAPTION_PATH )
        self.caption_list = []
        self.caption_num_list = []
        
        #for captions in list_captions:
        #    self.caption_num_list.append( min( len( captions ), cfg.DATASET.MAX_CAPTION_PER_IMAGE ) )
        #
        #    if len( captions ) < cfg.DATASET.MAX_CAPTION_PER_IMAGE:
        #        self.caption_list.append( captions + [[ 0 for _ in range( 100 ) ]] * ( cfg.DATASET.MAX_CAPTION_PER_IMAGE - len( captions ) ) )
        #    else:
        #        self.caption_list.append( captions[:cfg.DATASET.MAX_CAPTION_PER_IMAGE] )

        for captions in list_captions:
            self.caption_num_list.append( min( len( captions ), cfg.DATASET.MAX_CAPTION_PER_IMAGE ) )

            if len( captions ) < cfg.DATASET.MAX_CAPTION_PER_IMAGE:
                self.caption_list.append( np.concatenate( ( captions, np.zeros( ( cfg.DATASET.MAX_CAPTION_PER_IMAGE - len( captions ), cfg.TEXT.DIMENSION ) ) ) , axis=0 ) )
            else:
                self.caption_list.append( captions[:cfg.DATASET.MAX_CAPTION_PER_IMAGE] )
        
        self.lookup_table = tf.convert_to_tensor( self.lookup_table, dtype=tf.dtypes.int32 )
        self.caption_list = tf.convert_to_tensor( self.caption_list, dtype=tf.dtypes.float32 )
        self.caption_num_list = tf.convert_to_tensor( self.caption_num_list, dtype=tf.dtypes.int32 )

    def _read_image( self, image_path ):
        img = tf.io.read_file( image_path )
        img = tf.io.decode_jpeg( img, channels=3 )

        return img

    def _normalize( self, img ):
        img = ( img / 255 ) * 2 - 1

        return img

    def _random_flip( self, img ):
        img = tf.image.random_flip_left_right( img )
        img = tf.image.random_flip_up_down( img )

        return img

    def _resize_and_crop( self, img, crop_size, resize_size ):
        img = tf.image.resize( img, [ resize_size, resize_size ] )
        img = tf.image.random_crop( img, size = [ crop_size, crop_size, 3 ] )

        return img

    def _image_preprocessing( self, image_idx, image_path, wrong_path, captions, caption_num ): 
        img = self._read_image( image_path )
        img = self._normalize( img )
        img = self._random_flip( img )

        img64  = self._resize_and_crop( img,  64,  80 )
        img128 = self._resize_and_crop( img, 128, 160 )
        
        # wrong_id = tf.random.uniform( (), minval=0, maxval=1000, dtype=tf.dtypes.int32 )
        # wrong_img_id = wrong_path[ wrong_id ]
        # wrong_caption_id = self.lookup_table[ wrong_img_id ]
        # wrong_captions = self.caption_list[ wrong_caption_id[0] ]
        
        # w_c_id = tf.random.uniform( (), minval=0, maxval=self.caption_num_list[wrong_caption_id[0]], dtype=tf.dtypes.int32 )
        # wrong_caption = wrong_captions[ w_c_id ]
        
        # wrong_img_idx = tf.random.uniform( (), minval=1, maxval=7371, dtype=tf.dtypes.int32 )
        # wrong_img_path = tf.strings.as_string( wrong_img_idx, width=5, fill='0' )
        # wrong_img_path = tf.strings.join( ['./102flowers/image_', wrong_img_path, '.jpg'] )

        # wrong_img = self._read_image( wrong_img_path )
        # wrong_img = self._normalize( wrong_img )
        # wrong_img = self._random_flip( wrong_img )

        # wrong_img64  = self._resize_and_crop( wrong_img,  64,  80 )
        # wrong_img128 = self._resize_and_crop( wrong_img, 128, 160 )
        # wrong_img256 = self._resize_and_crop( wrong_img, 256, 320 )

        caption_id = tf.random.uniform( (), minval=0, maxval=caption_num, dtype=tf.dtypes.int32 )
        caption = captions[ caption_id ]
        
        wrong_caption_id = tf.random.uniform( (), minval=0, maxval=7370, dtype=tf.dtypes.int32)
        wrong_captions = self.caption_list[wrong_caption_id]

        # if wrong_captions == captions:
            # wrong_caption_id = random.randint(0, len(self.caption_list) - 1)
        
        # wrong_captions = self.caption_list[wrong_caption_id] 
        wrong_idx = tf.random.uniform( (), minval= 0, maxval=self.caption_num_list[wrong_caption_id], dtype=tf.dtypes.int32)
        wrong_caption = wrong_captions[wrong_idx]
        
        return (img64, img128), caption, wrong_caption

    def generate( self ):
        dataset = tf.data.Dataset.from_tensor_slices((
            self.image_idx,
            self.image_path,
            self.wrong_path,
            self.caption_list,
            self.caption_num_list
        ))
        dataset = dataset.map( self._image_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE )
        dataset = dataset.shuffle( 2000 ).batch( cfg.TRAIN.BATCH_SIZE, drop_remainder=True )
        dataset = dataset.prefetch( buffer_size=tf.data.experimental.AUTOTUNE )

        return dataset

if __name__ == '__main__':
    dataset = DatasetGenerator().generate()

    for real_imgs, wrong_imgs, caption, wrong_caption in dataset.take(3):
        for real_img in real_imgs:
            print( tf.shape( real_img ) )
        # for wrong_img in wrong_imgs:
            # print( tf.shape( wrong_img ) )
        print( tf.shape( caption ) )
        print( tf.shape( wrong_caption ))
        print( )
