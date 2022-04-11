import tensorflow as tf
from tensorflow.keras import layers

from config import *

class Conv3x3(layers.Layer):
    def __init__(self, out_channels):
        super(Conv3x3, self).__init__()
        self.conv = layers.Conv2D( out_channels, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=tf.keras.initializers.Orthogonal() )

    def call(self, x):
        return self.conv( x )

class Block3x3(layers.Layer):
    def __init__(self, out_channels):
        super(Block3x3, self).__init__()
        self.block = tf.keras.Sequential([
            Conv3x3( out_channels ),
            layers.BatchNormalization( gamma_initializer=tf.keras.initializers.TruncatedNormal(1.0, 0.02) ),
            layers.LeakyReLU( 0.2 )
        ])

    def call(self, x):
        return self.block( x )

class DownsampleBlock(layers.Layer):
    def __init__(self, out_channels):
        super(DownsampleBlock, self).__init__()
        self.block = tf.keras.Sequential([
            layers.Conv2D( out_channels, 4, 2, 'same', use_bias=False, kernel_initializer=tf.keras.initializers.Orthogonal() ),
            layers.BatchNormalization( gamma_initializer=tf.keras.initializers.TruncatedNormal(1.0, 0.02) ),
            layers.LeakyReLU( 0.2 )
        ])

    def call(self, x):
        return self.block( x )

class Discriminator64(tf.keras.Model):
    def __init__(self):
        super(Discriminator64, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.em_dim = cfg.GAN.EMBEDDING_DIM

        self.img_enc = tf.keras.Sequential([
            layers.Conv2D( self.df_dim, 4, 2, 'same', use_bias=False, kernel_initializer=tf.keras.initializers.Orthogonal() ),
            layers.LeakyReLU( 0.2 ),
            DownsampleBlock( self.df_dim * 2 ),
            DownsampleBlock( self.df_dim * 4 ),
            DownsampleBlock( self.df_dim * 8 )
        ])

        self.jointConv = Block3x3( self.df_dim * 8 )
        self.logits = tf.keras.Sequential([
            layers.Conv2D( 1, 4, strides=4, kernel_initializer=tf.keras.initializers.Orthogonal() ),
            layers.Activation('sigmoid')
        ])

        self.uncond_logits = tf.keras.Sequential([
            layers.Conv2D( 1, 4, strides=4, kernel_initializer=tf.keras.initializers.Orthogonal() ),
            layers.Activation('sigmoid')
        ])

    def call(self, x_var, c_code):
        x_code = self.img_enc( x_var )

        c_code = tf.reshape( c_code, (-1, 1, 1, self.em_dim) )
        c_code = tf.tile(c_code, (1, 4, 4, 1))
        h_c_code = tf.concat((c_code, x_code), 3)
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)
        output_uncond = self.uncond_logits(x_code)

        return [tf.reshape(output, [-1]), tf.reshape(output_uncond, [-1])]

class Discriminator128(tf.keras.Model):
    def __init__(self):
        super(Discriminator128, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.em_dim = cfg.GAN.EMBEDDING_DIM

        self.img_enc = tf.keras.Sequential([
            layers.Conv2D( self.df_dim, 4, 2, 'same', use_bias=False, kernel_initializer=tf.keras.initializers.Orthogonal() ),
            layers.LeakyReLU( 0.2 ),
            DownsampleBlock( self.df_dim *  2 ),
            DownsampleBlock( self.df_dim *  4 ),
            DownsampleBlock( self.df_dim *  8 ),
            DownsampleBlock( self.df_dim * 16 ),
            Block3x3( self.df_dim * 8 )
        ])

        self.jointConv = Block3x3( self.df_dim * 8 )
        self.logits = tf.keras.Sequential([
            layers.Conv2D( 1, 4, strides=4, kernel_initializer=tf.keras.initializers.Orthogonal() ),
            layers.Activation('sigmoid')
        ])

        self.uncond_logits = tf.keras.Sequential([
            layers.Conv2D( 1, 4, strides=4, kernel_initializer=tf.keras.initializers.Orthogonal() ),
            layers.Activation('sigmoid')
        ])

    def call(self, x_var, c_code):
        x_code = self.img_enc( x_var )

        c_code = tf.reshape( c_code, (-1, 1, 1, self.em_dim) )
        c_code = tf.tile(c_code, (1, 4, 4, 1))
        h_c_code = tf.concat((c_code, x_code), 3)
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)
        output_uncond = self.uncond_logits(x_code)

        return [tf.reshape(output, [-1]), tf.reshape(output_uncond, [-1])]

class Discriminator256(tf.keras.Model):
    def __init__(self):
        super(Discriminator256, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.em_dim = cfg.GAN.EMBEDDING_DIM

        self.img_enc = tf.keras.Sequential([
            layers.Conv2D( self.df_dim, 4, 2, 'same', use_bias=False, kernel_initializer=tf.keras.initializers.Orthogonal() ),
            layers.LeakyReLU( 0.2 ),
            DownsampleBlock( self.df_dim *  2 ),
            DownsampleBlock( self.df_dim *  4 ),
            DownsampleBlock( self.df_dim *  8 ),
            DownsampleBlock( self.df_dim * 16 ),
            DownsampleBlock( self.df_dim * 32 ),
            Block3x3( self.df_dim * 16 ),
            Block3x3( self.df_dim * 8 )
        ])

        self.jointConv = Block3x3( self.df_dim * 8 )
        self.logits = tf.keras.Sequential([
            layers.Conv2D( 1, 4, strides=4, kernel_initializer=tf.keras.initializers.Orthogonal() ),
            layers.Activation('sigmoid')
        ])

        self.uncond_logits = tf.keras.Sequential([
            layers.Conv2D( 1, 4, strides=4, kernel_initializer=tf.keras.initializers.Orthogonal() ),
            layers.Activation('sigmoid')
        ])

    def call(self, x_var, c_code):
        x_code = self.img_enc( x_var )

        c_code = tf.reshape( c_code, (-1, 1, 1, self.em_dim) )
        c_code = tf.tile(c_code, (1, 4, 4, 1))
        h_c_code = tf.concat((c_code, x_code), 3)
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)
        output_uncond = self.uncond_logits(x_code)

        return [tf.reshape(output, [-1]), tf.reshape(output_uncond, [-1])]

if __name__ == '__main__':
    Dnet64 = Discriminator64()
    Dnet128 = Discriminator128()
    Dnet256 = Discriminator256()

    img64 = tf.random.normal((3, 64, 64, 3))
    img128 = tf.random.normal((3, 128, 128, 3))
    img256 = tf.random.normal((3, 256, 256, 3))
    mu = tf.random.normal((3, 128))
    
    print(tf.shape(Dnet64(img64, mu)))
    print(tf.shape(Dnet128(img128, mu)))
    print(tf.shape(Dnet256(img256, mu)))
