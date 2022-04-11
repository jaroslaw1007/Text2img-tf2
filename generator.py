import tensorflow as tf
from tensorflow.keras import layers

from config import *

class GLU(layers.Layer):
    def __init__(self):
        super(GLU, self).__init__()

    def call(self, x):
        nc = tf.shape(x)[-1]
        nc = nc // 2
        return x[..., :nc] * tf.sigmoid( x[..., nc:] )

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
            # GLU()
            layers.LeakyReLU( 0.2 )
        ])

    def call(self, x):
        return self.block( x )

class UpsampleBlock(layers.Layer):
    def __init__(self, out_channels):
        super(UpsampleBlock, self).__init__()
        self.block = tf.keras.Sequential([
            layers.UpSampling2D( interpolation='nearest' ),
            Block3x3( out_channels )
        ])

    def call(self, x):
        return self.block( x )

class ResidualBlock(layers.Layer):
    def __init__(self, out_channels):
        super(ResidualBlock, self).__init__()
        self.block = tf.keras.Sequential([
          Block3x3( out_channels ),
          Conv3x3( out_channels ),
          layers.BatchNormalization( gamma_initializer=tf.keras.initializers.TruncatedNormal(1.0, 0.02) )
        ])

    def call(self, x):
        residual = x
        out = self.block( x )
        out += residual
        return out

class ConditioningAugmentation(tf.keras.Model):
    def __init__(self):
        super(ConditioningAugmentation, self).__init__()
        self.em_dim = cfg.GAN.EMBEDDING_DIM
        self.model = tf.keras.Sequential([
            layers.Dense( self.em_dim * 2 , kernel_initializer=tf.keras.initializers.Orthogonal() ),
            # GLU()
            layers.LeakyReLU( 0.2 )
        ])

    def encode(self, text_embedding):
        x = self.model( text_embedding )
        mu = x[:, :self.em_dim]
        logvar = x[:, self.em_dim:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = tf.math.exp( logvar / 2 )
        eps = tf.random.normal( tf.shape(std) )
        return mu + eps * std

    def call(self, text_embedding):
        mu, logvar = self.encode( text_embedding )
        c_code = self.reparameterize( mu, logvar )
        return c_code, mu, logvar

class StageIGenerator(tf.keras.Model):
    def __init__(self, ngf):
        super(StageIGenerator, self).__init__()
        self.gf_dim = ngf

        self.fc = tf.keras.Sequential([
            layers.Dense( ngf * 4 * 4 , use_bias=False, kernel_initializer=tf.keras.initializers.Orthogonal() ),
            layers.BatchNormalization( gamma_initializer=tf.keras.initializers.TruncatedNormal(1.0, 0.02) ),
            # GLU()
            layers.LeakyReLU( 0.2 )
        ])

        self.upsample = tf.keras.Sequential([
            UpsampleBlock( ngf //  2 ),
            UpsampleBlock( ngf //  4 ),
            UpsampleBlock( ngf //  8 ),
            UpsampleBlock( ngf // 16 )
        ])

    def call(self, z_code, c_code):
        in_code = tf.concat( [z_code, c_code], axis=1 )
        out_code = self.fc( in_code )

        out_code = tf.reshape( out_code, (-1, 4, 4, self.gf_dim) )
        out_code = self.upsample( out_code )

        return out_code

class StageIIGenerator(tf.keras.Model):
    def __init__(self, ngf):
        super(StageIIGenerator, self).__init__()
        self.gf_dim = ngf
        self.em_dim = cfg.GAN.EMBEDDING_DIM

        self.jointConv = Block3x3(ngf)
        self.residual = tf.keras.Sequential([
            ResidualBlock(ngf),
            ResidualBlock(ngf)
        ])
        self.upsample = UpsampleBlock( ngf // 2 )

    def call(self, h_code, c_code):
        f_size = tf.shape( h_code )[2]

        c_code = tf.reshape( c_code, (-1, 1, 1, self.em_dim) )
        c_code = tf.tile( c_code, (1, f_size, f_size, 1) )
        h_c_code = tf.concat( [c_code, h_code], axis=3 )

        out_code = self.jointConv( h_c_code )
        out_code = self.residual( out_code )
        out_code = self.upsample( out_code )

        return out_code

class GenImage(tf.keras.Model):
    def __init__(self):
        super(GenImage, self).__init__()
        self.conv = Conv3x3( 3 )

    def call(self, h_code):
        out = self.conv( h_code )
        return tf.math.tanh( out )

class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM

        self.ca_net = ConditioningAugmentation()

        # 64 * 64 images
        self.h_net1 = StageIGenerator( self.gf_dim * 16 )
        self.img_net1 = GenImage()

        # 128 * 128 images
        self.h_net2 = StageIIGenerator( self.gf_dim )
        self.img_net2 = GenImage()

        # 256 * 256 images
        # self.h_net3 = StageIIGenerator( self.gf_dim // 2 )
        # self.img_net3 = GenImage()

    def call(self, z_code, text_embedding):
        c_code, mu, logvar = self.ca_net( text_embedding )

        h_code1 = self.h_net1(z_code, c_code)
        fake_img1 = self.img_net1(h_code1)

        h_code2 = self.h_net2(h_code1, c_code)
        fake_img2 = self.img_net2(h_code2)

        # h_code3 = self.h_net3(h_code2, c_code)
        # fake_img3 = self.img_net3(h_code3)

        return [fake_img1, fake_img2], mu, logvar

if __name__ == '__main__':
    generator = Generator()

    noise = tf.random.normal((3, cfg.GAN.Z_DIM))
    text_embedding = tf.random.normal((3, cfg.TEXT.DIMENSION))

    out_data, mu, logvar = generator( noise, text_embedding )

    for od in out_data:
        print( tf.shape(od) )
    print( tf.shape(mu) )
    print( tf.shape(logvar) )
