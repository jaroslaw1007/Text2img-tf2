import tensorflow as tf
import numpy as np
import os
import time
import imageio
import pandas as pd

from config import *

def compute_mean_covariance(imgs):
    batch_size = tf.shape(imgs)[0]
    height = tf.shape(imgs)[1]
    width = tf.shape(imgs)[2]
    channel_num = tf.shape(imgs)[3]

    num_pixels = height * width

    mu = tf.math.reduce_mean(imgs, axis=1, keepdims=True)
    mu = tf.math.reduce_mean(mu, axis=2, keepdims=True)
    
    img_hat = imgs - tf.tile(mu, (1, height, width, 1))
    img_hat = tf.reshape(img_hat, (batch_size, num_pixels, channel_num))

    img_hat_transpose = tf.transpose(img_hat, perm=[0, 2, 1])
    covariance = tf.matmul(img_hat_transpose, img_hat)
    covariance = covariance / tf.cast(num_pixels, tf.float32)

    return mu, covariance

def KL_loss( mu, logvar ):
    KLD_element = 1 + logvar - tf.math.exp( logvar ) - mu ** 2
    KLD = -0.5 * tf.reduce_mean( KLD_element )
    return KLD

def update_avg_param_G( Gnet, avg_param_G ):
    new_avg = []

    if avg_param_G is None:
        for p in Gnet.weights:
            new_avg.append( tf.identity( p ) )
    else:
        for p, avg_p in zip( Gnet.weights, avg_param_G ):
            new_avg.append( avg_p * 0.999 + tf.identity( p ) * 0.001 )
    return new_avg

def save_model( Gnet, Dnets, avg_param_G, count ):
    Gnet.set_weights( avg_param_G )
    Gnet.save_weights( os.path.join( cfg.TRAIN.CKPT_DIR, 'ckpt_stackgan_generator_'+str(count) ) )
    Dnets[0].save_weights( os.path.join( cfg.TRAIN.CKPT_DIR, 'ckpt_stackgan_discriminator64_'+str(count) ) )
    Dnets[1].save_weights( os.path.join( cfg.TRAIN.CKPT_DIR, 'ckpt_stackgan_discriminator128_'+str(count) ) )
    # Dnets[2].save_weights( os.path.join( cfg.TRAIN.CKPT_DIR, 'ckpt_stackgan_discriminator256_'+str(count) ) )

def utPuzzle( imgs, count ):
    for image in imgs:
        image = ( ( image + 1.0 ) / 2.0 * 255.0 ).numpy().astype( np.uint8 )
        h, w, c = image[0].shape
        path = os.path.join( cfg.TRAIN.SNAPSHOT.IMAGE_DIR, 'stackgan_{:d}_size_{:d}.png'.format(count, h) )
        out = np.zeros( (h * cfg.TRAIN.SNAPSHOT.SAMPLE_ROW, w * cfg.TRAIN.SNAPSHOT.SAMPLE_COL, c), np.uint8 )
        for n, img in enumerate ( image ):
            j, i = divmod( n, cfg.TRAIN.SNAPSHOT.SAMPLE_COL )
            out[j * h : (j + 1) * h, i * w : (i + 1) * w, :] = img
        imageio.imwrite( path, out )

@tf.function
def discriminator_train_step( model, optimizer, criterion, 
                              real_imgs, fake_imgs, mu, wrong_mu ):
    batch_size = tf.shape( real_imgs )[0]

    real_labels = tf.ones( [batch_size] )
    fake_labels = tf.zeros( [batch_size] )

    with tf.GradientTape() as tape:
        real_logits = model( real_imgs, mu )
        wrong_logits = model( real_imgs, wrong_mu )
        fake_logits = model( fake_imgs, mu )

        errD_real = criterion( real_labels, real_logits[0] )
        errD_wrong = criterion( fake_labels, wrong_logits[0] )
        errD_fake = criterion( fake_labels, fake_logits[0] )

        if cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
            errD_real_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion( real_labels, real_logits[1] )
            errD_worng_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion( real_labels, wrong_logits[1] )
            errD_fake_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion( fake_labels, fake_logits[1] )

            errD_real += errD_real_uncond
            errD_wrong += errD_worng_uncond
            errD_fake += errD_fake_uncond

        errD = errD_real + 0.5 * ( errD_wrong + errD_fake )
        
    # print(errD_real, errD_wrong, errD_fake)
    grads = tape.gradient( errD, model.trainable_weights )
    optimizer.apply_gradients( zip( grads, model.trainable_weights ) )

    return errD

@tf.function
def generator_train_step( model, optimizer, criterion, 
                          Dnets, noise, text_embedding ):
    batch_size = tf.shape( text_embedding )[0]

    real_labels = tf.ones( [batch_size] )

    with tf.GradientTape() as tape:
        fake_imgs, mu, logvar = model( noise, text_embedding )
        
        errG_total = 0
        for fake_img, Dnet in zip( fake_imgs, Dnets ):
            logits = Dnet( fake_img, mu )
            errG = criterion( real_labels, logits[0] )
            if cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
                errG_patch = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                    criterion( real_labels, logits[1] )
                errG += errG_patch

            errG_total += errG

        if cfg.TRAIN.COEFF.COLOR_LOSS > 0:
            mu1, cov1 = compute_mean_covariance(fake_imgs[-1])
            mu2, cov2 = compute_mean_covariance(fake_imgs[-2])
            # mu3, cov3 = compute_mean_covariance(fake_imgs[-3])            
            
            like_mu1 = cfg.TRAIN.COEFF.COLOR_LOSS * tf.losses.MSE(mu1, mu2)
            like_cov1 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * tf.losses.MSE(cov1, cov2)
            # like_mu2 = cfg.TRAIN.COEFF.COLOR_LOSS * tf.losses.MSE(mu2, mu3)
            # like_cov2 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * tf.losses.MSE(cov2, cov3)
            errG_total += like_mu1 + like_cov1

        kl_loss = KL_loss( mu, logvar ) * cfg.TRAIN.COEFF.KL
        errG_total += kl_loss

    grads = tape.gradient( errG_total, model.trainable_weights )
    optimizer.apply_gradients( zip( grads, model.trainable_weights ) )

    return kl_loss, errG_total

def train( Gnet, Dnets, train_dataset ):
    Goptimizer = tf.keras.optimizers.Adam( learning_rate = cfg.TRAIN.GENERATOR_LR, beta_1 = 0.5, beta_2 = 0.999 )
    Doptimizers = [
        tf.keras.optimizers.Adam( learning_rate = cfg.TRAIN.DISCRIMINATOR_LR, beta_1 = 0.5, beta_2 = 0.999 ),
        tf.keras.optimizers.Adam( learning_rate = cfg.TRAIN.DISCRIMINATOR_LR, beta_1 = 0.5, beta_2 = 0.999 )
        # tf.keras.optimizers.Adam( learning_rate = cfg.TRAIN.DISCRIMINATOR_LR, beta_1 = 0.5, beta_2 = 0.999 )
    ]

    avg_param_G = None

    criterion = tf.keras.losses.BinaryCrossentropy()

    kl_train_loss = tf.keras.metrics.Mean( name = 'kl_loss' )
    generator_train_loss = tf.keras.metrics.Mean( name = 'generator_loss' )
    discriminator_train_loss = tf.keras.metrics.Mean( name = 'discriminator_loss' )

    fixed_noise = tf.random.normal( (cfg.TRAIN.BATCH_SIZE, cfg.GAN.Z_DIM) )
    fixed_text_embedding = pd.read_pickle( cfg.TEST.CAPTION_PATH )[:cfg.TRAIN.BATCH_SIZE]
    fixed_text_embedding = np.asarray(fixed_text_embedding)
    fixed_text_embedding = np.squeeze(fixed_text_embedding)

    count = 0
    for epoch in range( cfg.TRAIN.MAX_EPOCH ):
        start = time.time()

        kl_train_loss.reset_states()
        generator_train_loss.reset_states()
        discriminator_train_loss.reset_states()

        print( 'Epoch {:3d}'.format( epoch ) )

        for batch_idx, inputs in enumerate( train_dataset ):
            real_imgs, text_embedding, wrong_text_embedding = inputs
            batch_size = tf.shape( real_imgs[0] )[0]
            
            noise = tf.random.normal( ( batch_size, cfg.GAN.Z_DIM ) )
            fake_imgs, mu, _ = Gnet( noise, text_embedding )
            _, wrong_mu, _ = Gnet( noise, wrong_text_embedding )
            if avg_param_G is None: avg_param_G = Gnet.weights
            
            # error_all = []
            errD_total = 0
            for i in range( len( Dnets ) ):
                errD = discriminator_train_step( Dnets[i], Doptimizers[i], criterion,
                                                 real_imgs[i], fake_imgs[i],
                                                 mu, wrong_mu)
                # error_all.append(errD)
                errD_total += errD
            discriminator_train_loss( errD_total )

            kl_loss, errG_total = generator_train_step( Gnet, Goptimizer, criterion,
                                                        Dnets, noise, text_embedding )
            avg_param_G = update_avg_param_G( Gnet, avg_param_G )

            kl_train_loss( kl_loss )
            generator_train_loss( errG_total )

            count = count + 1

            if count % cfg.TRAIN.SNAPSHOT.INTERVAL == 0:
                save_model( Gnet, Dnets, avg_param_G, count )

                fake_imgs, _, _ = Gnet( fixed_noise, fixed_text_embedding )
                utPuzzle( fake_imgs, count )

            print('  Batch {:3d} generator_loss={:9.6f} discriminator_loss={:9.6f} kl_loss={:9.6f}'.format(batch_idx, \
                          generator_train_loss.result(), discriminator_train_loss.result(), kl_train_loss.result()), end='\r')
            
            # print(error_all[0], error_all[1], error_all[2])

        print ( '\nTime taken for 1 epoch {} sec\n'.format( time.time() - start ) )
