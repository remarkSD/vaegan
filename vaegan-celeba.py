from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from models_celebA import *
from celebA_loader import *

image_size = 64
channels = 3
# network parameters
input_shape = (image_size, image_size, channels)
batch_size = 64
kernel_size = 3
filters = np.array([64,32])
z_dim = 128
epochs = 100
lr = 0.0003
decay = 0
dir='/home/raimarc/Documents/img_align_celeba/'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    df_dim = 64
    height = np.array([64])
    width = np.array([64])

    # Instantiate encoder, decoder/generator, discriminator models
    inputs = Input(shape=input_shape)

    vaegan_encoder = encoder(num_filters=filters[0], ch=channels, rows=height, cols=width, z_dim=z_dim)
    vaegan_decoder = generator(num_filters=filters[1], z_dim=z_dim, ch=channels)
    vaegan_disc = discriminator(num_filters=32, z_dim=z_dim, ch=3, rows=height, cols=width)
    #vaegan_decoder = build_generator(Input(shape=(z_dim,)), image_size)
    #vaegan_disc = build_discriminator(inputs)
#    vaegan_decoder.summary()


    disc_optimizer = RMSprop(lr=lr, decay=decay)
    vaegan_disc.compile(loss='binary_crossentropy',
                        optimizer=disc_optimizer,
                        metrics=['accuracy'])
    vaegan_disc.summary()



    outputs = vaegan_decoder(vaegan_encoder(inputs)[2])
    print("outputshape", outputs.shape)
    vaegan_disc.trainable=False
    vae = Model(inputs, outputs, name='vae')

    # Design vaegan_encoder
    reconstruction_loss = mse(K.flatten(inputs), K.flatten(outputs))
    reconstruction_loss *= image_size * image_size * channels
    kl_loss = 1 + vaegan_encoder(inputs)[1] - K.square(vaegan_encoder(inputs)[0]) - K.exp(vaegan_encoder(inputs)[1])
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    rmsprop = RMSprop(lr=lr)
    vae.compile(optimizer=rmsprop)
    vae.summary()



    # Instantiate GAN model
    gan_input = Input(shape=(z_dim,))
    gan_output = vaegan_disc(vaegan_decoder(gan_input))
    print("gan_inputshape",  gan_input.shape)
    print("gan_outshape", gan_output.shape)

    gan_optimizer = RMSprop(lr=lr)
    vaegan_disc.trainable = False
    gan = Model(gan_input, gan_output, name='gan')
    gan.compile(loss='binary_crossentropy',
                optimizer=gan_optimizer,
                metrics=['accuracy'])
    gan.summary()


    #instantiate VAEGAN Model
    vaegan_output = vaegan_disc(vaegan_decoder(vaegan_encoder(inputs)[2]))
    vaegan = Model(inputs, vaegan_output, name='vaegan')
    vaegan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')
    vaegan.summary()

    if args.weights:
        #print("loading weights",args.weights)
        vaegan.load_weights(args.weights)
        #print(vae)


    save_interval = int(202599/batch_size)
    #epochs=1
    #save_interval=50
    img_loader = celeb_loader(dir=dir,
                            batch_size=batch_size, norm=True)
    for i in range (epochs):
        for j in range (int(save_interval)):
            # Load real images
            real_images, _ = next(img_loader)

            # Train encoder
            vae.train_on_batch(real_images, None)

            #vaegan_encoder.train_on_batch(real_images, None)
            # Generate fake images
            #noise = np.random.uniform(size=(batch_size, z_dim), low=-1.0, high=1.0)
            noise = np.random.normal(size=(batch_size, z_dim))
            fake_images = vaegan_decoder.predict(noise)
            ae_images = vaegan_decoder.predict(vaegan_encoder.predict(real_images)[2])
            #x = np.concatenate((real_images, fake_images))
            #print(x.shape)
            # Label real and fake images
            y1 = np.ones([batch_size, 1])
            y2 = np.zeros([batch_size, 1])
            #print(y.shape)
            # Train Discriminator
            metrics = vaegan_disc.train_on_batch(real_images, y1)
            loss = metrics[0]
            disc_acc = metrics[1]
            log = "%d-%d: [discriminator loss (real): %f, acc: %f]" % (i,j, loss, disc_acc)
            #print(log)

            metrics = vaegan_disc.train_on_batch(fake_images, y2)
            loss = metrics[0]
            disc_acc = metrics[1]
            log = "%s [discriminator loss (fake): %f, acc: %f]" % (log, loss, disc_acc)
            #print(log)

            metrics = vaegan_disc.train_on_batch(ae_images, y2)
            loss = metrics[0]
            disc_acc = metrics[1]
            log = "%s [discriminator loss (ae): %f, acc: %f]" % (log, loss, disc_acc)
            #print(log)

            # Generate fake image
            #noise = np.random.uniform(size=(batch_size, z_dim), low=-1.0, high=1.0)
            noise = np.random.normal(size=(batch_size, z_dim))
            # Label fake images as real
            y = np.ones([batch_size, 1])
            # Train the Adversarial network
            metrics = gan.train_on_batch(noise, y)
            loss = metrics[0]
            acc = metrics[1]
            logg = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
            print(logg)

            if j % 200 == 0:
                model_save_path = 'vaegan_checkpoints/vaegan-model-'+'{:05}'.format(i)+'-'+'{:05}'.format(j)+'.h5'
                print("Saving model to", model_save_path)
                vaegan.save_weights(model_save_path)

                # Predict Sample
                z_sample = np.random.uniform(size=(25,z_dim), low=-1.0, high=1.0)
                out_random = vaegan_decoder.predict(z_sample)
                # Unnormalize samples
                out_random = (out_random + 1)*127.5
                out_random = out_random.astype(np.uint8)
                print("MAX", np.max(out_random))
                print("MIN", np.min(out_random))

                # Put samples in grid
                fig = np.zeros((64*5,64*5,3))
                for k1 in range (5):
                    for k2 in range (5):
                        fig[64*k2:64*(k2+1),64*k1:64*(k1+1),:] = out_random[k1*5+k2]
                #cv2.imshow("image",out_random[0])
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                # Write samples
                out_filename = 'vaegan_checkpoints/out' + '{:05}'.format(i)+'-'+'{:05}'.format(j)+'.jpg'
                cv2.imwrite(out_filename, fig)






    #output sampling
    '''
    num_outputs = 10
    z_sample = np.random.uniform(size=(num_outputs,z_dim), low=-3.0, high=3.0)
    out_random = vaegan_decoder.predict(z_sample)
    print("min", np.min(out_random))
    print("max", np.max(out_random))
    for i in range (out_random.shape[0]):
        cv2.imshow("image",out_random[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    some_gen = celeb_loader(batch_size=128)
    data, _ = next(some_gen)
    #print(vae)
    out_enc = vaegan_encoder.predict(data)
    #out = vaegan_decoder.predict(out_enc[2])

    out = vae.predict(data)
    print("data", data.shape)
    print("out", out.shape)

    for i in range (data.shape[0]):
        cv2.imshow("image",data[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        cv2.imshow("out image",out[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    '''


    #plot_results(models, data, batch_size=batch_size, model_name="vae_dcnn_celebA")
