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


#
# def plot_results(models,
#                  data,
#                  batch_size=128,
#                  model_name="vae_DCNN_celebA"):
#     """Plots labels and MNIST digits as function of 2-dim latent vector
#
#     Arguments:
#         models (list): encoder and decoder models
#         data (list): test data and label
#         batch_size (int): prediction batch size
#         model_name (string): which model is using this function
#
#     Returns:
#         none
#     """
#
#     encoder, decoder = models
#     x_test, y_test = data
#     os.makedirs(model_name, exist_ok=True)
#     '''
#     filename = os.path.join(model_name, "vae_mean.png")
#     # display a 2D plot of the digit classes in the latent space
#     z_mean, _, _ = encoder.predict(x_test,
#                                    batch_size=batch_size)
#     plt.figure(figsize=(12, 10))
#     plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
#     plt.colorbar()
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.savefig(filename)
#     plt.show()
#     '''
#     filename = os.path.join(model_name, "digits_over_latent.png")
#     # display a 30x30 2D manifold of the digits
#     n = 1
#     digit_size = 64
#     figure = np.zeros((digit_size * n, digit_size * n,3))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-4, 4, n)
#     grid_y = np.linspace(-4, 4, n)[::-1]
#
#
#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
# #            z_sample = np.array([[xi, yi]])
#             z_sample = np.random.uniform(size=(2,128))
#
#             #print(z_sample.shape)
#             x_decoded = decoder.predict(z_sample)
#             print(x_decoded.shape)
#
#             '''
#             digit = x_decoded[0].reshape(digit_size, digit_size,3)
#             figure[i * digit_size: (i + 1) * digit_size,
#                    j * digit_size: (j + 1) * digit_size,:] = digit
#
#     plt.figure(figsize=(10, 10))
#     start_range = digit_size // 2
#     end_range = n * digit_size + start_range + 1
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     plt.xticks(pixel_range, sample_range_x)
#     plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imshow(figure, cmap='Greys_r')
#     plt.savefig(filename)
#     plt.show()
#     '''
#
image_size = 64
channels = 3
# network parameters
input_shape = (image_size, image_size, channels)
batch_size = 128
kernel_size = 3
filters = np.array([64,32])
z_dim = 128
epochs = 10
lr = 0.0002
decay = 6e-10

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
    vaegan_encoder.summary()
    vaegan_decoder.summary()


    disc_optimizer = RMSprop(lr=lr, decay=decay)
    vaegan_disc.compile(loss='binary_crossentropy',
                        optimizer=disc_optimizer,
                        metrics=['accuracy'])
    vaegan_disc.summary()

    # Instantiate VAE model
    models = (vaegan_encoder, vaegan_decoder)
    outputs = vaegan_decoder(vaegan_encoder(inputs)[2])
    print("outputshape", outputs.shape)
    vae = Model(inputs, outputs, name='vae')

    reconstruction_loss = binary_crossentropy(K.flatten(inputs),
                                              K.flatten(outputs))

    reconstruction_loss *= image_size * image_size
    kl_loss = 1 + vaegan_encoder(inputs)[1] - K.square(vaegan_encoder(inputs)[0]) - K.exp(vaegan_encoder(inputs)[1])
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)

    vae.add_loss(vae_loss)
    rmsprop = RMSprop(lr=0.0001)
    vae.compile(optimizer=rmsprop)
    vae.summary()
    #print(vae)
    #plot_model(vae, to_file='vae_dcnn.png', show_shapes=True)

    # Instantiate VAE-GAN model
    gan_input = Input(shape=(128,))
    gan_output = vaegan_disc(vaegan_decoder(gan_input))
    print("gan_inputshape",  gan_input.shape)
    print("gan_outshape", gan_output.shape)

    gan_optimizer = RMSprop(lr=lr*0.5, decay=decay)
    vaegan_disc.trainable = False
    gan = Model(gan_input, gan_output, name='gan')
    gan.compile(loss='binary_crossentropy',
                optimizer=gan_optimizer,
                metrics=['accuracy'])
    gan.summary()

    if args.weights:
        #print("loading weights",args.weights)
        gan.load_weights(args.weights)
        #print(vae)
    else:
        '''
        checkpoint_period = 5
        checkpoint_path = 'checkpoints/'
        checkpointer = ModelCheckpoint(filepath=checkpoint_path + 'model-{epoch:05d}.hdf5',
                                        verbose=1,
                                        save_weights_only=True,
                                        period=checkpoint_period)
        #vae.load_weights('checkpoints/model-00340.hdf5')
        vae.fit_generator(celeb_loader(dir='/home/airscan-razer04/Documents/datasets/img_align_celeba/',
                            randomize=True,
                            batch_size=batch_size,
                            height=image_size,
                            width=image_size),
                #epochs=1,
                #steps_per_epoch=1

                epochs=epochs,
                steps_per_epoch=int(20599/batch_size),
                callbacks=[checkpointer]
                #validation_data=(x_test, None)
                )
        vae.save_weights('vae_dcnn_celebA-02.h5')
        '''
        save_interval = int(20599/batch_size)
        #epochs=1
        #save_interval=10
        img_loader = celeb_loader(batch_size=batch_size)

        for i in range(1000):
            real_images_input, _ = next(img_loader)
            # Train VAE
            vae_metrics = vae.train_on_batch(real_images_input, None)
            print("VAE training metrics", vae_metrics)


        for i in range (epochs):
            for j in range (int(save_interval)):
                # Load real images
                real_images_input, _ = next(img_loader)
                # Train VAE
                vae_metrics = vae.train_on_batch(real_images_input, None)
                print("VAE metrics", vae_metrics)


                real_images = vae.predict(real_images_input)
                # Generate fake images
                noise = np.random.uniform(size=(batch_size, z_dim), low=-1.0, high=1.0)
                fake_images = vaegan_decoder.predict(noise)
                x = np.concatenate((real_images, fake_images))
                #print(x.shape)
                # Label real and fake images
                y = np.ones([2 * batch_size, 1])
                y[batch_size:, :] = 0
                #print(y.shape)
                # Train Discriminator
                metrics = vaegan_disc.train_on_batch(x, y)
                loss = metrics[0]
                disc_acc = metrics[1]
                log = "%d-%d: [discriminator loss: %f, acc: %f]" % (i,j, loss, disc_acc)
                #print(log)

                for k in range(1):
                    # Generate fake image
                    noise = np.random.uniform(size=(batch_size, z_dim), low=-1.0, high=1.0)
                    # Label fake images as real
                    y = np.ones([batch_size, 1])
                    # Train the Adversarial network
                    metrics = gan.train_on_batch(noise, y)
                    loss = metrics[0]
                    acc = metrics[1]
                    logg = "%s [adversarial loss: %f, acc: %f]" % (log, loss, acc)
                    print(logg)
            model_save_path = 'gan_checkpoints/gan-celebA-model-'+'{:05}'.format(i)+'-'+'{:05}'.format(j)+'.h5'
            print("Saving model to", model_save_path)
            gan.save_weights(model_save_path)




    #output sampling
    num_outputs = 10
    z_sample = np.random.uniform(size=(num_outputs,z_dim), low=-3.0, high=3.0)
    out_random = vaegan_decoder.predict(z_sample)
    for i in range (out_random.shape[0]):
        cv2.imshow("image",out_random[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    some_gen = celeb_loader(batch_size=128)
    data, _ = next(some_gen)
    #print(vae)
    out_enc = vaegan_encoder.predict(data)
    #out = vaegan_decoder.predict(out_enc[2])
    '''
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