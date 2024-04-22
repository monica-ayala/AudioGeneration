# AudioGeneration : Model Implementation and Evaluation
---
**TC3002B: Desarrollo de aplicaciones avanzadas de ciencias computacionales (Gpo 201)**

**Mónica Andrea Ayala Marrero - A01707439**

---
#### About the model

This project employs a **Variational Autoencoder for autonomous music generation** based on spectogram reconstruction. An overview of the proyect pipeline is the following:

Firstly, we will train our VAE Model to reconstruct spectograms of shape (512, 512, 1) that we get from preprocessing the audios in our dataset. 

*Note: To learn more about the dataset click [here](https://github.com/monica-ayala/AudioGeneration/blob/main/Preprocessing/README.md)*

![1](https://github.com/monica-ayala/AudioGeneration/assets/75228128/b9225c7a-f194-4c84-8264-dd5fb2eebafb)

Then, having succesfully trained our model, we can use the decoder part as a generator. This is because we will be able to sample a random vector of data and decode it to transform it into a spectogram that will follow the same distribution of data as the one in our dataser, but be new/different as the seed is a random sample.

![2](https://github.com/monica-ayala/AudioGeneration/assets/75228128/12f688b4-8294-4250-b220-499da1361153)

#### Implementation

Using tensorflow and tensorflow probability we were able to build our model.

We first start by defining our prior distribution that is defined by the number of components and the latent dimension.

```
def get_prior(num_modes, latent_dim):
  gm = tfp.distributions.MixtureSameFamily(
    mixture_distribution=tfp.distributions.Categorical(probs=[1.0/num_modes,]*num_modes),
    components_distribution = tfp.distributions.MultivariateNormalDiag(
      loc = tf.Variable(tf.random.normal(shape = [num_modes, latent_dim])),
      scale_diag = tfp.util.TransformedVariable(
        tf.Variable(tf.ones(shape = [num_modes, latent_dim])),
        bijector = tfp.bijectors.Softplus())
      )
    )
  return gm
```

This is the distribution that we will seek to train, it is defined as a mixture of Gaussian Distributions and it has fixed mixing coefficients but trainable mean and standard deviation.

For training a variational autoencoder we must define special loss functions. One of this is the KL_Regularizer. We must define this first as it will directly go into our decoder.

```
def get_kl_regularizer(prior_distribution):
    reg = tfp.layers.KLDivergenceRegularizer(
        prior_distribution,
        weight = 1.0,
        use_exact_kl = False,
        test_points_fn = lambda q : q.sample(3),
        test_points_reduce_axis = (0,1))

    return reg
```

##### Encoder

For the encoder, we define a convolutional network that recieves an input of shape (512, 512, 1) and applies several Convolutional layers, Batch Normalization and Max Pooling. Our model is ligther than it should, truly, better results would be attained without using strides in the last convolutional layer but this results in a heavy model that takes 2 days training.

After the last convolutional layer, we flatten it and pass it to a dense layer that is the size of our latent dimension (100, in this case). Finally we pass it to a MultivariateNormalTri Layer and pass the KL_Regularizer defined previously.

```
input_shape = (512,512,1)
    encoder = Sequential([
        Conv2D(filters = 32, kernel_size = 4, activation = 'relu',
               strides = 2, padding = 'SAME', input_shape = input_shape),
        BatchNormalization(),

        Conv2D(filters = 64, kernel_size = 4, activation = 'relu',
               strides = 2, padding = 'SAME'),
        BatchNormalization(),
        MaxPooling2D(pool_size=2, strides=2, padding='SAME'), 

        Conv2D(filters = 128, kernel_size = 4, activation = 'relu',
               strides = 2, padding = 'SAME'),
        BatchNormalization(),
        
        Conv2D(filters = 256, kernel_size = 4, strides = 2, activation = 'relu', padding = 'SAME'),
        BatchNormalization(),
        
        Conv2D(filters = 256, kernel_size = 4, strides = 2, activation = 'relu', padding = 'SAME'),
        BatchNormalization(),

        Flatten(),
        Dense(tfp.layers.MultivariateNormalTriL.params_size(latent_dim)),

        tfp.layers.MultivariateNormalTriL(latent_dim, activity_regularizer = kl_regularizer)
    ])
```

##### Decoder

For the decoder I had previously tried to use ConvTranspose2D Layers without success, so I finally decided to use UpSampling and Conv2D layers. We take the input of our Dense layer (16384) and resize it before starting to apply the upsampling and  convolutional layers. Again, a four times bigger input (65538) would be too much/take up too much memory, but would be best suited for our (512, 512, 1) spectograms.

```
decoder = Sequential([
        Dense(16384, activation = 'relu', input_shape = (latent_dim,)),
        Reshape((8, 8, 256)),
        UpSampling2D(size=(2, 2)),
        
        Conv2D(filters = 128, kernel_size = 3,
               activation = 'relu', padding = 'SAME'),
        BatchNormalization(),
        
        UpSampling2D(size=(2, 2)),
        Conv2D(filters = 64, kernel_size = 3,
               activation = 'relu', padding = 'SAME'),
        BatchNormalization(),
        
        UpSampling2D(size=(2, 2)),
        Conv2D(filters = 32 , kernel_size = 3,
               activation = 'relu', padding = 'SAME'),
        
        UpSampling2D(size=(2, 2)),
        Conv2D(filters = 128 , kernel_size = 3,
               activation = 'relu', padding = 'SAME'),
        
        UpSampling2D(size=(2, 2)),
        Conv2D(filters = 64 , kernel_size = 3,
               activation = 'relu', padding = 'SAME'),
        UpSampling2D(size=(2, 2)),
        Conv2D(filters = 32 , kernel_size = 3,
            activation = 'relu', padding = 'SAME'),
        
        Conv2D(filters = 1 , kernel_size = 3, padding = 'SAME'),
        Flatten(),
        tfp.layers.IndependentBernoulli(event_shape = (512, 512, 1))
    ])
```

We then define our other loss function, the reconstruction loss, which compares the differences between the input and output images. 

```
def reconstruction_loss(batch_of_images, decoding_dist):
    return -tf.reduce_sum(decoding_dist.log_prob(batch_of_images), axis = 0)
```

This is the one we pass to our final model defined below.

```
vae = Model(inputs=encoder.inputs, outputs=decoder(encoder.outputs))
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
vae.compile(optimizer=optimizer, loss=reconstruction_loss)
```

#### Evaluation

For this step we define a function that generates new samples of spectograms from the generative model, taking the prior distribution and the decoder to generate this data with random sampling.

```
def generate_music(prior, decoder, n_samples):
    z = prior.sample(n_samples)
    return decoder(z).mean()

n_samples = 5
sm = generate_music(prior, decoder, n_samples)
```

We finally use librosa to reconstruct the spectogram into music and also to create plots of our samples.

*Note: After much trouble with the preprocessing step, I am only begining to train my final model. With only 10 epochs completed, these are the results.*

**Sample 01**

*Spectogram:*
![image](https://github.com/monica-ayala/AudioGeneration/assets/75228128/2a45a31b-c7fe-42d0-bdb1-41b8014ad054)

*Audio:* [01](https://drive.google.com/file/d/1mbF8tjqJycRnt4WydzhHou_O8c5Tzpr0/view?usp=drive_link)

**Sample 02**

*Spectogram:*

*Audio:* [01]()

**Sample 03**

*Spectogram:*

*Audio:* [01]()

**Sample 04**

*Spectogram:*

*Audio:* [01]()

**Sample 05**

*Spectogram:*

*Audio:* [01]()

For the [previously implemented model](https://github.com/monica-ayala/AudioGeneration/blob/main/models/e150_01.md) that failed the spectograms would always look like this even after training for days:
![image](https://github.com/monica-ayala/AudioGeneration/assets/75228128/07c0e751-398e-48a0-bcc2-66a865a4e11c)

