![](examples/gump_example.gif)
# Faceswap-Autoencoder

This is a Pytorch implementation of a Face Swap Autoencoder, roughly based on  [Shaonlu's tensorflow implementation.](https://github.com/shaoanlu/faceswap-GAN). 

## Interesting Stuff

- Both the autoencoder and the discriminator are using spectral normalization
- Discriminator is being used only as a learned preceptual loss, not a direct adversarial loss
- The Conv2d operation has been modified to be compatible with the use of spectral normalization before a pixel-shuffle





