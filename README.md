# mp2p_solar_inpainting

This is a PyTorch code for Over-Exposure Region Recovery of Solar Image using a Mask-Pix2Pix Network.

# Train
You can train this model as follows:

    git clone https://github.com/phoenixtreesky7/mp2p_solar_inpainting.git
    cd mp2p_solar_inpainting/implementations/mp2p/
    python3 mask_pix2pix.py --dataset_name solar --datasave_name solar --batch_size 8

The code is built on the work of [pix2pix(PyTorch GAN)](https://github.com/eriklindernoren/PyTorch-GAN).
# Mask-Pix2Pix

The main improvements and contributions of this paper are: 

`(1)` Unlike the conventional Pix2Pix [1], our network utilizes the Convolution-SwitchNorm-LReLU/ReLU modules (LReLU for encoder and ReLU for decoder) rather than the Convolution-BatchNorm-ReLU ones. The former (i.e. `switchable normalization` [2]) can switch between `BatchNorm` [3], `LayerNorm` [4] and `InstanceNorm` [5] by learning their importance weights in an end-to-end manner. The improved architecture of our model boosts the robustness of the network.

`(2)` Our objective function contain an adversarial cGAN loss, a masked L1 loss and a edge mask loss/smoothness. The `adversarial cGAN loss` can capture the full entropy of the conditional distributions they model, and thereby produce highly realistic
textures. The `masked L1 loss` calculates the L1 loss only in masked regions (the missing regions), enforcing correctness at low frequencies which guarantees restoration of high fidelity for missing regions. The `edge mask loss` is used for smoothing edges of OERs and suppressing edge artifacts in final restored image. 

[1] P. Isola, J.-Y. Zhu, T. Zhou, A. A. Efros, Image-to-image translation with conditional adversarial networks, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 1125–1134

[2] P. Luo, J. Ren, Z. Peng, Differentiable learning-to-normalize via switchable normalization, arXiv preprint arXiv:1806.10779.

[3] S. Ioffe, C. Szegedy, Batch normalization: Accelerating deep network training by reducing internal covariate shift, arXiv preprint arXiv:1502.03167.

[4] J. Lei Ba, J. R. Kiros, G. E. Hinton, Layer normalization, arXiv preprint arXiv:1607.06450.

[5] D. Ulyanov, A. Vedaldi, V. Lempitsky, Instance normalization: The missing ingredient for fast stylization, arXiv preprint arXiv:1607.08022.
