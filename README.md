# WGAIN_gp
This is the mini task of van der Schaar Lab, 2023.

Please run the `myGAIN.py` and `myWGAIN_gp.py` directly.

`myGAIN.py` is a reprocution of GAIN (Generative Adversarial Imputation Nets) based on PyTorch.

`myWGAIN_gp.py` is an improvement of GAIN based on WGAN_gp(Wasserstein GAN with gradient penalty).

`utils.py` includes utility functions for GAIN and WGAIN_gp.
- rounding: Handlecategorical variables after imputation
- rmse_loss: Evaluate imputed data in terms of RMSE
- xavier_init: Xavier initialization
- sample_M: binary_sampler
- sample_Z: sample uniform random variables
- sample_batch_index: sample random batch index

Reference:
- J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets," ICML, 2018.
- M. Arjovsky, S. Chintala, L. Bottou, "Wasserstein Generative Adversarial Networks," PMLR, 2017.
- [GAIN by jsyoon0823](https://github.com/jsyoon0823/GAIN)