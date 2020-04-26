# Modelling mask particle capture efficiency

![](assets/penetration_gaussian_prior.png)

![](assets/linear_degradation.gif)


`test_comparison_polar.py` compares the impact of accounting for polarization effects in the penetration model.

The probabilistic examples require PyTorch and [Pyro](https://pyro.ai).

## Generate synthetic data

See `generate_synthetic_data.py`.

## References

1. Bałazy, A. et al. Manikin-Based Performance Evaluation of N95 Filtering-Facepiece Respirators Challenged with Nanoparticles. Ann Occup Hyg 50, 259–269 (2006).
2. Payet, S., Boulaud, D., Madelaine, G. & Renoux, A. Penetration and pressure drop of a HEPA filter during loading with submicron liquid particles. Journal of Aerosol Science 23, 723–735 (1992).
3. Bingham, E. et al. Pyro: Deep Universal Probabilistic Programming. arXiv:1810.09538 [cs, stat] (2018).
