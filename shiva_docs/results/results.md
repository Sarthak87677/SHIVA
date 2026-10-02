# SHIVA-1 results (seed 42, profile standard, v0.1.0)

## Noether validation
| universe | energy (pred / drift) | momentum | angular momentum | consistency |
|---|---|---|---|---|
| newtonian | conserved / 2.2e-06 | conserved / 4.3e-16 | conserved / 4.5e-16 | 100% |
| inverse-k 2.5 | conserved / 3.6e-05 | conserved / 3.1e-16 | conserved / 7.4e-16 | 100% |
| anisotropic gravity | conserved / 3.6e-06 | conserved / 4.3e-16 | broken / 7.1e-02 | 100% |
| axisymmetric gravity | conserved / 2.0e-06 | conserved / 2.2e-16 | conserved / 3.0e-16 | 100% |
| relativistic inertia c=1.5 | conserved / 1.3e-06 | conserved / 3.7e-16 | conserved / 7.4e-16 | 100% |
| anisotropic inertia | conserved / 2.5e-06 | conserved / 7.7e-16 | broken / 2.9e-02 | 100% |
| time-varying G | broken / 1.8e-01 | conserved / 3.3e-16 | conserved / 4.6e-16 | 100% |
| Lambda < 0 (AdS box) | conserved / 2.2e-06 | broken / 3.4e-16 | conserved / 2.2e-16 | 100% |
| Coulomb + B field | conserved / 1.4e-06 | conserved / 8.6e-08 | conserved / 5.0e-08 | 100% |
| non-reciprocal species | broken / 8.7e-03 | broken / 9.2e-02 | broken / 1.8e-02 | 100% |
| retarded gravity c=3 | broken / 3.1e-02 | broken / 4.0e-02 | broken / 1.9e-02 | 100% |
| Langevin bath | broken / 1.2e+00 | broken / 2.1e-01 | broken / 1.4e-01 | 100% |

## Gravity ladder (symbolic analysis)
| law | d_eff | stable circular orbit (r=1) | apsidal angle | stability bands | field eqs verified |
|---|---|---|---|---|---|
| newtonian (inverse-square) | 3.00 | True | 3.1416 | 1 | True |
| inverse-k (k = 2.5, d_eff = 3.5) | 3.50 | True | 4.4429 | 1 | True |
| tensor (anisotropic metric Q) | 3.00 | True | 3.1416 | 1 | True |
| fractal (log-periodic, eps = 0.4) | 3.00 | True | 0.4046 | 17 | True |
| yukawa (lambda = 1) | 3.00 | True | 4.4429 | 1 | True |

## Universe search
Evaluations: 59; best fitness 0.5764 (after CMA-ES: 0.5885).

```
PhysicsGenome shiva-best-518a60c477c7 (gen 4, dim 3)
  gravity : tensor, G=1, k=2.5, Q-eig=[0.319, 1.489, 2.106]
  em      : off (alpha/alpha0=1)
  inertia : newtonian, masses=equal
  thermo  : gamma=0, T=0, noise=gaussian(q=1), active=0@v0=0.5, adiabatic=1.4
  fluid   : nu=0.001, h=1, lambda=1, N-eig=[1.0, 1.0], drag=0|v|^2
  interact: instantaneous, species=1, nonreciprocity=0.000, sigma=0.3, core=0.347@0.0503
  cosmos  : Lambda=0.178, H=0
```

| genome | gravity law | fitness | richness | novelty | feasibility |
|---|---|---|---|---|---|
| 518a60c477c7 | tensor | 0.5764 | 0.603 | 0.0 | 0.828 |
| 11e760b54b9a | inverse_k | 0.5687 | 0.6046 | 0.2433 | 0.761 |
| 15f81c563fe3 | inverse_k | 0.5687 | 0.6046 | 0.0 | 0.761 |
| bff6de480328 | inverse_k | 0.5635 | 0.5963 | 0.0 | 0.78 |
| 2ef70fc5ed06 | inverse_k | 0.5613 | 0.5963 | 0.1925 | 0.766 |
| ea3b2e57a6b4 | tensor | 0.5575 | 0.579 | 0.2248 | 0.857 |
| b5893e8a8d76 | tensor | 0.5326 | 0.5454 | 0.2522 | 0.9 |
| d5424d059d46 | inverse_k | 0.5323 | 0.553 | 0.16 | 0.85 |
| 5de07d59c7a1 | inverse_k | 0.5298 | 0.5539 | 0.1702 | 0.826 |
| 39c99bfdf960 | inverse_k | 0.5227 | 0.542 | 0.132 | 0.857 |

Held-out validation at final resolution (fresh seeds):

| candidate | gravity law | search fitness | validated richness |
|---|---|---|---|
| 0c9ea847fffe | tensor | 0.589 | 0.501 +/- 0.073 |
| 518a60c477c7 | tensor | 0.576 | 0.654 +/- 0.019 |
| 11e760b54b9a | inverse_k | 0.569 | 0.621 +/- 0.087 |
| 15f81c563fe3 | inverse_k | 0.569 | 0.621 +/- 0.087 |

## Emergence: best universe vs Newtonian baseline
| phenomenon | best | baseline |
|---|---|---|
| self organization | 1.000 | 1.000 |
| oscillation | 0.344 | 0.000 |
| attractor | 0.034 | 0.127 |
| chaos | 0.993 | 0.866 |
| fractal | 0.957 | 0.872 |
| symmetry breaking | 0.500 | 0.674 |
| clustering | 0.199 | 0.122 |
| pattern formation | 0.997 | 0.997 |
| coherent vortices | 0.722 | 0.722 |
| rotational symmetry breaking | 0.500 | 0.500 |
| **richness R (seed 0)** | **0.667** | **0.612** |
| richness mean +/- sd, 5 held-out test seeds | 0.624 +/- 0.026 | 0.558 +/- 0.083 |

Welch t-test on held-out test seeds (best vs baseline richness): t = 1.69, p = 0.154.

- best: Lyapunov 4.935, D2 = 0.60 (Poisson ref 2.72), FoF groups 2, clustered fraction 0.32, xi slope 2.6371261484307857, dominant period 0.21729729729727942
- baseline: Lyapunov 2.008, D2 = 1.32 (Poisson ref 2.43), FoF groups 1, clustered fraction 0.24, xi slope 2.699021229793555, dominant period 0.535999999999956

## Technology champions (alien physics)
| domain | feasible | eff | stab | adapt | map | cosmic | pareto | PPCA joined front |
|---|---|---|---|---|---|---|---|---|
| propulsion_nozzle | True | 0.77 | 0.47 | 0.76 | 0.88 | 0.78 | 49 | 13 |
| structure_truss | True | 0.60 | 1.00 | 0.59 | 0.80 | 0.60 | 46 | 17 |
| energy_flywheel | True | 0.48 | 1.00 | 0.47 | 0.74 | 0.49 | 55 | 19 |
| drone_frame | True | 1.00 | 0.99 | 0.93 | 1.00 | 0.60 | 51 | 15 |
| pump_impeller | True | 0.82 | 1.00 | 0.82 | 0.91 | 0.35 | 43 | 7 |
| gear_pair | True | 0.76 | 0.99 | 0.75 | 0.88 | 0.76 | 47 | 11 |
| cosmic_habitat | True | 0.65 | 0.87 | 0.64 | 0.82 | 0.26 | 44 | 8 |
| cosmic_lightsail | True | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 40 | 7 |

Adaptive-design agent (mean improvement per 8-edit episode across 5 physics contexts):

| domain | trained policy | random edits | untrained |
|---|---|---|---|
| propulsion_nozzle | 0.3341 | 0.0508 | 0.0000 |
| structure_truss | 0.1980 | -0.0417 | 0.0000 |
| energy_flywheel | 0.1671 | -0.0232 | 0.0000 |

Generative bracket: fitness 0.157 under alien physics, 0.157 for the same shape on Earth; volume fraction 0.45.

## Real-world prototypes
| domain | material | feasible | fidelity | efficiency alien -> real | MC reliability | mass kg | cost |
|---|---|---|---|---|---|---|---|
| propulsion_nozzle | Stainless steel 304 | True | 0.959 | 0.77 -> 0.76 | 62% | 1.2 | $1,820 |
| structure_truss | CFRP quasi-isotropic laminate | True | 0.994 | 0.60 -> 0.67 | 100% | 2.96 | $248 |
| energy_flywheel | Aluminium 7075-T6 | True | 0.949 | 0.48 -> 0.56 | 69% | 1.43e+03 | $57,140 |
| drone_frame | Aluminium 6061-T6 | True | 0.961 | 1.00 -> 1.00 | 100% | 3.05 | $704 |
| pump_impeller | Aluminium 6061-T6 | True | 0.968 | 0.82 -> 0.82 | 100% | 1.33 | $1,936 |
| gear_pair | Alloy steel 4140 (Q&T) | True | 0.871 | 0.76 -> 0.74 | 100% | 9.39 | $1,019 |
| cosmic_habitat | CFRP unidirectional (fibre direction) | True | 0.999 | 0.65 -> 0.64 | 63% | 9.49e+06 | $474,465,952 |
| cosmic_lightsail | Polyimide film (Kapton HN) | True | 0.973 | 1.00 -> 1.00 | 100% | 332 | $761,711 |

Total wall time: 717 s.
