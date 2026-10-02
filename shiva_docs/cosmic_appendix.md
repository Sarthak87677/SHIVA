# SHIVA-1 Cosmic Appendix

*Theory, derivations and scaling laws behind the universe-scale results of the SHIVA-1 whitepaper.*

Every closed-form statement below is either verified symbolically by `shiva_core.physics_mutator`
(SymPy) or checked numerically by the test suite; the corresponding test is cited in brackets.

---

## A1. Units of the simulated universes

Particle universes use $G = M_{\rm tot} = R_0 = 1$ (total mass and initial radius), so the dynamical time is
$t_{\rm dyn} = \sqrt{R_0^3/GM} = 1$. A reference run of 4000 steps at $\Delta t = 2\times10^{-3}$ covers
$8\,t_{\rm dyn}$ — enough for violent relaxation of a cold collapse (Lynden-Bell 1967) and several crossing
times, not enough for two-body relaxation ($t_{\rm relax} \approx N t_{\rm dyn}/8\ln N \approx 4\,t_{\rm dyn}$ at
$N = 160$, so collisional effects are present and are part of the dynamics, as in small star clusters).
Gravitational softening $\epsilon = 0.05$ enters through $s \to s + \epsilon^2$ in every pair potential;
because the force is the exact gradient of the *softened* potential, energy conservation is unaffected.

## A2. The gravitational ladder as a family of field theories

The mutator's ladder is nested — each rung contains the previous one as a special case:

| rung | potential (unit masses) | field equation (verified) |
|---|---|---|
| inverse-square | $U = -G/r$ | $\nabla^2\Phi = 4\pi G\rho$ |
| inverse-$k$ | $U = -G\,r^{1-k}/(k-1)$ | Gauss law in $d_{\rm eff} = k+1$ dimensions |
| tensor | $U = -G\,\phi_k(\rho)$, $\rho^2 = x^{T}Qx$, $\det Q = 1$ | $\nabla\cdot(Q^{-1}\nabla\Phi) = 4\pi G\rho$ |
| fractal | $U = -G\,\phi_k(\rho)\,\big[1+\epsilon\sum_n\gamma^n\cos(\omega\beta^n\ln\tfrac{\rho}{r_0})\big]$ | running coupling $G_{\rm eff}(r)$ |
| Yukawa (branch) | $U = -G e^{-r/\lambda}/r$ | $(\nabla^2-\lambda^{-2})\Phi = 4\pi G\rho$ |

**Inverse-$k$ as dimension.** The radial Laplacian in $d$ dimensions, $\Phi'' + \frac{d-1}{r}\Phi'$,
annihilates $r^{2-d}$; a force $|F| \propto r^{-k}$ is therefore the Green's function of a Gauss law in
$d_{\rm eff} = k+1$ dimensions, continued to non-integer $d$ in the sense of Stillinger (1977). Mutating $k$ is
mutating the effective dimensionality felt by gravity *without* changing the dimension of the configuration
space. [`test_field_equations_verified`, `test_effective_dimension`]

**Tensor gravity.** Under $y = Q^{1/2}x$ the anisotropic operator becomes the ordinary Laplacian, so
$(x^TQx)^{-1/2}$ is its Green's function (checked on random points; residual $< 10^{-8}$). With $\det Q = 1$ the
mutation changes *shape*, not strength: the attraction along an eigen-direction with eigenvalue $\lambda$ is
$|F| = G/(\sqrt{\lambda}\,r^2)$.

**Fractal gravity and discrete scale invariance.** The modulation is a Weierstrass-type sum in $\ln r$, so the
law is (quasi-)invariant under the discrete dilations $r \to r\,e^{2\pi/\omega}$ — the discrete scale
invariance of Sornette (1998). It is equivalent to a scale-dependent coupling with "beta function"
$\beta(G_{\rm eff}) = \mathrm{d}G_{\rm eff}/\mathrm{d}\ln r$ (reported symbolically by the mutator). Note that the
*force* modulation is $\epsilon\sum_n\gamma^n\omega\beta^n$ times larger than the potential modulation: with
$\epsilon = 0.4$, $\omega = 6$, $\beta = 2$, $\gamma = 0.5$ and three octaves the force reverses sign in thin
shells (repulsive shells; see `figures/force_laws.png`) even though the potential envelope stays attractive.

## A3. Orbits

For a central potential the effective potential is $U_{\rm eff} = L^2/2r^2 + U(r)$. A circular orbit at $r_0$
exists when $U'(r_0) > 0$ and is stable when

$$\kappa^2 \equiv U_{\rm eff}''(r_0) = \frac{3U'(r_0)}{r_0} + U''(r_0) > 0 ,$$

with orbital and epicyclic frequencies $\Omega^2 = U'/r$ and $\kappa^2$; the apsidal angle (pericentre to
apocentre) of a near-circular orbit is $\psi = \pi\,\Omega/\kappa$. `SymbolicPhysics.orbital_analysis`
evaluates these for any mutated law and scans $r\in[0.05, 20]$ for stability *bands*.

* **Power laws.** $\psi = \pi/\sqrt{3-k}$; stable circular orbits exist iff $k < 3$, i.e. $d_{\rm eff} < 4$ —
  Ehrenfest's (1917) argument that planetary systems need three space dimensions, recovered continuously.
  Orbits close for *all* bound initial conditions only for $k = 2$ and Hooke's law (Bertrand 1873).
  [`test_power_law_apsidal_angle`, `test_power_law_k_above_3_has_no_stable_circular_orbit`]
* **Simulation check.** Two bodies integrated by the N-body code with the compiled SymPy force advance their
  pericentre by $2\psi$ per radial period to within 2 % for $k = 2$ and $k = 2.5$
  ($\psi = 4.443$ rad). [`test_two_body_precession_matches_symbolic_apsidal_angle`]
* **Yukawa.** Circular orbits are stable only for $r < \lambda(1+\sqrt5)/2$ — the golden ratio sets the edge of
  a screened gravitating system (reproduced to < 3 %). [`test_yukawa_stability_edge_matches_classical_result`]
* **Fractal gravity — closed form (derived here).** Write $U = -G\,M(x)/r$ with $x = \ln r$ and
  $M' = \mathrm{d}M/\mathrm{d}x$. Then $U' = G r^{-2}(M-M')$, $U'' = G r^{-3}(-2M+3M'-M'')$, hence

  $$\Omega^2 = \frac{G}{r^3}(M - M'),\qquad \kappa^2 = \frac{G}{r^3}(M - M''),\qquad
  \psi = \pi\sqrt{\frac{M-M'}{M-M''}} .$$

  Stability requires $M - M'' > 0$. For a single octave $M = 1+\epsilon\cos\omega x$ this is
  $1 + \epsilon(1+\omega^2)\cos\omega x > 0$: as soon as $\epsilon(1+\omega^2) > 1$ the radial axis splits into
  alternating stable and unstable bands with period $2\pi/\omega$ in $\ln r$ — **quantised orbital shells from
  purely classical gravity**. The showcase genome ($\epsilon(1+\omega^2) \approx 15$) has 17 bands between
  $r = 0.05$ and $20$, and at $r_0 = 1$ the formula gives $\psi = \pi\sqrt{1.7/102.5} = 0.4046$, identical to the
  SymPy-numeric value. [`test_fractal_orbit_closed_form`, `test_fractal_gravity_has_multiple_stability_bands`]

## A4. Virial theorem and the existence of bound structures

For a bounded system the Clausius virial theorem gives $\langle\sum p\cdot v\rangle = -\langle\sum x\cdot F\rangle$
for *any* force law; the simulator reports $Q_v = -\sum x\cdot F/\sum p\cdot v$ (equilibrium $Q_v = 1$), which the
`virialized` initial condition enforces for whatever law the mutator produced. For homogeneous potentials of
degree $n$ and quadratic kinetic energy, $2\langle T\rangle = n\langle U\rangle$; for inverse-$k$ gravity
$n = 1-k$, so

$$E = T + U = \frac{3-k}{2}\,U .$$

Bound virialised structures ($E < 0$ with $U < 0$) exist only for $k < 3$; at $k = 3$ they are marginal and for
$k > 3$ any virialised configuration has $E > 0$. The orbital criterion of A3 and the virial criterion of A4
agree: **stable self-gravitating structure requires $k < 3$ ($d_{\rm eff} < 4$)**. The Noether module reports
the homogeneity degree (`virial_degree`) whenever every active channel is a pure power law.

## A5. Gravitational instability (generalised Jeans analysis)

Linearising continuity and Euler's equation for a uniform gas of sound speed $c_s$ with an arbitrary pair
potential $\phi$ gives $\omega^2 = c_s^2q^2 + \rho_0 q^2\hat\phi(q)$. Using the three-dimensional Fourier transform
of $r^{-a}$ with $a = k-1\in(0,3)$,

$$\omega^2 = c_s^2 q^2 - G\rho_0 A_k\,q^{k-2},\qquad
A_k = \frac{\pi^{3/2}\,2^{4-k}\,\Gamma\!\big(\tfrac{4-k}{2}\big)}{(k-1)\,\Gamma\!\big(\tfrac{k-1}{2}\big)} ,$$

which reduces to the classical Jeans relation for $k = 2$ ($A_2 = 4\pi$). Exact values from
`SymbolicPhysics.jeans_analysis`:

| $k$ | $A_k$ (exact) | $A_k$ | $q_J$ ($G=\rho_0=c_s=1$) | preferred scale $2\pi/q_*$ |
|---|---|---|---|---|
| 1.5 | $2\sqrt2\,\pi^{3/2}$ | 15.750 | 3.01 | none (largest scales fastest) |
| 2.0 | $4\pi$ | 12.566 | 3.54 | none (scale-free growth for $q \ll q_J$) |
| 2.5 | $\tfrac{4\sqrt2}{3}\pi^{3/2}$ | 10.500 | 4.80 | 3.30 |
| 3.0 | $\pi^2$ | 9.870 | 9.87 | 1.27 |
| 3.5 | $\tfrac{8\sqrt2}{5}\pi^{3/2}$ | 12.600 | 158.8 | 0.070 |

Three regimes follow — a *prediction* of characteristic structure scales that SHIVA-1's clustering
estimators can test:

1. $k < 2$ (super-Newtonian range): the growth rate $\Gamma^2 = G\rho_0A_kq^{k-2} - c_s^2q^2$ diverges as
   $q\to 0$ — the universe collapses coherently on its largest scales.
2. $k = 2$: growth is scale-free on large scales (our universe).
3. $2 < k < 4$: growth peaks at $q_*^{4-k} = (k-2)G A_k\rho_0/2c_s^2$, a **preferred clustering wavelength**
   that shrinks rapidly as $k \to 3.5$: such universes fragment into many small clumps.

**Anisotropic (tensor) gravity.** With $\phi(x) = f(x^TQx)$ and $\det Q = 1$, the substitution $y = Q^{1/2}x$ gives
$\hat\phi(q) = \hat\phi_{\rm iso}\big(\sqrt{q^TQ^{-1}q}\big)$, hence

$$\omega^2 = c_s^2q^2 - G\rho_0A_k\,q^2\,(q^TQ^{-1}q)^{(k-4)/2}.$$

For a wave-vector along an eigen-direction with eigenvalue $\lambda$ the gravitational term is multiplied by
$\lambda^{(4-k)/2}$: for $k<4$ instability grows fastest for wave-vectors along the *largest* eigenvalue of $Q$,
i.e. matter collapses into sheets normal to that axis — an anisotropic analogue of Zel'dovich pancakes. For the
reference-run winner ($k = 2.5$, eigenvalues 0.32, 1.49, 2.11) the factor ranges from 0.42 to 1.75 between the
softest and stiffest axes, predicting strongly aspherical collapse (observed asphericity growth 0.39 versus 0.20
for Newtonian gravity; whitepaper §5.4). This prediction is derived here but not yet tested quantitatively.

**Screening switches structure formation off.** For Yukawa gravity
$\omega^2 = c_s^2q^2 - 4\pi G\rho_0\,q^2/(q^2+\lambda^{-2})$: the unstable band is $q^2 < k_J^2 - \lambda^{-2}$, so
if $\lambda k_J < 1$ no mode grows at all. [`test_jeans_analysis_limits`]

## A6. Newtonian cosmology

The cosmology gene adds the Newtonian cosmological-constant force $F = \tfrac{\Lambda}{3}m\,x$, i.e. the
non-relativistic limit of de Sitter ($\Lambda > 0$, accelerated expansion) or anti-de Sitter ($\Lambda < 0$,
a harmonic confining box) space-times. It is isotropic about the origin but not translation-invariant, so it
conserves energy and angular momentum while breaking momentum — exactly what the Noether battery measures
(in the reference run's Noether battery, $N = 64$ over $2\,t_{\rm dyn}$: energy drift $2.2\times10^{-6}$, angular-momentum drift $2.2\times10^{-16}$, momentum no longer protected).
A Hubble drag $\dot p = -Hp$ models peculiar-velocity decay in comoving coordinates; the proper energy
bookkeeping in that case is the Layzer–Irvine equation, left for future work.

## A7. The conservation-law budget of a universe

| mutation | energy | momentum | angular momentum | time reversal | Liouville |
|---|---|---|---|---|---|
| any central reciprocal law, separable $T(p)$ | ✓ | ✓ | ✓ | ✓ | ✓ |
| anisotropic metric $Q$ or mass tensor $M$ | ✓ | ✓ | only about common symmetry axes | ✓ | ✓ |
| $G(t)$ modulation | ✗ | ✓ | ✓ | ✗ | ✓ |
| $\Lambda \neq 0$ | ✓ | ✗ | ✓ | ✓ | ✓ |
| uniform $B$ field | ✓ | pseudo-momentum $\sum(p - q\,x\times B)$ | canonical $J_B = L_B + \sum \tfrac{q}{2}B\,r_\perp^2$ ($B$ = field strength) | ✗ | ✓ |
| non-reciprocal species $K \neq K^T$ | ✗ | ✗ | ✗ | ✓ | ✓ |
| retarded propagation | ✗ | ✗ | ✗ | ✗ | ✗ |
| friction / active matter / Hubble drag | ✗ | ✗ | ✗ | ✗ | ✗ |
| periodic box / reflective walls | ✓ / ✓ | ✓ / ✗ | ✗ / ✗ | ✓ | ✓ |

Two entries deserve comment. **Non-reciprocal forces** depend on positions only, so the flow
$\dot q = v(p)$, $\dot p = F(q)$ is divergence-free (Liouville holds) and time-reversal symmetric, yet energy and
momentum are not conserved — action ≠ reaction lets pairs self-propel, the mechanism behind non-reciprocal
phase transitions (Ivlev et al. 2015; Fruchart et al. 2021). **Retarded gravity** without the velocity-dependent
terms of general relativity produces aberration: the attraction points to where the source *was*, the orbit
gains or loses energy, and the system is no longer Hamiltonian in the particle variables. This is Laplace's
(1805) argument that Newtonian gravity must propagate (effectively) much faster than light; general relativity
evades it because velocity-dependent terms cancel the aberration almost exactly (Carlip 2000). SHIVA-1's
retarded universes show precisely this secular drift (≈ 3 % in energy over 2 $t_{\rm dyn}$ at $c_{\rm prop} = 3$).

## A8. Cosmic-structure estimators

* **Friends-of-friends** haloes with linking length $b = 0.2\,\bar\ell$ ($\bar\ell$ = mean interparticle
  separation of the initial state; Davis et al. 1985) — the standard halo definition (overdensity ≈ 180).
* **Two-point correlation** by the Landy–Szalay estimator $\xi = (DD - 2DR + RR)/RR$ with a 4× random catalogue,
  fitted by $\xi = (r/r_0)^{-\gamma}$; galaxy surveys give $\gamma \approx 1.8$.
* **Correlation dimension** $D_2$ by Grassberger–Procaccia over the most linear window spanning ≥ 0.6 decades.
  With $N \sim 10^2$ the estimator is biased low by edge and finite-size effects (128 uniform points in a cube
  give $D_2 \approx 2.6$, not 3), so SHIVA-1 always compares with Poisson samples of the same size measured over
  the *same* range and scores the *deficit* $1 - D_2/D_2^{\rm Poisson}$. Caveat: at these $N$ the deficit
  cannot separate genuine multifractality from smooth central concentration (a Plummer sphere also scores).

## A9. The fine-structure constant, chemistry and strength-limited cosmic engineering

With electron and nucleon masses held fixed, the Bohr radius scales as $a_0 \propto \alpha^{-1}$ and bond energies
as $\alpha^2$ (Press & Lightman 1983). Moduli and strengths are energy densities, so

$$E,\ \sigma \propto \frac{\alpha^2}{a_0^3}\propto\alpha^5,\qquad \rho \propto \frac{m_p}{a_0^3}\propto\alpha^3,
\qquad \boxed{\ \frac{\sigma}{\rho}\propto\alpha^2\ }$$

The mutator keeps $\alpha$ inside $1/180 < \alpha < 1/85$, the window usually quoted for stable chemistry and
long-lived stars (Barrow & Tipler 1986; Tegmark 1998). Specific strength governs every strength-limited
megastructure, which therefore scales as $\alpha^2/g$:

| structure | limit | scaling |
|---|---|---|
| rotating habitat (self-weight only) | $R_{\max} = \sigma/(SF\,\rho\,a)$ | $\alpha^2/a$ |
| flywheel (thin rim) | $e_{\max} = \sigma/2\rho$ | $\alpha^2$ |
| hanging tether / space elevator | breaking length $L = \sigma/\rho g$ | $\alpha^2/g$ |

With our-universe materials at 1 g and safety factor 2 (values from `real_world_mapper.MATERIALS`):

| material | $\sigma/\rho$ (kJ/kg) | habitat $R_{\max}$ | flywheel $e_{\max}$ (Wh/kg) | breaking length |
|---|---|---|---|---|
| Al 7075-T6 | 179 | 9.1 km | 25 | 18 km |
| Ti-6Al-4V | 199 | 10.1 km | 28 | 20 km |
| 4140 steel (Q&T) | 83 | 4.3 km | 12 | 8.5 km |
| CFRP (unidirectional) | 938 | 48 km | 130 | 96 km |
| Zylon PBO | 3718 | 190 km | 516 | 379 km |

A universe with $\alpha$ 20 % larger multiplies every one of these by 1.44; one at the lower edge of the chemistry
window ($\alpha/\alpha_0 = 0.76$) shrinks them by 0.58. The habitat domain's actual designs carry pressure and
2 t/m² of shielding, which lowers the attainable radius well below $R_{\max}$.

## A10. Light sails and the speed of light

Radiation pressure on a reflector at normal incidence is $P = (1+R)S/c$, so a sail's characteristic
acceleration $a_c = (1+R)SA/(mc)$ scales as $1/c$. In universes where the inertia gene lowers the speed limit
$c$ (and the technology context inherits it), sails become proportionally more effective; the lightness
number $\beta = a_c/g_\odot$ also inherits the mutated solar gravity. The thermal limit
$T = [(1-R)S/((\varepsilon_f+\varepsilon_b)\sigma_{SB})]^{1/4}$ bounds how close to the star a polyimide
film may operate.

## A11. Rigid bodies: spontaneous rotational symmetry breaking

The free rigid body's intermediate-axis rotation is unstable (tennis-racket / Dzhanibekov effect): the
splitting integrator reproduces periodic flips while conserving $|L|$ to $10^{-15}$. With internal dissipation at
constant $|L|$, energy decays until the body spins about its *maximum*-inertia axis — the "major-axis rule"
learned the hard way by Explorer 1 (1958). SHIVA-1 applies the dissipation with a compensating attitude rotation
so the spatial angular momentum stays fixed to $10^{-10}$ [`test_rigid_body_flips_and_major_axis_rule`], and the
habitat domain enforces the same rule as a design constraint ($I_{\rm spin}/I_\perp \ge 1.2$).

## A12. What these universes are — and are not

The SHIVA-1 universes are classical, non-relativistic (except for the kinetic-energy law), small-$N$ models.
They probe how *structural features* of laws — exponents, symmetries, screening, scale invariance,
reciprocity — shape emergent structure and engineering limits. They are not claims about other regions of
our cosmos, and the engineering mappings are order-of-magnitude scaling arguments, not an alternative
chemistry.

---

### References

Barrow, J. D. & Tipler, F. J. (1986) *The Anthropic Cosmological Principle*. OUP.
Bertrand, J. (1873) C. R. Acad. Sci. 77, 849–853.
Carlip, S. (2000) Aberration and the speed of gravity. Phys. Lett. A 267, 81–87.
Davis, M., Efstathiou, G., Frenk, C. S. & White, S. D. M. (1985) ApJ 292, 371–394.
Ehrenfest, P. (1917) Proc. Amsterdam Acad. 20, 200–209.
Fruchart, M., Hanai, R., Littlewood, P. B. & Vitelli, V. (2021) Non-reciprocal phase transitions. Nature 592, 363–369.
Grassberger, P. & Procaccia, I. (1983) Phys. Rev. Lett. 50, 346–349.
Ivlev, A. V. et al. (2015) Statistical mechanics where Newton's third law is broken. Phys. Rev. X 5, 011035.
Landy, S. D. & Szalay, A. S. (1993) ApJ 412, 64–71.
Laplace, P.-S. (1805) *Traité de mécanique céleste*, vol. 4.
Lynden-Bell, D. (1967) MNRAS 136, 101–121.
Press, W. H. & Lightman, A. P. (1983) Phil. Trans. R. Soc. A 310, 323–336.
Sornette, D. (1998) Discrete scale invariance and complex dimensions. Phys. Rep. 297, 239–270.
Stillinger, F. H. (1977) J. Math. Phys. 18, 1224–1234.
Tegmark, M. (1998) Is "the theory of everything" merely the ultimate ensemble theory? Ann. Phys. 270, 1–51.
