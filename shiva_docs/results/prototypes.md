# Real-world prototypes

### propulsion_nozzle prototype (Stainless steel 304)
- feasible on Earth: **True**  (Monte-Carlo reliability 62%)
- fidelity to alien design: 0.959; efficiency alien 0.766 -> real 0.757
- mass 1.2 kg, est. material+parts cost $1,820

| parameter | alien | real |
|---|---|---|
| chamber_pressure | 2e+07 | 1.506e+07 |
| expansion_ratio | 12.57 | 12.57 |
| throat_radius | 0.006977 | 0.0059 |
| chamber_temperature | 2344 | 2344 |
| molar_mass | 0.02789 | 0.02789 |
| half_angle | 14.98 | 15 |
| wall_thickness | 0.001876 | 0.0021 |

| predicted metric | mean | std |
|---|---|---|
| Isp_s | 214.8 | 1.5 |
| Isp_ideal_s | 283.8 | 1.9 |
| thrust_N | 2706 | 64 |
| mass_flow_kg_s | 1.285 | 0.031 |
| exit_mach | 3.503 | 0.0073 |
| exit_pressure_Pa | 1.319e+05 | 2.1e+03 |
| thrust_coefficient | 1.644 | 0.002 |
| c_star_m_s | 1281 | 8.8 |
| wall_SF | 1.524 | 0.085 |
| thrust_to_weight | 229.7 | 5.8 |
| mass_kg | 1.202 | 0.024 |
| length_m | 0.3119 | 0.00096 |

Build instructions:
1. Generate the contour from the SCAD/STL (throat r = 5.9 mm, expansion ratio 12.6, half-angle 15.0 deg).
2. Fabricate in Stainless steel 304 by laser powder-bed fusion with integral regenerative cooling channels, or machine from bar; minimum wall 2.10 mm.
3. Stress-relieve and HIP printed parts; inspect channels by CT scan; flow-test the coolant circuit.
4. Hydrostatically proof-test the chamber at 1.5x design chamber pressure.
5. Hot-fire testing only at a licensed propulsion test facility under its safety procedures.
6. Inspect all critical dimensions against the CAD model (tolerances as noted) before assembly.
7. Proof-test to 1.5x the design load/pressure/speed behind a barrier before use; record results.
8. Planning values only: confirm material certificates and have a qualified engineer review load-bearing, rotating or pressurised parts.

### structure_truss prototype (CFRP quasi-isotropic laminate)
- feasible on Earth: **True**  (Monte-Carlo reliability 100%)
- fidelity to alien design: 0.994; efficiency alien 0.600 -> real 0.669
- mass 2.96 kg, est. material+parts cost $248

| parameter | alien | real |
|---|---|---|
| h0 | 0.9847 | 0.985 |
| h1 | 0.7528 | 0.755 |
| h2 | 0.398 | 0.4 |
| area_top | 0.0001519 | 0.0001539 |
| area_bottom | 0.0003942 | 0.0004155 |
| area_web | 0.0001901 | 0.0002011 |

| predicted metric | mean | std |
|---|---|---|
| mass_kg | 2.952 | 0.046 |
| tip_deflection_mm | 3.503 | 0.18 |
| max_stress_MPa | 37.19 | 0.28 |
| buckling_utilisation | 0.5315 | 0.051 |
| payload_to_mass | 101.6 | 1.6 |

Build instructions:
1. Cut members to the joint-to-joint lengths in the BOM (add gusset overlap).
2. Machine gusset plates in CFRP quasi-isotropic laminate; drill M8 clearance holes per joint.
3. Assemble on a flat jig; bolt joints (pin-jointed assumption) and torque to spec.
4. Anchor joints 0 and 4 to a rigid wall; apply the payload gradually while monitoring tip deflection.
5. Acceptance: tip deflection under 300 kg below 8.0 mm.
6. Inspect all critical dimensions against the CAD model (tolerances as noted) before assembly.
7. Proof-test to 1.5x the design load/pressure/speed behind a barrier before use; record results.
8. Planning values only: confirm material certificates and have a qualified engineer review load-bearing, rotating or pressurised parts.

### energy_flywheel prototype (Aluminium 7075-T6)
- feasible on Earth: **True**  (Monte-Carlo reliability 69%)
- fidelity to alien design: 0.949; efficiency alien 0.479 -> real 0.562
- mass 1.43e+03 kg, est. material+parts cost $57,140

| parameter | alien | real |
|---|---|---|
| outer_radius | 0.6737 | 0.641 |
| inner_ratio | 0.1313 | 0.1308 |
| thickness | 0.4 | 0.4 |
| rpm | 7449 | 4800 |
| housing_pressure_ratio | 1e-05 | 1e-05 |

| predicted metric | mean | std |
|---|---|---|
| energy_kWh | 10.44 | 0.51 |
| specific_energy_Wh_kg | 7.34 | 0.21 |
| burst_SF | 2.061 | 0.12 |
| tip_speed_m_s | 322.3 | 4.5 |
| self_discharge_h | 78.53 | 1.5 |
| windage_W | 27.6 | 1.5 |
| bearing_W | 105.3 | 3.1 |
| mass_kg | 1422 | 40 |
| max_stress_MPa | 243.6 | 7.9 |

Build instructions:
1. Filament-wind (or machine) the rotor in Aluminium 7075-T6: R = 641 mm, bore = 84 mm, thickness 400 mm.
2. Cure per resin schedule; machine faces; press onto the hub with controlled interference.
3. Balance to ISO 21940 grade G2.5 or better.
4. Install in the burst-containment housing; evacuate to the design pressure ratio 1.0e-05.
5. Spin-test remotely in a spin pit up to 1.15x the design speed (5520 rpm) before any operation near people.
6. Inspect all critical dimensions against the CAD model (tolerances as noted) before assembly.
7. Proof-test to 1.1x the design load/pressure/speed behind a barrier before use; record results.
8. Planning values only: confirm material certificates and have a qualified engineer review load-bearing, rotating or pressurised parts.

### drone_frame prototype (Aluminium 6061-T6)
- feasible on Earth: **True**  (Monte-Carlo reliability 100%)
- fidelity to alien design: 0.961; efficiency alien 1.000 -> real 1.000
- mass 3.05 kg, est. material+parts cost $704

| parameter | alien | real |
|---|---|---|
| arm_length | 0.3097 | 0.31 |
| arm_od | 0.02808 | 0.028 |
| wall_ratio | 0.06001 | 0.05357 |
| prop_diameter | 0.3763 | 0.381 |
| n_arms | 3 | 3 |
| battery_wh | 309.8 | 325.6 |

| predicted metric | mean | std |
|---|---|---|
| hover_time_min | 47.63 | 1.2 |
| total_mass_kg | 3.047 | 0.0066 |
| hover_power_W | 348.8 | 8.4 |
| rotor_hz | 59.33 | 1.4 |
| arm_natural_hz | 99.61 | 2.7 |
| tip_mach | 0.2928 | 0.0071 |
| arm_SF | 34.91 | 1.9 |
| thrust_per_motor_N | 9.963 | 0.029 |
| mass_kg | 3.047 | 0.0066 |

Build instructions:
1. Cut 3 arm tubes (Aluminium 6061-T6) to 310 mm; deburr and scuff bond areas.
2. CNC-cut centre plates from the SCAD outline; clamp arms between plates with bonded inserts.
3. Print/machine motor mounts; mount motors at arm tips; route ESC wiring along arms.
4. Flash and configure the flight controller; set motor order and propeller direction.
5. Bench-test without propellers, then tethered hover test outdoors away from people; verify vibration spectrum shows no peak near the arm natural frequency.
6. Fly only in compliance with local aviation regulations.
7. Inspect all critical dimensions against the CAD model (tolerances as noted) before assembly.
8. Proof-test to 1.5x the design load/pressure/speed behind a barrier before use; record results.
9. Planning values only: confirm material certificates and have a qualified engineer review load-bearing, rotating or pressurised parts.

### pump_impeller prototype (Aluminium 6061-T6)
- feasible on Earth: **True**  (Monte-Carlo reliability 100%)
- fidelity to alien design: 0.968; efficiency alien 0.823 -> real 0.823
- mass 1.33 kg, est. material+parts cost $1,936

| parameter | alien | real |
|---|---|---|
| r1 | 0.03414 | 0.0341 |
| r2 | 0.1357 | 0.1357 |
| b2 | 0.006429 | 0.0064 |
| beta1 | 32.47 | 32.5 |
| beta2 | 21.55 | 21.5 |
| blades | 7 | 7 |
| rpm | 1645 | 1645 |

| predicted metric | mean | std |
|---|---|---|
| pressure_rise_kPa | 300.6 | 12 |
| head_m | 30.72 | 1.2 |
| efficiency | 0.8227 | 0.0013 |
| shaft_power_W | 4385 | 1.8e+02 |
| disk_friction_W | 299.5 | 17 |
| npsh_required_m | 1.407 | 0.032 |
| npsh_available_m | 11.11 | 0.2 |
| specific_speed_nq | 13.82 | 0.33 |
| de_haller | 0.9249 | 0.017 |
| tip_speed_m_s | 23.36 | 0.34 |
| mass_kg | 1.329 | 0.034 |

Build instructions:
1. Export the impeller SCAD/STL: r1 = 34.1 mm, r2 = 135.7 mm, b2 = 6.4 mm, 7 blades, beta1/beta2 = 32.5/21.5 deg.
2. Produce in Aluminium 6061-T6 by investment casting (production) or SLS/SLM (prototype); machine bore and wear rings.
3. Dynamically balance to ISO 21940 G6.3.
4. Assemble with volute sized for the design flow; prime the pump before start-up.
5. Measure the H-Q curve per ISO 9906 at 1645 rpm; verify 250 kPa at 12 L/s and absence of cavitation noise.
6. Inspect all critical dimensions against the CAD model (tolerances as noted) before assembly.
7. Proof-test to 1.5x the design load/pressure/speed behind a barrier before use; record results.
8. Planning values only: confirm material certificates and have a qualified engineer review load-bearing, rotating or pressurised parts.

### gear_pair prototype (Alloy steel 4140 (Q&T))
- feasible on Earth: **True**  (Monte-Carlo reliability 100%)
- fidelity to alien design: 0.871; efficiency alien 0.757 -> real 0.739
- mass 9.39 kg, est. material+parts cost $1,019

| parameter | alien | real |
|---|---|---|
| module | 0.001811 | 0.002 |
| z1 | 40 | 40 |
| face_factor | 15.66 | 14 |
| pressure_angle | 22.45 | 20 |
| shift | 0.3385 | 0.35 |

| predicted metric | mean | std |
|---|---|---|
| mesh_efficiency | 0.9953 | 2.5e-05 |
| contact_ratio | 1.734 | 2.2e-16 |
| bending_stress_MPa | 70 | 0.67 |
| contact_stress_MPa | 462.5 | 11 |
| power_density_W_kg | 532 | 11 |
| center_distance_mm | 160 | 0 |
| z2 | 120 | 0 |
| ratio | 3 | 0 |
| pitch_velocity_m_s | 6.283 | 0 |
| face_width_mm | 27.97 | 0.27 |
| mass_kg | 9.403 | 0.19 |

Build instructions:
1. Turn blanks in Alloy steel 4140 (Q&T); pinion z = 40, gear z = 120, module 2.00 mm, pressure angle 20.0 deg, profile shift x1 = +0.35 / x2 = -0.35.
2. Hob teeth (or wire-EDM/print for prototypes); through-harden or nitride as specified for steels.
3. Grind/finish flanks to ISO 1328 grade 6-7; deburr tips.
4. Mount on parallel shafts at centre distance 160.00 mm (no shift sum), check backlash and contact pattern with marking compound.
5. Run-in under 25 % load with ISO VG 220 oil, then load-test to 1.25x rated torque.
6. Inspect all critical dimensions against the CAD model (tolerances as noted) before assembly.
7. Proof-test to 1.2x the design load/pressure/speed behind a barrier before use; record results.
8. Planning values only: confirm material certificates and have a qualified engineer review load-bearing, rotating or pressurised parts.

### cosmic_habitat prototype (CFRP unidirectional (fibre direction))
- feasible on Earth: **True**  (Monte-Carlo reliability 63%)
- fidelity to alien design: 0.999; efficiency alien 0.645 -> real 0.643
- mass 9.49e+06 kg, est. material+parts cost $474,465,952

| parameter | alien | real |
|---|---|---|
| radius | 170.9 | 171 |
| rpm | 2.177 | 2.18 |
| thickness | 0.02366 | 0.024 |
| width | 189.7 | 190 |
| pressure | 5.256e+04 | 5.256e+04 |

| predicted metric | mean | std |
|---|---|---|
| gravity_g | 0.9068 | 0.02 |
| rim_speed_m_s | 38.98 | 0.54 |
| hoop_stress_MPa | 503.2 | 9.1 |
| hull_SF | 2.984 | 0.16 |
| habitable_area_km2 | 0.204 | 0.0027 |
| structure_mass_t | 9508 | 2.1e+02 |
| population_capacity | 2040 | 27 |
| coriolis_ratio | 0.05132 | 0.00071 |
| spin_inertia_ratio | 1.658 | 0.0082 |
| mass_kg | 9.508e+06 | 2.1e+05 |

Build instructions:
1. Megastructure: the full-scale design is a conceptual study (launch/in-space manufacturing not costed).
2. Print the 1:1000 scale model from the STL (R = 171 m -> 171 mm).
3. Spin-up schedule: ramp to 2.18 rpm over weeks for vestibular adaptation (Hall 1999).
4. Hull segments must be proof-pressurised before habitation; spin about the maximum-inertia axis.
5. Inspect all critical dimensions against the CAD model (tolerances as noted) before assembly.
6. Proof-test to 1.2x the design load/pressure/speed behind a barrier before use; record results.
7. Planning values only: confirm material certificates and have a qualified engineer review load-bearing, rotating or pressurised parts.

### cosmic_lightsail prototype (Polyimide film (Kapton HN))
- feasible on Earth: **True**  (Monte-Carlo reliability 100%)
- fidelity to alien design: 0.973; efficiency alien 1.000 -> real 1.000
- mass 332 kg, est. material+parts cost $761,711

| parameter | alien | real |
|---|---|---|
| film_thickness | 9.094e-07 | 9e-07 |
| side_length | 443.4 | 443.4 |
| reflectivity | 0.9313 | 0.91 |
| boom_linear_density | 0.06029 | 0.06029 |
| distance_au | 0.4438 | 0.4438 |

| predicted metric | mean | std |
|---|---|---|
| char_accel_mm_s2 | 26.05 | 0.67 |
| lightness_number | 0.8645 | 0.011 |
| sail_temperature_K | 360.4 | 2 |
| total_mass_kg | 331.8 | 7.1 |
| area_m2 | 1.962e+05 | 4e+03 |
| force_mN | 8642 | 2.7e+02 |
| mass_kg | 331.8 | 7.1 |

Build instructions:
1. Procure 0.9 um Polyimide film (Kapton HN) with vapour-deposited aluminium front coat and high-emissivity back coat.
2. Cut four triangular quadrants for a 443.4 m square sail; add rip-stop tape grid.
3. Fold (frog-leg or Miura) onto the deployer spools; integrate booms and tip vanes.
4. Ground deployment test in a gravity-offload rig; thermal-vacuum test the coated film.
5. Inspect all critical dimensions against the CAD model (tolerances as noted) before assembly.
6. Proof-test to 1x the design load/pressure/speed behind a barrier before use; record results.
7. Planning values only: confirm material certificates and have a qualified engineer review load-bearing, rotating or pressurised parts.

> Free dimensions were re-optimised after snapping to standard stock sizes.