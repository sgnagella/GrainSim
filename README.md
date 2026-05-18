# GrainSim — UAMMD Fork for Dense Granular Flows

[![Documentation Status](https://readthedocs.org/projects/uammd/badge/?version=latest)](https://uammd.readthedocs.io/en/latest/?badge=latest)

GrainSim is a fork of [UAMMD](https://github.com/RaulPPelaez/UAMMD) (Universally Adaptable Multiscale Molecular Dynamics) that extends the framework with a Velocity-Verlet integrator following the modified scheme of Groot & Warren (1997), adapted for contact-rich particulate flows such as dense granular systems. The fork adds rotational degrees of freedom, per-particle stress tracking, and periodic image flags as first-class particle properties.

---

## Motivation

Standard Velocity-Verlet integrators compute forces at particle positions and assume forces are velocity-independent. In **contact-rich flows** — dense granular packings, frictional suspensions, or DPD-like systems — forces such as viscous drag, lubrication, and tangential friction depend explicitly on the **relative velocities** of contacting pairs. If those velocities are stale (from the previous full step), the integrator is inconsistent and can produce significant energy drift or instability.

The **Groot-Warren (1997)** modification resolves this by introducing a half-step velocity prediction before force evaluation. Forces are then computed at the predicted velocities, and the full velocity update corrects using the new forces. This makes the scheme self-consistent for velocity-dependent contact forces while retaining second-order accuracy in position.

---

## What Changed from Upstream UAMMD

### 1. New Integrator: `ND::NewtonEuler` (`src/Integrator/VelocityVerlet.cuh/.cu`)

A new namespace `uammd::ND` (Newtonian Dynamics) is introduced alongside a class hierarchy:

```
uammd::Integrator
  └── uammd::ND::BaseVerletIntegrator   (handles setup, force compute, shear matrix)
        └── uammd::ND::NewtonEuler      (Groot-Warren velocity-Verlet per step)
```

**Integration algorithm per time step** (`NewtonEuler::forwardTime`):

```
Step 1 — Update positions (using forces from previous step):
  r(t+dt)    = r(t) + dt·v(t) + ½·dt²·F(t)/m

Step 2 — Half-step velocity prediction (stored in halfVel / halfAngVel):
  v̂(t+dt/2) = v(t) + ½·dt·F(t)/m
  ω̂(t+dt/2) = ω(t) + ½·dt·τ(t)/I

Step 3 — Force evaluation at new positions and predicted velocities:
  F(t+dt), τ(t+dt) = interactors(r(t+dt), v̂(t+dt/2), ω̂(t+dt/2))

Step 4 — Full velocity update (corrector):
  v(t+dt)  = v̂(t+dt/2) + ½·dt·F(t+dt)/m
  ω(t+dt)  = ω̂(t+dt/2) + ½·dt·τ(t+dt)/I
```

where `I = ½·m·R²` is the moment of inertia of a solid disc (2D) or sphere (3D).

This matches exactly the Groot-Warren scheme: interactors receive the predicted half-step velocity through `ParticleData`, so any velocity-dependent contact force (e.g., viscous normal damping, Coulomb-limited tangential friction) is evaluated at a consistent intermediate state.

**Key parameters** (`ND::Parameters`):

| Parameter | Type | Default | Description |
|---|---|---|---|
| `dt` | `real` | — | Time step size |
| `temperature` | `real` | `0` | Thermal noise amplitude (set >0 for stochastic noise on positions) |
| `viscosity` | `real` | `1` | Background fluid viscosity (sets self-mobility) |
| `hydrodynamicRadius` | `real` | `-1` (use `pd->radius`) | Uniform particle radius for mobility |
| `box` | `Box` | — | Simulation domain with periodic BCs |
| `is2D` | `bool` | `false` | Restrict dynamics to the xy-plane |
| `K` | `vector<real3>` | zero | 3×3 shear rate matrix (encoded as three row vectors) |

### 2. New and Extended Particle Properties (`src/ParticleData/ParticleData.cuh`)

The fork registers the following additional properties via `EXTRA_PARTICLE_PROPERTIES`:

| Property accessor | Type | Purpose |
|---|---|---|
| `getImage` | `real3` | Periodic image counter — tracks how many times each particle has crossed each box face, enabling unwrapped trajectory analysis |
| `getInitCenter` | `real3` | Initial center-of-mass position, useful for computing mean displacement relative to reference state |
| `getHalfVel` | `real3` | Half-step translational velocity `v̂(t+dt/2)` — written by the integrator before force evaluation |
| `getHalfAngVel` | `real4` | Half-step angular velocity `ω̂(t+dt/2)` — written by the integrator before force evaluation |
| `getStressX` | `real3` | Per-particle virial stress contribution, x-row of the stress tensor |
| `getStressY` | `real3` | Per-particle virial stress contribution, y-row of the stress tensor |
| `getStressZ` | `real3` | Per-particle virial stress contribution, z-row of the stress tensor |

The upstream properties `angVel` (real4), `torque` (real4), and `dir` (real4) are also fully wired into the integrator, making rotational dynamics a first-class citizen.

Interactors can read `halfVel` / `halfAngVel` during force evaluation to implement velocity-dependent contact forces that are consistent with the Groot-Warren scheme:

```cpp
// Inside a PairForces transverser — read half-step velocity
auto halfVel = pd->getHalfVel(access::gpu, access::read);
real3 vij = halfVel[i] - halfVel[j];  // relative velocity at t + dt/2
```

The stress arrays are zeroed by the integrator at the start of each step (alongside force and torque) and are available for post-processing or on-the-fly granular temperature / pressure calculations.

---

## Usage

### Building

GrainSim inherits the UAMMD build system. All modules are header-only; compile your simulation with `nvcc`:

```bash
# Using conda (recommended)
conda env create -f environment.yml -n grainsim
conda activate grainsim

# Install headers
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install
```

### Minimal Granular Simulation

```cpp
#include "uammd.cuh"
#include "Integrator/VelocityVerlet.cuh"
#include "Interactor/PairForces.cuh"
// ... your contact potential header

using namespace uammd;

int main() {
    int N = 10000;
    auto pd = make_shared<ParticleData>(N);

    // Set initial positions, masses, radii ...
    {
        auto pos  = pd->getPos(access::cpu, access::write);
        auto mass = pd->getMass(access::cpu, access::write);
        auto rad  = pd->getRadius(access::cpu, access::write);
        // fill arrays ...
    }

    ND::Parameters par;
    par.dt          = 1e-4;
    par.temperature = 0;           // athermal granular flow
    par.viscosity   = 1.0;
    par.box         = Box(make_real3(10, 10, 1));
    par.is2D        = true;

    auto integrator = make_shared<ND::NewtonEuler>(pd, par);

    // Add a contact potential (e.g., Hertz + viscous damping + Coulomb friction)
    // using PairForces — the transverser can read pd->getHalfVel() for
    // velocity-dependent terms consistent with the Groot-Warren scheme.
    auto contact = make_shared<PairForces<YourContactPotential>>(pd, box, potPar);
    integrator->addInteractor(contact);

    // Time loop
    for (int step = 0; step < 1000000; step++) {
        integrator->forwardTime();
        // Optionally read stressX/Y/Z for pressure output
    }
    return 0;
}
```

### Accessing Half-Step Velocities in a Custom Interactor

Any interactor that needs velocity-dependent forces should read `halfVel` / `halfAngVel` rather than `vel` / `angVel`. These are guaranteed to hold `v(t + dt/2)` when the interactor's `sum()` is called:

```cpp
struct ContactTransverser {
    real3 *halfVel;
    real4 *halfAngVel;
    // ...

    ContactTransverser(Box box, shared_ptr<ParticleData> pd) {
        auto hv  = pd->getHalfVel(access::gpu, access::read);
        auto hav = pd->getHalfAngVel(access::gpu, access::read);
        halfVel    = hv.raw();
        halfAngVel = hav.raw();
        // ...
    }
};
```

---

## Physics Background

### Groot-Warren Velocity-Verlet (1997)

The original Groot-Warren paper [1] introduced the modified velocity-Verlet scheme for Dissipative Particle Dynamics (DPD) to handle forces that depend on both positions **and** velocities. The problem is circular: the new velocity is needed to compute the new dissipative force, but the new force is needed to compute the new velocity.

The Groot-Warren fix is a predictor-corrector loop:

1. **Predict** a half-step velocity `v̂ = v + λ·dt·F/m` (with `λ = 0.5` here).
2. **Evaluate** all forces using positions `r(t+dt)` and predicted velocities `v̂`.
3. **Correct** the velocity with the updated force.

This makes the scheme self-consistent for velocity-dependent interactions (viscous drag, lubrication, tangential friction), while maintaining second-order accuracy in positions (same as standard Velocity-Verlet). Stability of dense, contact-rich systems is dramatically improved compared to a naive forward-Euler velocity update.

### Rotational Dynamics

Each particle carries angular velocity `ω` and is subject to torques `τ` arising from tangential contact forces. The integrator applies the same Groot-Warren half-step structure to `ω`, meaning tangential contact forces — which depend on the surface velocity `v_surface = v + ω × R·n̂` — are also evaluated at the predicted half-step angular velocity.

### Per-Particle Stress

The `stressX/Y/Z` arrays accumulate the per-particle virial stress contribution from each contact:

```
σ_αβ^i = ½ Σ_j  r_ij,α · F_ij,β
```

Summing over all particles gives the total virial pressure tensor. The arrays are zeroed at the start of each step so interactors can accumulate contributions additively.

---

## Repository Structure

```
src/
  Integrator/
    VelocityVerlet.cuh   — ND::Parameters, BaseVerletIntegrator, NewtonEuler declaration
    VelocityVerlet.cu    — GPU kernels and NewtonEuler::forwardTime implementation
    ...                  — upstream UAMMD integrators (BD, BDHI, LBM, VerletNVE/NVT, ...)
  Interactor/
    ...                  — upstream UAMMD interactors (PairForces, BondedForces, SPH, ...)
  ParticleData/
    ParticleData.cuh     — extended with image, initCenter, halfVel, halfAngVel, stressX/Y/Z
examples/
  ...                    — upstream UAMMD examples
docs/
  ...                    — upstream UAMMD documentation (readthedocs)
```

---

## Dependencies

Same as upstream UAMMD:

- **CUDA** (required)
- **lapacke** / **cblas** or **MKL** (some modules)
- All other dependencies bundled under `src/third_party/`

Install everything via conda:

```bash
conda env create -f environment.yml -n grainsim
```

---

## References

[1] R. D. Groot and P. B. Warren, "Dissipative particle dynamics: Bridging the gap between atomistic and mesoscopic simulation," *J. Chem. Phys.*, 107, 4423 (1997). https://doi.org/10.1063/1.474784

[2] R. P. Peláez et al., "Universally Adaptable Multiscale Molecular Dynamics (UAMMD)," *Comput. Phys. Commun.*, 306, 109363 (2025). https://doi.org/10.1016/j.cpc.2024.109363
