/**Raul P. Pelaez 2017-2020. Brownian Dynamics integrators

  Solves the following differential equation:
      X[t+dt] = dt(KX[t]+MF[t]) + sqrt(2*Tdt)*dW B
   Being:
     X - Positions
     M - Self Diffusion  coefficient -> 1/(6 \pi*viscosity*radius)
     K - Shear matrix
     dW- Noise vector
     B - sqrt(M)


OPTIONS:

ND::Parameters par;
par.K -> a std:vector<real3> of three elements, encoding a 3x3 shear Matrix.
zero(3,3) by default. par.temperature -> System Temperature par.viscosity ->
System Viscosity par.hydrodynamicRadius -> Particle radius (if all particles
have the same radius). Set this variable if pd->radius has not been set or you
want all particles to have the same diffusive radius and ignore pd->radius.
par.dt -> Time step size.
par.is2D -> Set to true if the system lives in 2D.

USAGE:
Use as any other Integrator.

  auto sys = make_shared<System>();
  auto pd = make_shared<ParticleData>(N, sys);
  ...
//Set initial state
  ...
  auto pg = make_shared<ParticleGroup>(pd, sys, "All");
  ND::Parameters par;
  par.temperature = std::stod(argv[7]); //For example
  par.viscosity = 1.0;
  par.hydrodynamicRadius = 1.0;
  par.dt = std::stod(argv[3]); //For example
*/
#ifndef NEWTONIANDYNAMICSINTEGRATOR_CUH
#define NEWTONIANDYNAMICSINTEGRATOR_CUH
#include "Integrator.cuh"
#include "global/defines.h"
#include "utils/Box.cuh"

namespace uammd {
namespace ND {
struct Parameters {
  // The 3x3 shear matrix is encoded as an array of 3 real3
  std::vector<real3> K = std::vector<real3>(3, real3());
  real temperature = 0;
  real viscosity = 1;
  real hydrodynamicRadius = -1.0;
  real dt = 0;
  Box box;
  bool is2D = false;
};

class BaseVerletIntegrator : public Integrator {
public:
  using Parameters = ND::Parameters;

  BaseVerletIntegrator(shared_ptr<ParticleGroup> pg, Parameters par);

  BaseVerletIntegrator(shared_ptr<ParticleData> pd, Parameters par)
      : BaseVerletIntegrator(std::make_shared<ParticleGroup>(pd, "All"),
                             par) {}

  ~BaseVerletIntegrator();

  virtual void forwardTime() override = 0;

  virtual real sumEnergy() override {
    // Sum 1.5*kT to each particle
    auto energy = pd->getEnergy(access::gpu, access::readwrite);
    auto energy_gr = pg->getPropertyIterator(energy);
    auto energy_per_particle =
        thrust::make_constant_iterator<real>(1.5 * temperature);
    thrust::transform(thrust::cuda::par, energy_gr,
                      energy_gr + pg->getNumberParticles(), energy_per_particle,
                      energy_gr, thrust::plus<real>());
    return 0;
  }

protected:
  real3 Kx, Ky, Kz; // shear matrix
  real selfMobility;
  real hydrodynamicRadius = real(-1.0);
  real temperature = real(0.0);
  real dt;
  bool is2D;
  cudaStream_t st;
  int steps;
  uint seed;
  Box box;

  void updateInteractors();
  void resetForces();
  // void copytoOldForces();
  // void storePastVelocities();
  void computeCurrentForces();
  real *getParticleRadiusIfAvailable();
};

class NewtonEuler : public BaseVerletIntegrator {
public:
  NewtonEuler(shared_ptr<ParticleGroup> pg, Parameters par)
      : BaseVerletIntegrator(pg, par) {
    sys->log<System::MESSAGE>("[ND::NewtonEuler] Initialized");
  }

  NewtonEuler(shared_ptr<ParticleData> pd, Parameters par)
      : NewtonEuler(std::make_shared<ParticleGroup>(pd, "All"), par) {}

  void forwardTime() override;

protected:
  void updatePositions();
  void updateVelocities(bool isHalf=false);
  // void halfUpdateVelocities();
};


} // namespace ND
} // namespace uammd

#include "VelocityVerlet.cu"
#endif
