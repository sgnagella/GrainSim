#include "VelocityVerlet.cuh"
#include "third_party/saruprng.cuh"
#include "utils/debugTools.h"
namespace uammd {
namespace ND {

BaseVerletIntegrator::BaseVerletIntegrator(shared_ptr<ParticleGroup> pg,
                                               Parameters par)
    : Integrator(pg, "ND::BaseVerletIntegrator"), Kx(make_real3(0)),
      Ky(make_real3(0)), Kz(make_real3(0)), temperature(par.temperature),
      dt(par.dt), is2D(par.is2D), steps(0), box(par.box) {
  sys->rng().next32();
  sys->rng().next32();
  seed = sys->rng().next32();
  sys->log<System::MESSAGE>("[ND::BaseVerletIntegrator] Initialized");
  int numberParticles = pg->getNumberParticles();
  this->selfMobility = 1.0 / (6.0 * M_PI * par.viscosity);
  if (par.hydrodynamicRadius != real(-1.0)) {
    this->selfMobility /= par.hydrodynamicRadius;
    this->hydrodynamicRadius = par.hydrodynamicRadius;
    if (pd->isRadiusAllocated()) {
      sys->log<System::WARNING>("[ND::BaseVerletIntegrator] Assuming all "
                                "particles have hydrodynamic radius %g",
                                par.hydrodynamicRadius);
    } else {
      sys->log<System::MESSAGE>(
          "[ND::BaseVerletIntegrator] Hydrodynamic radius: %g",
          par.hydrodynamicRadius);
    }
    sys->log<System::MESSAGE>("[ND::BaseVerletIntegrator] Self Mobility: %g",
                              selfMobility);
  } else if (pd->isRadiusAllocated()) {
    sys->log<System::MESSAGE>(
        "[ND::BaseVerletIntegrator] Hydrodynamic radius: particleRadius");
    sys->log<System::MESSAGE>(
        "[ND::BaseVerletIntegrator] Self Mobility: %g/particleRadius",
        selfMobility);
  } else {
    this->hydrodynamicRadius = real(1.0);
    sys->log<System::MESSAGE>(
        "[ND::BaseVerletIntegrator] Hydrodynamic radius: %g",
        hydrodynamicRadius);
    sys->log<System::MESSAGE>("[ND::BaseVerletIntegrator] Self Mobility: %g",
                              selfMobility);
  }
  sys->log<System::MESSAGE>("[ND::BaseVerletIntegrator] Temperature: %g",
                            temperature);
  sys->log<System::MESSAGE>("[ND::BaseVerletIntegrator] dt: %g", dt);
  sys->log<System::MESSAGE>("[ND::BaseVerletIntegrator] Box dimensions: %g %g %g",
                            box.boxSize.x, box.boxSize.y, box.boxSize.z);
  if (par.K.size() == 3) {
    int numberNonZero = std::count_if(par.K.begin(), par.K.end(), [](real3 k) {
      return k.x != 0 or k.y != 0 or k.z != 0;
    });
    if (numberNonZero > 0) {
      Kx = par.K[0];
      Ky = par.K[1];
      Kz = par.K[2];
      sys->log<System::MESSAGE>("[ND::BaseVerletIntegrator] Shear Matrix: [ "
                                "%g %g %g; %g %g %g; %g %g %g ]",
                                Kx.x, Kx.y, Kx.z, Ky.x, Ky.y, Ky.z, Kz.x, Kz.y,
                                Kz.z);
    }
  }
  if (is2D) {
    sys->log<System::MESSAGE>(
        "[ND::BaseVerletIntegrator] Starting in 2D mode");
  }
  CudaSafeCall(cudaStreamCreate(&st));
}

BaseVerletIntegrator::~BaseVerletIntegrator() {
  sys->log<System::MESSAGE>("[ND::BaseVerletIntegrator] Destroyed");
  cudaStreamDestroy(st);
}

void BaseVerletIntegrator::updateInteractors() {
  for (auto updatable : updatables) {
    updatable->updateSimulationTime(steps * dt);
  }
  if (steps == 1) {
    for (auto updatable : updatables) {
      updatable->updateTemperature(temperature);
      updatable->updateTimeStep(dt);
    }
  }
  CudaCheckError();
}

// void BaseVerletIntegrator::storePastVelocities() {
//   int numberParticles = pg->getNumberParticles();
//   auto force = pd->getForce(access::location::gpu, access::mode::read);
//   auto vel = pd->getOldForce(access::location::gpu, access::mode::readwrite);
//   auto force_gr = pg->getPropertyIterator(force);
//   auto vel_gr = pg->getPropertyIterator(vel);
//   thrust::copy(thrust::cuda::par.on(st), force_gr, force_gr + numberParticles,
//                vel_gr);
//   CudaCheckError();
// }

void BaseVerletIntegrator::resetForces() {
  int numberParticles = pg->getNumberParticles();
  auto force = pd->getForce(access::location::gpu, access::mode::write);
  auto torque = pd->getTorque(access::location::gpu, access::mode::write);
  auto forceGroup = pg->getPropertyIterator(force);
  auto torqueGroup = pg->getPropertyIterator(torque);
  auto stressX = pd->getStressX(access::location::gpu, access::mode::write);
  auto stressXGroup = pg->getPropertyIterator(stressX);
  auto stressY = pd->getStressY(access::location::gpu, access::mode::write);
  auto stressYGroup = pg->getPropertyIterator(stressY);
  auto stressZ = pd->getStressZ(access::location::gpu, access::mode::write);
  auto stressZGroup = pg->getPropertyIterator(stressZ);
  thrust::fill(thrust::cuda::par.on(st), forceGroup,
               forceGroup + numberParticles, real4());
  thrust::fill(thrust::cuda::par.on(st), torqueGroup,
               torqueGroup + numberParticles, real4());
  thrust::fill(thrust::cuda::par.on(st), stressXGroup,
               stressXGroup + numberParticles, real3());
  thrust::fill(thrust::cuda::par.on(st), stressYGroup,
               stressYGroup + numberParticles, real3());
  thrust::fill(thrust::cuda::par.on(st), stressZGroup,
               stressZGroup + numberParticles, real3());
  CudaCheckError();
}

void BaseVerletIntegrator::computeCurrentForces() {
  resetForces();
  for (auto forceComp : interactors)
    forceComp->sum({.force = true, .energy = false, .virial = false}, st);
  CudaCheckError();
}

// void BaseVerletIntegrator::copytoOldForces(){
//   int numberParticles = pg->getNumberParticles();
//   auto force = pd->getForce(access::location::gpu, access::mode::read);
//   auto oldforce = pd->getOldForce(access::location::gpu, access::mode::readwrite);
//   auto force_gr = pg->getPropertyIterator(force);
//   auto oldforce_gr = pg->getPropertyIterator(oldforce);
//   thrust::copy(thrust::cuda::par.on(st), force_gr, force_gr + numberParticles,
//                oldforce_gr);
//   CudaCheckError();
// }

real *BaseVerletIntegrator::getParticleRadiusIfAvailable() {
  real *d_radius = nullptr;
  if (hydrodynamicRadius == real(-1.0) && pd->isRadiusAllocated()) {
    auto radius = pd->getRadius(access::location::gpu, access::mode::read);
    d_radius = radius.raw();
    sys->log<System::DEBUG3>(
        "[ND::BaseVerletIntegrator] Using particle radius.");
  }
  return d_radius;
}

namespace NewtonEuler_ns {

// Implements the modified Velcity-Verlet algorithm for Newtonian dynamics 
// ref: https://arxiv.org/pdf/2005.12755
__global__ void integrateGPU(real4 *pos,
                             ParticleGroup::IndexIterator indexIterator,
                             const real4 *force, real3 Kx, real3 Ky, real3 Kz,
                             real selfMobility, real *radius, real dt,
                             bool is2D, real temperature, int N, uint stepNum,
                             uint seed, Box box, real3 *image, real3 *velocity, real *mass) {
  uint id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N)
    return;
  
  // int time = n * dt;                               
  int i = indexIterator[id];
  real3 V = make_real3(velocity[i]);
  real Mass = mass[i];
  real3 R = make_real3(pos[i]);
  real3 I = make_real3(image[i]);
  real3 F = make_real3(force[i]) / Mass;
  real M = selfMobility * (radius ? (real(1.0) / radius[i]) : real(1.0));

  // Step the positions
  // R(t + dt) = R(t) + dt * KR + dt * V(t) + 0.5 * dt**2 * force(t)/m
  // V(t + dt/2) = V(t) + 0.5 * dt * force(t)/m
  // force(t + dt)  = F(R(t), V(t + dt/2)) computed outside of this kernel
  // V(t + dt) = V(t) + 0.5 * dt * (force(t) + force(t + dt))/m

  R += dt * V + real(0.5) * dt * dt * F;
  if (temperature > 0) {
    Saru rng(i, stepNum, seed);
    real B = sqrt(real(2.0) * temperature * M * dt);
    real3 dW = make_real3(rng.gf(0, B), rng.gf(0, B).x);
    R += dW;
  }

  // Periodic BCs
  // if( Kx.y != real(0.0) ){
  //   R = box.apply_pbc_lees_edwards(R, Kx.y, time, &I);
  // }
  // else{
  //   R = box.apply_pbc(R, &I);
  // }

  R = box.apply_pbc(R, &I);
  image[i] = make_real3(I);

  pos[i].x = R.x;
  pos[i].y = R.y;
  if (!is2D)
    pos[i].z = R.z;

}

__global__ void integrateVelocitiesGPU(real4 *pos, 
                                ParticleGroup::IndexIterator indexIterator,
                                const real4 *force, 
                                const real4 *torque, 
                                real3 *newVelocity, 
                                real3 *oldVelocity, 
                                real4 *newAngVel, 
                                real4 *oldAngVel,
                                real *radius, 
                                real dt, 
                                bool is2D, 
                                real temperature, 
                                int N, uint stepNum, uint seed, Box box, real *mass) {
  uint id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= N)
    return;

  int i = indexIterator[id];
  real3 V = make_real3(oldVelocity[i]);
  real3 W = make_real3(oldAngVel[i]);
  real Mass = mass[i];
  real Radius = radius ? radius[i] : real(1.0);
  real3 R = make_real3(pos[i]);
  real3 F = make_real3(force[i]) / Mass;
  real3 T = make_real3(torque[i]) / (Mass * Radius * Radius);

  real3 newVel = V + 0.5 * dt * F;
  newVelocity[i].x = newVel.x;
  newVelocity[i].y = newVel.y;
  if(is2D){
    newVelocity[i].z = newVel.z;
  }

  real3 newW = W + dt * T; // Multiplied by 2 because we multiply by inverse of 
                           // moment of inertia, which is 0.5 * m * R^2 for solid circle

  newAngVel[i] = make_real4(newW.x, newW.y, newW.z, 0.0);
} // namespace NewtonEuler_ns

// __global__ void integrateHalfVelocitiesGPU(real4 *pos,
//                              ParticleGroup::IndexIterator indexIterator,
//                              const real4 *force, 
//                              const real4 *torque, 
//                              real3 *velocity, real3 *old_velocity, 
//                              real4 *ang_vel, real4 *old_ang_vel,
//                              real *radius, real dt, bool is2D, real temperature,
//                              int N, uint stepNum, uint seed, Box box, real *mass) {
//   uint id = blockIdx.x * blockDim.x + threadIdx.x;
//   if (id >= N)
//     return;

//   int i = indexIterator[id];
//   real3 V = make_real3(velocity[i]);
//   real3 old_V = real3();
//   real3 W = make_real3(ang_vel[i]);
//   real Mass = mass[i];
//   real Radius = radius ? radius[i] : real(1.0);
//   real invR = real(1.0) / Radius;
//   real3 R = make_real3(pos[i]);
//   real3 F = make_real3(force[i]) / Mass;
//   real3 T = make_real3(torque[i]) / (Mass * Radius * Radius);

//   // real Z = -real(0.5) * Radius * dt / Mass;
//   // real Z = Radius / Mass;
//   real Z = real(0.0);
//   // Step the velocities
//   // V(t + dt/2) = V(t) + 0.5 * dt * force(t)/m

//   // V += real(0.5) * dt * F; 
//   // V += real(0.5) * (dt * F + invR * Mass * F);
//   // V += Z * V; // adding drag contribution
//   // V += real(0.5) * dt * F;

//   W += dt * T; // Multiplied by 2 because we multiply by inverse of moment of inertia, which is 0.5 * m * R^2 for solid circle

//   old_V = V + 0.5 * dt * (F - Z * V);
//   old_velocity[i].x = old_V.x;
//   old_velocity[i].y = old_V.y;
//   if (!is2D)
//     old_velocity[i].z = old_V.z;
//   // old_velocity[i] = V + 0.5 * dt * (F - Z * V);
//   // printf("In kernel IntegrateHalfVelocities Particle %d: V = (%g, %g, %g)\n", i, old_velocity[i].x, old_velocity[i].y, old_velocity[i].z);

//   // velocity[i] = make_real3(V);
//   // old_ang_vel[i] = ang_vel[i];
//   // ang_vel[i] = make_real4(W);
  
//   // Store the current forces for the next step
//   // oldforce[i] = force[i];
//   // oldtorque[i] = torque[i];

// } // namespace NewtonEuler_ns

// __global__ void integrateVelocitiesGPU(real4 *pos,
//                              ParticleGroup::IndexIterator indexIterator,
//                              const real4 *oldforce, const real4 *force, 
//                              const real4 *oldtorque, const real4 *torque, 
//                              real3 *velocity, real3 *old_vel,
//                              real4 *ang_vel, real4 *old_ang_vel,
//                              real *mass, real *radius, real dt, bool is2D, real temperature,
//                              int N, uint stepNum, uint seed, Box box) {
//   uint id = blockIdx.x * blockDim.x + threadIdx.x;
//   if (id >= N)
//     return;

//   int i = indexIterator[id];
//   real3 V = make_real3(old_vel[i]); // v^{n+1/2} 
//   real3 old_V = real3(); 
//   real3 W = make_real3(ang_vel[i]);
//   // real3 old_W = make_real3(old_ang_vel[i]);

//   // printf("Particle %d: V before update = (%g, %g, %g), W before update = (%g, %g, %g)\n", i, V.x, V.y, V.z, W.x, W.y, W.z);
//   real Mass = mass[i];
//   real Radius = radius ? radius[i] : real(1.0);
//   real invR = real(1.0) / Radius;
//   real3 F = make_real3(force[i]); 
//   real3 T = make_real3(torque[i]);

//   // real Z = Radius / Mass;
//   real Z = real(0.0);
//   // real Z = -real(0.5) * Radius * dt / Mass;
//   // real Zp = real(1.0) / (real(1.0) - Z);

//   // printf("Particle %d: Force = (%g, %g, %g), Torque = (%g, %g, %g)\n", i, F.x, F.y, F.z, T.x, T.y, T.z);
//   // F += make_real3(oldforce[i]);
//   // T += make_real3(oldtorque[i]);
//   F = F / Mass;
//   T = T / (Mass * Radius * Radius);
//   // real3 F_old = make_real3(oldforce[i]);
//   // real3 F = make_real3(force[i]) / Mass;
//   // Step the velocities
//   // V(t + dt) = V(t) + 0.5 * dt * (force(t) + force(t + dt))/m
//   // V += real(0.5) * dt * (F + F_old);

//   // V *= Z;
//   // V += real(0.5) * dt * F + old_V * (real(1.0) + Z);
//   // old_V += Z * old_V; 
//   // old_V += real(0.5) * dt * F;
//   // V = Zp * old_V; 

//   // V += real(0.5) * dt * F;
//   // V += real(0.5) * (dt * F + invR * F * Mass);
//   W += dt * T; // Multiplied by 2 because we multiply by inverse of moment of inertia, which is 0.5 * m * R^2 for solid circle

//   // printf("Particle %d: V after update = (%g, %g, %g), W after update = (%g, %g, %g)\n", i, V.x, V.y, V.z, W.x, W.y, W.z);
//   old_V = V + 0.5 * dt * (F - Z * V);
//   velocity[i].x = old_V.x;
//   velocity[i].y = old_V.y;
//   if (!is2D)
//     velocity[i].z = old_V.z;

//   // printf("In kernel IntegrateVelocities Particle %d: V = (%g, %g, %g)\n", i, velocity[i].x, velocity[i].y, velocity[i].z);
//   // velocity[i] = make_real3(V);
//   // ang_vel[i] = make_real4(W);
// }

} // namespace NewtonEuler_ns

void NewtonEuler::forwardTime() {
  if( steps == 0 ){
    computeCurrentForces();

    // Copy to old forces for the first step
    // copytoOldForces();
  }
  steps++;
  sys->log<System::DEBUG1>("[ND::NewtonEuler] Performing integration step %d",
                           steps);
  updatePositions();
  // halfUpdateVelocities();
  // storeOldForces();
  updateVelocities(true);
  updateInteractors();
  computeCurrentForces();
  updateVelocities(false);
}

void NewtonEuler::updatePositions() {
  int numberParticles = pg->getNumberParticles();
  int BLOCKSIZE = 128;
  uint Nthreads = BLOCKSIZE < numberParticles ? BLOCKSIZE : numberParticles;
  uint Nblocks =
      numberParticles / Nthreads + ((numberParticles % Nthreads != 0) ? 1 : 0);
  real *d_radius = getParticleRadiusIfAvailable();
  auto groupIterator = pg->getIndexIterator(access::location::gpu);
  auto pos = pd->getPos(access::location::gpu, access::mode::readwrite);
  auto force = pd->getForce(access::location::gpu, access::mode::read);
  auto image = pd->getImage(access::location::gpu, access::mode::readwrite);
  auto velocity = pd->getVel(access::location::gpu, access::mode::readwrite);
  auto mass = pd->getMass(access::location::gpu, access::mode::read);
  NewtonEuler_ns::integrateGPU<<<Nblocks, Nthreads, 0, st>>>(
      pos.raw(), groupIterator, force.raw(), Kx, Ky, Kz, selfMobility, d_radius,
      dt, is2D, temperature, numberParticles, steps, seed, box, image.raw(), velocity.raw(), mass.raw());
}

void NewtonEuler::updateVelocities(bool isHalf){
  int numberParticles = pg->getNumberParticles(); 
    int BLOCKSIZE = 128; 
    uint Nthreads = BLOCKSIZE < numberParticles ? BLOCKSIZE : numberParticles; 
    uint Nblocks = 
        numberParticles / Nthreads + ((numberParticles % Nthreads != 0) ? 1 : 0); 
    real *d_radius = getParticleRadiusIfAvailable();
    auto groupIterator = pg->getIndexIterator(access::location::gpu); 
    auto pos = pd->getPos(access::location::gpu, access::mode::read);
    auto force = pd->getForce(access::location::gpu, access::mode::read);
    auto torque = pd->getTorque(access::location::gpu, access::mode::read);
    auto velocity = pd->getVel(access::location::gpu, access::mode::readwrite);
    auto ang_vel = pd->getAngVel(access::location::gpu, access::mode::readwrite);
    auto half_vel = pd->getHalfVel(access::location::gpu, access::mode::readwrite);
    auto half_ang_vel = pd->getHalfAngVel(access::location::gpu, access::mode::readwrite);
    auto mass = pd->getMass(access::location::gpu, access::mode::read);
    if(isHalf){
      NewtonEuler_ns::integrateVelocitiesGPU<<<Nblocks, Nthreads, 0, st>>>(
        pos.raw(), groupIterator, 
        force.raw(), torque.raw(), 
        half_vel.raw(), velocity.raw(),
        half_ang_vel.raw(), ang_vel.raw(), 
        d_radius, dt, is2D, temperature, numberParticles, steps, seed, box, mass.raw());
      return;
    }
    NewtonEuler_ns::integrateVelocitiesGPU<<<Nblocks, Nthreads, 0, st>>>(
        pos.raw(), groupIterator, 
        force.raw(), torque.raw(), 
        velocity.raw(), half_vel.raw(),
        ang_vel.raw(), half_ang_vel.raw(), 
        d_radius, dt, is2D, temperature, numberParticles, steps, seed, box, mass.raw());
}

// void NewtonEuler::halfUpdateVelocities(){
//     int numberParticles = pg->getNumberParticles(); 
//     int BLOCKSIZE = 128; 
//     uint Nthreads = BLOCKSIZE < numberParticles ? BLOCKSIZE : numberParticles; 
//     uint Nblocks = 
//         numberParticles / Nthreads + ((numberParticles % Nthreads != 0) ? 1 : 0); 
//     real *d_radius = getParticleRadiusIfAvailable();
//     auto groupIterator = pg->getIndexIterator(access::location::gpu); 
//     auto pos = pd->getPos(access::location::gpu, access::mode::read);
//     auto force = pd->getForce(access::location::gpu, access::mode::read);
//     auto oldforce = pd->getOldForce(access::location::gpu, access::mode::read);
//     auto torque = pd->getTorque(access::location::gpu, access::mode::read);
//     auto oldtorque = pd->getOldTorque(access::location::gpu, access::mode::read);
//     auto ang_vel = pd->getAngVel(access::location::gpu, access::mode::readwrite);
//     auto velocity = pd->getVel(access::location::gpu, access::mode::readwrite);
//     auto old_vel = pd->getOldVel(access::location::gpu, access::mode::readwrite);
//     auto old_ang_vel = pd->getOldAngVel(access::location::gpu, access::mode::readwrite);
//     auto mass = pd->getMass(access::location::gpu, access::mode::read);
//     NewtonEuler_ns::integrateHalfVelocitiesGPU<<<Nblocks, Nthreads, 0, st>>>(
//         pos.raw(), groupIterator, force.raw(), oldforce.raw(), 
//         torque.raw(), oldtorque.raw(), 
//         velocity.raw(), old_vel.raw(),
//         ang_vel.raw(), old_ang_vel.raw(), 
//         d_radius, dt, is2D, temperature, numberParticles, steps, seed, box, mass.raw());
// }

// void NewtonEuler::updateVelocities(){
//     int numberParticles = pg->getNumberParticles(); 
//     int BLOCKSIZE = 128; 
//     uint Nthreads = BLOCKSIZE < numberParticles ? BLOCKSIZE : numberParticles; 
//     uint Nblocks = 
//         numberParticles / Nthreads + ((numberParticles % Nthreads != 0) ? 1 : 0); 
//     real *d_radius = getParticleRadiusIfAvailable();
//     auto groupIterator = pg->getIndexIterator(access::location::gpu); 
//     auto pos = pd->getPos(access::location::gpu, access::mode::read);
//     auto force = pd->getForce(access::location::gpu, access::mode::read);
//     auto oldforce = pd->getOldForce(access::location::gpu, access::mode::read);
//     auto torque = pd->getTorque(access::location::gpu, access::mode::read);
//     auto oldtorque = pd->getOldTorque(access::location::gpu, access::mode::read);
//     auto ang_vel = pd->getAngVel(access::location::gpu, access::mode::readwrite);
//     auto velocity = pd->getVel(access::location::gpu, access::mode::readwrite);
//     auto old_vel = pd->getOldVel(access::location::gpu, access::mode::readwrite);
//     auto old_ang_vel = pd->getOldAngVel(access::location::gpu, access::mode::readwrite);
//     auto mass = pd->getMass(access::location::gpu, access::mode::read); 
//     NewtonEuler_ns::integrateVelocitiesGPU<<<Nblocks, Nthreads, 0, st>>>(
//         pos.raw(), groupIterator, 
//         oldforce.raw(), force.raw(), 
//         oldtorque.raw(), torque.raw(),
//         velocity.raw(), old_vel.raw(),
//         ang_vel.raw(), old_ang_vel.raw(), 
//         mass.raw(), d_radius, dt, is2D, temperature, numberParticles, steps, seed, box);
// }

} // namespace ND
} // namespace uammd