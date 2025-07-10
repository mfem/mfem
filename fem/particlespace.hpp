// Copyright (c) 2010-2025, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_PARTICLESPACE
#define MFEM_PARTICLESPACE

#include "../config/config.hpp"
#include "../linalg/linalg.hpp"
#include "fespace.hpp"
#include "gslib.hpp"

namespace mfem
{
class ParticleSpace
{
private:
   const Ordering::Type;
   FindPointsGSLIB finder;
   

public:
   ParticleSpace(Mesh &m, int num_initial_particles, Ordering::Type ordering=Ordering::byVDIM);










// -------------------------------------------------------------------------------------------

// temporary helper variables:
mutable Vector coords_p, mom_p, efield_p, bfield_p;
mutable real_t q, m;

// Algorithm step:
void BorisAlgorithm::Step(const ParticleFunction &mass, 
                           const ParticleFunction &charge,
                           ParticleFunction &coords,
                           ParticleFunction &momentum,
                           ParticleFunction &efield,
                           ParticleFunction &bfield)
{
   // Assume E_gf + B_gf are member variables
   // Interpolate electric + magnetic field onto particles
   if (E_gf)
      efield.Interpolate(*E_gf); // Precondition is that E_gf mesh matches ParticleSpace mesh....
   
   if (B_gf)
      bfield.Interpolate(*B_gf); // Precondition is that E_gf mesh matches ParticleSpace mesh....

   for (int p = 0; p < pspace.GetNP(); p++)
   {
      // double ParticleFunctionGetParticleData(int particle_index, int comp=0)
      m = mass.GetParticleData(p);
      q = charge.GetParticleData(p);

      // void ParticleFunction::GetParticleData(int particle_index, Vector &data)
      coords.GetParticleData(p, coords_p);
      momentum.GetParticleData(p, mom_p);
      efield.GetParticleData(p, efield_p);
      bfield.GetParticleData(p, bfield_p);

      // Only coords_P and mom_p are updated:
      ParticleStep(m, q, efield_p, bfield_p, coords_p, mom_p);

      // Set updated data:
      coords.SetParticleData(p, coords_p);
      momentum.SetParticleData(p, mom_p);
   }
}

// -------------------------------------------------------------------------------------------
// int main:
enum PData
{
   MASS,
   CHARGE,
   MOMENTUM,
   EFIELD,
   BFIELD,
   SIZE
}
ParticleSpace pspace(mesh, 1000, Ordering::byVDIM); // Define a ParticleSpace w/ 1000 particles randomly distributed in mesh bounding box

ParticleFunction mass(pspace); mass = 1.0; 
ParticleFunction charge(pspace); charge = 1.0;

ParticleFunction momentum(pspace);
momentum.ProjectCoefficient(inputMomentumCoeff); // Use a coefficient to set particle momentums based on location

ParticleFunction efield(pspace) efield = 0.0;
ParticleFunction bfield(pspace) bfield = 0.0;

BorisAlgorithm boris(pspace, E_gf, B_gf);

for (int n = 0; n < nsteps; n++)
{
   boris.Step(mass, charge, pspace.Coords(), momentum, efield, bfield);
   // Coordinate change doesn't impact anything!

   if (i % rm_lost_particles_freq == 0)
   {
      pspace.RemoveLostParticles(______);
      // Either pass all ParticleFunctions as an argument, or can get an update operator, or just:
      mass.Update();
   }

   if (i % add_particles_freq == 0)
   {
      Vector new_coords; 
      Vector new_mom;
      ...
      pspace.AddParticles(new_coords);

      // Need to update ALL ParticleFunctions w/ new data
      mass.Update(1.0); // Fill all new particles to have mass of 1
      charge.Update(1.0);

      momentum.Update(new_mom); // Checks ensure that new_mom size is correct
   }

   if (i % redist_freq == 0)
   {
      pspace.Redistribute( _____ );
      // Must provide all Particle Functions to also be redistributed!
      // Allows flexibility to write code to redistribute efield, bfield OR not.
   }

}


}



} // namespace mfem


#endif // MFEM_PARTICLESPACE
