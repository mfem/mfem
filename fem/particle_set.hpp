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

#ifndef MFEM_PARTICLE_SET
#define MFEM_PARTICLE_SET


#ifdef MFEM_USE_GSLIB // Must compile with GSLib!

namespace mfem
{


class ParticleSet
{
    public:
        enum class Init
        {
            UNIFORM,
            RANDOM
        };

    protected:
        FindPointsGSLib finder;
        Array<real_t> xs, ys, zs;

    public:
        /// Initialize a ParticleSet on Mesh \p m. Note: the input mesh \p m must have Nodes set.
        explicit ParticleSet(const Mesh &m)
        { finder.Setup(m); };

#ifdef MFEM_USE_MPI
        
        // TODO. Will start with serial. Need to consider when a particle leaves one rank and goes to next
        ParticleSet(const Mesh &m, MPI_Comm comm)
        : finder(comm) { finder.Setup(m); };

#endif // MFEM_USE_MPI

        /** Initialize particles using an initialization method.

            @param[in] init            Initialization method (see ParticleSet::Init for options).
            @param[in] numParticles    Number of particles to add to ParticleSet.*/
        void Initialize(ParticleSet::Init init, int numParticles);

        /// Add an individual particle at specified location \p coords.
        void AddParticle(const Array<real_t> &coords);

        


}

} // namespace mfem


#endif // MFEM_USE_GSLIB
#endif // MFEM_PARTICLE_SET