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

#include "mfem.hpp"
#include "unit_tests.hpp"

using namespace mfem;
using namespace std;

static constexpr int VDIM = 7;
static_assert(VDIM > 1);
static constexpr int NV_rm = 12;
static constexpr int NV = 27;
static_assert(NV_rm < NV);
static constexpr int VDIM_INC = 3;

void TestSetGetValues(Ordering::Type ordering)
{
   ParticleVector mv(VDIM, ordering, NV);
   Vector vecs[NV];
   for (int i = 0; i < NV; i++)
   {
      vecs[i].SetSize(VDIM);
      vecs[i].Randomize(i);
      mv.SetValues(i, vecs[i]);
   }

   int wrong_vec_count = 0;
   Vector aux(VDIM);
   for (int i = 0; i < NV; i++)
   {
      mv.GetValues(i, aux);
      if (!(aux.DistanceTo(vecs[i]) == MFEM_Approx(0,0)))
      {
         wrong_vec_count++;
      }
   }
   REQUIRE(wrong_vec_count == 0);

}

void TestSetGetComponents(Ordering::Type ordering)
{
   ParticleVector mv(VDIM, ordering, NV);
   Vector vecs[NV];
   for (int i = 0; i < NV; i++)
   {
      vecs[i].SetSize(VDIM);
      vecs[i].Randomize(i);
   }

   // Set components individually
   Vector comps[VDIM];
   for (int vd = 0; vd < VDIM; vd++)
   {
      comps[vd].SetSize(NV);
      for (int i = 0; i < NV; i++)
      {
         comps[vd][i] = vecs[i][vd];
      }
      mv.SetComponents(vd, comps[vd]);
   }

   // Verify get component
   int wrong_comp_count = 0;
   Vector aux_comp(NV);
   for (int vd = 0; vd < VDIM; vd++)
   {
      mv.GetComponents(vd, aux_comp);
      if (!(aux_comp.DistanceTo(comps[vd]) == MFEM_Approx(0,0)))
      {
         wrong_comp_count++;
      }
   }
   REQUIRE(wrong_comp_count == 0);

   // Verify all vectors correct
   int wrong_vec_count = 0;
   Vector aux(VDIM);
   for (int i = 0; i < NV; i++)
   {
      mv.GetValues(i, aux);
      if (!(aux.DistanceTo(vecs[i]) == MFEM_Approx(0,0)))
      {
         wrong_vec_count++;
      }
   }
   REQUIRE(wrong_vec_count == 0);

}

void TestResize(Ordering::Type ordering)
{
   SECTION((ordering == Ordering::byNODES ? "byNODES" : "byVDIM"))
   {
      Vector all_data(NV*VDIM);
      all_data.Randomize(1234);
      const ParticleVector mv_all(VDIM, ordering, all_data);

      // Start with mv_test = mv_all
      ParticleVector mv_test(VDIM, ordering, NV);
      mv_test = mv_all;
      REQUIRE(mv_test.DistanceTo(mv_all) == MFEM_Approx(0.0));

      // Remove N_rm vectors from mv_test + save them into vecs_diff
      std::vector<Vector> vecs_diff(NV_rm);
      for (int i = 0; i < NV_rm; i++)
      {
         mv_all.GetValues(NV - NV_rm + i, vecs_diff[i]);
      }
      mv_test.SetNumParticles(NV-NV_rm);
      REQUIRE(mv_test.GetNumParticles() == NV - NV_rm);

      // Resize mv_test back
      mv_test.SetNumParticles(NV);
      REQUIRE(mv_test.GetNumParticles() == NV);

      // Ensure that vectors post-shrink match those in mv_all
      int wrong_shrink_vec_count = 0;
      Vector v1, v2;
      for (int i = 0; i < NV-NV_rm; i++)
      {
         mv_all.GetValues(i, v1);
         mv_test.GetValues(i, v2);
         if (!(v1.DistanceTo(v2) == MFEM_Approx(0,0)))
         {
            wrong_shrink_vec_count++;
         }
      }
      REQUIRE(wrong_shrink_vec_count == 0);

      // Set vectors back to mv_test, and then check equality
      mv_test.SetNumParticles(NV);
      for (int i = 0; i < NV_rm; i++)
      {
         mv_test.SetValues(i+(NV-NV_rm), vecs_diff[i]);
      }
      REQUIRE(mv_test.DistanceTo(mv_all) == MFEM_Approx(0.0));
   }
}

void TestSetVDim(Ordering::Type ordering)
{
   Vector data_vecs[NV];
   Vector data_comps[VDIM_INC]; // last VDIM_INC comps
   Vector data_vecs_red[NV]; // Vectors of reduced vdim
   ParticleVector mv(VDIM+VDIM_INC, ordering, NV);
   for (int i = 0; i < NV; i++)
   {
      data_vecs[i].SetSize(VDIM+VDIM_INC);
      data_vecs[i].Randomize(i);
      data_vecs_red[i].NewDataAndSize(data_vecs[i].GetData(), VDIM);

      mv.SetValues(i, data_vecs[i]);
   }
   for (int vd = 0; vd < VDIM_INC; vd++)
   {
      data_comps[vd].SetSize(NV);
      mv.GetComponents(vd+VDIM, data_comps[vd]);
   }

   // Reduce vdim + compare against data_vecs_red
   mv.SetVDim(VDIM);
   Vector aux(VDIM);
   int wrong_vec_red_count = 0;
   for (int i = 0; i < NV; i++)
   {
      mv.GetValues(i, aux);
      if (aux.DistanceTo(data_vecs_red[i]) != MFEM_Approx(0.0))
      {
         wrong_vec_red_count++;
      }
   }
   REQUIRE(wrong_vec_red_count == 0);

   // Increase vdim, update components, + compare against data_vecs
   mv.SetVDim(VDIM+VDIM_INC);
   for (int vd = 0; vd < VDIM_INC; vd++)
   {
      mv.SetComponents(vd+VDIM, data_comps[vd]);
   }
   int wrong_vec_count = 0;
   aux.SetSize(VDIM+VDIM_INC);
   for (int i = 0; i < NV; i++)
   {
      mv.GetValues(i, aux);
      if (aux.DistanceTo(data_vecs[i]) != MFEM_Approx(0.0))
      {
         wrong_vec_count++;
      }
   }
   REQUIRE(wrong_vec_count == 0);

}

TEST_CASE("ParticleVector set/get values", "[ParticleVector]")
{
   TestSetGetValues(Ordering::byNODES);
   TestSetGetValues(Ordering::byVDIM);
}

TEST_CASE("ParticleVector set/get components", "[ParticleVector]")
{
   TestSetGetComponents(Ordering::byNODES);
   TestSetGetComponents(Ordering::byVDIM);
}

TEST_CASE("ParticleVector resize", "[ParticleVector]")
{
   TestResize(Ordering::byNODES);
   TestResize(Ordering::byVDIM);
}

TEST_CASE("ParticleVector set vdim","[ParticleVector]")
{
   TestSetVDim(Ordering::byNODES);
   TestSetVDim(Ordering::byVDIM);
}
