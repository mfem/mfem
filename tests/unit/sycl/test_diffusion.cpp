// Copyright (c) 2010-2023, Lawrence Livermore National Security, LLC. Produced
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
using namespace mfem;

#include "unit_tests.hpp"

#ifdef MFEM_USE_SYCL

#include <cassert>

#include "general/debug.hpp"

void sycl_diffusion()
{
   // #warning Diffusion kernel version set to 2!
   setenv("VERSION","2",1);
   const int p = GENERATE(1,2);
   const auto mesh_filename =
      GENERATE("../../data/star-q3.mesh", // 2D
               "../../data/fichera-q3.mesh"); // 3D
   dbg("[SYCL] Diffusion p=%d %s",p,mesh_filename);
   Mesh mesh = Mesh::LoadFromFile(mesh_filename);

   H1_FECollection fec(p, mesh.Dimension());
   FiniteElementSpace fes(&mesh, &fec);

   const IntegrationRule &ir =
      IntRules.Get(fes.GetFE(0)->GetGeomType(), 2*p+1);
   const Vector &J = mesh.GetGeometricFactors(ir, GeometricFactors::JACOBIANS)->J;
   const double dotJ = J*J;
   REQUIRE((std::isfinite(dotJ) && !std::isnan(dotJ)));

   ConstantCoefficient one(1.0);
   BilinearForm a_pa(&fes), a_fa(&fes);
   a_pa.AddDomainIntegrator(new DiffusionIntegrator(one));
   a_fa.AddDomainIntegrator(new DiffusionIntegrator(one));
   a_pa.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a_pa.Assemble(), a_fa.Assemble();
   a_fa.Finalize();

   GridFunction x(&fes), y_fa(&fes), y_pa(&fes);
   x.Randomize(1);

   a_pa.Mult(x, y_pa);
   a_fa.Mult(x, y_fa);
   y_pa -= y_fa;
   REQUIRE(y_pa.Norml2() == MFEM_Approx(0.0));
}

#endif // MFEM_USE_SYCL
