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
#include "linalg/dtensor.hpp"


using namespace mfem;


TEST_CASE("BandMatrix", "[BandMatrix]")
{
   int size = GENERATE(4,6,8);
   int bandwidth =GENERATE(0,1,2);

   DenseMatrix dm(size,size);
   dm = 0.0;

   for (int i = 0; i < dm.Height(); i++)
   {
      for (int j  = std::max(i - bandwidth,0) ;
               j <= std::min(i + bandwidth, size-1) ; j++)
      {
         dm(i,j) = 10*(i+1) + (j+1);
      }
   }
   BandMatrix bm(dm);

   SECTION("DenseMatrix Constructor")
   {
      DenseMatrix diff(size);
      Add(dm,  bm, -1.0,diff);
      REQUIRE(diff.FNorm() == MFEM_Approx(0.0));
      Add(bm,  dm, -1.0, diff);
      REQUIRE(diff.FNorm() == MFEM_Approx(0.0));
   }

   SECTION("ToDenseMatrix")
   {
      DenseMatrix diff(size);
      DenseMatrix dm2 = bm.ToDenseMatrix();
      Add(bm, dm2, -1.0, diff);
      REQUIRE(diff.FNorm() == MFEM_Approx(0.0));
   }

   SECTION("Mult")
   {
      Vector x(size);
      for (int i = 0; i < dm.Height(); i++)
      {
         x[i] = i+1;
      }

      Vector bd(size), bb(size);
      bm.Mult(x, bb);
      dm.Mult(x, bd);

      Vector diff(size);
      add(bd, -1.0, bb, diff);

      REQUIRE(diff.Norml2() == MFEM_Approx(0.0));
   }

   MatrixInverse *idm = dm.Inverse();
   MatrixInverse *ibm = bm.Inverse();

   SECTION("Det")
   {
      REQUIRE(ibm->Det() == MFEM_Approx(dm.Det(), 1e-6, 1e-6));
      REQUIRE(ibm->Det() == MFEM_Approx(idm->Det(), 1e-6, 1e-6));
   }

   SECTION("Solve")
   {
      Vector x(size);
      for (int i = 0; i < dm.Height(); i++)
      {
         x[i] = i+1;
      }

      Vector bd(size), bb(size);
      ibm->Mult(x, bb);
      idm->Mult(x, bd);

      Vector diff(size);
      add(bd, -1.0, bb, diff);

      REQUIRE(diff.Norml2() == MFEM_Approx(0.0, 1e-6, 1e-6));
   }
   delete ibm;
   delete idm;

   SECTION("Inverse")
   {
      DenseMatrix inv(size);
      bm.Inverse(inv);
      dm.Invert();
      DenseMatrix diff(size);
      Add(dm, inv, -1.0, diff);
      REQUIRE(diff.FNorm() == MFEM_Approx(0.0, 1e-6, 1e-6));
   }
}

TEST_CASE("Approximate inverse of BandMatrix", "[BandMatrix]")
{
   int size = GENERATE(16,32);
   real_t diag = GENERATE(4.0,8.0);
   int bandwidth = 1;
   Vector val(3);
   val[0] = val[2] = 1.0;
   val[1] = diag;

   DenseMatrix dm(size,size);
   for (int i = 0; i < dm.Height(); i++)
   {
      for (int j  = std::max(i - bandwidth,0) ;
               j <= std::min(i + bandwidth, size-1) ; j++)
      {
         dm(i,j) = val[i - j + bandwidth];
      }
   }
   BandMatrix bm(dm);

   SECTION("DenseMatrix Constructor")
   {
      DenseMatrix diff(size);
      Add(dm,  bm, -1.0,diff);
      REQUIRE(diff.FNorm() == MFEM_Approx(0.0));
      Add(bm,  dm, -1.0, diff);
      REQUIRE(diff.FNorm() == MFEM_Approx(0.0));
   }

   SECTION("Inverse")
   {
      DenseMatrix inv(size);
      bm.Inverse(inv);
      dm.Invert();
      DenseMatrix diff(size);
      Add(dm, inv, -1.0, diff);
      REQUIRE(diff.FNorm() == MFEM_Approx(0.0, 1e-6, 1e-6));
   }

   SECTION("Approx Inverse")
   {
      DenseMatrix I = DenseMatrix::Identity(size);
      DenseMatrix ans(size);
      
      real_t tol = 1;
      int steps  = 3;
      real_t fac = 1e2;
      mfem::out<<"Approximate inverse of band matrix"<<std::endl;
      mfem::out<<"req.tol --> bandwidth & achieved tol"<<std::endl;
      for (int i = 0; i < steps; i++, tol/=fac)
      {
          BandMatrix ainv(bm);
          ainv.Invert(tol);
          mfem::Mult(ainv, bm, ans);
          DenseMatrix diff(size);
          Add(ans, I, -1.0, diff);
          mfem::out<<std::setw(8)<<tol
                   <<" --> "<<std::setw(3)<<ainv.GetBandWidth()
                   <<" : "<<diff.FNorm()<<std::endl;
          REQUIRE(diff.FNorm() <tol);
      }
   }
}

