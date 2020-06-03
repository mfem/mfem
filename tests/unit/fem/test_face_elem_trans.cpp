// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
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
#include "catch.hpp"

using namespace mfem;

namespace face_elem_trans
{

TEST_CASE("3D FaceElementTransformations",
          "[FaceElementTransformations]")
{
   int log = 0;
   int n = 1;
   int dim = 3;
   int order = 1;

   Mesh mesh(n, n, n, Element::TETRAHEDRON, 1, 2.0, 3.0, 5.0);

   SECTION("Transform")
   {
      int npts = 0;
      for (int f=0; f<mesh.GetNFaces(); f++)
      {
         if (log > 0)
         {
            std::cout << "Getting trans for face " << f << std::endl;
         }
         FaceElementTransformations *T =
            mesh.GetInteriorFaceTransformations(f);

         if (T != NULL)
         {
            const IntegrationRule &ir = IntRules.Get(T->GetGeometryType(),
                                                     2*order + 2);
            if (log > 0)
            {
               std::cout << f << " " << T->Elem1No
                         << " " << T->Elem2No << std::endl;
            }

            double tip_data[3];
            double tip1_data[3];
            double tip2_data[3];
            Vector tip(tip_data, 3);
            Vector tip1(tip1_data, 3);
            Vector tip2(tip2_data, 3);

            for (int j=0; j<ir.GetNPoints(); j++)
            {
               npts++;
               const IntegrationPoint &ip = ir.IntPoint(j);
               IntegrationPoint eip1, eip2;

               T->SetIntPoint(&ip);
               T->Transform(ip, tip);

               T->Loc1.Transform(ip, eip1);
               T->Elem1->Transform(eip1, tip1);

               tip1 -= tip;

               REQUIRE(tip1.Norml2() == Approx(0.0));

               if (T->Elem2)
               {
                  T->Loc2.Transform(ip, eip2);
                  T->Elem2->Transform(eip2, tip2);

                  tip2 -= tip;

                  REQUIRE(tip2.Norml2() == Approx(0.0));
               }
            }
         }
         if (log > 0)
         {
            std::cout << "Checked " << npts << " points within face "
                      << f << std::endl;
         }
      }
   }
}

} // namespace face_elem_trans
