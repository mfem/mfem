#include "mfem.hpp"
#include "unit_tests.hpp"
#include "../mesh/mesh_test_utils.hpp"
using namespace mfem;
#ifdef MFEM_USE_MPI

TEST_CASE("ProjectBdrCoefficientTangent face DOF orientation",
          "[Parallel][ND][DofTransformation]")
{
   Mesh smesh = DividingPlaneMesh(true, true, true);
   int int_bdr_attr = smesh.bdr_attributes.Max();
   int order = 2;
   ND_FECollection fec(order, 3);

   // Serial reference
   FiniteElementSpace sfes(&smesh, &fec);
   GridFunction sgf(&sfes);
   sgf = 0.0;
   VectorFunctionCoefficient coeff(3, [](const Vector &x, Vector &v)
   {
      v.SetSize(3);
      v[0] =  1.234 * x[1] - 2.357 * x[2];
      v[1] = -1.234 * x[0] + 3.572 * x[2];
      v[2] =  2.357 * x[0] - 3.572 * x[1];
   });
   Array<int> sbdr(smesh.bdr_attributes.Max()); sbdr = 0;
   sbdr[int_bdr_attr - 1] = 1;
   sgf.ProjectBdrCoefficientTangent(coeff, sbdr);

   // Evaluate serial at element centroids
   std::vector<real_t> serial_vals;
   IntegrationPoint ip_c;
   ip_c.Set3(0.25, 0.25, 0.25);
   for (int e = 0; e < smesh.GetNE(); e++)
   {
      Vector v(3);
      sgf.GetVectorValue(e, ip_c, v);
      serial_vals.push_back(v.Norml2());
   }

   // Parallel
   ParMesh pmesh(MPI_COMM_WORLD, smesh);
   ParFiniteElementSpace pfes(&pmesh, &fec);
   ParGridFunction pgf(&pfes);
   pgf = 0.0;
   int pbm = pmesh.bdr_attributes.Max();
   Array<int> pbdr(pbm); pbdr = 0;
   if (int_bdr_attr <= pbm) { pbdr[int_bdr_attr - 1] = 1; }
   Array<int> gm(pbm);
   MPI_Allreduce(pbdr.GetData(), gm.GetData(), pbm, MPI_INT, MPI_MAX,
                 MPI_COMM_WORLD);
   pbdr = gm;
   pgf.ProjectBdrCoefficientTangent(coeff, pbdr);

   // Evaluate parallel at element centroids and compare against serial
   int local_ne = pmesh.GetNE();
   real_t max_err = 0.0;
   for (int e = 0; e < local_ne; e++)
   {
      Vector v(3);
      pgf.GetVectorValue(e, ip_c, v);
      real_t par_val = v.Norml2();

      // Match to serial element by centroid
      Vector center(3);
      pmesh.GetElementCenter(e, center);
      int best = -1;
      real_t best_dist = 1e20;
      for (int se = 0; se < smesh.GetNE(); se++)
      {
         Vector sc(3);
         smesh.GetElementCenter(se, sc);
         real_t d = std::abs(sc[0]-center[0]) + std::abs(sc[1]-center[1]) +
                    std::abs(sc[2]-center[2]);
         if (d < best_dist) { best_dist = d; best = se; }
      }
      if (best >= 0 && serial_vals[best] > 1e-15)
      {
         real_t rel_err = std::abs(par_val - serial_vals[best]) / serial_vals[best];
         if (rel_err > max_err) { max_err = rel_err; }
      }
   }
   real_t global_max_err = 0.0;
   MPI_Allreduce(&max_err, &global_max_err, 1, MFEM_MPI_REAL_T, MPI_MAX,
                 MPI_COMM_WORLD);

   CAPTURE(global_max_err);
   REQUIRE(global_max_err < 1e-12);
}
#endif
