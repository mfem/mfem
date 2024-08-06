#include "manihyp.hpp"

namespace mfem
{

Mesh sphericalMesh(const double r, Element::Type element_type, int order,
                   const int level_serial, const int level_parallel, bool parallel)
{
   parallel = parallel || level_parallel > 0;
#ifndef MFEM_USE_MPI
   if (parallel) {mfem_error("Parallel MFEM is not built but parallel flag is on.")}
#endif

   VectorFunctionCoefficient sphere_cf(3, [r](const Vector &x,
   Vector &new_x) {new_x = x; new_x *= r/x.Norml2();});
   int Nvert = 8, Nelem = 6;
   if (element_type == Element::Type::TRIANGLE)
   {
      Nvert = 6;
      Nelem = 8;
   }
   Mesh mesh(2, Nvert, Nelem, 0, 3);

   switch (element_type)
   {
      case Element::Type::TRIANGLE:
      {
         const real_t tri_v[6][3] =
         {
            { 1,  0,  0}, { 0,  1,  0}, {-1,  0,  0},
            { 0, -1,  0}, { 0,  0,  1}, { 0,  0, -1}
         };
         const int tri_e[8][3] =
         {
            {0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4},
            {1, 0, 5}, {2, 1, 5}, {3, 2, 5}, {0, 3, 5}
         };

         for (int j = 0; j < Nvert; j++)
         {
            mesh.AddVertex(tri_v[j]);
         }
         for (int j = 0; j < Nelem; j++)
         {
            int attribute = j + 1;
            mesh.AddTriangle(tri_e[j], attribute);
         }
         mesh.FinalizeTriMesh(1, 1, true);
         break;
      }
      case Element::Type::QUADRILATERAL:
      {
         const real_t quad_v[8][3] =
         {
            {-1, -1, -1}, {+1, -1, -1}, {+1, +1, -1}, {-1, +1, -1},
            {-1, -1, +1}, {+1, -1, +1}, {+1, +1, +1}, {-1, +1, +1}
         };
         const int quad_e[6][4] =
         {
            {3, 2, 1, 0}, {0, 1, 5, 4}, {1, 2, 6, 5},
            {2, 3, 7, 6}, {3, 0, 4, 7}, {4, 5, 6, 7}
         };

         for (int j = 0; j < Nvert; j++)
         {
            mesh.AddVertex(quad_v[j]);
         }
         for (int j = 0; j < Nelem; j++)
         {
            int attribute = j + 1;
            mesh.AddQuad(quad_e[j], attribute);
         }
         mesh.FinalizeQuadMesh(1, 1, true);
         break;
      }
      default:
         mfem_error("Only triangle and quadrilateral are supported.");
   }
   mesh.SetCurvature(order, false, 3);
   mesh.GetNodes()->ProjectCoefficient(sphere_cf);

   for (int i=0; i<level_serial; i++)
   {
      mesh.UniformRefinement();
      mesh.GetNodes()->ProjectCoefficient(sphere_cf);
   }
   if (!parallel) { return mesh; }

#ifdef MFEM_USE_MPI
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int i=0; i<level_parallel; i++)
   {
      pmesh.UniformRefinement();
      pmesh.GetNodes()->ProjectCoefficient(sphere_cf);
   }
   return pmesh;
#endif

}
} // end of namespace mfem
