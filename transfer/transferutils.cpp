#include <assert.h>
#include "transferutils.hpp"

#include <iostream>

namespace mfem
{

namespace private_
{

void MaxCol(const DenseMatrix &mat, double *vec, bool include_vec_elements)
{
   int n = mat.Height();
   int start = 0;
   if (!include_vec_elements)
   {
      for (int i = 0; i < n; ++i)
      {
         vec[i] = mat.Elem(i, 0);
      }

      start = 1;
   }

   for (int i = 0; i < mat.Height(); ++i)
   {
      for (int j = start; j < mat.Width(); ++j)
      {
         const double e = mat.Elem(i, j);

         if (vec[i] < e)
         {
            vec[i] = e;
         }
      }
   }
}

void MinCol(const DenseMatrix &mat, double *vec, bool include_vec_elements)
{
   int n = mat.Height();
   int start = 0;
   if (!include_vec_elements)
   {
      for (int i = 0; i < n; ++i)
      {
         vec[i] = mat.Elem(i, 0);
      }

      start = 1;
   }

   for (int i = 0; i < mat.Height(); ++i)
   {
      for (int j = start; j < mat.Width(); ++j)
      {
         const double e = mat.Elem(i, j);

         if (vec[i] > e)
         {
            vec[i] = e;
         }
      }
   }
}

Element * NewElem(const int type, const int *cells_data, const int attr)
{
   switch (type)
   {
      case Geometry::TRIANGLE:      return new Triangle(cells_data, attr);
      case Geometry::TETRAHEDRON:   return new Tetrahedron(cells_data, attr);
      case Geometry::SQUARE:        return new Quadrilateral(cells_data, attr);
      case Geometry::CUBE:          return new Hexahedron(cells_data, attr);

      default:
      {
         assert(false && "unknown type");
         mfem::err <<  "NewElem: unknown type " << type << std::endl;
         return nullptr;
      }
   }
}

int MaxVertsXFace(const int type)
{
   switch (type)
   {
      case Geometry::TRIANGLE:      return 2;
      case Geometry::TETRAHEDRON:   return 3;
      case Geometry::SQUARE:        return 2;
      case Geometry::CUBE:          return 4;

      default:
      {
         assert(false && "unknown type");
         mfem::err <<  "NewElem: unknown type " << type << std::endl;
         return -1;
      }
   }
}

void Finalize(Mesh &mesh, const bool generate_edges)
{
   //based on the first element
   int type = mesh.GetElement(0)->GetGeometryType();

   switch (type)
   {
      case Geometry::TRIANGLE:      return mesh.FinalizeTriMesh(generate_edges);
      case Geometry::SQUARE:        return mesh.FinalizeQuadMesh(generate_edges);
      case Geometry::CUBE:          return mesh.FinalizeHexMesh(generate_edges);
      case Geometry::TETRAHEDRON:   return mesh.FinalizeTetMesh(generate_edges);

      default:
      {
         assert(false && "unknown type");
         mfem::err <<  "Finalize: unknown type " << type << std::endl;
         return;
      }
   }
}

double Sum(const DenseMatrix &mat)
{
   Vector rs(mat.Width());
   mat.GetRowSums(rs);
   return rs.Sum();
}
}

}