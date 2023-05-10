#include "mfem.hpp"
#include <chrono>


using namespace mfem;
int main()
{
   const int order = 2;
   const int dim = 2;
   for (int i=0; i< 5; i++)
   {
      Mesh mesh = Mesh::MakeCartesian2D(static_cast<int>(std::pow(2, i)),
                                        static_cast<int>(std::pow(2, i)), Element::Type::QUADRILATERAL);
      mesh.EnsureNCMesh();
      out<<mesh.GetNE() <<std::endl;

      FiniteElementCollection *fec = new RT_FECollection(order, dim);
      FiniteElementSpace *fes = new FiniteElementSpace(&mesh, fec);

      VectorFunctionCoefficient v(2, [](const Vector &x, double t, Vector &y)
      {
         y(0) = 1.0;
         y(1) = 0.0;
      });

      GridFunction x(fes);
      x.ProjectCoefficient(v);
      out << "(" << x.Min() << ", " << x.Max() << ")" << std::endl;
      delete fes;
      delete fec;
   }
   return 0;
}