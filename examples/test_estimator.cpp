#include "mfem.hpp"

using namespace mfem;
int main()
{
   const int order = 2;
   const int dim = 2;
   for (int i=0; i<7; i++)
   {
      Mesh mesh = Mesh::MakeCartesian2D(static_cast<int>(std::pow(2, i)),
                                        static_cast<int>(std::pow(2, i)), Element::Type::QUADRILATERAL);
      mesh.EnsureNCMesh();
      out<<mesh.GetNE() <<std::endl;

      FiniteElementCollection *fec = new DG_FECollection(order, dim);
      FiniteElementSpace *fes = new FiniteElementSpace(&mesh, fec, 2,
                                                       Ordering::byNODES);

      VectorFunctionCoefficient v(2, [](const Vector &x, double t, Vector &y)
      {
         y(0) = std::sin(x(0))*std::sin(x(1));
         y(1) = std::cos(x(0))*std::cos(x(1));
      });

      GridFunction x(fes);
      x.ProjectCoefficient(v);

      ProjectionErrorEstimator estimator(x);
      auto & estimators = estimator.GetLocalErrors();

      PRefDiffEstimator estimator_Katen(x, -1);
      auto & estimators_Katen = estimator_Katen.GetLocalErrors();

      // double total_error = estimators.Norml2();
      out << estimator.GetTotalError() << std::endl;
      out << estimator_Katen.GetTotalError() << std::endl;
      out << x.ComputeL2Error(v) << std::endl;
      out << estimator.GetTotalError() / x.ComputeL2Error(v) << std::endl;
      out << estimator_Katen.GetTotalError() / x.ComputeL2Error(v) << std::endl;
      // out << estimators.Max() << std::endl;
      delete fes;
      delete fec;
   }

   return 0;

}