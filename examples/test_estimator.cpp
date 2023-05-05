#include "mfem.hpp"
#include <chrono>


using namespace mfem;
int main()
{
   const int order = 2;
   const int dim = 2;
   for (int i=0; i<12; i++)
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

      out << "Estimation" << std::endl;
      auto start_1 = std::chrono::high_resolution_clock::now();
      ProjectionErrorEstimator estimator(x);
      auto & estimators = estimator.GetLocalErrors();
      out << "estimated error: " << estimator.GetTotalError() << std::endl;
      auto stop_1 = std::chrono::high_resolution_clock::now();
      auto duration_1 = std::chrono::duration_cast<std::chrono::microseconds>
                        (stop_1 - start_1);
      out << "wall time: " << duration_1.count() / 1000.0 << "ms" << std::endl;

      out << "Estimation" << std::endl;
      auto start_2 = std::chrono::high_resolution_clock::now();
      PRefDiffEstimator estimator_Katen(x, -1);
      auto & estimators_Katen = estimator_Katen.GetLocalErrors();
      out << "estimated error: " << estimator_Katen.GetTotalError() << std::endl;
      auto stop_2 = std::chrono::high_resolution_clock::now();
      auto duration_2 = std::chrono::duration_cast<std::chrono::microseconds>
                        (stop_2 - start_2);
      out << "wall time: " << duration_2.count() / 1000.0 <<  "ms" << std::endl;

      // double total_error = estimators.Norml2();
      out << x.ComputeL2Error(v) << std::endl;
      out << estimator.GetTotalError() / x.ComputeL2Error(v) << std::endl;
      out << estimator_Katen.GetTotalError() / x.ComputeL2Error(v) << std::endl;
      // out << estimators.Max() << std::endl;
      delete fes;
      delete fec;
   }

   return 0;

}