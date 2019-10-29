#ifndef DRL4AMR_HPP
#define DRL4AMR_HPP

#include "mfem.hpp"

using namespace std;
using namespace mfem;

class Drl4Amr
{
private:
   const int n = 8;
   const Element::Type type = Element::QUADRILATERAL;
   const bool generate_edges = true;
   const double sx = 1.0;
   const double sy = 1.0;
   const bool sfc = false; // space-filling curve ordering
   const int order = 2;
   const bool pa = true;
   const char *device_config = "cpu";
   const bool visualization = true;
   const char *vishost = "localhost";
   const int visport = 19916;
   socketstream sol_sock;
   const int max_dofs = 500;

   Device device;
   Mesh mesh;
   const int dim;
   const int sdim;
   H1_FECollection fec;
   FiniteElementSpace fespace;
   BilinearForm a;
   LinearForm b;
   ConstantCoefficient one;
   ConstantCoefficient zero;
   BilinearFormIntegrator *integ;
   GridFunction x;
   Array<int> ess_bdr;
   int iteration;
   FiniteElementSpace flux_fespace;
   ZienkiewiczZhuEstimator estimator;
   ThresholdRefiner refiner;
public:
   Drl4Amr();
   int Compute();
   int Refine();
   int Update();
};

#endif // DRL4AMR_HPP
