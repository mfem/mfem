#ifndef DRL4AMR_HPP
#define DRL4AMR_HPP

#include "mfem.hpp"

using namespace std;
using namespace mfem;

class Drl4Amr
{
private:
   const int nx = 8;
   const int ny = 8;
   const int max_dofs = 50000;
   const int max_depth = 2;
   const Element::Type type = Element::QUADRILATERAL;
   const bool generate_edges = true;
   const double sx = 1.0;
   const double sy = 1.0;
   const bool sfc = false; // space-filling curve ordering
   const char *device_config = "cpu";
   const bool visualization = true;
   const char *vishost = "localhost";
   const int visport = 19916;
   const int visw = 480;
   const int vish = 480;
   socketstream vis[3];
   Vector image;

   int order;
   Device device;
   Mesh mesh;
   int dim;
   int sdim;
   H1_FECollection fec;
   FiniteElementSpace fespace;
   ConstantCoefficient one;
   ConstantCoefficient zero;
   BilinearFormIntegrator *integ;
   FunctionCoefficient xcoeff;
   GridFunction x;
   int iteration;
   FiniteElementSpace flux_fespace;
   ZienkiewiczZhuEstimator estimator;
   ThresholdRefiner refiner;

public:
   Drl4Amr(int order);
   int Compute();
   int Refine(int el =-1);
   int GetNDofs() { return fespace.GetNDofs();}
   int GetNE() { return fespace.GetNE(); }
   double GetNorm();
   double *GetImage();
   void ShowImage();
   int GetImageX() const { return 1 + order * (nx << max_depth); }
   int GetImageY() const { return 1 + order * (ny << max_depth); }
   int GetImageSize() const { return GetImageX() * GetImageY(); }
};

#endif // DRL4AMR_HPP
