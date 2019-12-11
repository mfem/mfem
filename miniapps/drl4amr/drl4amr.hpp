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
   const Element::Type elem_type = Element::QUADRILATERAL;
   const double sx = 1.0;
   const double sy = 1.0;

   const int oversample = 8;
   const int context = 4;

   Mesh mesh;

   int order;
   int dim;

   Device device;
   socketstream vis[3];

   Vector image;
   Vector local_image;

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

   double *GetFullImage();
   int GetFullWidth() const { return 1 + order * (nx << max_depth); }
   int GetFullHeight() const { return GetFullWidth(); }
   void ShowFullImage();

   double* GetLocalImage(int element);
   int GetLocalWidth() const { return oversample*order + 2*context; }
   int GetLocalHeight() const { return GetLocalWidth(); }

   void RandomRefine(double p = 0.5);
};

#endif // DRL4AMR_HPP
