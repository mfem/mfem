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
   const int max_depth = 2;
   const int max_dofs = 5000;
   const Element::Type quads = Element::QUADRILATERAL;
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
   socketstream vis[5];

   const int order;
   const long int seed;
   Device device;
   Mesh mesh, image_mesh;
   const int dim;
   const int sdim;
   H1_FECollection h1fec;
   L2_FECollection l2fec;
   FiniteElementSpace h1fes, l2fes, image_fes;
   ConstantCoefficient one;
   ConstantCoefficient zero;
   BilinearFormIntegrator *integ;
   FunctionCoefficient xcoeff;
   GridFunction solution, elem_id, elem_depth;
   Vector solution_image, elem_id_image, elem_depth_image;
   FiniteElementSpace flux_fespace;
   ZienkiewiczZhuEstimator estimator;
   ThresholdRefiner refiner;
   int iteration;

public:
   Drl4Amr(int order, int seed =0);

   int Compute();

   /// If el_to_refine is positive or null, will refine this element only,
   /// if el_to_refine is strictly negative, it will use the built-in refiner.
   /// Return:
   ///    1 if max_depth was hit
   ///   -1 if the stopping criterion satisfied (only with the built-in refiner)
   ///    0 otherwise
   int Refine(int el_to_refine =-1);

   int GetNE() { return h1fes.GetNE(); }
   int GetNDofs() { return h1fes.GetNDofs();}

   double GetNorm();
   double *GetImage();
   double *GetIdField();
   double *GetDepthField();

   int GetImageX() const { return order * (nx << max_depth);}
   int GetImageY() const { return order * (ny << max_depth);}
   int GetImageSize() const { return GetImageX() * GetImageY(); }

private:
   void GetImage(GridFunction&, Vector&);
};

#endif // DRL4AMR_HPP
