#ifndef DRL4AMR_HPP
#define DRL4AMR_HPP

#include "mfem.hpp"

using namespace std;
using namespace mfem;

class Drl4Amr
{
private:
   const int nx = 4;
   const int ny = 4;
   const int max_dofs = 500;
   const int max_depth = 2;
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
   Mesh mesh, node_image_mesh, elem_image_mesh;
   const int dim;
   const int sdim;
   H1_FECollection h1fec;
   L2_FECollection l2fec;
   FiniteElementSpace h1fes, l2fes, node_image_l2fes, elem_image_l2fes;
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
   Drl4Amr(int order, long int seed =0);

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
   double *GetElemIdField();
   double *GetElemDepthField();

   int GetNodeImageX() const { return 1 + GetElemImageX();}
   int GetNodeImageY() const { return 1 + GetElemImageY();}
   int GetNodeImageSize() const { return GetNodeImageX() * GetNodeImageY(); }

   int GetElemImageX() const { return order * (nx << max_depth);}
   int GetElemImageY() const { return order * (ny << max_depth);}
   int GetElemImageSize() const { return GetElemImageX() * GetElemImageY(); }

private:
   void GetH1Image(GridFunction&, Vector&);
   void GetL2Image(GridFunction&, Vector&);
};

#endif // DRL4AMR_HPP
