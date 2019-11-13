//                                MFEM DRL4AMR

#include "mfem.hpp"
#include "drl4amr.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// *****************************************************************************
#define dbg(...) { if (getenv("DBG")) {\
  printf("\n\033[33m"); printf(__VA_ARGS__); printf("\033[m"); fflush(0); }}

// *****************************************************************************
enum TestFunction {STEPS, ZZFAIL};

static int discs;
static double theta;
static Array<double> offsets;
constexpr int nb_discs_max = 6;
constexpr double sharpness = 100.0;
static TestFunction test_func = STEPS;

double x0_steps(const Vector &x)
{
   double result = 0.0;
   const double t = x[0] + tan(theta)*x[1];
   for (double o: offsets) { result += 1.0 + tanh(sharpness*(o - t)); }
   return result/static_cast<double>(discs<<1); // should be in [0,1]
}

double sgn(double val)
{
   const double eps = 1.e-10;
   if (val < -eps) { return -1.0; }
   if (val > +eps) { return +1.0; }
   return 0.;
}

double x0_zzfail(const Vector &x)
{
   const double f = 16.0;
   const double twopi = 8*atan(1.);
   const double w0 = sgn( sin(f*twopi*x[0]) );
   const double w1 = sgn( sin(f*twopi*x[1]) );
   if (x[0] < 0.5) { return w0*w1; }
   return 0;
}

double x0(const Vector &x)
{
   switch (test_func)
   {
      case STEPS: return x0_steps(x);
      case ZZFAIL: return x0_zzfail(x);
   }
   printf("unknown function: %d\n",test_func);
   return 0;
}

// *****************************************************************************
Drl4Amr::Drl4Amr(int order, int seed):
   order(order),
   seed(seed != 0 ? seed : time(NULL)),
   device(device_config),
   mesh(nx, ny, quads, generate_edges, sx, sy, sfc),
   node_image_mesh(GetNodeImageX(), GetNodeImageY(), quads, false, sx, sy, false),
   elem_image_mesh(GetElemImageX(), GetElemImageY(), quads, false, sx, sy, false),
   dim(mesh.Dimension()),
   sdim(mesh.SpaceDimension()),
   h1fec(order, dim, BasisType::Positive),
   l2fec(0, dim),
   h1fes(&mesh, &h1fec),
   l2fes(&mesh, &l2fec),
   node_image_l2fes(&node_image_mesh, &l2fec),
   elem_image_l2fes(&elem_image_mesh, &l2fec),
   one(1.0),
   zero(0.0),
   integ(new DiffusionIntegrator(one)),
   xcoeff(x0),
   solution(&h1fes),
   elem_id(&elem_image_l2fes),
   elem_depth(&elem_image_l2fes),
   solution_image(GetNodeImageSize()),
   elem_id_image(GetElemImageSize()),
   elem_depth_image(GetElemImageSize()),
   flux_fespace(&mesh, &h1fec, sdim),
   estimator(*integ, solution, flux_fespace),
   refiner(estimator),
   iteration(0)
{
   MFEM_VERIFY(node_image_l2fes.GetNE()==GetNodeImageSize(),"");
   MFEM_VERIFY(elem_image_l2fes.GetNE()==GetElemImageSize(),"");
   dbg("Drl4Amr order: %d\n", order);
   device.Print();

   mesh.EnsureNCMesh();
   mesh.SetCurvature(order, false, sdim, Ordering::byNODES);
   mesh.PrintCharacteristics();

   h1fes.Update();
   l2fes.Update();
   solution.Update();

   // Connect to GLVis.
   if (visualization)
   {
      vis[0].open(vishost, visport);
      vis[1].open(vishost, visport);
      vis[2].open(vishost, visport);
      vis[3].open(vishost, visport);
      vis[4].open(vishost, visport);
   }

   // Initialize theta, offsets and x from x0_coeff
   srand48(seed);
   theta = M_PI*drand48()/2.0;
   discs = static_cast<int>(1 + nb_discs_max*drand48());
   offsets.SetSize(discs);
   for (int i=0; i < discs; i++) { offsets[i] = drand48(); }
   offsets.Sort();
   printf("\ntheta = %f, discontinuities:%d", theta, discs);
   for (double offset: offsets) { printf("\n%f ", offset); }
   solution.ProjectCoefficient(xcoeff); // TODO: call Compute?

   if (visualization && vis[0].good())
   {
      vis[0].precision(8);
      vis[0] << "solution" << endl << mesh << solution << flush;
      vis[0] << "window_title '" << "Solution" << "'" << endl
             << "window_geometry "
             << 0 << " " << 0 << " " << visw << " " << vish << endl
             << "keys mgA" << endl;
   }

   if (visualization && vis[1].good())
   {
      vis[1].precision(8);
      vis[1] << "mesh" << endl << mesh << flush;
      vis[1] << "window_title '" << "Mesh" << "'" << endl
             << "window_geometry "
             << (vish + 10) << " " << 0
             << " " << visw << " " << vish  << endl
             << "keys mgA" << endl;
   }

   if (visualization && vis[2].good())
   {
      solution_image = 0.0;
      GridFunction gf(&node_image_l2fes, solution_image.GetData());
      vis[2].precision(8);
      vis[2] << "solution" << endl << node_image_mesh << gf << flush;
      vis[2] << "window_title '" << "Node Image" << "'" << endl
             << "window_geometry "
             <<  (2 * vish + 10) << " " << 0
             << " " << visw << " " << vish << endl
             << "keys RjgA" << endl;
   }

   if (visualization && vis[3].good())
   {
      elem_depth_image = 0.0;
      GridFunction gf(&elem_image_l2fes, elem_depth_image.GetData());
      vis[3].precision(8);
      vis[3] << "solution" << endl << elem_image_mesh << gf << flush;
      vis[3] << "window_title '" << "Elem Depth" << "'" << endl
             << "window_geometry "
             <<  (3 * vish + 10) << " " << 0
             << " " << visw << " " << vish << endl
             << "keys RjgA" << endl;
   }

   if (visualization && vis[4].good())
   {
      elem_id_image = 0.0;
      GridFunction gf(&elem_image_l2fes, elem_id_image.GetData());
      vis[4].precision(8);
      vis[4] << "solution" << endl << elem_image_mesh << gf << flush;
      vis[4] << "window_title '" << "Elem Id" << "'" << endl
             << "window_geometry "
             <<  (4 * vish + 10) << " " << 0
             << " " << visw << " " << vish << endl
             << "keys RjgA" << endl;
   }

   // Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   // that uses the ComputeElementFlux method of the DiffusionIntegrator to
   // recover a smoothed flux (gradient) that is subtracted from the element
   // flux to get an error indicator. We need to supply the space for the
   // smoothed flux: an (H1)^sdim (i.e., vector-valued) space is used here.
   estimator.SetAnisotropic();

   // A refiner selects and refines elements based on a refinement strategy.
   // The strategy here is to refine elements with errors larger than a
   // fraction of the maximum element error. Other strategies are possible.
   // The refiner will call the given error estimator.
   refiner.SetTotalErrorFraction(0.7);
}


// *****************************************************************************
int Drl4Amr::Compute()
{
   iteration++;
   const int cdofs = h1fes.GetTrueVSize();
   mfem::out << "\nAMR iteration " << iteration << endl;
   mfem::out << "Number of unknowns: " << cdofs << endl;

   // TODO: it would be more proper to actually solve here
   solution.ProjectCoefficient(xcoeff);

   // constrain slave nodes
   if (h1fes.GetProlongationMatrix())
   {
      Vector y(h1fes.GetTrueVSize());
      h1fes.GetRestrictionMatrix()->Mult(solution, y);
      h1fes.GetProlongationMatrix()->Mult(y, solution);
   }

   // Send solution by socket to the GLVis server.
   if (visualization && vis[0].good())
   {
      vis[0] << "solution\n" << mesh << solution << flush;
      fflush(0);
   }
   if (cdofs > max_dofs)
   {
      mfem::out << "Reached the maximum number of dofs." << endl;
      return -1;
   }
   return 0;
}


// *****************************************************************************
int Drl4Amr::Refine(int el_to_refine)
{
   if (el_to_refine >= 0)
   {
      mesh.EnsureNCMesh();
      const int depth = mesh.ncmesh->GetElementDepth(el_to_refine);
      if (depth == max_depth) { return 1; }
      MFEM_VERIFY(depth <= max_depth, "max_amr_depth error");
      dbg("Refine el:%d, depth:%d", el_to_refine, depth);
      Array<Refinement> refinements(1);
      refinements[0] = Refinement(el_to_refine);
      mesh.GeneralRefinement(refinements, 1, 1);
      // Send solution by socket to the GLVis server.
      if (visualization && vis[1].good())
      {
         vis[1] << "mesh\n" << mesh << flush;
         fflush(0);
      }
   }
   else
   {
      dbg("Refine with refiner");
      // Call the refiner to modify the mesh. The refiner calls the error
      // estimator to obtain element errors, then it selects elements to be
      // refined and finally it modifies the mesh. The Stop() method can be
      // used to determine if a stopping criterion was met.
      refiner.Apply(mesh);
      if (refiner.Stop())
      {
         cout << "Stopping criterion satisfied. Stop." << endl;
         return -1;
      }
   }

   // Update the space to reflect the new state of the mesh.
   h1fes.Update();
   l2fes.Update();
   solution.Update();
   dbg("done");
   return 0;
}


// *****************************************************************************
double Drl4Amr::GetNorm()
{
   dbg("GetNorm");
   // Setup all integration rules for any element type
   const int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   { irs[i] = &(IntRules.Get(i, order_quad)); }
   const double err_x  = solution.ComputeL2Error(xcoeff, irs);
   const double norm_x = ComputeLpNorm(2., xcoeff, mesh, irs);
   return err_x / norm_x;
}


// *****************************************************************************
void Drl4Amr::GetImage(GridFunction &gf, Vector &image, bool isH1)
{
   dbg("GetImage");
   Vector vals;
   Array<int> vert;
   const int offset = isH1 ? 0 : 1;
   const int width = GetNodeImageX() - offset;
   // rasterize each element into the image
   for (int k = 0; k < mesh.GetNE(); k++)
   {
      Geometry::Type geom = Geometry::SQUARE;
      int depth = mesh.ncmesh->GetElementDepth(k);
      int times = (1 << (max_depth - depth)) * order;
      IntegrationRule &ir = GlobGeometryRefiner.Refine(geom, times)->RefPts;
      gf.GetValues(k, ir, vals);
      mesh.GetElementVertices(k, vert);
      double *v = mesh.GetVertex(vert[0]);
      int ox = int(v[0] * nx*(1 << max_depth) * order);
      int oy = int(v[1] * ny*(1 << max_depth) * order);
      int N = times - offset;
      for (int i = 0; i <= N; i++)
      {
         for (int j = 0; j <= N; j++)
         {
            int n = i * (times + 1) + j;
            int m = (oy + i) * width + (ox + j);
            image(m) = vals(n);
         }
      }
   }
}


// *****************************************************************************
double *Drl4Amr::GetImage()
{
   dbg("GetImage: solution");
   GetImage(solution, solution_image, true);
   if (visualization && vis[2].good())
   {
      GridFunction gf(&node_image_l2fes, solution_image.GetData());
      vis[2] << "solution" << endl << node_image_mesh << gf << flush;
   }
   return solution_image.GetData();
}
\

// *****************************************************************************
double *Drl4Amr::GetElemIdField()
{
   GridFunction id(&l2fes);

   Array<int> dofs;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      l2fes.GetElementDofs(i, dofs);
      for (int k = 0; k < dofs.Size(); k++)
      {
         id[ dofs[k] ] = i;
      }
   }
   GetImage(id, elem_id, false);
   if (visualization && vis[4].good())
   {
      GridFunction gf(&elem_image_l2fes, elem_id.GetData());
      vis[4] << "solution" << endl << elem_image_mesh << gf << flush;
      fflush(0);
   }
   return elem_id.GetData();
}


// *****************************************************************************
double *Drl4Amr::GetElemDepthField()
{
   GridFunction level(&l2fes);

   Array<int> dofs;
   for (int i = 0; i < mesh.GetNE(); i++)
   {
      const int depth = mesh.ncmesh->GetElementDepth(i);
      l2fes.GetElementDofs(i, dofs);
      for (int k = 0; k < dofs.Size(); k++)
      {
         level[ dofs[k] ] = depth;
      }
   }
   GetImage(level, elem_depth, false);
   if (visualization && vis[3].good())
   {
      GridFunction gf(&elem_image_l2fes, elem_depth.GetData());
      vis[3] << "solution" << endl << elem_image_mesh << gf << flush;
      fflush(0);
   }
   return elem_depth.GetData();
}


// *****************************************************************************
extern "C" {
   Drl4Amr* Ctrl(int order, int seed) { return new Drl4Amr(order, seed); }

   int Compute(Drl4Amr *ctrl) { return ctrl->Compute(); }
   int Refine(Drl4Amr *ctrl, int el) { return ctrl->Refine(el); }

   int GetNE(Drl4Amr *ctrl) { return ctrl->GetNE(); }
   int GetNDofs(Drl4Amr *ctrl) { return ctrl->GetNDofs(); }

   double GetNorm(Drl4Amr *ctrl) { return ctrl->GetNorm(); }

   double *GetImage(Drl4Amr *ctrl) { return ctrl->GetImage(); }
   double *GetElemIdField(Drl4Amr *ctrl) { return ctrl->GetElemIdField(); }
   double *GetElemDepthField(Drl4Amr *ctrl) { return ctrl->GetElemDepthField(); }

   int GetNodeImageX(Drl4Amr *ctrl) { return ctrl->GetNodeImageX(); }
   int GetNodeImageY(Drl4Amr *ctrl) { return ctrl->GetNodeImageY(); }
   int GetNodeImageSize(Drl4Amr *ctrl) { return ctrl->GetNodeImageSize(); }

   int GetElemImageX(Drl4Amr *ctrl) { return ctrl->GetElemImageX(); }
   int GetElemImageY(Drl4Amr *ctrl) { return ctrl->GetElemImageY(); }
   int GetElemImageSize(Drl4Amr *ctrl) { return ctrl->GetElemImageSize(); }
}
