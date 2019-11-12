//                                MFEM DRL4AMR

#include "mfem.hpp"
#include "drl4amr.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

// *****************************************************************************
#define dbg(...) \
   { printf("\n\033[33m"); printf(__VA_ARGS__); printf("\033[m"); fflush(0); }

// *****************************************************************************
enum TestFunction { STEPS, ZZFAIL};

static int discs;
static double theta;
static Array<double> offsets;
constexpr int nb_discs_max = 6;
constexpr double sharpness = 100.0;
static TestFunction test_func = ZZFAIL;

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
Drl4Amr::Drl4Amr(int order):
   order(order),
   device(device_config),
   mesh(nx, ny, type, generate_edges, sx, sy, sfc),
   imesh(GetImageX(), GetImageY(), type, false, sx, sy, false),
   dim(mesh.Dimension()),
   sdim(mesh.SpaceDimension()),
   fec(order, dim, BasisType::Positive),
   fec0(0, dim, BasisType::Positive),
   ifec(0, dim),
   fespace(&mesh, &fec),
   ifes(&imesh, &ifec),
   fespace0(&mesh, &fec0),
   one(1.0),
   zero(0.0),
   integ(new DiffusionIntegrator(one)),
   xcoeff(x0),
   x(&fespace),
   iteration(0),
   flux_fespace(&mesh, &fec, sdim),
   estimator(*integ, x, flux_fespace),
   refiner(estimator),
   nefr(GetImageSize()),
   v_level_no(&ifes),
   v_elem_id(&ifes),
   i_level_no(GetImageSize()),
   i_elem_id(GetImageSize())
{
   dbg("Drl4Amr order: %d\n", order);
   device.Print();

   mesh.EnsureNCMesh();
   mesh.SetCurvature(order, false, sdim, Ordering::byNODES);
   mesh.PrintCharacteristics();

   fespace.Update();
   x.Update();

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
   srand48(time(NULL));
   theta = M_PI*drand48()/2.0;
   discs = static_cast<int>(1 + nb_discs_max*drand48());
   offsets.SetSize(discs);
   for (int i=0; i < discs; i++)
   {
      offsets[i] = drand48();
   }
   offsets.Sort();
   printf("\ntheta = %f, discontinuities:%d", theta, discs);
   for (double offset: offsets)
   {
      printf("\n%f ", offset);
   }
   x.ProjectCoefficient(xcoeff); // TODO: call Compute?

   if (visualization && vis[0].good())
   {
      vis[0].precision(8);
      vis[0] << "solution" << endl << mesh << x << flush;
      vis[0] << "window_title '" << "Solution" << "'" << endl
             << "window_geometry " << 0 << " " << 0 << " " << visw << " " << vish << endl
             << "keys mgA" << endl;

      vis[1].precision(8);
      vis[1] << "mesh" << endl << mesh << flush;
      vis[1] << "window_title '" << "Mesh" << "'" << endl
             << "window_geometry "
             << (vish + 10) << " " << 0
             << " " << visw << " " << vish  << endl
             << "keys mgA" << endl;

      vis[2].precision(8);
      vis[2] << "solution" << endl << mesh << x << flush;
      vis[2] << "window_title '" << "Image" << "'" << endl
             << "window_geometry "
             <<  (2 * vish + 10) << " " << 0
             << " " << visw << " " << vish << endl
             << "keys RjgA" << endl;

      vis[3].precision(8);
      vis[3] << "solution" << endl << mesh << x << flush;
      vis[3] << "window_title '" << "Level" << "'" << endl
             << "window_geometry "
             <<  (3 * vish + 10) << " " << 0
             << " " << visw << " " << vish << endl
             << "keys RjgA" << endl;

      vis[4].precision(8);
      vis[4] << "solution" << endl << mesh << x << flush;
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
   //dbg("~Drl4Amr order: %d\n",o);
}


// *****************************************************************************
int Drl4Amr::Compute()
{
   iteration ++;
   const int cdofs = fespace.GetTrueVSize();
   cout << "\nAMR iteration " << iteration << endl;
   cout << "Number of unknowns: " << cdofs << endl;

   // TODO: it would be more proper to actually solve here
   x.ProjectCoefficient(xcoeff);

   // constrain slave nodes
   if (fespace.GetProlongationMatrix())
   {
      Vector y(fespace.GetTrueVSize());
      fespace.GetRestrictionMatrix()->Mult(x, y);
      fespace.GetProlongationMatrix()->Mult(y, x);
   }

   // Send solution by socket to the GLVis server.
   if (visualization && vis[0].good())
   {
      vis[0] << "solution\n" << mesh << x << flush;
      fflush(0);
   }
   if (cdofs > max_dofs)
   {
      cout << "Reached the maximum number of dofs. Stop." << endl;
      exit(0);
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
      if (depth == max_depth)
      {
         return 0;
      }
      MFEM_VERIFY(depth <= max_depth, "max_amr_depth error");
      //dbg("Refine el:%d, depth:%d", el_to_refine, depth);
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
      //dbg("Refine with refiner");
      // Call the refiner to modify the mesh. The refiner calls the error
      // estimator to obtain element errors, then it selects elements to be
      // refined and finally it modifies the mesh. The Stop() method can be
      // used to determine if a stopping criterion was met.
      refiner.Apply(mesh);
      if (refiner.Stop())
      {
         cout << "Stopping criterion satisfied. Stop." << endl;
         return 1;
      }
   }

   // Update the space to reflect the new state of the mesh.
   fespace.Update();
   x.Update();
   return 0;
}


// *****************************************************************************
double Drl4Amr::GetNorm()
{
   // Setup all integration rules for any element type
   const int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }
   const double err_x  = x.ComputeL2Error(xcoeff, irs);
   const double norm_x = ComputeLpNorm(2., xcoeff, mesh, irs);
   return err_x / norm_x;
}


// *****************************************************************************
double *Drl4Amr::GetImage()
{
   Vector sln;
   Array<int> vert;
   int imwidth = GetImageX();
   image.SetSize(GetImageSize());
   // rasterize each element into the image
   for (int k = 0; k < mesh.GetNE(); k++)
   {
      int subdiv = (1 << (max_depth - mesh.ncmesh->GetElementDepth(k))) * order;
      const IntegrationRule &ir =
         GlobGeometryRefiner.Refine(Geometry::SQUARE, subdiv)->RefPts;
      x.GetValues(k, ir, sln);
      mesh.GetElementVertices(k, vert);
      const double *v = mesh.GetVertex(vert[0]);
      int ox = int(v[0] * nx*(1 << max_depth) * order);
      int oy = int(v[1] * ny*(1 << max_depth) * order);
      for (int i = 0; i <= subdiv; i++)
         for (int j = 0; j <= subdiv; j++)
         {
            int n = i*(subdiv+1) + j;
            int m = (oy + i)*imwidth + (ox + j);
            image(m) = sln(n);
         }
   }
   if (visualization) { ShowImage(); }
   return image.GetData();
}

// *****************************************************************************
void Drl4Amr::GetImage(GridFunction &gf, Vector &v_result, Array<int> &i_result)
{
   Vector vals;
   Array<int> vert;
   const int width = GetImageX();

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
      for (int i = 0; i <= times; i++)
      {
         for (int j = 0; j <= times; j++)
         {
            int n = i * (times + 1) + j;
            int m = (oy + i) * width + (ox + j);
            i_result[m] = static_cast<int>(vals(n));
            v_result(m) = vals(n);
         }
      }
   }
}

// *****************************************************************************
int *Drl4Amr::GetLevelField()
{
   //Mesh msh(mesh, true);
   FiniteElementSpace fes0(&mesh, &fec0);
   GridFunction level(&fes0);

   Array<int> dofs;
   for (int i = 0; i < mesh.GetNE(); i++)
   {

      const int depth = mesh.ncmesh->GetElementDepth(i);

      fes0.GetElementDofs(i, dofs);
      for (int k = 0; k < dofs.Size(); k++)
      {
         level[ dofs[k] ] = depth;
      }
   }

   GetImage(level, v_level_no, i_level_no);

   if (visualization && vis[3].good())
   {
      GridFunction gf(&ifes, v_level_no.GetData());
      vis[3] << "solution" << endl << imesh << gf << flush;
      fflush(0);
   }

   return i_level_no;
}

// *****************************************************************************
int *Drl4Amr::GetElemIdField()
{
   Mesh msh(mesh, true);
   FiniteElementSpace fes0(&msh, &fec0);
   GridFunction elem(&fes0);

   Array<int> dofs;
   for (int i = 0; i < msh.GetNE(); i++)
   {
      fes0.GetElementDofs(i, dofs);
      for (int k = 0; k < dofs.Size(); k++)
      {
         elem[ dofs[k] ] = i;
      }
   }

   GetImage(elem, v_elem_id, i_elem_id);

   if (visualization && vis[4].good())
   {
      GridFunction gf(&ifes, v_elem_id.GetData());
      vis[4] << "solution" << endl << imesh << gf << flush;
      fflush(0);
   }

   return i_elem_id;
}


// *****************************************************************************
void Drl4Amr::ShowImage()
{
   if (!vis[2].good()) { return; }
   GridFunction gf(&ifes, image.GetData());
   vis[2] << "solution" << endl << imesh << gf << flush;
}


// *****************************************************************************
extern "C" {
   Drl4Amr* Ctrl(int order) { return new Drl4Amr(order); }

   int Compute(Drl4Amr *ctrl) { return ctrl->Compute(); }
   int Refine(Drl4Amr *ctrl, int el) { return ctrl->Refine(el); }

   int GetNE(Drl4Amr *ctrl) { return ctrl->GetNE(); }
   int GetNEFullyRefined(Drl4Amr *ctrl) { return ctrl->GetNEFullyRefined(); }
   int GetNDofs(Drl4Amr *ctrl) { return ctrl->GetNDofs(); }

   double GetNorm(Drl4Amr *ctrl) { return ctrl->GetNorm(); }
   double *GetImage(Drl4Amr *ctrl) { return ctrl->GetImage(); }

   int* GetLevelField(Drl4Amr *ctrl) { return ctrl->GetLevelField(); }
   int* GetElemIdField(Drl4Amr *ctrl) { return ctrl->GetElemIdField(); }

   int GetImageX(Drl4Amr *ctrl) { return ctrl->GetImageX(); }
   int GetImageY(Drl4Amr *ctrl) { return ctrl->GetImageY(); }
   int GetImageSize(Drl4Amr *ctrl) { return ctrl->GetImageSize(); }
}
