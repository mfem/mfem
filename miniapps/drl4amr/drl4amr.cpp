//                                MFEM DRL4AMR

#include "mfem.hpp"
#include "drl4amr.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#define dbg(...) \
   { printf("\n\033[33m"); printf(__VA_ARGS__); printf("\033[m"); fflush(0); }

static int discs;
static double theta;
static Array<double> offsets;
constexpr int nb_discs_max = 6;
constexpr double sharpness = 100.0;

// *****************************************************************************
double x0(const Vector &x)
{
   double result = 0.0;
   const double t = x[0] + tan(theta)*x[1];
   for (double o: offsets) { result += 1.0 + tanh(sharpness*(o - t)); }
   return result/static_cast<double>(discs<<1); // should be in [0,1]
}

// *****************************************************************************
Drl4Amr::Drl4Amr(int order):
   order(order),
   device(device_config),
   mesh(nx, ny, type, generate_edges, sx, sy, sfc),
   dim(mesh.Dimension()),
   sdim(mesh.SpaceDimension()),
   fec(order, dim, BasisType::Positive),
   fespace(&mesh, &fec),
   one(1.0),
   zero(0.0),
   integ(new DiffusionIntegrator(one)),
   xcoeff(x0),
   x(&fespace),
   iteration(0),
   flux_fespace(&mesh, &fec, sdim),
   estimator(*integ, x, flux_fespace),
   refiner(estimator)
{
   //dbg("Drl4Amr order:%d",o);
   device.Print();

   mesh.EnsureNCMesh();
   mesh.PrintCharacteristics();

   fespace.Update();

   // Connect to GLVis.
   if (visualization)
   {
      vis[0].open(vishost, visport);
      vis[1].open(vishost, visport);
      vis[2].open(vishost, visport);
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
             << "keys RjgA" << endl; // mn
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
      //mesh.PrintCharacteristics();
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
   //dbg("GetNorm");
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
   Array<int> vert;
   Vector sln;

   image.SetSize(GetImageSize());
   int imwidth = GetImageX();

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
void Drl4Amr::ShowImage()
{
   if (!vis[2].good()) { return; }

   Mesh imesh(GetImageX(), GetImageY(), type, false, sx, sy, false);

   L2_FECollection fec(0, imesh.Dimension());
   FiniteElementSpace fes(&imesh, &fec);
   GridFunction gridfn(&fes, image.GetData());

   vis[2] << "solution" << endl << imesh << gridfn << flush;
}


// *****************************************************************************
extern "C" {
   Drl4Amr* Ctrl(int order) { return new Drl4Amr(order); }
   int Compute(Drl4Amr *ctrl) { return ctrl->Compute(); }
   int Refine(Drl4Amr *ctrl, int el) { return ctrl->Refine(el); }
   int GetNDofs(Drl4Amr *ctrl) { return ctrl->GetNDofs(); }
   int GetNE(Drl4Amr *ctrl) { return ctrl->GetNE(); }
   double GetNorm(Drl4Amr *ctrl) { return ctrl->GetNorm(); }
   double *GetImage(Drl4Amr *ctrl) { return ctrl->GetImage(); }
   int GetImageSize(Drl4Amr *ctrl) { return ctrl->GetImageSize(); }
   int GetImageX(Drl4Amr *ctrl) { return ctrl->GetImageX(); }
   int GetImageY(Drl4Amr *ctrl) { return ctrl->GetImageY(); }
}
