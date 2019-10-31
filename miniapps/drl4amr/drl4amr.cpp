//                                MFEM DRL4AMR

#include "mfem.hpp"
#include "drl4amr.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

//#define dbg(...) \
//   { printf("\n\033[33m"); printf(__VA_ARGS__); printf("\033[m"); fflush(0); }

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
   for (double o: offsets)
   {
      result += tanh(sharpness*(o - t));
   }
   return result/static_cast<double>(discs);
}

// *****************************************************************************
Drl4Amr::Drl4Amr(int o):
   order(o),
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

   mesh.EnsureNodes();
   mesh.PrintCharacteristics();
   mesh.SetCurvature(order, false, sdim, Ordering::byNODES);

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
   x.ProjectCoefficient(xcoeff);

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

   x.ProjectCoefficient(xcoeff);

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
      if (depth == max_amr_depth)
      {
         return 0;
      }
      MFEM_VERIFY(depth <= max_amr_depth, "max_amr_depth error");
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
class NCM: public NCMesh
{
public:
   NCM(NCMesh *n): NCMesh(*n) {}

   int GetMaxDepth()
   {
      int max_depth = -1;
      for (int i = 0; i < leaf_elements.Size(); i++)
      {
         const int depth = GetElementDepth(i);
         max_depth = std::max(depth, max_depth);
      }
      return max_depth;
   }

   bool IsAllMaxDepth()
   {
      const int max_depth = GetMaxDepth();
      for (int i = 0; i < leaf_elements.Size(); i++)
      {
         const int depth = GetElementDepth(i);
         if (depth < max_depth)
         {
            return false;
         }
      }
      return true;
   }

   void FullRefine(Mesh *image)
   {
      const int max_depth = GetMaxDepth();
      const char ref_type = 7; // iso
      Array<Refinement> refinements;
      for (int i = 0; i < leaf_elements.Size(); i++)
      {
         if (GetElementDepth(i) < max_depth)
         {
            refinements.Append(Refinement(i, ref_type));
         }
      }
      image->GeneralRefinement(refinements, 1, 0);
   }
};


// *****************************************************************************
void Drl4Amr::GetImage()
{
   //dbg("GetImage");
   //mesh.ncmesh->PrintStats();
   bool done = false;
   Mesh msh(mesh, true);
   FiniteElementSpace fes(&msh, &fec);
   GridFunction X(&fes);

   X.ProjectCoefficient(xcoeff);

   while (!done)
   {
      NCM nc(msh.ncmesh);
      nc.FullRefine(&msh);
      done = nc.IsAllMaxDepth();
      fes.Update();
      X.Update();
   }

   if (visualization && vis[2].good())
   {
      vis[2] << "solution" << endl << msh << X << flush;
      fflush(0);
   }
   //static int nexit = 0;
   //if (nexit++ == 8) { exit(0); }
}


// *****************************************************************************
extern "C" {
   Drl4Amr* Ctrl(int order)
   {
      return new Drl4Amr(order);
   }
   int Compute(Drl4Amr *ctrl)
   {
      return ctrl->Compute();
   }
   int Refine(Drl4Amr *ctrl, int el)
   {
      return ctrl->Refine(el);
   }
   int GetTrueVSize(Drl4Amr *ctrl)
   {
      return ctrl->GetTrueVSize();
   }
   int GetNE(Drl4Amr *ctrl)
   {
      return ctrl->GetNE();
   }
   double GetNorm(Drl4Amr *ctrl)
   {
      return ctrl->GetNorm();
   }
   void GetImage(Drl4Amr *ctrl)
   {
      return ctrl->GetImage();
   }
}
