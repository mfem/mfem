//                                MFEM DRL4AMR

#include "mfem.hpp"
#include "drl4amr.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#define dbg(...) \
   { printf("\n\033[33m"); printf(__VA_ARGS__); printf("\033[m"); fflush(0); }

enum TestFunction {
   STEPS,
   ZZFAIL,
};

static TestFunction test_func = ZZFAIL;

// *****************************************************************************

static int discs;
static double theta;
static Array<double> offsets;
constexpr int nb_discs_max = 6;
constexpr double sharpness = 100.0;

double x0_steps(const Vector &x)
{
   double result = 0.0;
   const double t = x[0] + tan(theta)*x[1];
   for (double o: offsets) { result += 1.0 + tanh(sharpness*(o - t)); }
   return result/static_cast<double>(discs<<1); // should be in [0,1]
}

double sgn(double val)
{
   double eps = 1.e-10;
   if (val < -eps) return -1.;
   if (val > +eps) return +1.;
   return 0.;
}

double x0_zzfail(const Vector &x)
{
   double f = 16;
   double twopi = 8*atan(1.);
   double w0 = sgn( sin(f*twopi*x[0]) );
   double w1 = sgn( sin(f*twopi*x[1]) );
   if (x[0] < 0.5) return w0*w1;
   return 0;
}

double x0(const Vector &x)
{
   switch (test_func) {
   case STEPS:
      return x0_steps(x);
      break;
   case ZZFAIL:
      return x0_zzfail(x);
      break;
   }
   printf("unknown function: %d\n",test_func);
   return 0;
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

   bool IsAllMaxDepth(int max_depth)
   {
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

   // If max_amr_depth =-1, use runtime GetMaxDepth as target depth
   void FullRefine(Mesh *image, const int max_amr_depth =-1)
   {
      const int max_depth = (max_amr_depth > 0) ? max_amr_depth : GetMaxDepth();
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
Drl4Amr::Drl4Amr(int o):
   order(o),
   device(device_config),
   mesh(nx, ny, type, generate_edges, sx, sy, sfc),
   dim(mesh.Dimension()),
   sdim(mesh.SpaceDimension()),
   fec(order, dim, BasisType::Positive),
   fec0(0, dim, BasisType::Positive),
   fespace(&mesh, &fec),
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
   level_no(0),
   elem_id(0)
{
   //dbg("Drl4Amr order: %d\n",o);
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

      vis[3].precision(8);
      vis[3] << "solution" << endl << mesh << x << flush;
      vis[3] << "window_title '" << "Level" << "'" << endl
             << "window_geometry "
             <<  (2 * vish + 10) << " " << 0
             << " " << visw << " " << vish << endl
             << "keys RjgA" << endl; // mn

      vis[4].precision(8);
      vis[4] << "solution" << endl << mesh << x << flush;
      vis[4] << "window_title '" << "Elem Id" << "'" << endl
             << "window_geometry "
             <<  (2 * vish + 10) << " " << 0
             << " " << visw << " " << vish << endl
             << "keys RjgA" << endl; // mn
   }

   // Make fully refined mesh to get its size, so we can allocate
   // statically sized metadata arrays.
   mesh.EnsureNCMesh();
   Mesh msh(mesh, true);
   bool done = false;
   while (!done)
   {
      NCM nc(msh.ncmesh);
      nc.FullRefine(&msh, max_depth);
      done = nc.IsAllMaxDepth(max_depth);
   }
   nefr = msh.GetNE();
   level_no = new int[nefr];
   elem_id = new int[nefr];

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
      nc.FullRefine(&msh, max_depth);
      done = nc.IsAllMaxDepth(max_depth);
      fes.Update();
      X.Update();
   }

   if (visualization && vis[2].good())
   {
      vis[2] << "solution" << endl << msh << X << flush;
      fflush(0);
   }

   image.SetSize(X.Size());
   image = X;
   Vector R(X.Size());

   //dbg("GetImageSize: %d", image.Size());
   //for (int i=0; i< image.Size(); i++) { dbg("%f", image[i]); }
   //dbg("GetImageSize: %dx%d = %d", GetImageX(), GetImageY(), GetImageSize());

   IntegrationRules irs(0, Quadrature1D::ClosedUniform);
   constexpr int ir_order = 5;
   const IntegrationRule &ir = irs.Get(Geometry::SQUARE, ir_order);
   MFEM_VERIFY(order <= ir_order, "");
   // number of the points in the integration rule
   const int nip = ir.GetNPoints();
   MFEM_VERIFY(nip == 25,"");
   //dbg("IntegrationPoints (%d):", nip);
   /*for (int k = 0; k < nip; k++)
   {
      const IntegrationPoint &ip = ir.IntPoint(k);
      dbg("x:%f y:%f", ip.x, ip.y);
   }*/
   const int NX = GetImageX();
   const int NY = GetImageY();
   //dbg("\033[37morder=%d NX=%d, NY=%d", order, NX, NY);

   for (int e=0; e < msh.GetNE(); ++e)
   {
      //dbg("\033[31mel#%d", e);
      Array<int> v;
      msh.GetElementVertices(e,v);
      MFEM_VERIFY(v.Size() == 4,"");
      const double *A = msh.GetVertex(v[0]);
      const double *B = msh.GetVertex(v[1]);
      const double *C = msh.GetVertex(v[2]);
      const double *D = msh.GetVertex(v[3]);
      Vector V;
      X.GetValues(e, ir, V);
      //dbg("V (%d):\n", V.Size()); V.Print();
      MFEM_VERIFY(order ==1 || order ==2, "");
      if (order == 1)
      {
         const int xa = static_cast<int>((A[0] / sx) * (NX-1));
         const int ya = static_cast<int>((A[1] / sy) * (NY-1));
         const int idA = xa*NY + ya;
         R[idA] = fabs(V[0]);
         //dbg("A(%d, %d)[%d] = %f", xa, ya, idA, R[idA]);

         const int xb = static_cast<int>((B[0] / sx) * (NX-1));
         const int yb = static_cast<int>((B[1] / sy) * (NY-1));
         const int idB = xb*NY + yb;
         R[idB] = fabs(V[4]);
         //dbg("B(%d, %d)[%d] = %f", xb, yb, idB, R[idB]);

         const int xc = static_cast<int>((C[0] / sx) * (NX-1));
         const int yc = static_cast<int>((C[1] / sy) * (NY-1));
         const int idC = xc*NY + yc;
         R[idC] = fabs(V[24]);
         //dbg("C(%d, %d)[%d] = %f", xc, yc, idC, R[idC]);

         const int xd = static_cast<int>((D[0] / sx) * (NX-1));
         const int yd = static_cast<int>((D[1] / sy) * (NY-1));
         const int idD = xd*NY + yd;
         R[idD] = fabs(V[20]);
         //dbg("D(%d, %d)[%d] = %f", xd, yd, idD, R[idD]);
      }
      if (order == 2)
      {
         const int xa = static_cast<int>((A[0] / sx) * (NX-1));
         const int ya = static_cast<int>((A[1] / sy) * (NY-1));
         const int idA = xa*NY + ya;
         R[idA] = fabs(V[0]);
         //dbg("A(%d, %d)[%d] = %f", xa, ya, idA, R[idA]);

         const int xe = static_cast<int>((((A[0]+B[0])/2.0)/sx) * (NX-1));
         const int ye = static_cast<int>(((A[1]+0.0) / sy) * (NY-1));
         const int idE = xe*NY + ye;
         R[idE] = fabs(V[2]);
         //dbg("E(%d, %d)[%d] = %f", xe, ye, idE, R[idE]);

         const int xb = static_cast<int>((B[0] / sx) * (NX-1));
         const int yb = static_cast<int>((B[1] / sy) * (NY-1));
         const int idB = xb*NY + yb;
         R[idB] = fabs(V[4]);
         //dbg("B(%d, %d)[%d] = %f", xb, yb, idB, R[idB]);

         const int xf = static_cast<int>((B[0] / sx) * (NX-1));
         const int yf = static_cast<int>((((B[1]+C[1])/2.0)/sy) * (NY-1));
         const int idF = xf*NY + yf;
         R[idF] = fabs(V[14]);
         //dbg("F(%d, %d)[%d] = %f", xf, yf, idF, R[idF]);

         const int xc = static_cast<int>((C[0] / sx) * (NX-1));
         const int yc = static_cast<int>((C[1] / sy) * (NY-1));
         const int idC = xc*NY + yc;
         R[idC] = fabs(V[24]);
         //dbg("C(%d, %d)[%d] = %f", xc, yc, idC, R[idC]);

         const int xg = static_cast<int>((((C[0]+D[0])/2.0)/sx) * (NX-1));
         const int yg = static_cast<int>((C[1] / sy) * (NY-1));
         const int idG = xg*NY + yg;
         R[idG] = fabs(V[22]);
         //dbg("G(%d, %d)[%d] = %f", xg, yg, idG, R[idG]);

         const int xd = static_cast<int>((D[0] / sx) * (NX-1));
         const int yd = static_cast<int>((D[1] / sy) * (NY-1));
         const int idD = xd*NY + yd;
         R[idD] = fabs(V[20]);
         //dbg("D(%d, %d)[%d] = %f", xd, yd, idD, R[idD]);

         const int xh = static_cast<int>((D[0] / sx) * (NX-1));
         const int yh = static_cast<int>((((D[1]+A[1])/2.0)/sy) * (NY-1));
         const int idH = xh*NY + yh;
         R[idH] = fabs(V[10]);
         //dbg("H(%d, %d)[%d] = %f", xh, yh, idH, R[idH]);

         const int xi = static_cast<int>((((A[0]+B[0])/2.0)/sx) * (NX-1));
         const int yi = static_cast<int>((((D[1]+A[1])/2.0)/sy) * (NY-1));
         const int idI = xi*NY + yi;
         R[idI] = fabs(V[12]);
         //dbg("I(%d, %d)[%d] = %f", xi, yi, idI, R[idI]);
      }
   }
   image = R;
   //static int nexit = 0;
   //if (nexit++ == 1) { exit(0); }
   return image.GetData();
}

// *****************************************************************************
int *Drl4Amr::GetLevelField()
{
   bool done = false;
   Mesh msh(mesh, true);

   FiniteElementSpace fes0(&msh, &fec0);
   GridFunction level(&fes0);

   Array<int> dofs;
   for (int i = 0; i < msh.GetNE(); i++) {

      const int depth = msh.ncmesh->GetElementDepth(i);
      
      fes0.GetElementDofs(i, dofs);
      for (int k = 0; k < dofs.Size(); k++) {
         level[ dofs[k] ] = depth;
      }
   }

   while (!done)
   {
      NCM nc(msh.ncmesh);
      nc.FullRefine(&msh, max_depth);
      done = nc.IsAllMaxDepth(max_depth);
      fes0.Update();
      level.Update();
   }

   if (visualization && vis[3].good())
   {
      vis[3] << "solution" << endl << msh << level << flush;
      fflush(0);
   }

   std::copy(level.GetData(), level.GetData()+msh.GetNE(), level_no);

   return level_no;
}

// *****************************************************************************
int *Drl4Amr::GetElemIdField()
{
   bool done = false;
   Mesh msh(mesh, true);

   FiniteElementSpace fes0(&msh, &fec0);
   GridFunction elem(&fes0);

   Array<int> dofs;
   for (int i = 0; i < msh.GetNE(); i++) {
      fes0.GetElementDofs(i, dofs);
      for (int k = 0; k < dofs.Size(); k++) {
         elem[ dofs[k] ] = i;
      }
   }

   while (!done)
   {
      NCM nc(msh.ncmesh);
      nc.FullRefine(&msh, max_depth);
      done = nc.IsAllMaxDepth(max_depth);
      fes0.Update();
      elem.Update();
   }

   if (visualization && vis[4].good())
   {
      vis[4] << "solution" << endl << msh << elem << flush;
      fflush(0);
   }

   std::copy(elem.GetData(), elem.GetData()+msh.GetNE(), elem_id);

   return elem_id;
}


// *****************************************************************************
int Drl4Amr::GetImageSize()
{
   MFEM_VERIFY( GetImageX() * GetImageY() == image.Size(), "");
   return GetImageX() * GetImageY();
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

   int GetNEFR(Drl4Amr *ctrl) { return ctrl->GetNEFR(); } // # elements fully refined
   int* GetLevelField(Drl4Amr *ctrl) { return ctrl->GetLevelField(); }
   int* GetElemIdField(Drl4Amr *ctrl) { return ctrl->GetElemIdField(); }
}
