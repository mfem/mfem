//                                MFEM DRL4AMR

#include "mfem.hpp"
#include "drl4amr.hpp"

#include "linalg/dtensor.hpp"

#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

#define dbg(...) \
   { printf("\n\033[33m"); printf(__VA_ARGS__); printf("\033[m"); fflush(0); }

const char *device_config = "cpu";

static int discs;
static double theta;
static Array<double> offsets;
constexpr int nb_discs_max = 6;
constexpr double sharpness = 100.0;

const bool visualization = true;
const char *vishost = "localhost";
const int visport = 19916;
const int visw = 480;
const int vish = 480;

// *****************************************************************************
double x0(const Vector &x)
{
   double result = 0.0;
   const double t = x[0] + tan(theta)*x[1];
   for (double o : offsets)
   {
      result += 1.0 + tanh(sharpness*(o - t));
   }
   return result / (discs << 1); // should be in [0,1]
}

// *****************************************************************************
Drl4Amr::Drl4Amr(int order):
   mesh(nx, ny, elem_type, true, sx, sy, false),
   order(order),
   dim(mesh.Dimension()),
   device(device_config),
   fec(order, dim, BasisType::Positive),
   fespace(&mesh, &fec),
   one(1.0),
   zero(0.0),
   integ(new DiffusionIntegrator(one)),
   xcoeff(x0),
   x(&fespace),
   iteration(0),
   flux_fespace(&mesh, &fec),
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
//      vis[1].open(vishost, visport);
//      vis[2].open(vishost, visport);
   }

   // Initialize theta, offsets and x from x0_coeff
   srand48(4);//time(NULL));
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
   //estimator.SetAnisotropic();

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
double *Drl4Amr::GetFullImage()
{
   Array<int> vert;
   Vector sln;

   int width = GetFullWidth();
   int height = GetFullHeight();

   image.SetSize(width * height);

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
         int m = (oy + i)*width + (ox + j);

         image(m) = sln(n);
      }
   }

   if (visualization) { ShowFullImage(); }

   return image.GetData();
}

// *****************************************************************************

static void GetFaceGradients(const GridFunction &gf,
                            int face, int side, const IntegrationRule &ir,
                            DenseMatrix &grad/*, DenseMatrix &tr*/)
{
   IntegrationRule tmp(ir.GetNPoints());
   FaceElementTransformations *Transf;
   Mesh *mesh = gf.FESpace()->GetMesh();

   if (side == 0)
   {
      Transf = mesh->GetFaceElementTransformations(face, 4);
      Transf->Loc1.Transform(ir, tmp);
      gf.GetGradients(Transf->Elem1No, tmp, grad);
   }
   else
   {
      Transf = mesh->GetFaceElementTransformations(face, 8);
      Transf->Loc2.Transform(ir, tmp);
      gf.GetGradients(Transf->Elem2No, tmp, grad);
   }
}


double* Drl4Amr::GetLocalImage(int element)
{
   NCMesh *ncmesh = mesh.ncmesh;
   MFEM_ASSERT(ncmesh, "");

   int width = GetLocalWidth();
   int height = GetLocalHeight();
   int subdiv = oversample*order;

   local_image.SetSize(width*height*3);
   local_image = 0.0;

   auto im = Reshape(local_image.Write(false), width, height, 3);

   Vector sln;
   DenseMatrix grad;

   // rasterize element interior
   {
      const IntegrationRule &ir =
         GlobGeometryRefiner.Refine(Geometry::SQUARE, subdiv)->RefPts;

      x.GetValues(element, ir, sln);
      x.GetGradients(element, ir, grad);

      for (int i = 0; i <= subdiv; i++)
      for (int j = 0; j <= subdiv; j++)
      {
         int k = i*(subdiv+1) + j;

         im(j+1, i+1, 0) = sln(k);
         im(j+1, i+1, 1) = grad(0, k);
         im(j+1, i+1, 2) = grad(1, k);
      }
   }

   // rasterize neighbor edges
   {
      Array<int> edges, ori;
      mesh.GetElementEdges(element, edges, ori);

      IsoparametricTransformation T;
      T.SetFE(&SegmentFE);

      const IntegrationRule &ir =
         GlobGeometryRefiner.Refine(Geometry::SEGMENT, subdiv)->RefPts;

      for (int e = 0; e < edges.Size(); e++)
      {
         int type;
         const NCMesh::NCList &elist = ncmesh->GetEdgeList();
         const NCMesh::MeshId &eid = elist.LookUp(edges[e], &type);

         int side = 1;
         if (type == 0 || type == 2) // conforming or slave face
         {
            int el1, el2;
            mesh.GetFaceElements(eid.index, &el1, &el2);
            side = (element == el1 && el2 >= 0) ? 1 : 0;

            DenseMatrix tr;
            x.GetFaceValues(eid.index, side, ir, sln, tr);
            GetFaceGradients(x, eid.index, side, ir, grad);
         }
         else if (type == 1) // master
         {
            sln = 0.0;
            grad = 0.0;
         }
         else
         {
            MFEM_ABORT("invalid edge type");
         }

         const int edges[4][2][4] =
         {
            { { 1, 0, 1, 0 }, { width-2, 0, -1, 0 } },
            { { width-1, 1, 0, 1 }, { width-1, height-2, 0, -1 } },
            { { width-2, height-1, -1, 0 }, { 1, height-1, 1, 0 } },
            { { 0, height-2, 0, -1 }, { 0, 1, 0, 1 } }
         };

         // store edge values
         int o = side ? 0 : 1;
         int x = edges[e][o][0];
         int y = edges[e][o][1];

         for (int i = 0; i <= subdiv; i++)
         {
            im(x, y, 0) = sln(i);
            im(x, y, 1) = grad(0, i);
            im(x, y, 2) = grad(1, i);

            x += edges[e][o][2];
            y += edges[e][o][3];
         }

         cout << "edge " << e << ", type = " << type << ", o = " << o << endl;
      }
   }
   return local_image.GetData();
}

// *****************************************************************************
void Drl4Amr::ShowFullImage()
{
   if (!vis[2].good()) { return; }

   Mesh imesh(GetFullWidth(), GetFullHeight(), elem_type, false, sx, sy, false);

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

double *GetFullImage(Drl4Amr *ctrl) { return ctrl->GetFullImage(); }
int GetFullWidth(Drl4Amr *ctrl) { return ctrl->GetFullWidth(); }
int GetFullHeight(Drl4Amr *ctrl) { return ctrl->GetFullHeight(); }

} // extern "C"
