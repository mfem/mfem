#include "../../mfem.hpp"
#include "../common/mfem-common.hpp"
#include <fstream>
#include <iostream>
#include "mesh-fitting.hpp"
//make p-adaptivity -j && ./p-adaptivity -slstype 4 -rs 0 -hiter 3 -o 5 -n 0 -dist
// make p-adaptivity -j && ./p-adaptivity -slstype 5 -rs 2 -hiter 0 -o 5 -n 4
// make p-adaptivity -j && ./p-adaptivity -slstype 6 -rs 3 -hiter 0 -o 5 -n 0 -ndiff 1
// make p-adaptivity -j && ././p-adaptivity -m apollo_initial_tri.mesh -slstype 7 -rs 4 -hiter 2 -o 2 -ndiff 1

using namespace mfem;
using namespace std;


//rectangle at center xc,yc with length lx and width ly
double r_rectangle(const Vector &x, Vector &x_center, Vector &lengths)
{
   double xc = x_center(0);
   double yc = x_center(1);
   double xv = x(0),
          yv = x(1);
   double lx = lengths(0);
   double ly = lengths(1);
   double phi_vertical = std::pow(lx*0.5, 2.0) - std::pow(xv-xc, 2.0);
   double phi_horizontal = std::pow(ly*0.5, 2.0) - std::pow(yv-yc, 2.0);

   double phi_rectangle = r_intersect(phi_vertical, phi_horizontal);
   return phi_rectangle;
}

double rectangle(const Vector &x)
{
   Vector xc(x.Size());
   xc = 0.5;
   Vector lengths(x.Size());
   lengths(0) = 0.2;
   lengths(1) = 0.4;
   return r_rectangle(x, xc, lengths);
}

double rectangle_and_circle(const Vector &x)
{
   int op = 1;
   Vector xc(x.Size());
   xc = 0.5;
   Vector lengths(x.Size());
   lengths(0) = 0.3;
   lengths(1) = 0.8;

   return op == 0 ? r_intersect(r_rectangle(x, xc, lengths), -r_circle(x, xc,
                                                                       0.3)) :
          r_union(r_rectangle(x, xc, lengths), -r_circle(x, xc, 0.3));
}

class PRefinementTransfer
{
private:
   FiniteElementSpace *src;

public:
   /// @brief Constructs a transfer operator from \p lFESpace to \p hFESpace
   /// which have different FE collections.
   /** No matrices are assembled, only the action to a vector is being computed.
       The underlying finite elements need to implement the GetTransferMatrix
       methods. */
   PRefinementTransfer(const FiniteElementSpace& src_);

   /// Destructor
   ~PRefinementTransfer();

   /// Update source FiniteElementSpace used to construct the
   /// PRefinementTransfer operator.
   void SetSourceFESpace(const FiniteElementSpace& src_);

   /// @brief Interpolation or prolongation of a vector \p x corresponding to
   /// the coarse space to the vector \p y corresponding to the fine space.
   void Transfer(GridFunction &targf);
};

PRefinementTransfer::PRefinementTransfer(const FiniteElementSpace &src_)
{
   src = new FiniteElementSpace(src_);
}

PRefinementTransfer::~PRefinementTransfer()
{
   delete src;
}

void PRefinementTransfer::SetSourceFESpace(const FiniteElementSpace &src_)
{
   if (src) { delete src; }
   src = new FiniteElementSpace(src_);
}

void PRefinementTransfer::Transfer(GridFunction &targf)
{
   MFEM_VERIFY(targf.GetSequence() != targf.FESpace()->GetSequence(),
               "GridFunction should not be updated prior to UpdateGF.");
   Vector srcgf = targf;
   targf.Update();
   PRefinementTransferOperator preft =
      PRefinementTransferOperator(*src, *(targf.FESpace()));
   preft.Mult(srcgf, targf);
}

GridFunction* ProlongToMaxOrder(const GridFunction *x, const int fieldtype)
{
   const FiniteElementSpace *fespace = x->FESpace();
   Mesh *mesh = fespace->GetMesh();
   const FiniteElementCollection *fec = fespace->FEColl();
   const int vdim = fespace->GetVDim();

   // find the max order in the space
   int max_order = fespace->GetMaxElementOrder();

   // create a visualization space of max order for all elements
   FiniteElementCollection *fecInt = NULL;
   if (fieldtype == 0)
   {
      fecInt = new H1_FECollection(max_order, mesh->Dimension());
   }
   else if (fieldtype == 1)
   {
      fecInt = new L2_FECollection(max_order, mesh->Dimension());
   }
   FiniteElementSpace *spaceInt = new FiniteElementSpace(mesh, fecInt,
                                                         fespace->GetVDim(),
                                                         fespace->GetOrdering());

   IsoparametricTransformation T;
   DenseMatrix I;

   GridFunction *xInt = new GridFunction(spaceInt);

   // interpolate solution vector in the larger space
   for (int i = 0; i < mesh->GetNE(); i++)
   {
      Geometry::Type geom = mesh->GetElementGeometry(i);
      T.SetIdentityTransformation(geom);

      Array<int> dofs;
      fespace->GetElementVDofs(i, dofs);
      Vector elemvect(0), vectInt(0);
      x->GetSubVector(dofs, elemvect);
      DenseMatrix elemvecMat(elemvect.GetData(), dofs.Size()/vdim, vdim);

      const auto *fe = fec->GetFE(geom, fespace->GetElementOrder(i));
      const auto *feInt = fecInt->GetFE(geom, max_order);

      feInt->GetTransferMatrix(*fe, T, I);

      spaceInt->GetElementVDofs(i, dofs);
      vectInt.SetSize(dofs.Size());
      DenseMatrix vectIntMat(vectInt.GetData(), dofs.Size()/vdim, vdim);

      //      I.Mult(elemvecMat, vectIntMat);
      Mult(I, elemvecMat, vectIntMat);
      xInt->SetSubVector(dofs, vectInt);
   }

   xInt->MakeOwner(fecInt);
   return xInt;
}

void VisualizeFESpacePolynomialOrder(FiniteElementSpace &fespace,
                                     const char *title)
{
   Mesh *mesh = fespace.GetMesh();
   L2_FECollection order_coll = L2_FECollection(0, mesh->Dimension());
   FiniteElementSpace order_space = FiniteElementSpace(mesh, &order_coll);
   GridFunction order_gf = GridFunction(&order_space);

   for (int e = 0; e < mesh->GetNE(); e++)
   {
      order_gf(e) = fespace.GetElementOrder(e);
   }

   socketstream vis1;
   common::VisualizeField(vis1, "localhost", 19916, order_gf, title,
                          400, 0, 400, 400, "RjmAcp");
}

int main(int argc, char *argv[])
{
   // 0. Set the method's default parameters.
   const char *mesh_file = "square01.mesh";
   int init_order        = 4;
   int rs_levels         = 1;
   int surf_ls_type      = 3;
   bool comp_dist        = false;
   int estimator_type    = 0;
   bool visualization    = true;
   int h_iters           = 0;
   bool normalization    = false;
   int pmin              = 1;
   int pmax              = 8;
   double e_upper        = 1e-05;
   double e_lower        = 1e-13;
   int n_iters           = 10;
   int ndiff             = 0;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&init_order, "-o", "--order",
                  "Polynomial degree of mesh finite element space.");
   args.AddOption(&rs_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&surf_ls_type, "-slstype", "--surf-ls-type",
                  "1 - Circle (DEFAULT), 2 - Squircle");
   args.AddOption(&comp_dist, "-dist", "--comp-dist",
                  "-no-dist","--no-comp-dist",
                  "Compute distance from 0 level set or not.");
   args.AddOption(&estimator_type, "-est", "--est-type",
                  "0 - PRefDiff.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&h_iters, "-hiter", "--h-iter",
                  "Number of amr iterations on background mesh");
   args.AddOption(&normalization, "-nor", "--normalization", "-no-nor",
                  "--no-normalization",
                  "Make all terms in the optimization functional unitless.");
   args.AddOption(&pmax, "-pmax", "--pmax",
                  "Upper limit on polynomial order.");
   args.AddOption(&pmin, "-pmin", "--pmin",
                  "Lower limit on polynomial order.");
   args.AddOption(&e_upper, "-eu", "--eu",
                  "Refine until error reaches this limit.");
   args.AddOption(&e_lower, "-el", "--el",
                  "Derefine until error reaches this limit.");
   args.AddOption(&n_iters, "-n", "--niters",
                  "Number of iterations.");
   args.AddOption(&ndiff, "-ndiff", "--ndiff",
                  "Number of diffusion iterations.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 2. Initialize and refine the starting mesh.
   Mesh mesh(mesh_file, 1, 1, false);
   for (int lev = 0; lev < rs_levels; lev++) { mesh.UniformRefinement(); }
   const int dim = mesh.Dimension();
   mesh.EnsureNCMesh(true);

   H1_FECollection fec(init_order, dim);
   FiniteElementSpace fespace(&mesh, &fec);
   GridFunction x(&fespace);

   L2_FECollection fecl2(0, dim);
   FiniteElementSpace fespacel2(&mesh, &fecl2);
   GridFunction elgf(&fespacel2);

   FunctionCoefficient *ls_coeff = NULL;
   if (surf_ls_type == 1) //Circle
   {
      ls_coeff = new FunctionCoefficient(circle_level_set);
   }
   else if (surf_ls_type == 2) // reactor
   {
      ls_coeff = new FunctionCoefficient(reactor);
      comp_dist = true;
      std::cout << "Forcing comp dist to true for reactor level-set\n";
   }
   else if (surf_ls_type == 3) // squircle
   {
      ls_coeff = new FunctionCoefficient(squircle_level_set);
   }
   else if (surf_ls_type == 4) // inclined_line
   {
      ls_coeff = new FunctionCoefficient(inclined_line);
   }
   else if (surf_ls_type == 5) // inclined_line
   {
      ls_coeff = new FunctionCoefficient(rectangle);
   }
   else if (surf_ls_type == 6) // inclined_line
   {
      ls_coeff = new FunctionCoefficient(rectangle_and_circle);
   }
   else if (surf_ls_type == 7) // inclined_line
   {
      ls_coeff = new FunctionCoefficient(apollo_level_set);
   }
   else
   {
      MFEM_ABORT("Surface fitting level set type not implemented yet.")
   }

   if (surf_ls_type == 7)
   {
      mesh.SetCurvature(1);
      Vector p_min(dim), p_max(dim);
      mesh.GetBoundingBox(p_min, p_max);

      GridFunction &x_bg = *(mesh.GetNodes());
      GridFunction dx(x_bg);
      const int num_nodes = x_bg.Size() / dim;
      for (int i = 0; i < num_nodes; i++)
      {
         for (int d = 0; d < dim; d++)
         {
            if (d == 0)
            {
               //                dx(i + d*num_nodes) = 0.5;
               dx(i*dim + d) = 0.5;
            }
            else if (d == 1)
            {
               //                  dx(i + d*num_nodes) = 2.5;
               dx(i*dim + d) = 2.5;
            }
         }
      }
      x_bg -= dx;
   }

   {
      ofstream mesh_ofs("apollo_input_mesh_tri.mesh");
      mesh.Print(mesh_ofs);
   }



   x.ProjectCoefficient(*ls_coeff);

   //   double dx = AvgElementSize(mesh);
   //   PDEFilter filter(mesh, dx);
   //   GridFunctionCoefficient gfc(x);
   //   ParGridFunction x2(x);
   //   filter.Filter(gfc, x2);
   //   DiffuseField(x, 20);



   x.ProjectCoefficient(*ls_coeff);

   // Do AMR iterations
   if (h_iters > 0)
   {
      OptimizeMeshWithAMRAroundZeroLevelSet(mesh, *ls_coeff,
                                            h_iters, x);
      x.ProjectCoefficient(*ls_coeff);
   }
   if (ndiff > 0)
   {
      DiffuseField(x, ndiff);
   }
   else if (ndiff < 0)
   {
      double dx = AvgElementSize(mesh);
      PDEFilter filter(mesh, dx);
      GridFunctionCoefficient gfc(&x);
      GridFunction x2(x);
      filter.Filter(gfc, x2);
      x = x2;
   }

   if (visualization)
   {
      socketstream vis1;
      common::VisualizeField(vis1, "localhost", 19916, x,
                             "Pre-distance Level-set function",
                             00, 0, 700, 600, "Rjmc");
   }

   if (comp_dist)
   {
      if (visualization)
      {
         socketstream vis1;
         common::VisualizeField(vis1, "localhost", 19916, x,
                                "Pre-distance Level-set function",
                                00, 0, 700, 600, "Rjmc");
      }
      ComputeScalarDistanceFromLevelSet(mesh, *ls_coeff, x, 0, 5, 500, false, 1);
   }



   if (visualization)
   {
      socketstream vis1;
      common::VisualizeField(vis1, "localhost", 19916, x, "Level-set function",
                             00, 0, 700, 600, "Rjmc");
   }
   MFEM_ABORT(" ");

   {
      ofstream mesh_ofs("apollo_amr.mesh");
      mesh.Print(mesh_ofs);
   }
   ofstream gf_ofs("apollo_dist.gf");
   x.Save(gf_ofs);

   MFEM_ABORT(" ");

   // Now do p-adaptivity itrations
   PRefDiffEstimator prdiff(x, -1, normalization);
   Vector error_estimates = prdiff.GetLocalErrors();
   elgf = error_estimates;

   if (visualization)
   {
      socketstream vis1;
      common::VisualizeField(vis1, "localhost", 19916, elgf, "Error estimate",
                             700, 00, 700, 600, "RjmcL");
      VisualizeFESpacePolynomialOrder(fespace,"Polynomial order");
   }


   int fespace_order_max = fespace.GetMaxElementOrder();
   int fespace_order_min = fespace.GetMinElementOrder();
   int count = 1;

   for (int iter = 0; iter < n_iters &&
        fespace_order_max < pmax &&
        fespace_order_min > pmin &&
        count > 0; iter++)
   {
      error_estimates = prdiff.GetLocalErrors();
      count = 0;

      PRefinementTransfer preft_fespace = PRefinementTransfer(fespace);
      for (int e = 0; e < mesh.GetNE(); e++)
      {

         double elem_error = error_estimates(e);
         double elem_order = fespace.GetElementOrder(e);
         //           if (elem_error > e_upper && elem_order < pmax) {
         //               fespace.SetElementOrder(e, elem_order+1);
         //               count++;
         //           }
         if (elem_error < e_lower && elem_order > pmin)
         {
            fespace.SetElementOrder(e, elem_order-1);
            count++;
         }
      }

      if (count == 0) { break; }
      fespace.Update(false);
      preft_fespace.Transfer(x);
      std::cout << count << " elements were p-refined in this iteration\n";
      fespace_order_max = fespace.GetMaxElementOrder();
      fespace_order_min = fespace.GetMinElementOrder();
   }

   if (visualization)
   {
      socketstream vis1;
      common::VisualizeField(vis1, "localhost", 19916, x, "Solution",
                             700, 00, 700, 600, "RjmcL");
      VisualizeFESpacePolynomialOrder(fespace,"Polynomial order");
   }

   error_estimates = prdiff.GetLocalErrors();
   elgf = error_estimates;

   if (visualization)
   {
      socketstream vis1;
      common::VisualizeField(vis1, "localhost", 19916, elgf, "Error estimate",
                             700, 00, 700, 600, "RjmcL");
   }



}
