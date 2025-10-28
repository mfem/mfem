//                                MFEM Example 41
//
// Compile with: make ex41p
//
// Sample runs: ex41 -r 3
//              ex41 -m ../data/hexagon.mesh -r 3 -o 3
//              ex41 -m ../data/square-mixed.mesh -r 2 -eta 5
//              ex41 -m ../data/l-shape.mesh -r 3
//
// Description: This example solves the clamped biharmonic equation,
//
//                      ∇⁴u = f in Ω,  u = 0 and ∇u⋅n = 0 on ∂Ω,
//
//              in 2D using just H¹-conforming finite elements by employing the interior penalty
//              method outlined in [1]. This example demonstrates an approach to solving higher-order
//              PDEs in MFEM and implementation of custom domain and face integrators to solve the
//              weak form
//
//                      (H(u), H(v))_D - <{{n^T⋅H(u)⋅n}}, [[∇v⋅n]]>_F
//                                     - <{{n^T⋅H(v)⋅n}}, [[∇u⋅n]]>_F
//                                     + (η/h_e)<[[∇u⋅n]], [[∇v⋅n]]>_F = (f,v)_D ,
//
//               where (⋅,⋅)_D is domain integration, <⋅,⋅>_F is face
//               integration, and H(⋅) is the Hessian.
//
//              [1] Brenner, Susanne & Sung, Li-yeng. (2005). C0 Interior Penalty Methods
//                  for Fourth Order Elliptic Boundary Value Problems on Polygonal Domains.
//                  Journal of Scientific Computing. 22-23. 83-118. 10.1007/s10915-004-4135-7.

#include <mfem.hpp>

using namespace mfem;
using namespace std;

class BiharmonicIntegrator : public BilinearFormIntegrator
{
private:
   Coefficient &D;

   inline static const Vector factors_2D{1.0, 2.0, 1.0};
   mutable DenseMatrix hessian;
   mutable Vector factors;
public:
   BiharmonicIntegrator(Coefficient &D_) : D(D_) {}

   void AssembleElementMatrix(const FiniteElement &el,
                              ElementTransformation &Trans, DenseMatrix &elmat) override;
};

class C0InteriorPenaltyIntegrator : public BilinearFormIntegrator
{
private:
   const double eta;

   mutable Vector normal[2], dnshape[2], nv[2], nd2nshape[2];
   mutable DenseMatrix dshape[2], hessian[2], blockJ[2][2], blockC[2][2], elmatJ_p,
           elmatC_p;
public:
   C0InteriorPenaltyIntegrator(double eta_) : eta(eta_) {};

   void AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2,
                           FaceElementTransformations &Trans, DenseMatrix &elmat) override;
};

int main(int argc, char *argv[])
{
   // Parse command line args
   const char *mesh_file = "../data/star.mesh";
   int order = 2;
   int ref_levels = 0;
   real_t eta = 10;
   int max_it = 10000;
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&ref_levels, "-r", "--refs",
                  "Number of h-refinements.");
   args.AddOption(&eta, "-eta", "--penalty-coeff",
                  "Penalty coefficient.");
   args.AddOption(&max_it, "-mi", "--max-it",
                  "Maximum number of iterations");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // Read the mesh file
   Mesh mesh(mesh_file);
   mesh.SetCurvature(order); // ensure CalcPhys_ are of high-order
   int dim = mesh.Dimension();
   MFEM_ASSERT(dim == 2, "This example only supports 2D meshes.");

   // Refine the mesh
   for (int i = 0; i < ref_levels; i++)
   {
      mesh.UniformRefinement();
   }

   // Initialize the FE collection and FiniteElementSpace
   H1_FECollection fe_coll(order, dim);
   FiniteElementSpace fespace(&mesh, &fe_coll, 1);

   // Get the degrees-of-freedom (DOFs) associated with the sides of the panel
   Array<int> all_bdr_marker(mesh.bdr_attributes.Size());
   all_bdr_marker = 1; // Mark all sides
   Array<int> ess_tdof_list;
   fespace.GetEssentialTrueDofs(all_bdr_marker, ess_tdof_list);

   ConstantCoefficient one(1.0);

   // Initialize the bilinear form
   BilinearForm a(&fespace);
   a.AddDomainIntegrator(new BiharmonicIntegrator(one));
   a.AddInteriorFaceIntegrator(new C0InteriorPenaltyIntegrator(eta));
   a.AddBdrFaceIntegrator(new C0InteriorPenaltyIntegrator(eta));
   a.Assemble();

   // Initialize the linear form f=1.0
   LinearForm b(&fespace);
   b.AddDomainIntegrator(new DomainLFIntegrator(one));
   b.Assemble();

   // Form the linear system
   GridFunction x(&fespace);
   x = 0.0; // initial guess
   SparseMatrix A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);

   // Solve the system using CG with symmetric Gauss-Seidel preconditioner
   GSSmoother M(A);
   PCG(A, M, B, X, 1, max_it, 1e-12, 0.0);

   // Recover solution and visualize
   a.RecoverFEMSolution(X, B, x);

   if (visualization)
   {
      char vishost[] = "localhost";
      int visport    = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x << flush;
   }

   return 0;
}


void BiharmonicIntegrator::AssembleElementMatrix(const FiniteElement &el,
                                                 ElementTransformation &Trans, DenseMatrix &elmat)
{
   int ndof = el.GetDof();
   int dim = el.GetDim();

   MFEM_ASSERT(dim == 2, "Dimension must be 2.");

   hessian.SetSize(ndof, dim * (dim + 1) / 2);
   elmat.SetSize(ndof);
   factors.SetSize(dim * (dim + 1) / 2);

   elmat = 0.0;

   const IntegrationRule *ir = GetIntegrationRule(el, Trans);
   if (ir == NULL)
   {
      int order = 2*el.GetOrder();
      ir = &IntRules.Get(el.GetGeomType(), order);
   }

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const mfem::IntegrationPoint &ip = ir->IntPoint(i);
      Trans.SetIntPoint(&ip);

      el.CalcPhysHessian(Trans, hessian);

      factors = factors_2D;
      factors *= D.Eval(Trans, ip) * ip.weight * Trans.Weight();

      AddMultADAt(hessian, factors, elmat);
   }
}

void C0InteriorPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &el1,
                                                     const FiniteElement &el2, FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int dim = el1.GetDim();
   MFEM_ASSERT(dim == 2, "Dimension must be 2.");

   int ndof[2] = {el1.GetDof(), 0};
   int num_elems = 1;
   if (Trans.Elem2No >= 0)
   {
      ndof[1] = el2.GetDof();
      num_elems++;
   }

   for (int i = 0; i < num_elems; i++)
   {
      normal[i].SetSize(dim);
      dshape[i].SetSize(ndof[i], dim);
      hessian[i].SetSize(ndof[i], dim * (dim + 1) / 2);
      nv[i].SetSize(dim * (dim + 1) / 2);
      dnshape[i].SetSize(ndof[i]);
      nd2nshape[i].SetSize(ndof[i]);
   }

   for (int i = 0; i < num_elems; i++)
   {
      for (int j = 0; j < num_elems; j++)
      {
         blockJ[i][j].SetSize(ndof[i], ndof[j]);
         blockC[i][j].SetSize(ndof[i], ndof[j]);
      }
   }

   elmatJ_p.SetSize(ndof[0] + ndof[1]);
   elmatC_p.SetSize(ndof[0] + ndof[1]);
   elmat.SetSize(ndof[0] + ndof[1]);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * max(el1.GetOrder(), ndof[1] ? el2.GetOrder() : 0);
      ir = &IntRules.Get(Trans.GetGeometryType(), order);
   }

   // Compute edge length
   double h_e  = 0.0;
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      Trans.SetAllIntPoints(&ip);
      h_e += ip.weight * Trans.Weight();
   }

   const FiniteElement *els[2] = {&el1, &el2};
   ElementTransformation *el_trans[2] = {Trans.Elem1, Trans.Elem2};

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      elmatJ_p = 0.0;
      elmatC_p = 0.0;

      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Compute normal gradients + Hessians
      for (int i = 0; i < num_elems; i++)
      {
         if (i == 0)
         {
            CalcOrtho(Trans.Jacobian(), normal[i]);
            normal[i] /= normal[i].Norml2();
         }
         else
         {
            normal[i] = normal[0];
            normal[i] *= -1;
         }
         els[i]->CalcPhysDShape(*el_trans[i], dshape[i]);
         els[i]->CalcPhysHessian(*el_trans[i], hessian[i]);
         dshape[i].Mult(normal[i], dnshape[i]);
         nv[i][0] = normal[i][0]*normal[i][0];
         nv[i][1] = 2*normal[i][0]*normal[i][1];
         nv[i][2] = normal[i][1]*normal[i][1];
         hessian[i].Mult(nv[i], nd2nshape[i]);
      }

      // Compute blocks
      for (int i = 0; i < num_elems; i++)
      {
         for (int j = 0; j < num_elems; j++)
         {
            blockJ[i][j] = 0.0;
            blockC[i][j] = 0.0;
            AddMult_a_VWt(-1.0, dnshape[i], nd2nshape[j], blockJ[i][j]);
            elmatJ_p.SetSubMatrix(i*ndof[0], j*ndof[0], blockJ[i][j]);

            AddMult_a_VWt(eta/h_e, dnshape[i], dnshape[j], blockC[i][j]);
            elmatC_p.SetSubMatrix(i*ndof[0], j*ndof[0], blockC[i][j]);
         }
      }

      // Symmetrize the jump term
      elmatJ_p.Symmetrize();
      if (!ndof[1])
      {
         elmatJ_p *= 2;
      }

      // Add penalty term
      elmatJ_p += elmatC_p;
      elmatJ_p *= ip.weight * Trans.Weight();
      elmat += elmatJ_p;
   }
}
