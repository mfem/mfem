//                                MFEM Example 40
//
// Compile with: make ex41p
//
// Sample runs: ex41 -r 2 -eta 1e2
//              ex41 -m ../data/square-mixed.mesh -r 2 -eta 5
//              ex41 -m ../data/l-shape.mesh -r 2 -eta 1e2
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
//                      (H(u), H(v))_D + <{{n^T⋅H(u)⋅n}}, [[∇v⋅n]]>_F
//                                     + <{{n^T⋅H(v)⋅n}}, [[∇u⋅n]]>_F
//                                     + (η/h_e)<[[∇u⋅n]], [[∇v⋅n]]>_F = (f,v)_D ,
//
//               where (⋅,⋅)_D is domain integration, <⋅,⋅>_F is face
//               integration, and H(⋅) is the Hessian.
//
//              [1] Brenner, Susanne & Sung, Li-yeng. (2005). C0 Interior Penalty Methods
//                  for Fourth Order Elliptic Boundary Value Problems on Polygonal Domains.
//                  Journal of Scientific Computing. 22-23. 83-118. 10.1007/s10915-004-4135-7..

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
   const real_t eta;

   // AssembleBlock Helpers:
   mutable Vector n_b, dnshape_a, dnshape_b, nd2nshape_b, nv;

   // AssembleFaceMatrix Helpers:
   mutable Vector normal_1, normal_2;
   mutable DenseMatrix dshape_1, dshape_2, hessian_1, hessian_2, block11, block12,
           block21, block22, elmat_p;

public:
   C0InteriorPenaltyIntegrator(real_t eta_) : eta(eta_) {};

   void AssembleBlock(const DenseMatrix &dshape_a, const DenseMatrix &dshape_b,
                      const DenseMatrix &hessian_b, const Vector &n_a, const Vector &n_b, real_t h_e,
                      DenseMatrix &elmat_ab);

   void AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2,
                           FaceElementTransformations &Trans, DenseMatrix &elmat) override;
};

int main(int argc, char *argv[])
{
   // Parse command line args
   const char *mesh_file = "../data/star.mesh";
   int order = 2;
   int ref_levels = 0;
   real_t eta = 1e2;
   int max_it = 2000;
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

   real_t c, w;

   hessian.SetSize(ndof, dim * (dim + 1) / 2);
   elmat.SetSize(ndof);
   factors.SetSize(dim * (dim + 1) / 2);

   elmat = 0.0;

   const IntegrationRule *ir = GetIntegrationRule(el, Trans);
   if (ir == NULL)
   {
      int order = 2*el.GetOrder() + Trans.OrderW();
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

/** Compute: q^(a,b) + p^(a,b)

         q^(a,b) = [d\phi^(a)/dn^(a)][d^2\phi^b/dn^(a)^2], or
         q^(a,b) = [(grad \phi^a) dot n^(a)]*[n^(b)^T dot hess(\phi)^b dot n^(b)]

         and

         p^(a,b) = (eta/h_e)[d\phi^(a)/dn^(a)][d\phi^(b)/dn^(b)] or
                 = (eta/h_e)[(grad \phi^a) dot n^(a)][(grad \phi^b) dot n^(b)]
*/
void C0InteriorPenaltyIntegrator::AssembleBlock(const DenseMatrix &dshape_a,
                                                const DenseMatrix &dshape_b, const DenseMatrix &hessian_b, const Vector &n_a,
                                                const Vector &n_b, real_t h_e, DenseMatrix &elmat_ab)
{
   elmat_ab = 0.0;

   dnshape_a.SetSize(dshape_a.NumRows()); // ndofs_a
   dnshape_b.SetSize(dshape_b.NumRows()); // ndofs_b
   nd2nshape_b.SetSize(hessian_b.NumRows()); // ndofs_b
   nv.SetSize(hessian_b.NumCols());

   // dshape_a = (dof, dim)
   dshape_a.Mult(n_a, dnshape_a);

   // dshape_b = (dof, dim)
   dshape_b.Mult(n_b, dnshape_b);

   // hessian_b = (dof, [3 in 2D])
   nv[0] = pow(n_b[0],2);
   nv[1] = 2*n_b[0]*n_b[1];
   nv[2] = n_b[1]*n_b[1];

   hessian_b.Mult(nv, nd2nshape_b);

   // Consistency term:
   AddMult_a_VWt(1.0, dnshape_a, nd2nshape_b, elmat_ab);

   // Penalty term (symmetric):
   AddMult_a_VWt(eta/h_e, dnshape_a, dnshape_b, elmat_ab);
}

void C0InteriorPenaltyIntegrator::AssembleFaceMatrix(const FiniteElement &el1,
                                                     const FiniteElement &el2, FaceElementTransformations &Trans, DenseMatrix &elmat)
{
   int dim = el1.GetDim();
   int ndof1 = el1.GetDof();
   int ndof2 = 0;
   MFEM_ASSERT(dim == 2, "Dimension must be 2.");

   // For boundary face integration:
   if (Trans.Elem2No >= 0)
   {
      ndof2 = el2.GetDof();
   }

   normal_1.SetSize(dim);
   dshape_1.SetSize(ndof1, dim);
   hessian_1.SetSize(ndof1, dim * (dim + 1) / 2);
   block11.SetSize(ndof1, ndof1);
   if (ndof2 > 0)
   {
      dshape_2.SetSize(ndof2, dim);
      normal_2.SetSize(dim);
      hessian_2.SetSize(ndof2, dim * (dim + 1) / 2);
      block12.SetSize(ndof1, ndof2);
      block21.SetSize(ndof2, ndof1);
      block22.SetSize(ndof2, ndof2);
   }
   elmat.SetSize(ndof1 + ndof2);
   elmat_p.SetSize(ndof1 + ndof2);
   elmat = 0.0;

   const IntegrationRule *ir = IntRule;
   if (ir == NULL)
   {
      int order = 2 * max(el1.GetOrder(), ndof2 ? el2.GetOrder() : 0);
      ir = &IntRules.Get(el1.GetGeomType(), order);
   }


   // Compute edge length
   real_t h_e  = 0.0;
   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);
      Trans.SetAllIntPoints(&ip);
      h_e += ip.weight * Trans.Face->Weight();
   }

   for (int p = 0; p < ir->GetNPoints(); p++)
   {
      const IntegrationPoint &ip = ir->IntPoint(p);

      // Set the integration point in the face and the neighboring elements
      Trans.SetAllIntPoints(&ip);

      // Compute normals, derivatives, and hessians
      CalcOrtho(Trans.Face->Jacobian(), normal_1);
      normal_1 /= normal_1.Norml2();
      el1.CalcPhysDShape(*Trans.Elem1, dshape_1);
      el1.CalcPhysHessian(*Trans.Elem1, hessian_1);
      if (ndof2)
      {
         normal_2 = normal_1;
         normal_2 *= -1.0;
         el2.CalcPhysDShape(*Trans.Elem2, dshape_2);
         el2.CalcPhysHessian(*Trans.Elem2, hessian_2);
      }

      // (1,1) block
      AssembleBlock(dshape_1, dshape_1, hessian_1, normal_1, normal_1, h_e, block11);
      elmat_p.SetSubMatrix(0, 0, block11);
      if (ndof2 > 0)
      {
         // (1,2) block
         AssembleBlock(dshape_1, dshape_2, hessian_2, normal_1, normal_2, h_e, block12);
         elmat_p.SetSubMatrix(0, ndof1, block12);

         // (2,1) block
         AssembleBlock(dshape_2, dshape_1, hessian_1, normal_2, normal_1, h_e, block21);
         elmat_p.SetSubMatrix(ndof1, 0, block21);

         // (2,2) block
         AssembleBlock(dshape_2, dshape_2, hessian_2, normal_2, normal_2, h_e, block22);
         elmat_p.SetSubMatrix(ndof1, ndof1, block22);
      }

      // Apply 1/2 factor and symmetry term
      elmat_p.Symmetrize();

      elmat_p *= ip.weight * Trans.Face->Weight();

      elmat += elmat_p;
   }
}