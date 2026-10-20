#include "mfem.hpp"
#include "mg_agglom.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>


using namespace std;
using namespace mfem;


class AcousticUpwindFaceIntegrator : public BilinearFormIntegrator
{
private:
   const double K0, rho0, c;
   Vector shape1, shape2, nor, nhat;
   Vector pjump, pavg, unjump, unavg;

public:
   AcousticUpwindFaceIntegrator(double K0_, double rho0_)
      : K0(K0_), rho0(rho0_), c(std::sqrt(K0_/rho0_)) {}

   using BilinearFormIntegrator::AssembleFaceMatrix;

   void AssembleFaceMatrix(const FiniteElement &el1,
                           const FiniteElement &el2,
                           FaceElementTransformations &Trans,
                           DenseMatrix &elmat) override
   {
      const int dim = el1.GetDim();
      const int nc  = dim + 1;                       // p, u_0, ..., u_{dim-1}
      const int nd1 = el1.GetDof();
      const bool interior = (Trans.Elem2No >= 0);
      const int nd2 = interior ? el2.GetDof() : 0;
      const int ndofs = nc*(nd1 + nd2);

      elmat.SetSize(ndofs); elmat = 0.0;
      shape1.SetSize(nd1); shape2.SetSize(nd2);
      nor.SetSize(dim); nhat.SetSize(dim);
      pjump.SetSize(ndofs); pavg.SetSize(ndofs);
      unjump.SetSize(ndofs); unavg.SetSize(ndofs);

      // local index of dof k of component comp on side s (0 = el1, 1 = el2)
      auto idx = [&](int s, int comp, int k)
      { return (s == 0) ? comp*nd1 + k : nc*nd1 + comp*nd2 + k; };

      const IntegrationRule *ir = IntRule;
      if (!ir)
      {
         const int p = interior ? std::max(el1.GetOrder(), el2.GetOrder())
                                : el1.GetOrder();
         ir = &IntRules.Get(Trans.GetGeometryType(), 2*p + 2);
      }

      for (int q = 0; q < ir->GetNPoints(); q++)
      {
         const IntegrationPoint &ip = ir->IntPoint(q);
         Trans.SetAllIntPoints(&ip);

         el1.CalcShape(Trans.GetElement1IntPoint(), shape1);
         if (interior) { el2.CalcShape(Trans.GetElement2IntPoint(), shape2); }

         if (dim == 1) { nor(0) = 2.0*Trans.GetElement1IntPoint().x - 1.0; }
         else          { CalcOrtho(Trans.Jacobian(), nor); }
         const double area = nor.Norml2();           // face Jacobian determinant
         nhat.Set(1.0/area, nor);                    // unit normal out of el1
         const double w = ip.weight*area;

         // Trace functionals: [p], {q}, [u.n], {v.n} as vectors over local dofs
         pjump = 0.0; pavg = 0.0; unjump = 0.0; unavg = 0.0;
         for (int s = 0; s < (interior ? 2 : 1); s++)
         {
            const Vector &sh = (s == 0) ? shape1 : shape2;
            const double sgn = (s == 0) ? 1.0 : -1.0;
            for (int k = 0; k < sh.Size(); k++)
            {
               pjump(idx(s, 0, k)) = sgn*sh(k);
               pavg (idx(s, 0, k)) = 0.5*sh(k);
               for (int d = 0; d < dim; d++)
               {
                  unjump(idx(s, 1+d, k)) = sgn*nhat(d)*sh(k);
                  unavg (idx(s, 1+d, k)) = 0.5*nhat(d)*sh(k);
               }
            }
         }

         // elmat(test, trial):  AddMult_a_VWt(a, test_vec, trial_vec, elmat)
         if (interior)
         {
            AddMult_a_VVt(0.5*c*w, pjump,  elmat);           //  (c/2)[p][q]
            AddMult_a_VVt(0.5*c*w, unjump, elmat);           //  (c/2)[u.n][v.n]
            AddMult_a_VWt(-K0*w,   pavg,  unjump, elmat);    // -K0 [u.n]{q}
            AddMult_a_VWt(-w/rho0, unavg, pjump,  elmat);    // -(1/rho0)[p]{v.n}
         }
         else // rigid wall: pjump = q trace, unjump = (u.n) trace
         {
            AddMult_a_VWt(-K0*w, pjump, unjump, elmat);      // -K0 (u.n) q
            AddMult_a_VVt(c*w,   unjump, elmat);             //  c (u.n)(v.n)
         }
      }
   }
};

void uFun_init(const Vector & x, Vector & u);
real_t pFun_init(const Vector & x);


int main(int argc, char *argv[])
{
    // 1. Parse command-line options.
    const char *mesh_file = "../../data/inline-quad.mesh";
    int order = 1;
    int num_levels = 2;
    double bulk_modulus = 0.1;
    double fluid_density = 2.0;
    double dt = 1.0; 


    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
                    "Mesh file to use.");
    args.AddOption(&order, "-o", "--order",
                    "Finite element order (polynomial degree).");
    args.AddOption(&num_levels, "-nl", "--levels", "Number of Multigrid Levels.");
    args.AddOption(&bulk_modulus, "-bm", "--bulk-modulus", "Bulk modulus.");
    args.AddOption(&fluid_density, "-fd", "--fluid-density", "Fluid density.");
    args.AddOption(&dt, "-dt", "--time-step", "Time step.");
    args.ParseCheck();

    Mesh mesh(mesh_file);
    const int dim = mesh.Dimension();
    int ncoarse = pow(2, dim); 
    int ref_levels = num_levels - 2;
    for (int i = 0; i < ref_levels; ++i) { mesh.UniformRefinement(); }


    // Define the Finite Element Spaces. We use the same DG Collection for pressure and velocity
    DG_FECollection fec(order, dim, BasisType::GaussLobatto);
    FiniteElementSpace p_fes(&mesh, &fec);
    FiniteElementSpace v_fes(&mesh, &fec, dim, Ordering::byNODES); // velocity terms only
    FiniteElementSpace w_fes(&mesh, &fec, dim + 1, Ordering::byNODES); // face terms only


    //    Define the BlockStructure of the problem, i.e. define the array of
    //    offsets for each variable. The last component of the Array is the sum
    //    of the dimensions of each block.
    Array<int> block_offsets(3); // number of variables + 1
    block_offsets[0] = 0;
    block_offsets[1] = p_fes.GetVSize();
    block_offsets[2] = v_fes.GetVSize();
    block_offsets.PartialSum();

    std::cout << "***********************************************************\n";
    std::cout << "dim(R) = " << block_offsets[1] - block_offsets[0] << "\n";
    std::cout << "dim(W) = " << block_offsets[2] - block_offsets[1] << "\n";
    std::cout << "dim(R+W) = " << block_offsets.Last() << "\n";
    std::cout << "***********************************************************\n";

    
    // Coefficients 
    ConstantCoefficient bulk_coeff(bulk_modulus);
    ConstantCoefficient dens_recip_coeff(1.0 / fluid_density);

    // Mass Matrices for the time-stepping scheme
    // Mass blocks are scaled by 1/dt
    BilinearForm mp(&p_fes), mu(&v_fes);
    mp.AddDomainIntegrator(new MassIntegrator);
    mu.AddDomainIntegrator(new VectorMassIntegrator);
    mp.Assemble(); mp.Finalize();
    mu.Assemble(); mu.Finalize();
    SparseMatrix Mp_dt(mp.SpMat()); Mp_dt *= 1.0/dt;
    SparseMatrix Mu_dt(mu.SpMat()); Mu_dt *= 1.0/dt;
 

    MixedBilinearForm cdiv(&v_fes, &p_fes);                 //  K0 (div u, q)
    cdiv.AddDomainIntegrator(new VectorDivergenceIntegrator(bulk_coeff));
    cdiv.Assemble(); cdiv.Finalize();

    MixedBilinearForm cgrad(&p_fes, &v_fes);                // (1/rho0)(grad p, v)
    cgrad.AddDomainIntegrator(new GradientIntegrator(dens_recip_coeff));
    cgrad.Assemble(); cgrad.Finalize();

    // upwinding
    BilinearForm flux(&w_fes);
    flux.AddInteriorFaceIntegrator(new AcousticUpwindFaceIntegrator(bulk_modulus, fluid_density));
    flux.AddBdrFaceIntegrator(new AcousticUpwindFaceIntegrator(bulk_modulus, fluid_density));
    flux.Assemble(); flux.Finalize();

    //      [ M_p/dt + F_pp    K0 D + F_pu ]
    //      [ (1/rho0) G + F_up  M_u/dt + F_uu ]
    BlockMatrix A_vol(block_offsets);
    A_vol.SetBlock(0, 0, &Mp_dt);
    A_vol.SetBlock(0, 1, &cdiv.SpMat());
    A_vol.SetBlock(1, 0, &cgrad.SpMat());
    A_vol.SetBlock(1, 1, &Mu_dt);
    SparseMatrix *A_mono = A_vol.CreateMonolithic();
    SparseMatrix *LHS = Add(*A_mono, flux.SpMat());

    // GSSmoother prec(*LHS);
    SmoothedAggregationGMG mg(w_fes, *LHS, ncoarse, num_levels, false);
    mg.SetCycleType(mfem::MultigridBase::CycleType::VCYCLE, 3, 3);
    GMRESSolver solver;
    solver.SetOperator(*LHS);
    solver.SetPreconditioner(mg);
    solver.SetRelTol(1e-10); solver.SetMaxIter(500); solver.SetKDim(50);
    solver.SetPrintLevel(1);

    // State and time loop
    BlockVector w(block_offsets), rhs(block_offsets);
    GridFunction p(&p_fes), u(&v_fes);
    VectorFunctionCoefficient u_init(dim, uFun_init);
    FunctionCoefficient p_init(pFun_init);
    p.ProjectCoefficient(p_init);
    u.ProjectCoefficient(u_init);
    w.GetBlock(0) = p; w.GetBlock(1) = u;
    p.MakeRef(&p_fes, w.GetBlock(0), 0);
    u.MakeRef(&v_fes, w.GetBlock(1), 0);

    Mp_dt.Mult(w.GetBlock(0), rhs.GetBlock(0));
    Mu_dt.Mult(w.GetBlock(1), rhs.GetBlock(1));
    solver.Mult(rhs, w);   

    // char vishost[] = "localhost";
    // int  visport   = 19916;
    // socketstream u_sock(vishost, visport);
    // u_sock.precision(8);
    // u_sock << "solution\n" << mesh << u << "window_title 'Velocity'" << endl;
    // socketstream p_sock(vishost, visport);
    // p_sock.precision(8);
    // p_sock << "solution\n" << mesh << p << "window_title 'Pressure'" << endl;
}

void uFun_init(const Vector & x, Vector & u)
{
    const double pi = 3.14159265358979323846;
    const double k = 2.0*pi;
    u(0) = sin(k*x(0))*cos(k*x(1));
    u(1) = -cos(k*x(0))*sin(k*x(1));
}

real_t pFun_init(const Vector & x)
{
    const double pi = 3.14159265358979323846;
    const double k = 2.0*pi;
    return sin(k*x(0))*sin(k*x(1));
}