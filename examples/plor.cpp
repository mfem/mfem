#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;
using namespace mfem;

void f_exact(const Vector &x, Vector &f)
{
   double kappa = 2*M_PI;
   if (x.Size() == 3)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(2));
      f(2) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else if (x.Size() == 2)
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(1));
      f(1) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
   else
   {
      f(0) = (1. + kappa * kappa) * sin(kappa * x(0));
   }
}

struct HybridizationSolver : Solver
{
   Solver &solv;
   Hybridization &h;
   mutable Vector b_r, x_r;
   HybridizationSolver(Solver &solv_, Hybridization &h_)
   : Solver(solv_.Height()), solv(solv_), h(h_) { }
   void SetOperator(const Operator&) { }
   void Mult(const Vector &b, Vector &x) const
   {
      h.ReduceRHS(b, b_r);
      x_r.SetSize(b_r.Size());
      x_r = 0.0;
      solv.Mult(b_r, x_r);
      h.ComputeSolution(b, x_r, x);
   }
};

struct PermutedSolver : Solver
{
   Solver &solv;
   Array<int> p;
   mutable Vector bp, xp;
   PermutedSolver(Solver &solv_, const Array<int> &p_)
   : Solver(solv_.Height()), solv(solv_), p(p_), bp(p.Size()), xp(p.Size()) { }
   void SetOperator(const Operator&) { }
   void Mult(const Vector &b, Vector &x) const
   {
      for (int i=0; i<b.Size(); ++i) { bp[i] = p[i] < 0 ? -b[-1-p[i]] : b[p[i]]; }
      solv.Mult(bp, xp);
      for (int i=0; i<x.Size(); ++i)
      {
         int pi = p[i];
         int s = pi < 0 ? -1 : 1;
         x[pi < 0 ? -1-pi : pi] = s*xp[i];
      }
   }
};

const Array<int> &GetDofMap(FiniteElementSpace &fes, int i)
{
   const FiniteElement *fe = fes.GetFE(i);
   auto tfe = dynamic_cast<const TensorBasisElement*>(fe);
   MFEM_ASSERT(tfe != NULL, "");
   return tfe->GetDofMap();
}

Array<int> ComputeVectorFE_LORPermutation(
   FiniteElementSpace &fes_ho,
   FiniteElementSpace &fes_lor,
   FiniteElement::MapType type)
{
   // Given an index `i` of a LOR dof, `perm[i]` is the index of the
   // corresponding HO dof.
   Array<int> perm(fes_lor.GetVSize());
   Array<int> vdof_ho, vdof_lor;

   Mesh &mesh_lor = *fes_lor.GetMesh();
   int dim = mesh_lor.Dimension();
   const CoarseFineTransformations &cf_tr = mesh_lor.GetRefinementTransforms();
   for (int ilor=0; ilor<mesh_lor.GetNE(); ++ilor)
   {
      int iho = cf_tr.embeddings[ilor].parent;
      int lor_index = cf_tr.embeddings[ilor].matrix;

      int p = fes_ho.GetOrder(iho);
      int p1 = p+1;
      int ndof_per_dim = (dim == 2) ? p*p1 :
                         type == FiniteElement::H_CURL ? p*p1*p1 : p*p*p1;

      fes_ho.GetElementVDofs(iho, vdof_ho);
      fes_lor.GetElementVDofs(ilor, vdof_lor);

      const Array<int> &dofmap_ho = GetDofMap(fes_ho, iho);
      const Array<int> &dofmap_lor = GetDofMap(fes_lor, ilor);

      int off_x = lor_index % p;
      int off_y = (lor_index / p) % p;
      int off_z = (lor_index / p) / p;

      auto absdof = [](int i) { return i < 0 ? -1-i : i; };

      auto set_perm = [&](int off_lor, int off_ho, int n1, int n2)
      {
         for (int i1=0; i1<2; ++i1)
         {
            int m = (dim == 2 || type == FiniteElement::H_DIV) ? 1 : 2;
            for (int i2=0; i2<m; ++i2)
            {
               int i;
               i = dofmap_lor[off_lor + i1 + i2*2];
               int s1 = i < 0 ? -1 : 1;
               int idof_lor = vdof_lor[absdof(i)];
               i = dofmap_ho[off_ho + i1*n1 + i2*n2];
               int s2 = i < 0 ? -1 : 1;
               int idof_ho = vdof_ho[absdof(i)];
               int s3 = idof_lor < 0 ? -1 : 1;
               int s4 = idof_ho < 0 ? -1 : 1;
               int s = s1*s2*s3*s4;
               i = absdof(idof_ho);
               perm[absdof(idof_lor)] = s < 0 ? -1-absdof(i) : absdof(i);
            }
         }
      };

      int offset;

      if (type == FiniteElement::H_CURL)
      {
         // x
         offset = off_x + off_y*p + off_z*p*p1;
         set_perm(0, offset, p, p*p1);
         // y
         offset = ndof_per_dim + off_x + off_y*(p1) + off_z*p1*p;
         set_perm(dim == 2 ? 2 : 4, offset, 1, p*p1);
         // z
         if (dim == 3)
         {
            offset = 2*ndof_per_dim + off_x + off_y*p1 + off_z*p1*p1;
            set_perm(8, offset, 1, p+1);
         }
      }
      else
      {
         // x
         offset = off_x + off_y*p1 + off_z*p*p1;
         set_perm(0, offset, 1, 0);
         // y
         offset = ndof_per_dim + off_x + off_y*p + off_z*p1*p;
         set_perm(2, offset, p, 0);
         // z
         if (dim == 3)
         {
            offset = 2*ndof_per_dim + off_x + off_y*p + off_z*p*p;
            set_perm(4, offset, p*p, 0);
         }
      }
   }

   return perm;
}

int main(int argc, char *argv[])
{
   MPI_Session mpi;

   // 1. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int ref_levels = 0;
   int order = 3;
   const char *fe = "n";
   bool hybridization = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine", "Uniform refinements.");
   args.AddOption(&order, "-o", "--order", "Polynomial degree.");
   args.AddOption(&fe, "-fe", "--fe-type", "FE type. n for Hcurl, r for Hdiv");
   args.AddOption(&hybridization, "-hb", "--hybridization", "-no-hb",
                  "--no-hybridization", "Enable hybridization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   bool ND = false;
   if (string(fe) == "n") { ND = true; }
   else if (string(fe) == "r") { ND = false; }
   else { MFEM_ABORT("Bad FE type. Must be 'n' or 'r'."); }
   bool RT = !ND;

   Mesh serial_mesh(mesh_file, 1, 1);
   int dim = serial_mesh.Dimension();
   for (int l = 0; l < ref_levels; l++) { serial_mesh.UniformRefinement(); }
   ParMesh mesh(MPI_COMM_WORLD, serial_mesh);
   serial_mesh.Clear();

   int btype = BasisType::GaussLobatto;
   ParMesh mesh_lor(&mesh, order, btype);

   unique_ptr<FiniteElementCollection> fec_ho, fec_lor, fec_h;
   unique_ptr<ParFiniteElementSpace> fes_h;
   if (ND)
   {
      fec_ho.reset(new ND_FECollection(order, dim, BasisType::GaussLobatto, BasisType::Integrated));
      fec_lor.reset(new ND_FECollection(1, dim, BasisType::GaussLobatto, BasisType::Integrated));
   }
   else
   {
      fec_ho.reset(new RT_FECollection(order-1, dim, BasisType::GaussLobatto, BasisType::Integrated));
      fec_lor.reset(new RT_FECollection(0, dim, BasisType::GaussLobatto, BasisType::Integrated));
      if (hybridization)
      {
         fec_h.reset(new DG_Interface_FECollection(0, dim));
         fes_h.reset(new ParFiniteElementSpace(&mesh_lor, fec_h.get()));
      }
   }

   ParFiniteElementSpace fes_ho(&mesh, fec_ho.get());
   ParFiniteElementSpace fes_lor(&mesh_lor, fec_lor.get());

   Array<int> ess_tdof_list;
   ConstantCoefficient one(1.0);
   Vector ones_vec(dim);
   ones_vec = 1.0;
   VectorFunctionCoefficient coeff(dim, f_exact);

   ParBilinearForm a_ho(&fes_ho), a_lor(&fes_lor);
   // ParBilinearForm a_ho(&fes_lor), a_lor(&fes_lor);
   a_ho.AddDomainIntegrator(new VectorFEMassIntegrator);
   a_lor.AddDomainIntegrator(new VectorFEMassIntegrator);
   if (ND)
   {
      a_ho.AddDomainIntegrator(new CurlCurlIntegrator);
      a_lor.AddDomainIntegrator(new CurlCurlIntegrator);
   }
   else
   {
      a_ho.AddDomainIntegrator(new DivDivIntegrator);
      a_lor.AddDomainIntegrator(new DivDivIntegrator);
      if (hybridization)
      {
         a_lor.EnableHybridization(fes_h.get(), new NormalTraceJumpIntegrator, ess_tdof_list);
      }
   }
   a_ho.SetAssemblyLevel(AssemblyLevel::PARTIAL);
   a_ho.Assemble();
   a_lor.Assemble();
   a_lor.Finalize();

   LinearForm b_lor(&fes_lor);
   b_lor.AddDomainIntegrator(new VectorFEDomainLFIntegrator(coeff));
   b_lor.Assemble();

   LinearForm b_ho(&fes_ho);
   b_ho.AddDomainIntegrator(new VectorFEDomainLFIntegrator(coeff));
   b_ho.Assemble();

   GridFunction x_ho(&fes_ho), x_lor(&fes_lor);
   x_ho = 0.0;
   x_lor = 0.0;

   Vector X_ho, B_ho, X_lor, B_lor;
   OperatorHandle A_ho, A_lor;
   a_ho.FormLinearSystem(ess_tdof_list, x_ho, b_ho, A_ho, X_ho, B_ho);
   a_lor.FormLinearSystem(ess_tdof_list, x_lor, b_lor, A_lor, X_lor, B_lor);

   unique_ptr<Solver> solv_lor, amg;
   SparseMatrix diag;
   A_lor.As<HypreParMatrix>()->GetDiag(diag);
   if (RT && hybridization)
   {
      amg.reset(new HypreBoomerAMG(*A_lor.As<HypreParMatrix>()));
      // amg.reset(new UMFPackSolver(diag));
      solv_lor.reset(new HybridizationSolver(*amg, *a_lor.GetHybridization()));
   }
   else if (RT && dim == 3)
   {
      solv_lor.reset(new HypreADS(&fes_lor));
   }
   else
   {
      solv_lor.reset(new HypreAMS(&fes_lor));
   }
   solv_lor->SetOperator(*A_lor);

   FiniteElement::MapType t = ND ? FiniteElement::H_CURL : FiniteElement::H_DIV;
   Array<int> perm = ComputeVectorFE_LORPermutation(fes_ho, fes_lor, t);

   // TEMP
   // for (int i=0; i<perm.Size(); ++i) { perm[i] = i; }

   PermutedSolver solv_lor_perm(*solv_lor, perm);

   CGSolver cg(MPI_COMM_WORLD);
   cg.SetAbsTol(0.0);
   cg.SetRelTol(1e-12);
   cg.SetMaxIter(100);
   cg.SetPrintLevel(1);

   cg.SetOperator(*A_ho);
   cg.SetPreconditioner(solv_lor_perm);
   // cg.SetPreconditioner(*solv_lor);
   cg.Mult(B_ho, X_ho);

   // cg.SetOperator(*A_lor);
   // cg.SetPreconditioner(*amg);
   // cg.Mult(B_lor, X_lor);

   a_ho.RecoverFEMSolution(X_ho, b_ho, x_ho);

   ParaViewDataCollection dc("LOR", &mesh);
   dc.SetPrefixPath("ParaView");
   dc.SetHighOrderOutput(true);
   dc.SetLevelsOfDetail(order);
   dc.RegisterField("u", &x_ho);
   dc.SetCycle(0);
   dc.SetTime(0.0);
   dc.Save();

   // solv_lor.Mult(B_lor, X_lor);
   // a_lor.RecoverFEMSolution(X_lor, b_lor, x_lor);
   x_lor = x_ho;

   for (int i=0; i<perm.Size(); ++i)
   {
      int pi = perm[i];
      int s = pi < 0 ? -1 : 1;
      x_lor[i] = s*x_ho[pi < 0 ? -1-pi : pi];
   }

   dc.SetMesh(&mesh_lor);
   dc.DeregisterField("u");
   dc.RegisterField("u", &x_lor);
   dc.SetLevelsOfDetail(1);
   dc.SetCycle(1);
   dc.SetTime(1.0);
   dc.Save();

   return 0;
}
