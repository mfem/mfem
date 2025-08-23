//               MFEM Example 2 - Nonlinear elasticity + patch partial assembly
//
// Compile with: make nurbs_patch_ex2_nl
//
// Sample runs:  nurbs_patch_ex2 -incdeg 2 -rf 4 -int 0
//               nurbs_patch_ex2 -incdeg 2 -rf 4 -int 1
//
// Description:  This example code solves a simple nonlinear elasticity problem
//               describing a multi-material cantilever beam.
//
//                                 +----------+----------+
//                    boundary --->| material | material |<--- boundary
//                    attribute 1  |    1     |    2     |     attribute 2
//                    (fixed)      +----------+----------+     (pull down)
//
//               This example is a specialization of ex2 which demonstrates
//               patch-wise partial assembly on NURBS meshes.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include "fem/integ/bilininteg_patch.hpp"

using namespace std;
using namespace mfem;

using mfem::future::tensor;
using mfem::future::make_tensor;


class NeoHookeanNLFIntegrator : public NonlinearFormIntegrator
// class NeoHookeanNLFIntegrator : public Operator
{
private:
   // Data for NURBS patch PA
   // Set in PatchElasticitySetup3D: [numPatches x [ NQ[patch] x 12 ]]
   std::vector<Vector> ppa_data;
   // Vector [numPatches] of structs containing basis info for each patch
   std::vector<PatchBasisInfo> pbinfo;
   const FiniteElementSpace *fespace;   ///< Not owned.
   Vector pa_data;
   PWConstCoefficient *C1, *D1;
   int numPatches;
   static constexpr int vdim = 3;
   static constexpr int dim = 3;

   NURBSMeshRules *patchRules = nullptr;

   // cached values of u (for computing gradient)
   mutable Vector ucache;

public:
   NeoHookeanNLFIntegrator(PWConstCoefficient &c, PWConstCoefficient &d, NURBSMeshRules *pr)
   {
      C1 = &c; D1 = &d;
      patchRules = pr;
   }

   void AssemblePA(const FiniteElementSpace &fes);

   void MultPatch(const Vector &pa_data, const PatchBasisInfo &pb,
                  const Vector &x, Vector &y) const;
   void Mult(const Vector &x, Vector &y) const;

   void MultGradPatch(const Vector &pa_data, const PatchBasisInfo &pb,
                      const Vector &u, const Vector &x, Vector &y) const;
   void MultGrad(const Vector &x, Vector &y) const;

   void Update(const Vector &x);

};

// Computes everything we need at quadrature points (jacobian + coeffs) for
// partial assembly on patches
void NeoHookeanNLFIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   Mesh &mesh = *fes.GetMesh();
   fespace = &fes;
   numPatches = mesh.NURBSext->GetNP();
   ppa_data.resize(numPatches);
   for (unsigned int patch=0; patch<numPatches; ++patch)
   {
      // AssemblePatchPA(p, fes);
      PatchBasisInfo pb(&mesh, patch, patchRules);
      pbinfo.push_back(pb);

      // Quadrature points in each dimension for this patch
      const Array<int>& Q1D = pbinfo[patch].Q1D;
      // Total quadrature points
      const int nq = pbinfo[patch].NQ;

      Vector weightsv(nq);
      auto weights = Reshape(weightsv.HostReadWrite(), Q1D[0], Q1D[1], Q1D[2]);
      IntegrationPoint ip;

      Vector jacv(nq * vdim * vdim);  // Computed as in GeometricFactors::Compute
      auto jac = Reshape(jacv.HostReadWrite(), Q1D[0], Q1D[1], Q1D[2], vdim, vdim);
      Vector coeffsv(nq * 2);        // C1, D1 at quad points
      auto coeffs = Reshape(coeffsv.HostReadWrite(), Q1D[0], Q1D[1], Q1D[2], 2);

      MFEM_VERIFY(Q1D.Size() == 3, "Only 3D for now");

      for (int qz=0; qz<Q1D[2]; ++qz)
      {
         for (int qy=0; qy<Q1D[1]; ++qy)
         {
            for (int qx=0; qx<Q1D[0]; ++qx)
            {
               const int e = patchRules->GetPointElement(patch, qx, qy, qz);
               ElementTransformation *tr = mesh.GetElementTransformation(e);
               patchRules->GetIntegrationPointFrom1D(patch, qx, qy, qz, ip);

               weights(qx,qy,qz) = ip.weight;

               coeffs(qx,qy,qz,0) = C1->Eval(*tr, ip);
               coeffs(qx,qy,qz,1) = D1->Eval(*tr, ip);

               tr->SetIntPoint(&ip);

               const DenseMatrix& Jp = tr->Jacobian();
               for (int i=0; i<vdim; ++i)
               {
                  for (int j=0; j<vdim; ++j)
                  {
                     jac(qx,qy,qz,i,j) = Jp(i,j);
                  }
               }
            }
         }
      }

   // Computes values at quadrature points
   PatchElasticitySetup3D(Q1D[0], Q1D[1], Q1D[2], weightsv, jacv, coeffsv,
                          ppa_data[patch]);
   }
}

void NeoHookeanNLFIntegrator::MultPatch(const Vector &pa_data,
                                        const PatchBasisInfo &pb,
                                        const Vector &x,
                                        Vector &y) const
{
   // Unpack patch basis info
   const Array<int>& Q1D = pb.Q1D;
   const int NQ = pb.NQ;
   MFEM_VERIFY((pb.dim == 3) && (vdim == 3), "Dimension mismatch.");

   // gradu(i,j,q): d/d(x_j) u_i(x_q)
   Vector graduv(vdim*vdim*NQ);
   graduv = 0.0;
   auto gradu = Reshape(graduv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1], Q1D[2]);

   // S[i,j,q] = D( gradu )
   //          = stress[i,j,q] * J^{-T}[q]
   Vector Sv(vdim*vdim*NQ);
   Sv = 0.0;
   auto S = Reshape(Sv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1], Q1D[2]);

   Vector sumXYv(vdim*vdim*pb.MAX1D[0]*pb.MAX1D[1]);
   Vector sumXv(vdim*vdim*pb.MAX1D[0]);

   // 1) Interpolate U at dofs to gradu in reference quadrature space
   PatchG3D<3>(pb, x, sumXYv, sumXv, gradu);

   // 2) Apply the "D" operator at each quadrature point: D( gradu )
   PatchApplyKernel3D(pb, pa_data, NeoHookeanStress<3>, gradu, S);
   // PatchApplyKernel3D(pb, pa_data, LinearElasticStress<3>, gradu, S);

   // 3) Apply test function gradv
   PatchGT3D<3>(pb, S, sumXYv, sumXv, y);
   // cout << "done add mult" << endl;

}

void NeoHookeanNLFIntegrator::Mult(const Vector &x, Vector &y) const
{
   Vector xp, yp;

   for (int p=0; p<numPatches; ++p)
   {
      Array<int> vdofs;
      fespace->GetPatchVDofs(p, vdofs);

      x.GetSubVector(vdofs, xp);
      yp.SetSize(vdofs.Size());
      yp = 0.0;

      MultPatch(ppa_data[p], pbinfo[p], xp, yp);

      y.AddElementVector(vdofs, yp);
   }
}

void NeoHookeanNLFIntegrator::MultGradPatch(const Vector &pa_data,
                                            const PatchBasisInfo &pb,
                                            const Vector &u,
                                            const Vector &x,
                                            Vector &y) const
{
   // Unpack patch basis info
   const Array<int>& Q1D = pb.Q1D;
   const int NQ = pb.NQ;
   MFEM_VERIFY((pb.dim == 3) && (vdim == 3), "Dimension mismatch.");

   Vector graduv(vdim*vdim*NQ);
   graduv = 0.0;
   auto gradu = Reshape(graduv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1], Q1D[2]);

   // direction of action
   Vector dgraduv(vdim*vdim*NQ);
   dgraduv = 0.0;
   auto dgradu = Reshape(dgraduv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1], Q1D[2]);

   Vector Sv(vdim*vdim*NQ);
   Sv = 0.0;
   auto S = Reshape(Sv.HostReadWrite(), vdim, vdim, Q1D[0], Q1D[1], Q1D[2]);

   Vector sumXYv(vdim*vdim*pb.MAX1D[0]*pb.MAX1D[1]);
   Vector sumXv(vdim*vdim*pb.MAX1D[0]);

   // 1) Interpolate U at dofs to gradu in reference quadrature space
   PatchG3D<3>(pb, u, sumXYv, sumXv, gradu);
   sumXYv = 0.0; sumXv = 0.0;
   PatchG3D<3>(pb, x, sumXYv, sumXv, dgradu);

   // 2) Apply the "D" operator at each quadrature point: D( gradu )
   PatchApplyGradKernel3D(pb, pa_data, GradNeoHookeanStress<3>, gradu, dgradu, S);
   // PatchApplyKernel3D(pb, pa_data, LinearElasticStress<3>, dgradu, S);

   // 3) Apply test function gradv
   PatchGT3D<3>(pb, S, sumXYv, sumXv, y);
   // cout << "done add mult" << endl;

}

void NeoHookeanNLFIntegrator::MultGrad(const Vector &x, Vector &y) const
{
   Vector xp, yp;
   Vector up; // evaluation point of gradient

   for (int p=0; p<numPatches; ++p)
   {
      Array<int> vdofs;
      fespace->GetPatchVDofs(p, vdofs);

      x.GetSubVector(vdofs, xp);
      ucache.GetSubVector(vdofs, up);
      yp.SetSize(vdofs.Size());
      yp = 0.0;

      MultGradPatch(ppa_data[p], pbinfo[p], up, xp, yp);

      y.AddElementVector(vdofs, yp);
   }
   // cout << "done mult grad" << endl;
}

void NeoHookeanNLFIntegrator::Update(const Vector &x)
{
   ucache = x;
}

class Residual : public Operator
{
private:
   NeoHookeanNLFIntegrator *F1; // domain terms
   LinearForm *F2; // boundary terms
   Array<int> ess_tdofs;
   // NonlinearForm *F1 = nullptr; // domain terms

   class Jacobian : public Operator
   {
   public:
      Jacobian(const Residual *res) :
         Operator(res->Height()),
         residual(res),
         z(res->Height())
      { }

      void Mult(const Vector &x, Vector &y) const override
      {
         z = x;
         z.SetSubVector(residual->ess_tdofs, 0.0);
         y = 0.0;

         residual->F1->MultGrad(z, y);

         auto d_y = y.HostReadWrite();
         const auto d_x = x.HostRead();
         for (int i = 0; i < residual->ess_tdofs.Size(); i++)
         {
            d_y[residual->ess_tdofs[i]] = d_x[residual->ess_tdofs[i]];
         }

      }

      // Pointer to the wrapped Residual operator
      const Residual *residual = nullptr;

      // Temporary vector
      mutable Vector z;
   };

public:
   Residual(NeoHookeanNLFIntegrator *F1_, FiniteElementSpace &fespace, LinearForm *F2_, Array<int> &ess_tdof_list);

   void Mult(const Vector &x, Vector &y) const override;

   Operator &GetGradient(const Vector &x) const override;

   mutable std::shared_ptr<Jacobian> J;
};

Residual::Residual(NeoHookeanNLFIntegrator *F1_, FiniteElementSpace &fespace, LinearForm *F2_, Array<int> &ess)
   : Operator(fespace.GetTrueVSize()), F1(F1_), F2(F2_), ess_tdofs(ess)
{ }

void Residual::Mult(const Vector &x, Vector &y) const
{
   // cout << "debugging, x = " << endl; x.Print();
   y = 0.0;

   F1->Mult(x, y);

   // Add the boundary terms
   const real_t* v = F2->HostRead();
   for (int i = 0; i < F2->Size(); ++i)
   {
      y[i] -= v[i];
   }


   y.SetSubVector(ess_tdofs, 0.0);

   // cout << endl << endl << "debugging, y = " << endl; y.Print();
   // cout << endl << endl << "debugging, esstdofs = " << endl; ess_tdofs.Print();

   // cout << "Residual::Mult" << endl;
}

Operator &Residual::GetGradient(const Vector &x) const
{
   // action of gradient is calculated by the NeoHookeanNLFIntegrator, so update its data
   F1->Update(x);
   J = std::make_shared<Jacobian>(this);
   // cout << "Residual::GetGradient" << endl;
   return *J;
}


int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   const char *mesh_file = "../../data/beam-hex-nurbs.mesh";
   int refinement_factor = 0;
   int nurbs_degree_increase = 0;  // Elevate the NURBS mesh degree by this
   int spline_integration_type = 0;
   int visport = 19916;
   bool csv_info = 0;
   bool visualization = 1;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&refinement_factor, "-rf", "--refinement-factor",
                  "Refinement factor for the NURBS mesh.");
   args.AddOption(&nurbs_degree_increase, "-incdeg", "--nurbs-degree-increase",
                  "Elevate NURBS mesh degree by this amount.");
   args.AddOption(&spline_integration_type, "-int", "--integration-type",
                  "Integration rule type: 0 - full order Gauss Legendre, "
                  "1 - reduced order Gaussian Legendre");
   args.AddOption(&csv_info, "-csv", "--csv-info", "-no-csv",
                  "--no-csv-info",
                  "Enable or disable dump of info into csv.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();

   // Print & verify options
   args.PrintOptions(cout);
   MFEM_VERIFY(spline_integration_type >= 0 && spline_integration_type < 2,
               "Spline integration type must be 0 or 1 for this example");

   bool pa = true;
   bool patchAssembly = true;

   // 2. Read the mesh from the given mesh file.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   // Verify mesh is valid for this problem
   MFEM_VERIFY(mesh.IsNURBS(), "Example is for NURBS meshes");
   MFEM_VERIFY(mesh.GetNodes(), "NURBS mesh must have nodes");
   // if (mesh.attributes.Max() < 2 || mesh.bdr_attributes.Max() < 2)
   // {
   //    cout << "\nInput mesh should have at least two boundary"
   //         << "attributes! (See schematic in ex2.cpp)\n"
   //         << endl;
   // }

   if (nurbs_degree_increase > 0) { mesh.DegreeElevate(nurbs_degree_increase); }
   if (refinement_factor > 1) { mesh.NURBSUniformRefinement(refinement_factor); }

   // 5. Define a finite element space on the mesh.
   FiniteElementCollection * fec = mesh.GetNodes()->OwnFEC();
   FiniteElementSpace fespace = FiniteElementSpace(&mesh, fec, dim, Ordering::byVDIM);
   // FiniteElementSpace fespace = FiniteElementSpace(&mesh, fec, dim);
   cout << "Finite Element Collection: " << fec->Name() << endl;
   cout << "Number of finite element unknowns: " << fespace.GetTrueVSize() << endl;
   cout << "Number of elements: " << fespace.GetNE() << endl;
   cout << "Number of patches: " << mesh.NURBSext->GetNP() << endl;

   // 6. Determine the list of true (i.e. conforming) essential boundary dofs.
   Array<int> ess_tdof_list, ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 0;
   ess_bdr[0] = 1;
   fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Set up the linear form b(.)
   VectorArrayCoefficient f(dim);
   for (int i = 0; i < dim-1; i++)
   {
      f.Set(i, new ConstantCoefficient(0.0));
   }
   {
      Vector pull_force(mesh.bdr_attributes.Max());
      pull_force = 0.0;
      pull_force(1) = -1.0e-2;
      f.Set(dim-1, new PWConstCoefficient(pull_force));
   }

   LinearForm b(&fespace);
   b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   cout << "Assembling RHS ... " << flush;
   b.Assemble();
   cout << "done." << endl;

   // 8. Define the solution vector x as a finite element grid function
   GridFunction x(&fespace);
   x = 0.0;

   // 9. Set up the nonlinear form F

   // Lame parameters -> Mooney-Rivlin coefficients
   Vector C1(mesh.attributes.Max());
   Vector D1(mesh.attributes.Max());
   // int lambda1 = 1.0;
   // int lambda2 = lambda1*50;
   // int mu1 = 1.0;
   // int mu2 = mu1*50;
   // C1 = 0.5 * mu1; C1(0) = 0.5 * mu2;
   // D1 = 0.5 * (lambda1 + 2.0/3 * mu1); D1(0) = 0.5 * (lambda2 + 2.0/3 * mu2);

   C1 = 0.5; D1 = 2.83333;
   // C1 = 5.0; D1 = 1.0; //lambda, mu

   PWConstCoefficient C1_func(C1);
   PWConstCoefficient D1_func(D1);

   cout << "Assembling F ... " << flush;
   // Integration rule
   SplineIntegrationRule splineRule(spline_integration_type);
   NURBSMeshRules* meshRules = new NURBSMeshRules(mesh, splineRule);
   // Integrator
   NeoHookeanNLFIntegrator nhi(C1_func, D1_func, meshRules);
   nhi.AssemblePA(fespace);

   // Define nonlinear form
   Residual F(&nhi, fespace, &b, ess_tdof_list);
   cout << "done." << endl;

   // Form system
   CGSolver lsolver;
   lsolver.SetAbsTol(0);
   lsolver.SetRelTol(1e-4);
   lsolver.SetMaxIter(500);
   lsolver.SetPrintLevel(2);

   // GMRESSolver lsolver;
   // lsolver.iterative_mode = true;
   // lsolver.SetRelTol(1e-12);
   // lsolver.SetAbsTol(0);
   // lsolver.SetMaxIter(300);

   // Set up the nonlinear solver
   NewtonSolver newton;
   newton.iterative_mode = true;
   newton.SetOperator(F);
   newton.SetAbsTol(0);
   // newton.SetAbsTol(1e-6);
   newton.SetRelTol(1e-6);
   newton.SetMaxIter(20);
   newton.SetSolver(lsolver);
   newton.SetPrintLevel(1);

   // Solve
   Vector zero;
   Vector X(fespace.GetTrueVSize());
   // fespace.GetRestrictionMatrix()->Mult(x, X);
   // fespace.GetRestrictionOperator()->Mult(x, X);
   newton.Mult(zero, X);

   // debugging - linear
   // X[4] = 0.01;
   // cout << "debugging, X = " << endl; X.Print();
   // Vector Y(fespace.GetTrueVSize());
   // F.Mult(X,Y);
   // cout << "debugging, F(X) = " << endl;
   // Y.Print();

   // fespace.GetProlongationMatrix()->Mult(X, x);
   // fespace.GetProlongationOperator()->Mult(X, x);

   // IdentityOperator I(fespace.GetTrueVSize());
   // I.Mult(Y, x);
   // x.SyncMemory(Y);
   // x = Y;
   x = X;

   cout << " x = " << endl; x.Print();

   // OperatorPtr A;
   // a.FormLinearSystem(ess_tdof_list, x, b, A, X, B);
   // cout << "done. " << "(size = " << fespace.GetTrueVSize() << ")" << endl;

   // 11. Define the solver + preconditioner
   // CGSolver solver;
   // solver.SetOperator(*A);
   // solver.SetMaxIter(1e4);
   // solver.SetPrintLevel(1);
   // solver.SetRelTol(sqrt(1e-6));
   // solver.SetAbsTol(sqrt(1e-14));


   // 12. Solve the linear system A X = B.
   // cout << "Solving linear system ... " << endl;
   // solver.Mult(B, X);
   // cout << "Done solving system." << endl;
   // // Recover the solution as a finite element grid function.
   // a.RecoverFEMSolution(X, b, x);

   {
      cout << "Saving mesh and solution to file..." << endl;
      GridFunction *nodes = mesh.GetNodes();
      *nodes += x;
      x *= -1;
      ofstream mesh_ofs("displaced.mesh");
      mesh_ofs.precision(8);
      mesh.Print(mesh_ofs);
      ofstream sol_ofs("sol.gf");
      sol_ofs.precision(16);
      x.Save(sol_ofs);
   }

   // 15. Send the data by socket to a GLVis server.
   if (visualization)
   {
      // send to socket
      char vishost[] = "localhost";
      socketstream sol_sock(vishost, visport);
      sol_sock.precision(8);
      sol_sock << "solution\n" << mesh << x;
      sol_sock << "window_geometry " << 100 << " " << 800 << " "
               << 600 << " " << 600 << "\n"
               << "keys agc\n" << std::flush;
   }





   return 0;
}
