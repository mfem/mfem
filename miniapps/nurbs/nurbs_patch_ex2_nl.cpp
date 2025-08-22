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


void PatchElasticitySetup3D(const int Q1Dx,
                            const int Q1Dy,
                            const int Q1Dz,
                            const Vector &w,
                            const Vector &j,
                            const Vector &c,
                            Vector &d)
{
   // computes [J^{-T}, lambda*W*detJ, mu*W*detJ] at quadrature points
   const auto W = Reshape(w.Read(), Q1Dx,Q1Dy,Q1Dz);
   const auto J = Reshape(j.Read(), Q1Dx,Q1Dy,Q1Dz,3,3);
   const auto C = Reshape(c.Read(), Q1Dx,Q1Dy,Q1Dz,2);
   // nq * [9 (J^{-T}) + 1 (WdetJ) + 1 (C1) + 1 (D1)]
   d.SetSize(Q1Dx * Q1Dy * Q1Dz * 11);
   auto D = Reshape(d.Write(), Q1Dx,Q1Dy,Q1Dz, 11);
   const int NE = 1;
   MFEM_FORALL_3D(e, NE, Q1Dx, Q1Dy, Q1Dz,
   {
      MFEM_FOREACH_THREAD(qx,x,Q1Dx)
      {
         MFEM_FOREACH_THREAD(qy,y,Q1Dy)
         {
            MFEM_FOREACH_THREAD(qz,z,Q1Dz)
            {
               const real_t J11 = J(qx,qy,qz,0,0);
               const real_t J21 = J(qx,qy,qz,1,0);
               const real_t J31 = J(qx,qy,qz,2,0);
               const real_t J12 = J(qx,qy,qz,0,1);
               const real_t J22 = J(qx,qy,qz,1,1);
               const real_t J32 = J(qx,qy,qz,2,1);
               const real_t J13 = J(qx,qy,qz,0,2);
               const real_t J23 = J(qx,qy,qz,1,2);
               const real_t J33 = J(qx,qy,qz,2,2);
               const real_t detJ = J11 * (J22 * J33 - J32 * J23) -
               /* */               J21 * (J12 * J33 - J32 * J13) +
               /* */               J31 * (J12 * J23 - J22 * J13);
               const real_t wdetj = W(qx,qy,qz) * detJ;
               // adj(J)
               const real_t A11 = (J22 * J33) - (J23 * J32);
               const real_t A12 = (J32 * J13) - (J12 * J33);
               const real_t A13 = (J12 * J23) - (J22 * J13);
               const real_t A21 = (J31 * J23) - (J21 * J33);
               const real_t A22 = (J11 * J33) - (J13 * J31);
               const real_t A23 = (J21 * J13) - (J11 * J23);
               const real_t A31 = (J21 * J32) - (J31 * J22);
               const real_t A32 = (J31 * J12) - (J11 * J32);
               const real_t A33 = (J11 * J22) - (J12 * J21);

               // store J^{-T} = adj(J)^T / detJ
               D(qx,qy,qz,0) = A11 / detJ;
               D(qx,qy,qz,1) = A21 / detJ;
               D(qx,qy,qz,2) = A31 / detJ;
               D(qx,qy,qz,3) = A12 / detJ;
               D(qx,qy,qz,4) = A22 / detJ;
               D(qx,qy,qz,5) = A32 / detJ;
               D(qx,qy,qz,6) = A13 / detJ;
               D(qx,qy,qz,7) = A23 / detJ;
               D(qx,qy,qz,8) = A33 / detJ;
               // Coefficients
               D(qx,qy,qz,9) = C(qx,qy,qz,0) * wdetj; // C1 * w * detj
               D(qx,qy,qz,10) = C(qx,qy,qz,1) * wdetj; // D1 * w * detj
            }
         }
      }
   });
}

template <int dim>
tensor<real_t, dim, dim>
LinearElasticStress(const tensor<real_t, dim, dim> Jinvt,
                    const real_t lambda,
                    const real_t mu,
                    const tensor<real_t, dim, dim> gradu_ref)
{
   // cout << "graduref[0,0] = " << gradu_ref(0,0) << endl;
   // cout << "Jinvt[0,0] = " << Jinvt(0,0) << endl;
   // Convert gradu_ref to physical space
   const auto gradu = gradu_ref * transpose(Jinvt);
   // Compute stress
   constexpr auto I = mfem::future::IsotropicIdentity<dim>();
   const tensor<real_t, dim, dim> strain = sym(gradu);
   const tensor<real_t, dim, dim> stress =
      lambda * tr(strain) * I + static_cast<real_t>(2.0) * mu * strain;
   // Transform back to reference space
   return stress * Jinvt;
}

template <int dim>
tensor<real_t, dim, dim>
NeoHookeanStress(const tensor<real_t, dim, dim> Jinvt,
                 const real_t C1,
                 const real_t D1,
                 const tensor<real_t, dim, dim> gradu_ref)
{
   static constexpr auto I = mfem::future::IsotropicIdentity<dim>();
   const auto gradu = gradu_ref * transpose(Jinvt);
   const auto J = det(I + gradu);
   const auto p = -2.0 * D1 * J * (J - 1);
   const auto devB = dev(gradu + transpose(gradu) + dot(gradu, transpose(gradu)));
   auto stress = -(p / J) * I + 2 * (C1 / pow(J, 5.0 / 3.0)) * devB;
   // Transform back to reference space
   return stress * Jinvt;
}

template <int dim>
tensor<mfem::real_t, dim, dim>
GradNeoHookeanStress(const tensor<real_t, dim, dim> Jinvt,
                      const real_t C1,
                      const real_t D1,
                      const tensor<real_t, dim, dim> gradu_ref)
{
   static constexpr auto I = mfem::future::IsotropicIdentity<dim>();
   const auto gradu = gradu_ref * transpose(Jinvt);

   tensor<mfem::real_t, dim, dim> F = I + gradu;
   tensor<mfem::real_t, dim, dim> invF = inv(F);
   tensor<mfem::real_t, dim, dim> devB =
      dev(gradu + transpose(gradu) + dot(gradu, transpose(gradu)));
   mfem::real_t j = det(F);
   mfem::real_t coef = (C1 / pow(j, 5.0 / 3.0));
   const auto dsigma = make_tensor<dim, dim, dim, dim>([&](int i, int j, int k, int l)
   {
      return 2 * (D1 * j * (i == j) -
                          mfem::real_t(5.0 / 3.0) * coef * devB[i][j]) *
                     invF[l][k] +
                     2 * coef *
                     ((i == k) * F[j][l] + F[i][l] * (j == k) -
                      mfem::real_t(2.0 / 3.0) * ((i == j) * F[k][l]));
   });

   // Transform back to reference space
   // return dsigma * Jinvt;
   return ddot(dsigma, Jinvt);
}

template <typename Kernel>
void PatchApplyKernel3D(const PatchBasisInfo &pb,
                        const Vector &pa_data,
                        Kernel&& kernel,
                        DeviceTensor<5, real_t> &gradu,
                        DeviceTensor<5, real_t> &S)
{
   // Unpack patch basis info
   static constexpr int dim = 3;
   const Array<int>& Q1D = pb.Q1D;
   const int NQ = pb.NQ;
   // Quadrature data. 11 values per quadrature point.
   // First 9 entries are J^{-T}; then C1*W*detJ and D1*W*detJ
   const auto qd = Reshape(pa_data.HostRead(), NQ, 11);

   for (int qz = 0; qz < Q1D[2]; ++qz)
   {
      for (int qy = 0; qy < Q1D[1]; ++qy)
      {
         for (int qx = 0; qx < Q1D[0]; ++qx)
         {
            const int q = qx + ((qy + (qz * Q1D[1])) * Q1D[0]);
            const real_t C1 = qd(q,9);
            const real_t D1 = qd(q,10);
            const auto Jinvt = make_tensor<dim, dim>(
            [&](int i, int j) { return qd(q, i*dim + j); });
            const auto gradu_q = make_tensor<dim, dim>(
            [&](int i, int j) { return gradu(i,j,qx,qy,qz); });
            const auto Sq = kernel(Jinvt, C1, D1, gradu_q);

            // cout << "Sq[1,1] = " << Sq(1,1) << endl;

            for (int i = 0; i < dim; ++i)
            {
               for (int j = 0; j < dim; ++j)
               {
                  S(i,j,qx,qy,qz) = Sq(i,j);
               }
            }
         } // qx
      } // qy
   } // qz
}

class NeoHookeanNLFIntegrator : public NonlinearFormIntegrator
// class NeoHookeanNLFIntegrator : public Operator
// class NeoHookeanNLFIntegrator : public BilinearFormIntegrator
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

   // const IntegrationRule *IntRule;
   NURBSMeshRules *patchRules = nullptr;

public:
   NeoHookeanNLFIntegrator(PWConstCoefficient &c, PWConstCoefficient &d, NURBSMeshRules *pr)
   {
      C1 = &c; D1 = &d;
      patchRules = pr;
   }

   void AssembleNURBSPA(const FiniteElementSpace &fes);
   void AssemblePA(const FiniteElementSpace &fes);

   template <typename Kernel>
   void MultPatchPA3D(const Vector &pa_data, const PatchBasisInfo &pb, Kernel&& kernel,
                         const Vector &x, Vector &y) const;
   void Mult(const Vector &x, Vector &y) const;
   void AddMult(const Vector &x, Vector &y) const;

   void MultGradPatchPA3D(const Vector &pa_data, const PatchBasisInfo &pb,
                         const Vector &x, Vector &y) const;
   void MultGradPA(const Vector &x, Vector &y) const;

};

void NeoHookeanNLFIntegrator::AssembleNURBSPA(const FiniteElementSpace &fes)
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

void NeoHookeanNLFIntegrator::AssemblePA(const FiniteElementSpace &fes)
{
   AssembleNURBSPA(fes);
}

template <typename Kernel>
void NeoHookeanNLFIntegrator::MultPatchPA3D(const Vector &pa_data,
                                               const PatchBasisInfo &pb,
                                               Kernel&& kernel,
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
   PatchApplyKernel3D(pb, pa_data, kernel, gradu, S);

   // 3) Apply test function gradv
   PatchGT3D<3>(pb, S, sumXYv, sumXv, y);
   cout << "done add mult" << endl;

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

      MultPatchPA3D(ppa_data[p], pbinfo[p], NeoHookeanStress<3>, xp, yp);

      y.AddElementVector(vdofs, yp);
   }
}

void NeoHookeanNLFIntegrator::AddMult(const Vector &x, Vector &y) const
{
   NeoHookeanNLFIntegrator::Mult(x, y);
}

void NeoHookeanNLFIntegrator::MultGradPA(const Vector &x, Vector &y) const
{
   Vector xp, yp;

   for (int p=0; p<numPatches; ++p)
   {
      Array<int> vdofs;
      fespace->GetPatchVDofs(p, vdofs);

      x.GetSubVector(vdofs, xp);
      yp.SetSize(vdofs.Size());
      yp = 0.0;

      MultPatchPA3D(ppa_data[p], pbinfo[p], GradNeoHookeanStress<3>, xp, yp);

      y.AddElementVector(vdofs, yp);
   }
}

class Residual : public Operator
{
private:
   NeoHookeanNLFIntegrator *F1i; // domain terms
   LinearForm *F2; // boundary terms
   Array<int> ess_tdofs;
   NonlinearForm *F1 = nullptr; // domain terms

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

         residual->F1->Mult(z, y);

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
   Residual(NeoHookeanNLFIntegrator *F1i_, FiniteElementSpace &fespace, LinearForm *F2_, Array<int> &ess_tdof_list);

   void Mult(const Vector &x, Vector &y) const override;

   Operator &GetGradient(const Vector &x) const override;

   mutable std::shared_ptr<Jacobian> J;
};

Residual::Residual(NeoHookeanNLFIntegrator *F1i_, FiniteElementSpace &fespace, LinearForm *F2_, Array<int> &ess)
   : Operator(fespace.GetTrueVSize()), F1i(F1i_), F2(F2_), ess_tdofs(ess)
{
   F1 = new NonlinearForm(&fespace);
   F1->SetAssemblyLevel(AssemblyLevel::PARTIAL);
   F1->AddDomainIntegrator(F1i);
   F1->SetEssentialBC(ess_tdofs);
   F1->Setup();
}

void Residual::Mult(const Vector &x, Vector &y) const
{
   F1->Mult(x, y);

   // Add the boundary terms
   const real_t* v = F2->HostRead();
   for (int i = 0; i < F2->Size(); ++i)
   {
      y[i] -= v[i];
   }

   y.SetSubVector(ess_tdofs, 0.0);
}

Operator &Residual::GetGradient(const Vector &x) const
{
   J = std::make_shared<Jacobian>(this);
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
   if (mesh.attributes.Max() < 2 || mesh.bdr_attributes.Max() < 2)
   {
      cout << "\nInput mesh should have at least two boundary"
           << "attributes! (See schematic in ex2.cpp)\n"
           << endl;
   }

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
   int lambda1 = 1.0;
   int lambda2 = lambda1*50;
   int mu1 = 1.0;
   int mu2 = mu1*50;
   C1 = 0.5 * mu1; C1(0) = 0.5 * mu2;
   D1 = 0.5 * (lambda1 + 2.0/3 * mu1); D1(0) = 0.5 * (lambda2 + 2.0/3 * mu2);

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
   CGSolver krylov;
   krylov.SetAbsTol(0.0);
   krylov.SetRelTol(1e-4);
   krylov.SetMaxIter(500);
   krylov.SetPrintLevel(2);

   // Set up the nonlinear solver
   NewtonSolver newton;
   newton.SetOperator(F);
   newton.SetAbsTol(0.0);
   newton.SetRelTol(1e-6);
   newton.SetMaxIter(10);
   newton.SetSolver(krylov);
   newton.SetPrintLevel(1);

   // Solve
   // H1.GetRestrictionMatrix()->Mult(u, X);
   Vector zero;
   Vector X(fespace.GetTrueVSize());
   newton.Mult(zero, X);
   // H1.GetProlongationMatrix()->Mult(X, u);
   // Vector B, X;
   // F.SetEssentialTrueDofs(ess_tdof_list);



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





   return 0;
}
