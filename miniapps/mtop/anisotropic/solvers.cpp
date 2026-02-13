
#include <memory>
#include "solvers.hpp"
#include "linear_anisotropic_elasticity.hpp"

using namespace mfem;

using mfem::future::dual;
using mfem::future::tuple;
using mfem::future::tensor;

using mfem::future::Weight;
using mfem::future::Gradient;
using mfem::future::Identity;

///////////////////////////////////////////////////////////////////////////////
/// \brief The QFunction struct defining the linear elasticity operator at
/// integration points which is valid in 2D and 3D
template <int DIM=2, int DESIGN_DIM=3, typename scalar_t=real_t> struct
   QFunction
{
   using matd_t = tensor<scalar_t, DIM, DIM>;
   using vecd_t = tensor<scalar_t, DIM>;
   using vecdsgn_t = tensor<scalar_t, DESIGN_DIM>;


   struct Elasticity
   {

      const real_t* aniso_tensor;
      int simp_exp;

      Elasticity(mfem::Vector& tm, int simp_exp_): simp_exp(simp_exp_)
      {
         aniso_tensor=tm.Read();
         MFEM_VERIFY(simp_exp_>=4 &&
                     simp_exp%2 == 0, "SIMP exponent should be an even integer >= 4");
      }
      void SetSIMPExponent(int simp_exp_)
      {
         MFEM_VERIFY(simp_exp_>=4 &&
                     simp_exp/2 == 0, "SIMP exponent should be an even integer >= 4");
         simp_exp=simp_exp_;
      }

      MFEM_HOST_DEVICE inline auto operator()(const matd_t &dudxi,
                                              const vecdsgn_t &eta,
                                              const real_t &L1, const real_t &M1,
                                              const real_t &L2, const real_t &M2,
                                              const matd_t &J,
                                              const real_t &w) const
      {
         const matd_t JxW = mfem::future::adjugateT(J) * w;
         constexpr auto I = mfem::future::IsotropicIdentity<DIM>();
         const auto eps = dudxi * mfem::future::inv(J);


         mfem::future::tensor<real_t, 3,3> C;
         C(0,0)=aniso_tensor[0];
         C(0,1)=aniso_tensor[1]; C(1,0)=aniso_tensor[1];
         C(0,2)=aniso_tensor[2]; C(2,0)=aniso_tensor[2];
         C(1,1)=aniso_tensor[3];
         C(1,2)=aniso_tensor[4]; C(2,1)=aniso_tensor[4];
         C(2,2)=aniso_tensor[5];

         matd_t R;
         R(0,0)=eta[1]; R(0,1)=eta[2];
         R(1,0)=-eta[2]; R(1,1)=eta[1];

         const auto resp=mfem::future::transpose(R)*eps*R;
         const auto espv=mfem::voigt::StrainTensorToEngVoigt(resp);

         const scalar_t r2 = eta[1]*eta[1] + eta[2]*eta[2];
         const auto sigv=simp_exp > 4 ? C*espv*pow(r2, (simp_exp-4)/2) : C*espv;

         const auto asig=mfem::voigt::VoigtToStressTensor(sigv);
         //compute the anisotropic contribution
         auto stress=R*asig*mfem::future::transpose(R);
         const scalar_t s_p = pow(eta[0],simp_exp);
         const scalar_t L = L1*(1.0-s_p)+L2*s_p;
         const scalar_t M = M1*(1.0-s_p)+M2*s_p;

         //compute stress for the isotropic case
         stress=stress+(L * mfem::future::tr(eps) * I + 2.0 * M * eps);

         return tuple{stress * JxW};
      }
   };
};

AnisoLinElasticSolver::AnisoLinElasticSolver(ParMesh *mesh, int vorder):
   pmesh(mesh),
   dim(mesh->Dimension()),
   spaceDim(mesh->SpaceDimension()),
   vfec(new H1_FECollection(vorder, dim)),
   vfes(new ParFiniteElementSpace(pmesh, vfec, dim, Ordering::byNODES)),
   sol(vfes->GetTrueVSize()),
   adj(vfes->GetTrueVSize()),
   rhs(vfes->GetTrueVSize()),
   fdisp(vfes),
   adisp(vfes),
   fe(vfes->GetFE(0)),
   nodes((pmesh->EnsureNodes(),
          static_cast<ParGridFunction *>(pmesh->GetNodes()))),
   mfes(nodes->ParFESpace()),
   ir(IntRules.Get(fe->GetGeomType(),
                   fe->GetOrder() + fe->GetOrder() + fe->GetDim() - 1))
{
   sol = 0.0;
   rhs = 0.0;
   adj = 0.0;

   fdisp = 0.0;
   adisp = 0.0;

   SetLinearSolver();

   Operator::width = vfes->GetTrueVSize();
   Operator::height = vfes->GetTrueVSize();

   lcsurf_load = std::make_unique<SurfaceLoad>(dim, load_coeff);
   glsurf_load = std::make_unique<SurfaceLoad>(dim, surf_loads);

   if (pmesh->attributes.Size() > 0)
   {
      domain_attributes.SetSize(pmesh->attributes.Max());
      domain_attributes = 1;
   }

   //set the anisotrpic tensor
   if (2==dim)
   {
      aniso_tensor.SetSize(6); //store upper diagonal
   }
   else if (3==dim)
   {
      aniso_tensor.SetSize(21); //store upper diagonal
   }
   aniso_tensor=0.0;
   aniso_tensor.UseDevice(true);

   qs.reset(new QuadratureSpace(*pmesh,ir));
   ups.reset(new future::UniformParameterSpace(
                *pmesh, ir, 1, false /* used_in_tensor_product */));
}

AnisoLinElasticSolver::~AnisoLinElasticSolver()
{
   delete vfes;
   delete vfec;
}

void  AnisoLinElasticSolver::SetLinearSolver(real_t rtol, real_t atol,
                                             int miter)
{
   linear_rtol = rtol;
   linear_atol = atol;
   linear_iter = miter;
}

void AnisoLinElasticSolver::AddDispBC(int id, int dir, real_t val)
{
   if (dir == 0)
   {
      bcx[id] = ConstantCoefficient(val);
      AddDispBC(id, dir, bcx[id]);
   }
   else if (dir == 1)
   {
      bcy[id] = ConstantCoefficient(val);
      AddDispBC(id, dir, bcy[id]);
   }
   else if (dir == 2)
   {
      bcz[id] = ConstantCoefficient(val);
      AddDispBC(id, dir, bcz[id]);
   }
   else if (dir == -1)
   {
      bcx[id] = ConstantCoefficient(val);
      bcy[id] = ConstantCoefficient(val);
      bcz[id] = ConstantCoefficient(val);
      AddDispBC(id, 0, bcx[id]);
      AddDispBC(id, 1, bcy[id]);
      AddDispBC(id, 2, bcz[id]);
   }
   else
   {
      MFEM_ABORT("Invalid BC direction: "
                 "0(x), 1(y), 2(z), or -1(all), got " << dir);
   }
}

void AnisoLinElasticSolver::DelDispBC()
{
   bccx.clear();
   bccy.clear();
   bccz.clear();

   bcx.clear();
   bcy.clear();
   bcz.clear();

   ess_tdofv.DeleteAll();
}

void AnisoLinElasticSolver::AddDispBC(int id, int dir, Coefficient &val)
{
   if (dir == 0) { bccx[id] = &val; }
   else if (dir == 1) { bccy[id] = &val; }
   else if (dir == 2) { bccz[id] = &val; }
   else if (dir == -1)
   {
      bccx[id] = &val;
      bccy[id] = &val;
      bccz[id] = &val;
   }
   else
   {
      MFEM_ABORT("Invalid BC direction: "
                 "0(x), 1(y), 2(z), or -1(all), got " << dir);
   }
   if (pmesh->Dimension() == 2) { bccz.clear(); }
}

void AnisoLinElasticSolver::SetVolForce(real_t fx, real_t fy, real_t fz)
{
   delete lvforce;
   Vector ff(dim);
   ff(0) = fx;
   ff(1) = fy;
   if (dim == 3) { ff(2) = fz; }
   lvforce = new VectorConstantCoefficient(ff);
   volforce = lvforce;
}

void AnisoLinElasticSolver::SetVolForce(VectorCoefficient &fv)
{
   volforce = &fv;
}


void AnisoLinElasticSolver::SetEssTDofs(int j,
                                        ParFiniteElementSpace& scalar_space,
                                        Array<int> &ess_dofs)
{
   // Set the BC
   ess_dofs.DeleteAll();

   auto cbcc = &bccx;
   if (j == 1) { cbcc = &bccy; }
   else if (j == 2) { cbcc = &bccz; }

   Array<int> ess_bdr(pmesh->bdr_attributes.Max()); ess_bdr = 0;

   for (auto it = cbcc->begin(); it != cbcc->end(); it++)
   {
      ess_bdr[it->first - 1] = 1;
   }
   scalar_space.GetEssentialTrueDofs(ess_bdr,ess_dofs);
}


void AnisoLinElasticSolver::SetEssTDofs(Vector &bsol, Array<int> &ess_dofs)
{
   // Set the BC
   ess_tdofv.DeleteAll();

   Array<int> ess_tdofx, ess_tdofy, ess_tdofz;

   for (auto it = bccx.begin(); it != bccx.end(); it++)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[it->first - 1] = 1;
      Array<int> ess_tdof_list;
      vfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list, 0);
      ess_tdofx.Append(ess_tdof_list);

      VectorArrayCoefficient pcoeff(dim);
      pcoeff.Set(0, it->second, false);
      fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
   }
   // copy tdofsx from displacement grid function
   {
      Vector &vc = fdisp.GetTrueVector();
      const int Net = ess_tdofx.Size();
      const auto d_vc = vc.Read();
      const auto d_ess_tdofx = ess_tdofx.Read();
      auto d_bsol = bsol.ReadWrite();
      mfem::forall(Net, [=] MFEM_HOST_DEVICE(int ii)
      {
         d_bsol[d_ess_tdofx[ii]] = d_vc[d_ess_tdofx[ii]];
      });
   }
   ess_tdofx.HostReadWrite(), ess_dofs.HostReadWrite();
   ess_dofs.Append(ess_tdofx);
   ess_tdofx.DeleteAll();

   for (auto it = bccy.begin(); it != bccy.end(); it++)
   {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[it->first - 1] = 1;
      Array<int> ess_tdof_list;
      vfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list, 1);
      ess_tdofy.Append(ess_tdof_list);

      VectorArrayCoefficient pcoeff(dim);
      pcoeff.Set(1, it->second, false);
      fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
   }
   // copy tdofsy from displacement grid function
   {
      Vector &vc = fdisp.GetTrueVector();
      const int Net = ess_tdofy.Size();
      const auto d_vc = vc.Read();
      const auto d_ess_tdofy = ess_tdofy.Read();
      auto d_bsol = bsol.ReadWrite();
      mfem::forall(Net, [=] MFEM_HOST_DEVICE(int ii)
      {
         d_bsol[d_ess_tdofy[ii]] = d_vc[d_ess_tdofy[ii]];
      });
      ess_tdofy.HostReadWrite(), ess_dofs.HostReadWrite();
   }
   ess_dofs.Append(ess_tdofy);
   ess_tdofy.DeleteAll();

   if (dim == 3)
   {
      for (auto it = bccz.begin(); it != bccz.end(); it++)
      {
         Array<int> ess_bdr(pmesh->bdr_attributes.Max());
         ess_bdr = 0;
         ess_bdr[it->first - 1] = 1;
         Array<int> ess_tdof_list;
         vfes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list, 2);
         ess_tdofz.Append(ess_tdof_list);

         VectorArrayCoefficient pcoeff(dim);
         pcoeff.Set(2, it->second, false);
         fdisp.ProjectBdrCoefficient(pcoeff, ess_bdr);
      }

      // copy tdofsz from displacement grid function
      {
         Vector &vc = fdisp.GetTrueVector();
         for (int ii = 0; ii < ess_tdofz.Size(); ii++)
         {
            bsol[ess_tdofz[ii]] = vc[ess_tdofz[ii]];
         }
      }
      ess_dofs.Append(ess_tdofz);
      ess_tdofz.DeleteAll();
   }
}

void AnisoLinElasticSolver::Mult(const Vector &x, Vector &y) const
{

}

void AnisoLinElasticSolver::MultTranspose(const Vector &x,
                                          Vector &y) const
{

}

void AnisoLinElasticSolver::Assemble()
{
   // define the differentiable operator
   drhs = std::make_unique<mfem::future::DifferentiableOperator>(
   std::vector<mfem::future::FieldDescriptor> {{ FDispl, vfes }},
   std::vector<mfem::future::FieldDescriptor>
   {
      { Indicator, ups.get()},
      { Lambda1, ups.get()},
      { Mu1, ups.get()},
      { Lambda2, ups.get()},
      { Mu2, ups.get()},
      { Coords, mfes }
   },
   *pmesh);

   //dfem_mass_op->SetParameters({ dens1.get(), dens2.get(), density.get(), nodes });
   //drhs->SetParameters({});

   const auto finputs =
      mfem::future::tuple
   {
      mfem::future::Gradient<FDispl>{},
      mfem::future::Identity<Indicator>{},
      mfem::future::Identity<Lambda1>{},
      mfem::future::Identity<Mu1>{},
      mfem::future::Identity<Lambda2>{},
      mfem::future::Identity<Mu2>{},
      mfem::future::Gradient<Coords>{},
      mfem::future::Weight{}
   };


   const auto foutputs =
      mfem::future::tuple
   {
      mfem::future::Gradient<FDispl>{}
   };

   auto derivatives = std::integer_sequence<size_t, FDispl> {};

   if (2 == spaceDim)
   {
      using mfem::future::dual;
      using dual_t = dual<real_t, real_t>;

      QFunction<2, 3, dual_t>::Elasticity elasticity_func(aniso_tensor, 4);
      drhs->AddDomainIntegrator(elasticity_func, finputs, foutputs, ir,
                                domain_attributes, derivatives);
   }

   /*
   drhs->SetParameters({eta.get(),
                        lambda1.get(),mu1.get(),
                        lambda2.get(),mu2.get(),
                        nodes});
   */

   fdisp=0.0;

   dr_du=drhs->GetDerivative(FDispl, {&fdisp},
   { eta.get(), l1.get(), m1.get(), l2.get(), m2.get(), nodes });

   dr_du->Assemble(K);

   //set BC to the matrix

}


void AnisoLinElasticSolver::FSolve()
{}
