
#include <memory>
#include "solvers.hpp"

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
template <int DIM, typename scalar_t=real_t> struct QFunction
{
   using matd_t = tensor<scalar_t, DIM, DIM>;


   struct Elasticity
   {

      real_t* aniso_tensor;

      Elasticity(mfem::Vector& tm)
      {
         aniso_tensor=tm.Read();
      }

      MFEM_HOST_DEVICE inline auto operator()(const matd_t &dudxi,
                                              const scalar_t s,
                                              const scalar_t a, const  scalar_t b, 
                                              const real_t &L1, const real_t &M1,
                                              const real_t &L2, const real_t &M2,
                                              const matd_t &J,
                                              const real_t &w) const
      {
         const matd_t JxW = transpose(inv(J)) * det(J) * w;
         constexpr auto I = mfem::future::IsotropicIdentity<DIM>();
         const auto eps = mfem::future::sym(dudxi * mfem::future::inv(J));

         //compute the anisotropic contribution
         scalar_t r2=a*a+b*b;
         //rotation tensor 
         //rotate strain
         //compute stress
         auto stress=0.0*eps;
         //rotate back stress
         //add to isotropic


         //compute stress for the isotropic case
         stress=stress+(L1 * tr(eps) * I + 2.0 * M1 * eps)*(1.0-s)+
                        (L2 * tr(eps) * I + 2.0 * M2 * eps)*s;


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
                   fe->GetOrder() + fe->GetOrder() + fe->GetDim() - 1)),
    qs(*pmesh, ir)
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
   if(2==dim)
   {
      aniso_tensor.SetSize(6); //store upper diagonal
   }
   else
   if(3==dim)
   {
      aniso_tensor.SetSize(21); //store upper diagonal
   }
   aniso_tensor=0.0;
   aniso_tensor.UseDevice(true);
}

AnisoLinElasticSolver::~AnisoLinElasticSolver()
{
   delete vfes;
   delete vfec;  
}

void  AnisoLinElasticSolver::SetLinearSolver(real_t rtol, real_t atol, int miter)
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

}

void AnisoLinElasticSolver::FSolve()
{

}
