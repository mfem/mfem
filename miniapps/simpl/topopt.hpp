#ifndef TOPOPT_HPP
#define TOPOPT_HPP

#include "mfem.hpp"
#include "funs.hpp"
#include "linear_solver.hpp"


namespace mfem
{

class GLVis
{
   Array<socketstream*> sockets;
   Array<GridFunction*> gfs;
   Array<Mesh*> meshes;
   bool parallel;
   const char *hostname;
   const int port;
   bool secure;
public:
#ifdef MFEM_USE_GNUTLS
   static const bool secure_default = true;
#else
   static const bool secure_default = false;
#endif
   GLVis(const char hostname[], int port, bool parallel,
         bool secure = secure_default)
      :sockets(0), gfs(0), meshes(0), parallel(parallel),
       hostname(hostname), port(port), secure(secure_default) {}

   ~GLVis() {sockets.DeleteAll();}

   void Append(GridFunction &gf, const char window_title[]=nullptr,
               const char keys[]=nullptr)
   {
      socketstream *socket = new socketstream(hostname, port, secure);
      if (!socket->is_open())
      {
         return;
      }
      sockets.Append(socket);
      Mesh *mesh = gf.FESpace()->GetMesh();
      gfs.Append(&gf);
      meshes.Append(mesh);
      socket->precision(8);
#ifdef MFEM_USE_MPI
      if (parallel)
      {
         *socket << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() << "\n";
      }
#endif
      *socket << "solution\n" << *mesh << gf;
      if (keys)
      {
         *socket << "keys " << keys << "\n";
      }
      if (window_title)
      {
         *socket << "window_title '" << window_title <<"'\n";
      }
      *socket << std::flush;
   }

   void Update()
   {
      for (int i=0; i<sockets.Size(); i++)
      {
         if (!sockets[i]->is_open())
         {
            continue;
         }
#ifdef MFEM_USE_MPI
         if (parallel)
         {
            *sockets[i] << "parallel " << Mpi::WorldSize() << " " << Mpi::WorldRank() <<
                        "\n";
         }
#endif
         *sockets[i] << "solution\n" << *meshes[i] << *gfs[i];
         *sockets[i] << std::flush;
      }
   }

   socketstream &GetSocket(int i) {return *sockets[i];}
};

class DesignDensity
{
private:
   FiniteElementSpace &fes_control;
   const real_t tot_vol;
   const real_t min_vol;
   const real_t max_vol;
   bool hasPassiveElements;
   LegendreEntropy *entropy;
   std::unique_ptr<GridFunction> zero;
public:
   DesignDensity(
      FiniteElementSpace &fes_control, const real_t tot_vol,
      const real_t min_vol, const real_t max_vol,
      LegendreEntropy *entropy=nullptr);

   real_t ApplyVolumeProjection(GridFunction &x, bool use_entropy);
   bool hasEntropy() {return entropy?true:false;}
};

class StrainEnergyDensityCoefficient : public Coefficient
{
protected:
   Coefficient &lambda;
   Coefficient &mu;
   Coefficient &der_simp_cf;
   GridFunction &state_gf; // displacement
   GridFunction *adjstate_gf; // adjoint displacement
   DenseMatrix grad; // auxiliary matrix, used in Eval
   DenseMatrix adjgrad; // auxiliary matrix, used in Eval

public:
   StrainEnergyDensityCoefficient(Coefficient &lambda, Coefficient &mu,
                                  Coefficient &der_simp_cf,
                                  GridFunction &state_gf, GridFunction *adju_gf=nullptr)
      :lambda(lambda), mu(mu), der_simp_cf(der_simp_cf),
       state_gf(state_gf), adjstate_gf(adju_gf)
   { }

   real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override
   {
      real_t L = lambda.Eval(T, ip);
      real_t M = mu.Eval(T, ip);

      state_gf.GetVectorGradient(T, grad);
      if (adjstate_gf) { adjstate_gf->GetVectorGradient(T, adjgrad); }
      else {adjgrad.UseExternalData(grad.GetData(), grad.NumCols(), grad.NumRows());}

      real_t density = L*grad.Trace()*adjgrad.Trace();
      int dim = T.GetSpaceDim();
      for (int i=0; i<dim; i++)
      {
         for (int j=0; j<dim; j++)
         {
            density += M*grad(i,j)*(adjgrad(i,j)+adjgrad(j,i));
         }
      }
      return -der_simp_cf.Eval(T, ip)*density;
   }
};


class DensityBasedTopOpt
{
private:
   DesignDensity &density;
   GridFunction &control_gf;
   GridFunction &grad_control;
   HelmholtzFilter &filter;
   GridFunction &filter_gf;
   GridFunction &grad_filter;
   GridFunctionCoefficient grad_filter_cf;
   ElasticityProblem &elasticity;
   GridFunction &state_gf;
   std::unique_ptr<GridFunction> adj_state_gf;
   LinearForm &obj;

   std::unique_ptr<L2Projection> L2projector;
   real_t objval;
   real_t current_volume;
public:
   DensityBasedTopOpt(
      DesignDensity &density, GridFunction &gf_control, GridFunction &grad_control,
      HelmholtzFilter &filter, GridFunction &gf_filter, GridFunction &grad_filter,
      ElasticityProblem &elasticity, GridFunction &gf_state)
      :density(density), control_gf(gf_control), grad_control(grad_control),
       filter(filter), filter_gf(gf_filter), grad_filter(grad_filter),
       elasticity(elasticity), state_gf(gf_state),
       obj(elasticity.HasAdjoint() ? *elasticity.GetAdjLinearForm():
           *elasticity.GetLinearForm())
   {
      Array<int> empty(gf_control.FESpace()->GetMesh()->bdr_attributes.Max());
      empty = 0;
      L2projector.reset(new L2Projection(*gf_control.FESpace(), empty));
      grad_filter_cf.SetGridFunction(&grad_filter);
      L2projector->GetLinearForm()->AddDomainIntegrator(new DomainLFIntegrator(
                                                           grad_filter_cf));
   }

   real_t GetCurrentVolume() {return current_volume;}
   real_t GetCurrentObjectValue() {return objval;}

   real_t Eval()
   {
      current_volume = density.ApplyVolumeProjection(control_gf,
                                                     density.hasEntropy());
      filter.Solve(filter_gf);
      elasticity.Solve(state_gf);
      if (elasticity.IsParallel())
      {
#ifdef MFEM_USE_MPI
         objval = InnerProduct(elasticity.GetComm(), obj, state_gf);
#endif
      }
      else
      {
         objval = InnerProduct(obj, state_gf);
      }
      return objval;
   }

   void UpdateGradient()
   {
      if (elasticity.HasAdjoint())
      {
         elasticity.SolveAdjoint(*adj_state_gf);
      }
      filter.SolveAdjoint(grad_filter);
      L2projector->Solve(grad_control);
   }
};

} // end of namespace mfem
#endif
