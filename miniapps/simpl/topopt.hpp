#include "mfem.hpp"
#include "funs.hpp"
#include "linear_solver.hpp"


namespace mfem
{

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

   real_t ApplyVolumeProjection(GridFunction &x);
};

class DensityBasedTopOpt
{
private:
   DesignDensity &density;
   GridFunction &gf_control;
   GridFunction &grad_control;
   HelmholtzFilter &filter;
   GridFunction &gf_filter;
   GridFunction &grad_filter;
   GridFunctionCoefficient grad_filter_cf;
   ElasticityProblem &elasticity;
   GridFunction &gf_state;
   std::unique_ptr<GridFunction> gf_adj_state;
   LinearForm &obj;

   OperatorHandle L2projector;
   std::unique_ptr<LinearForm> grad_filter_form;
   real_t objval;
public:
   DensityBasedTopOpt(
      DesignDensity &density, GridFunction &gf_control, GridFunction &grad_control,
      HelmholtzFilter &filter, GridFunction &gf_filter, GridFunction &grad_filter,
      ElasticityProblem &elasticity, GridFunction &gf_state)
      :density(density), gf_control(gf_control), grad_control(grad_control),
       filter(filter), gf_filter(gf_filter), grad_filter(grad_filter),
       elasticity(elasticity), gf_state(gf_state),
       obj(elasticity.HasAdjoint() ? *elasticity.GetAdjLinearForm():
           *elasticity.GetLinearForm())
   {
      // setup L2Projector
      FiniteElementSpace *fes_control = gf_control.FESpace();
      bool parallel = false;
      std::unique_ptr<BilinearForm> L2projector_bilf;
#ifdef MFEM_USE_MPI
      ParFiniteElementSpace *pfes_control
         = dynamic_cast<ParFiniteElementSpace*>(fes_control);
      if (pfes_control)
      {
         L2projector_bilf.reset(new ParBilinearForm(pfes_control));
         grad_filter_form.reset(new ParLinearForm(pfes_control));
      }
      else
      {
         L2projector_bilf.reset(new BilinearForm(fes_control));
         grad_filter_form.reset(new LinearForm(fes_control));
      }
#else
      L2projector_bilf.reset(new BilinearForm(fes_control));
      grad_filter_form.reset(new LinearForm(fes_control));
#endif
      L2projector_bilf->AddDomainIntegrator(new InverseIntegrator(
                                               new MassIntegrator()));
      L2projector_bilf->Assemble();
      if (parallel)
      {
#ifdef MFEM_USE_MPI
         L2projector.Reset<HypreParMatrix>(
            static_cast<ParBilinearForm*>(L2projector_bilf.get())->ParallelAssemble(),
            true);
#endif
      }
      else
      {
         L2projector.Reset<SparseMatrix>(L2projector_bilf->LoseMat(), true);
      }
      grad_filter_cf.SetGridFunction(&grad_filter);
      grad_filter_form->AddDomainIntegrator(new DomainLFIntegrator(grad_filter_cf));

      if (elasticity.HasAdjoint())
      {
         FiniteElementSpace *fes_state = gf_state.FESpace();
         if (elasticity.IsParallel())
         {
#ifdef MFEM_USE_MPI
            ParFiniteElementSpace *pfes_state
               = static_cast<ParFiniteElementSpace*>(fes_state);
            gf_adj_state.reset(new ParGridFunction(pfes_state));
#endif
         }
         else
         {
            gf_adj_state.reset(new GridFunction(fes_state));
         }
      }
   }

   real_t Eval()
   {
      density.ApplyVolumeProjection(gf_control);
      filter.Solve(gf_filter);
      elasticity.Solve(gf_state);
      if (elasticity.IsParallel())
      {
#ifdef MFEM_USE_MPI
         objval = InnerProduct(elasticity.GetComm(), obj, gf_state);
#endif
      }
      else
      {
         objval = InnerProduct(obj, gf_state);
      }
      return objval;
   }

   void UpdateGradient()
   {
      if (elasticity.HasAdjoint())
      {
         elasticity.SolveAdjoint(*gf_adj_state);
      }
      filter.SolveAdjoint(grad_filter);
      grad_filter_form->Assemble();
      if (filter.IsParallel())
      {
#ifdef MFEM_USE_MPI
         std::unique_ptr<HypreParVector> v(static_cast<ParLinearForm*>
                                           (grad_filter_form.get())->ParallelAssemble());
         L2projector->Mult(*v, grad_control.GetTrueVector());
#endif
      }
      else
      {
         L2projector->Mult(*grad_filter_form, grad_control.GetTrueVector());
      }
      grad_control.SetFromTrueVector();
   }
};

} // end of namespace mfem
