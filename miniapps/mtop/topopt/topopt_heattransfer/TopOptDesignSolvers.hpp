#ifndef HT_DESIGNSOLVERS_HPP
#define HT_DESIGNSOLVERS_HPP

#include "mfem.hpp"
#include <cmath>
#include <memory>
#include <vector>
#include <iomanip>
#include <iostream>
#include "TopOptIMEXIntegrators.hpp"
#include "../../pde_filter.hpp"

namespace mfem
{
    class DesignSolver
{
   private:
   // Finite Element Spaces
   ParFiniteElementSpace state_fes;
   ParFiniteElementSpace filter_fes;
   ParFiniteElementSpace control_fes;
   IMEXAdvectionDiffusionSolver *oper;
   std::vector<Vector> states;
   std::vector<real_t> times;

   // Design Optimization
   toopt::PDEFilter &filter;
   HeatTransferObjectiveFunction &objective;
   Vector dJ_drho_tilde;
   SIMPCoefficient SIMP_cf;

   // Boundary Conditions
   Array<int> ess_tdof_list;
   Array<int> ess_bdr_attr;
   Array<int> inflow_bdr;

   // PDE Coefficients
   VectorFunctionCoefficient v_base;
   real_t dt_diff_term;
   FunctionCoefficient raw_inflow;
   real_t raw_diff_term;

   // Time Integration 
   int nsteps;
   real_t dt;
   real_t t_final;
   ParGridFunction &rho;         // working density (also the driver's ParaView field)
   ParGridFunction &rho_tilde;   // filtered density
   GridFunctionCoefficient q0;          // initial condition
   ParGridFunction q_gf;
   HypreParVector *q_vec;

   bool paraview_vis;


   int outer_it;
   int vis_steps;

   MPI_Comm comm;
   int imex_integrator;
   int problem_type;

   public:
   DesignSolver(ParFiniteElementSpace &state_fes_,
                         ParFiniteElementSpace &filter_fes_,
                         ParFiniteElementSpace &control_fes_,
                         toopt::PDEFilter &filter_,
                         Array<int> &ess_bdr_attr_,
                         Array<int> &inflow_bdr_,
                         HeatTransferObjectiveFunction &objective_,
                         VectorFunctionCoefficient &v_base_,
                         real_t raw_diff_term_,
                         real_t dt_diff_term_,
                         FunctionCoefficient &raw_inflow_,
                         GridFunctionCoefficient &q0_,
                         int nsteps_, real_t dt_, real_t t_final_,
                         ParGridFunction &rho_,
                         ParGridFunction &rho_tilde_,
                         SIMPCoefficient &SIMP_cf_, 
                         int imex_integrator_, int vis_steps_, int problem_type_, MPI_Comm comm_)
      : state_fes(state_fes_), filter_fes(filter_fes_), control_fes(control_fes_),
        filter(filter_),
        ess_bdr_attr(ess_bdr_attr_), inflow_bdr(inflow_bdr_),
        objective(objective_), 
        dt_diff_term(dt_diff_term_), raw_inflow(raw_inflow_), q0(q0_), raw_diff_term(raw_diff_term_),
        nsteps(nsteps_), dt(dt_), t_final(t_final_),
        rho(rho_), rho_tilde(rho_tilde_), q_gf(&state_fes_), imex_integrator(imex_integrator_), v_base(v_base_), SIMP_cf(SIMP_cf_),
        q_vec(nullptr), oper(nullptr), vis_steps(vis_steps_), problem_type(problem_type_), comm(comm_)
   { 
      outer_it = 0;
      dJ_drho_tilde.SetSize(filter_fes.GetTrueVSize());
      dJ_drho_tilde = 0.0;
   }

   ~DesignSolver() 
   { 
      if (oper) delete oper; 
      if (q_vec) delete q_vec;
   }

   int NumSteps() const {return nsteps;}
   real_t Time_Step() const {return dt;}

   // 1. Forward Filter. Raw control density -> filtered density (Helmholtz solve).
   void FilterFSolve(const Vector &rho_tv)
   {
      rho.SetFromTrueDofs(rho_tv);
      filter.Mult(rho, rho_tilde);
      rho_tilde.ExchangeFaceNbrData();
   }

   // 2. Forward physics: (re)assemble the operator for the current rho_tilde_, run
   //    the IMEX Forward Integration, store the trajectory, return J.
   real_t PhysicsFSolve()
   {
      if (oper) { delete oper; oper = nullptr; }
      if (q_vec) { delete q_vec; q_vec = nullptr; }
      std::unique_ptr<TopOptRKIMEXSolver> ode_solver = TopOptRKIMEXSolver::SelectTopOptRKIMEX(imex_integrator);
      oper = new IMEXAdvectionDiffusionSolver(state_fes, 
         raw_inflow, 
         v_base, dt_diff_term, 
         raw_diff_term, q0, 
         rho_tilde, dt, 
         t_final, SIMP_cf,
         comm, inflow_bdr, ess_bdr_attr);
      if (problem_type == 1){oper->InitializeInjectionProblem();}
      else if (problem_type == 2){oper->InitializeFlowProblem();}
      else{MFEM_ABORT("Unknown Problem Type: " << problem_type );}
      objective.Reset();

      q_gf = oper->Getq();
      // q_gf.ExchangeFaceNbrData();
      q_vec = q_gf.GetTrueDofs();
      real_t acc = objective.AccumulateTimestep(q_gf, dt, 0, nsteps);
      ParaViewDataCollection *pd = NULL;
      if (paraview_vis)
      {
         pd = new ParaViewDataCollection("forward", state_fes.GetParMesh());
         pd->SetPrefixPath("ParaView");
         pd->RegisterField("solution", &q_gf);
         pd->SetLevelsOfDetail(state_fes.GetOrder(0));
         pd->SetDataFormat(VTKFormat::BINARY);
         pd->SetHighOrderOutput(false);
         pd->SetCycle(0);
         pd->SetTime(0.0);
         pd->Save();
      }
      real_t t = 0.0;
      times.resize(nsteps);
      ode_solver->Init(*oper);
      oper->SetTime(t);
      bool done = false;
      int myrank;
      MPI_Comm_rank(comm, &myrank);
      // std::cout << "my_rank = " << myrank << "time step initial " << ", time: 0 " << ", ||q|| = " << q_vec->Norml2() << std::endl;
      for (int ti = 0; !done; )
      {
         real_t dt_real = std::min(dt, t_final - t);  
         oper->UpdateDt(dt_real);
         times[ti] = dt_real;
         ode_solver->Step(*q_vec, t, dt_real);
         q_gf.SetFromTrueDofs(*q_vec);
         acc = objective.AccumulateTimestep(q_gf, dt_real, ti, nsteps);
         // real_t current_obj = objective.GetObjective();
         // std::cout << "contribution = " << acc << " objective so far = " << current_obj << " contrib check = " << current_obj - acc << std::endl;
         // std::cout << "my_rank = " << myrank << "time step: " << ti << ", time: " << t << ", ||q|| = " << q_vec->Norml2() << std::endl;
         ti++;
         oper->SetStep(ti);
         oper->StoreTraj(ti, *q_vec);
         oper->SetTime(t);
         done = (t >= t_final - 1e-8*dt); 
         if (done || ti % vis_steps == 0)
         {
         q_gf.SetFromTrueDofs(*q_vec);
         if (paraview_vis)
         {
            pd->SetCycle(ti);
            pd->SetTime(t);
            pd->Save();
         }
         }
      }
      q_gf.SetFromTrueDofs(*q_vec);
      oper->Updateq(q_gf);
      //acc = objective.AccumulateTimestep(q_gf, dt, nsteps-1, nsteps);
      // objective.ComputeObjective(q_gf);
      return objective.GetObjective();
   }

   // 3. Adjoint physics: backward discrete-adjoint sweep -> dJ/d(rho_tilde).
   void PhysicsASolve()
   {
      std::unique_ptr<TopOptRKIMEXSolver> ode_solver = TopOptRKIMEXSolver::SelectTopOptRKIMEX(imex_integrator);
      MFEM_VERIFY(oper, "PhysicsASolve() requires a preceding PhysicsFSolve().");
      const int myid = Mpi::WorldRank();
      ParGridFunction lam_gf(&state_fes);
      ParLinearForm grad_form(&state_fes);
      objective.ComputeObjectiveGradient(q_gf, times[nsteps-1], nsteps-1, nsteps,grad_form);
      HypreParVector* grad_vec = grad_form.ParallelAssemble();

      // 3. Set the primal GridFunction from the True-Dofs
      HypreParVector lam_vec = *grad_vec;
      lam_vec *= -1.0;
      lam_gf.SetFromTrueDofs(lam_vec);
      oper->SetStep(nsteps);
      oper->TakeAdjoint();
      ode_solver->Init(*oper);
      ParaViewDataCollection *pd_adj = NULL;
      if (paraview_vis)
      {
         pd_adj = new ParaViewDataCollection("adjoint", state_fes.GetParMesh());
         pd_adj->SetPrefixPath("ParaView");
         pd_adj->RegisterField("solution", &lam_gf);
         pd_adj->SetLevelsOfDetail(state_fes.GetOrder(0));
         pd_adj->SetDataFormat(VTKFormat::BINARY);
         pd_adj->SetHighOrderOutput(false); 
         pd_adj->SetCycle(0);
         pd_adj->SetTime(t_final);
         pd_adj->Save();
      } 
      real_t t = t_final;
      bool done = false;
      for (int ti = 0; !done;)
      {
         real_t dti = times[nsteps-ti-1]; 
         oper->UpdateDt(dti);
         real_t t_dummy = t;
         oper->GetTraj(oper->GetStep() - 1, *q_vec);
         q_gf.SetFromTrueDofs(*q_vec);
         ode_solver->AdjointStep(lam_vec,*q_vec, dJ_drho_tilde, t_dummy, dti);
         ParLinearForm grad_form2(&state_fes);
         objective.ComputeObjectiveGradient(q_gf, times[nsteps-ti-2], nsteps - ti - 2, nsteps, grad_form2);
         grad_vec = grad_form2.ParallelAssemble();
         lam_vec.Add(-1.0, *grad_vec);
         ti++;
         oper->SetStep(nsteps-ti);
         t -= dti;
         oper->SetTime(t);
         done = (t <= 1e-8*dt); 
         if (done || ti % vis_steps == 0)
         {
            if (Mpi::Root())
            {
               // std::cout << "time step: " << ti << ", time: " << t << ", dt = " << dti << std::endl;  
            }
            // lam_gf = *lambda;
            lam_gf.SetFromTrueDofs(lam_vec);
            if (paraview_vis)
            {
               pd_adj->SetCycle(ti);
               pd_adj->SetTime(t_final-t);
               pd_adj->Save();
            }
         }
      }
      //dJ_drho_tilde = oper->GetDesignGrad();
      delete grad_vec;
   } 

   // 4. Adjoint filter: transpose the filter, dJ/d(rho_tilde) -> dJ/d(rho).
   void FilterASolve(Vector &dJ_drho)
   {
      filter.MultTranspose(dJ_drho_tilde, dJ_drho);
      MFEM_VERIFY(dJ_drho.Size() == control_fes.GetTrueVSize(),
                  "Raw design gradient has unexpected size.");

   }

   // Convenience: the four steps in sequence (forward filter + physics, adjoint
   // physics + filter). Returns J and fills dJ_drho.
   real_t ObjectiveAndGradient(const Vector &rho_tv, Vector &dJ_drho,
                               int outer_it = -1)
   {
      // FilterFSolve(rho_tv);
      // const real_t J = PhysicsFSolve();
      // PhysicsASolve();
      // FilterASolve(dJ_drho);
      const real_t J = 0.0;
      std::cout << "Not implemented " << std::endl;
      return J;
   }

   // Forward-only objective J(rho) (no gradient / no stored trajectory).
   real_t Objective(const Vector &rho_tv)
   {
      return 0.0;
      // return EvaluateDesignObjective(
      //           rho_tv, x0_, state_fes_, control_fes_, rho_, rho_tilde_, filter_,
      //           gamma_coef_, exterior_bdr_attr_, ess_bdr_attr_, objective_, mat_,
      //           load_spec_, load_coef_, impedance_, nsteps_, h_, mass_type_);
   }
};
}
#endif 

