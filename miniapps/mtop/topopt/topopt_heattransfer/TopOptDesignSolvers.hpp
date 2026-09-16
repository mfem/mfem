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
   ParFiniteElementSpace qoi_fes;
   ParFiniteElementSpace filter_fes;
   ParFiniteElementSpace control_fes;
 
   // Physics Operator
   std::unique_ptr<MixedMultiPhysicsOperator> oper;  

   std::vector<real_t> times;

   // Design Optimization
   toopt::PDEFilter &filter;
   HeatTransferObjectiveFunction &objective;   
   Vector dJ_drho_tilde;

   // Time Integration 
   int nsteps;
   real_t dt;
   real_t t_final;
   ParGridFunction &rho;         // working density (also the driver's ParaView field)
   ParGridFunction &rho_tilde;   // filtered density
   ParGridFunction qoi_gf; // to pass into objective
   BlockVector state_vec;
   BlockVector lam_vec;

   GridFunctionCoefficient q0;
   const Array<int> offsets;


   bool paraview_vis;


   int outer_it;
   int vis_steps;

   MPI_Comm comm;
   int imex_integrator;

   public:
   DesignSolver(ParFiniteElementSpace &qoi_fes_,
                         ParFiniteElementSpace &filter_fes_,
                         ParFiniteElementSpace &control_fes_,  
                         std::unique_ptr<MixedMultiPhysicsOperator> &oper_, 
                         toopt::PDEFilter &filter_,
                         HeatTransferObjectiveFunction &objective_,
                         GridFunctionCoefficient &q0_,
                         int nsteps_, real_t dt_, real_t t_final_,
                         ParGridFunction &rho_,
                         ParGridFunction &rho_tilde_,
                         int imex_integrator_, int vis_steps_, MPI_Comm comm_)
      : qoi_fes(qoi_fes_), filter_fes(filter_fes_), control_fes(control_fes_),
        filter(filter_),
        objective(objective_),  q0(q0_),
        nsteps(nsteps_), dt(dt_), t_final(t_final_),
        rho(rho_), rho_tilde(rho_tilde_), qoi_gf(&qoi_fes_), imex_integrator(imex_integrator_),
      oper(std::move(oper_)), vis_steps(vis_steps_), comm(comm_), offsets(oper->GetSystemOffsets())
   { 
      outer_it = 0;
      dJ_drho_tilde.SetSize(filter_fes.GetTrueVSize());
      dJ_drho_tilde = 0.0;  
      // state_vec = new BlockVector(&); 
   }

   ~DesignSolver() 
   { 
      //if (oper) delete oper; 
      //if (state_vec) delete state_vec;
   }

   void SetNewInitialCondition(GridFunctionCoefficient &q0_new){q0 = q0_new;}
 
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
      // 1. Allocate memory owned by state_vec using the correct block offsets
      state_vec.Update(offsets);  

      // 2. Safely copy the values from the temporary object into state_vec
      state_vec = oper->InitializeOperators(rho_tilde);

      FiniteElementCollection *fluid_heat_fec = new DG_FECollection(2, 2, BasisType::GaussLobatto);
      ParFiniteElementSpace *fluid_heat_fes = new ParFiniteElementSpace(qoi_fes.GetParMesh(), fluid_heat_fec); 
      ParGridFunction qoi2_gf(fluid_heat_fes);

      std::unique_ptr<TopOptRKIMEXSolver> ode_solver = TopOptRKIMEXSolver::SelectTopOptRKIMEX(imex_integrator);
      objective.Reset();

      qoi_gf.SetFromTrueDofs(state_vec.GetBlock(0));
      qoi2_gf.SetFromTrueDofs(state_vec.GetBlock(1));
 
      real_t acc = objective.AccumulateTimestep(qoi_gf, dt, 0, nsteps);
      ParaViewDataCollection *pd = NULL;
      if (paraview_vis)
      {
         pd = new ParaViewDataCollection("forward", qoi_fes.GetParMesh());
         pd->SetPrefixPath("ParaView");
         pd->RegisterField("solid", &qoi_gf);
         pd->RegisterField("fluid", &qoi2_gf);
         pd->SetLevelsOfDetail(fluid_heat_fes->GetOrder(0)); 
         pd->SetDataFormat(VTKFormat::BINARY);
         pd->SetHighOrderOutput(true);
         pd->SetCycle(0);
         pd->SetTime(0.0);
         pd->Save();
      }
      real_t t = 0.0;
      times.clear(); // Clear the vector instead of using resize()
      ode_solver->Init(*oper);
      oper->SetTime(t);
      bool done = false;
      if(Mpi::Root()){std::cout<<"Time: " << t << "; ||q|| = " << qoi_gf.Norml2() << std::endl;}
      int ti = 0;
      for (; !done; )
      {
         real_t dt_real = std::min(dt, t_final - t);  
         oper->UpdateDt(dt_real);
         times.push_back(dt_real);
         ode_solver->Step(state_vec, t, dt_real);
         qoi_gf.SetFromTrueDofs(state_vec.GetBlock(0));
         qoi2_gf.SetFromTrueDofs(state_vec.GetBlock(1));
         acc = objective.AccumulateTimestep(qoi_gf, dt_real, ti, nsteps);
         ti++;
         oper->SetStep(ti);
         oper->StoreTraj(ti, state_vec);
         oper->SetTime(t);
         done = (t >= t_final - 1e-8*dt); 
         if (done || ti % vis_steps == 0)
         {
            if(Mpi::Root()){std::cout<<"Time: " << t << "; ||qs|| = " << qoi_gf.Norml2() << std::endl;}
            if(Mpi::Root()){std::cout<<"Time: " << t << "; ||qf|| = " << qoi2_gf.Norml2() << std::endl;}
            if (paraview_vis)
            {
               pd->SetCycle(ti);
               pd->SetTime(t);
               pd->Save();
            }
         }
      }
      nsteps = times.size();
      qoi_gf.SetFromTrueDofs(state_vec.GetBlock(0));
      delete pd;
      return objective.GetObjective();
   }

   // 3. Adjoint physics: backward discrete-adjoint sweep -> dJ/d(rho_tilde).
   void PhysicsASolve()
   {
      std::unique_ptr<TopOptRKIMEXSolver> ode_solver = TopOptRKIMEXSolver::SelectTopOptRKIMEX(imex_integrator);
      MFEM_VERIFY(oper, "PhysicsASolve() requires a preceding PhysicsFSolve().");

      ParGridFunction lam_gf(&qoi_fes);
      ParLinearForm grad_form(&qoi_fes);
      objective.ComputeObjectiveGradient(qoi_gf, times[nsteps-1], nsteps-1, nsteps, grad_form);
      Vector* grad_vec = grad_form.ParallelAssemble();

      // 3. Set the primal GridFunction from the True-Dofs
      lam_vec.Update(oper->GetSystemOffsets());
      lam_vec.GetBlock(0) = *grad_vec;
      delete grad_vec;
      for (int idx = 1; idx < oper->GetSystemOffsets().Size() - 1; idx++)
      {
         lam_vec.GetBlock(idx) = 0.0;
      }
      lam_vec *= -1.0;

      lam_gf.SetFromTrueDofs(lam_vec.GetBlock(0));

      oper->SetStep(nsteps);
      ode_solver->Init(*oper);

      // ParaViewDataCollection *pd_adj = NULL;
      // if (paraview_vis)
      // {
      //    pd_adj = new ParaViewDataCollection("adjoint", qoi_fes.GetParMesh());
      //    pd_adj->SetPrefixPath("ParaView");
      //    pd_adj->RegisterField("solution", &lam_gf);
      //    pd_adj->SetLevelsOfDetail(qoi_fes.GetOrder(0));
      //    pd_adj->SetDataFormat(VTKFormat::BINARY);
      //    pd_adj->SetHighOrderOutput(false); 
      //    pd_adj->SetCycle(0);
      //    pd_adj->SetTime(t_final);
      //    pd_adj->Save();
      // } 
      real_t t = t_final;
      bool done = false;
      for (int ti = 0; !done;)
      {
         real_t dti = times[nsteps-ti-1]; 
         oper->UpdateDt(dti);
         real_t t_dummy = t;
         oper->GetTraj(oper->GetStep() - 1, state_vec);

         BlockVector pristine_state(offsets);
         pristine_state = state_vec;

         ode_solver->AdjointStep(lam_vec, state_vec, dJ_drho_tilde, t_dummy, dti);
         ParLinearForm grad_form2(&qoi_fes);
         qoi_gf.SetFromTrueDofs(pristine_state.GetBlock(0));
         int prev_step_idx = nsteps - ti - 2;
         real_t prev_dt = (prev_step_idx >= 0) ? times[prev_step_idx] : dt;
         objective.ComputeObjectiveGradient(qoi_gf, prev_dt, prev_step_idx, nsteps, grad_form2);
         Vector* loop_grad_vec = grad_form2.ParallelAssemble();
         lam_vec.GetBlock(0).Add(-1.0, *loop_grad_vec);
         delete loop_grad_vec;
         ti++; 
         oper->SetStep(nsteps-ti);
         t -= dti;
         oper->SetTime(t);
         done = (t <= 1e-8*dt); 
         if (done || ti % vis_steps == 0)
         {
            lam_gf.SetFromTrueDofs(lam_vec.GetBlock(0));
            // if (paraview_vis)
            // {
            //    pd_adj->SetCycle(ti);
            //    pd_adj->SetTime(t_final-t);
            //    pd_adj->Save();
            // }
         }
      }
      oper->AddStaticDesignGradient(lam_vec, dJ_drho_tilde);
      //delete pd_adj;
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

