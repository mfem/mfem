
#ifndef mechanics_operator_hpp
#define mechanics_operator_hpp

#include "mfem.hpp"
#include "mechanics_coefficient.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_solver.hpp"
#include "option_parser.hpp"
#include <iostream>

using namespace std;
using namespace mfem;

class SimVars
{
protected:
   double time;
   double dt;
public:
   double GetTime() const { return time; }
   double GetDTime() const { return dt; }
   
   void SetTime(double t) { time = t; }
   void SetDt(double dtime) { dt = dtime; }
};

//The NonlinearMechOperator class is what really drives the entire system.
//It's responsible for calling the Newton Rhapson solver along with several of
//our post-processing steps. It also contains all of the relevant information
//related to our Krylov iterative solvers.
class NonlinearMechOperator : public TimeDependentOperator
{
public:
   SimVars solVars;
protected:
   ParFiniteElementSpace &fe_space;
   
   ParNonlinearForm *Hform;
   mutable Operator *Jacobian;
   const Vector *x;
   
   /// Newton solver for the operator
   ExaNewtonSolver newton_solver;
   /// Solver for the Jacobian solve in the Newton method
   Solver *J_solver;
   /// Preconditioner for the Jacobian
   Solver *J_prec;
   /// nonlinear model
   ExaModel *model;
   /// Variable telling us if we should use the UMAT specific
   /// stuff
   bool umat_used;
   int newton_iter;
   int myid;
   
public:
   NonlinearMechOperator(ParFiniteElementSpace &fes,
                         Array<int> &ess_bdr,
                         ExaOptions &options,
                         QuadratureFunction &q_matVars0,
                         QuadratureFunction &q_matVars1,
                         QuadratureFunction &q_sigma0,
                         QuadratureFunction &q_sigma1,
                         QuadratureFunction &q_matGrad,
                         QuadratureFunction &q_kinVars0,
                         QuadratureFunction &q_vonMises,
                         ParGridFunction &beg_crds,
                         ParGridFunction &end_crds,
                         ParMesh *&pmesh,
                         Vector &matProps,
                         int nStateVars);
   
   /// Required to use the native newton solver
   virtual Operator &GetGradient(const Vector &x) const;
   virtual void Mult(const Vector &k, Vector &y) const;
   //We need the solver to update the end coords after each iteration has been complete
   //We'll also want to have a way to update the coords before we start running the simulations.
   //It might also allow us to set a velocity at every point, so we could test the models almost
   //as if we're doing a MPS.
   void UpdateEndCoords(const Vector& vel) const;
   /// Driver for the newton solver
   void Solve(Vector &x) const;
   
   /// Solve the Newton system for the 1st time step
   /// It was found that for large meshes a ramp up to our desired applied BC might
   /// be needed. It should be noted that this is no longer a const function since
   /// we modify several values/objects held by our class.
   void SolveInit(Vector &x);
   
   /// Get essential true dof list, if required
   const Array<int> &GetEssTDofList();
   
   /// Get FE space
   const ParFiniteElementSpace *GetFESpace() { return &fe_space; }
   
   /// routine to update beginning step model variables with converged end
   /// step values
   void UpdateModel(const Vector &x);
   /// Computes a volume average tensor/vector of some quadrature function
   /// it returns the vol avg value.
   void ComputeVolAvgTensor(const ParFiniteElementSpace* fes,
                            const QuadratureFunction* qf,
                            Vector& tensor,
                            int size);
   
   void ProjectModelStress(ParGridFunction &s);
   void ProjectVonMisesStress(ParGridFunction &vm);
   
   void SetTime(const double t);
   void SetDt(const double dt);
   void SetModelDebugFlg(const bool dbg);
   
   void DebugPrintModelVars(int procID, double time);
   /// Tests the deformation gradient function
   void testFuncs(const Vector &x0, ParFiniteElementSpace *fes);
   
   virtual ~NonlinearMechOperator();
};


#endif /* mechanics_operator_hpp */
