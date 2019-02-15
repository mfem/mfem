

#include "mechanics_operator.hpp"
#include "mfem.hpp"
#include "mechanics_coefficient.hpp"
#include "mechanics_integrators.hpp"
#include "mechanics_solver.hpp"
#include "option_parser.hpp"
#include <iostream>

using namespace std;
using namespace mfem;


NonlinearMechOperator::NonlinearMechOperator(ParFiniteElementSpace &fes,
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
                                             int nStateVars)
: TimeDependentOperator(fes.TrueVSize()), fe_space(fes),
newton_solver(fes.GetComm())
{
   Vector * rhs;
   rhs = NULL;
   
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   
   umat_used = options.umat;
   
   // Define the parallel nonlinear form
   Hform = new ParNonlinearForm(&fes);
   
   // Set the essential boundary conditions
   Hform->SetEssentialBCPartial(ess_bdr, rhs);
   
   if (options.umat) {
      //Our class will initialize our deformation gradients and
      //our local shape function gradients which are taken with respect
      //to our initial mesh when 1st created.
      model = new AbaqusUmatModel(&q_sigma0, &q_sigma1, &q_matGrad, &q_matVars0, &q_matVars1,
                                  &q_kinVars0, &beg_crds, &end_crds, pmesh,
                                  &matProps, options.nProps, nStateVars, &fes);
      
      // Add the user defined integrator
      Hform->AddDomainIntegrator(new ExaNLFIntegrator(dynamic_cast<AbaqusUmatModel*>(model)));
      
   }
   
   model->setVonMisesPtr(&q_vonMises);
   
   if (options.solver == KrylovSolver::GMRES) {
      
      HypreBoomerAMG *prec_amg = new HypreBoomerAMG();
      HYPRE_Solver h_amg = (HYPRE_Solver) *prec_amg;
      HYPRE_Real st_val = 0.90;
      HYPRE_Real rt_val = -10.0;
      HYPRE_Real om_val = 1.0;
      
      //
      int ml = HYPRE_BoomerAMGSetMaxLevels(h_amg, 30);
      int ct = HYPRE_BoomerAMGSetCoarsenType(h_amg, 0);
      int mt = HYPRE_BoomerAMGSetMeasureType(h_amg, 0);
      int st = HYPRE_BoomerAMGSetStrongThreshold(h_amg, st_val);
      int ns = HYPRE_BoomerAMGSetNumSweeps(h_amg, 3);
      int rt = HYPRE_BoomerAMGSetRelaxType(h_amg, 8);
      //int rwt = HYPRE_BoomerAMGSetRelaxWt(h_amg, rt_val);
      //int ro = HYPRE_BoomerAMGSetOuterWt(h_amg, om_val);
      //Dimensionality of our problem
      int ss = HYPRE_BoomerAMGSetNumFunctions(h_amg, 3);
      int smt = HYPRE_BoomerAMGSetSmoothType(h_amg, 3);
      int snl = HYPRE_BoomerAMGSetSmoothNumLevels(h_amg, 3);
      int sns = HYPRE_BoomerAMGSetSmoothNumSweeps(h_amg, 3);
      int sv = HYPRE_BoomerAMGSetVariant(h_amg, 0);
      int so = HYPRE_BoomerAMGSetOverlap(h_amg, 0);
      int sdt = HYPRE_BoomerAMGSetDomainType(h_amg, 1);
      int srw = HYPRE_BoomerAMGSetSchwarzRlxWeight(h_amg, rt_val);
      
      prec_amg->SetPrintLevel(0);
      
      J_prec = prec_amg;
      
      GMRESSolver *J_gmres = new GMRESSolver(fe_space.GetComm());
      //These tolerances are currently hard coded while things are being debugged
      //but they should eventually be moved back to being set by the options
      //      J_gmres->iterative_mode = false;
      //The relative tolerance should be at this point or smaller
      J_gmres->SetRelTol(options.krylov_rel_tol);
      //The absolute tolerance could probably get even smaller then this
      J_gmres->SetAbsTol(options.krylov_abs_tol);
      J_gmres->SetMaxIter(options.krylov_iter);
      J_gmres->SetPrintLevel(0);
      J_gmres->SetPreconditioner(*J_prec);
      J_solver = J_gmres;
      
   }else if (options.solver == KrylovSolver::PCG){
      
      HypreBoomerAMG *prec_amg = new HypreBoomerAMG();
      HYPRE_Solver h_amg = (HYPRE_Solver) *prec_amg;
      HYPRE_Real st_val = 0.90;
      HYPRE_Real rt_val = -10.0;
      HYPRE_Real om_val = 1.0;
      //
      int ml = HYPRE_BoomerAMGSetMaxLevels(h_amg, 30);
      int ct = HYPRE_BoomerAMGSetCoarsenType(h_amg, 0);
      int mt = HYPRE_BoomerAMGSetMeasureType(h_amg, 0);
      int st = HYPRE_BoomerAMGSetStrongThreshold(h_amg, st_val);
      int ns = HYPRE_BoomerAMGSetNumSweeps(h_amg, 3);
      int rt = HYPRE_BoomerAMGSetRelaxType(h_amg, 8);
      //int rwt = HYPRE_BoomerAMGSetRelaxWt(h_amg, rt_val);
      //int ro = HYPRE_BoomerAMGSetOuterWt(h_amg, om_val);
      //Dimensionality of our problem
      int ss = HYPRE_BoomerAMGSetNumFunctions(h_amg, 3);
      int smt = HYPRE_BoomerAMGSetSmoothType(h_amg, 3);
      int snl = HYPRE_BoomerAMGSetSmoothNumLevels(h_amg, 3);
      int sns = HYPRE_BoomerAMGSetSmoothNumSweeps(h_amg, 3);
      int sv = HYPRE_BoomerAMGSetVariant(h_amg, 0);
      int so = HYPRE_BoomerAMGSetOverlap(h_amg, 0);
      int sdt = HYPRE_BoomerAMGSetDomainType(h_amg, 1);
      int srw = HYPRE_BoomerAMGSetSchwarzRlxWeight(h_amg, rt_val);
      
      prec_amg->SetPrintLevel(0);
      J_prec = prec_amg;
      
      CGSolver *J_pcg = new CGSolver(fe_space.GetComm());
      //These tolerances are currently hard coded while things are being debugged
      //but they should eventually be moved back to being set by the options
      //The relative tolerance should be at this point or smaller
      J_pcg->SetRelTol(options.krylov_rel_tol);
      //The absolute tolerance could probably get even smaller then this
      J_pcg->SetAbsTol(options.krylov_abs_tol);
      J_pcg->SetMaxIter(options.krylov_iter);
      J_pcg->SetPrintLevel(0);
      J_pcg->iterative_mode = true;
      J_pcg->SetPreconditioner(*J_prec);
      J_solver = J_pcg;
      
   }//The SuperLU capabilities were gotten rid of due to the size of our systems
   //no longer making it a viable option to keep 1e6+ dof systems
   //Also, a well tuned PCG should be much faster than SuperLU for systems roughly
   //5e5 and up.
   else {
      printf("using minres solver \n");
      HypreSmoother *J_hypreSmoother = new HypreSmoother;
      J_hypreSmoother->SetType(HypreSmoother::l1Jacobi);
      J_hypreSmoother->SetPositiveDiagonal(true);
      J_prec = J_hypreSmoother;
      
      MINRESSolver *J_minres = new MINRESSolver(fe_space.GetComm());
      J_minres->SetRelTol(options.krylov_rel_tol);
      J_minres->SetAbsTol(options.krylov_abs_tol);
      J_minres->SetMaxIter(options.krylov_iter);
      J_minres->SetPrintLevel(-1);
      J_minres->SetPreconditioner(*J_prec);
      J_solver = J_minres;
      
   }
   //We might want to change our # iterations used in the newton solver
   //for the 1st time step. We'll want to swap back to the old one after this
   //step.
   newton_iter = options.newton_iter;
   
   // Set the newton solve parameters
   newton_solver.iterative_mode = true;
   newton_solver.SetSolver(*J_solver);
   newton_solver.SetOperator(*this);
   newton_solver.SetPrintLevel(1);
   newton_solver.SetRelTol(options.newton_rel_tol);
   newton_solver.SetAbsTol(options.newton_abs_tol);
   newton_solver.SetMaxIter(options.newton_iter);
}

const Array<int> &NonlinearMechOperator::GetEssTDofList()
{
   return Hform->GetEssentialTrueDofs();
}

// Solve the Newton system
void NonlinearMechOperator::Solve(Vector &x) const
{
   Vector zero;
   //We provide an initial guess for what our current coordinates will look like
   //based on what our last time steps solution was for our velocity field.
//   if(!model->GetEndCoordsMesh()){
//      model->SwapMeshNodes();
//   }
   //The end nodes are updated before the 1st step of the solution here so we're good.
   newton_solver.Mult(zero, x);
   //Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
   //back to the current configuration...
//   if(!model->GetEndCoordsMesh()){
//      model->SwapMeshNodes();
//   }
   //Once the system has finished solving, our current coordinates configuration are based on what our
   //converged velocity field ended up being equal to.
   MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge.");
}

// Solve the Newton system for the 1st time step
// It was found that for large meshes a ramp up to our desired applied BC might
// be needed.
void NonlinearMechOperator::SolveInit(Vector &x)
{
   Vector zero;
   Vector init_x(x);
   //We shouldn't need more than 5 NR to converge to a solution during our
   //initial step in our solution.
   //We'll change this back to the old value at the end of the function.
   newton_solver.SetMaxIter(5);
   //We provide an initial guess for what our current coordinates will look like
   //based on what our last time steps solution was for our velocity field.
//   if(!model->GetEndCoordsMesh()){
//      model->SwapMeshNodes();
//   }
   //The end nodes are updated before the 1st step of the solution here so we're good.
   newton_solver.Mult(zero, x);
   //Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
   //back to the current configuration...
//   if(!model->GetEndCoordsMesh()){
//      model->SwapMeshNodes();
//   }
   
   //If the step didn't converge we're going to do a ramp up to the applied
   //velocity that we want. The assumption being made here is that our 1st time
   //step should be in the linear elastic regime. Therefore, we should be able
   //to go from our reduced solution to the desired solution. This has been noted
   //to be a problem when really increasing the mesh size.
   if(!newton_solver.GetConverged()){
      //We're going to reset our initial applied BCs to being 1/64 of the original
      if(myid == 0) mfem::out << "Solution didn't converge. Reducing initial condition to 1/4 original value\n";
      x = init_x;
      x *= 0.25;
      //We're going to keep track of how many cuts we need to make. Hopefully we
      //don't have to reduce it anymore then 3 times total.
      int i = 1;
      
      //We provide an initial guess for what our current coordinates will look like
      //based on what our last time steps solution was for our velocity field.
//      if(!model->GetEndCoordsMesh()){
//         model->SwapMeshNodes();
//      }
      //The end nodes are updated before the 1st step of the solution here so we're good.
      newton_solver.Mult(zero, x);
      //Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
      //back to the current configuration...
//      if(!model->GetEndCoordsMesh()){
//         model->SwapMeshNodes();
//      }
      
      if(!newton_solver.GetConverged()){
         //We're going to reset our initial applied BCs to being 1/16 of the original
         if(myid == 0) mfem::out << "Solution didn't converge. Reducing initial condition to 1/16 original value\n";
         x = init_x;
         x *= 0.0625;
         //We're going to keep track of how many cuts we need to make. Hopefully we
         //don't have to reduce it anymore then 3 times total.
         i++;
         
         //We provide an initial guess for what our current coordinates will look like
         //based on what our last time steps solution was for our velocity field.
//         if(!model->GetEndCoordsMesh()){
//            model->SwapMeshNodes();
//         }
         //The end nodes are updated before the 1st step of the solution here so we're good.
         newton_solver.Mult(zero, x);
         //Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
         //back to the current configuration...
//         if(!model->GetEndCoordsMesh()){
//            model->SwapMeshNodes();
//         }
         
         if(!newton_solver.GetConverged()){
            //We're going to reset our initial applied BCs to being 1/64 of the original
            if(myid == 0) mfem::out << "Solution didn't converge. Reducing initial condition to 1/64 original value\n";
            x = init_x;
            x *= 0.015625;
            //We're going to keep track of how many cuts we need to make. Hopefully we
            //don't have to reduce it anymore then 3 times total.
            i++;
            
            //We provide an initial guess for what our current coordinates will look like
            //based on what our last time steps solution was for our velocity field.
//            if(!model->GetEndCoordsMesh()){
//               model->SwapMeshNodes();
//            }
            //The end nodes are updated before the 1st step of the solution here so we're good.
            newton_solver.Mult(zero, x);
            //Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
            //back to the current configuration...
//            if(!model->GetEndCoordsMesh()){
//               model->SwapMeshNodes();
//            }
            
            MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge after 1/64 reduction of applied BCs.");
         }// end of 1/64 reduction case
      }// end of 1/16 reduction case
      
      //Here we're upscaling our previous converged solution to the next level.
      //The upscaling should be a good initial guess, since everything we're doing
      //is linear in this first step.
      //We then have the solution try and converge again with our better initial
      //guess of the solution.
      //It might be that this process only needs to occur once and we can directly
      //upscale from the lowest level to our top layer since we're dealing with
      //supposedly a linear elastic type problem here.
      for (int j = 0; j < i; j++) {
         if(myid == 0) mfem::out << "Upscaling previous solution by factor of 4\n";
         x *= 4.0;
         //We provide an initial guess for what our current coordinates will look like
         //based on what our last time steps solution was for our velocity field.
//         if(!model->GetEndCoordsMesh()){
//            model->SwapMeshNodes();
//         }
         //The end nodes are updated before the 1st step of the solution here so we're good.
         newton_solver.Mult(zero, x);
         //Just gotta be safe incase something in the solver wasn't playing nice and didn't swap things
         //back to the current configuration...
//         if(!model->GetEndCoordsMesh()){
//            model->SwapMeshNodes();
//         }
         
         //Once the system has finished solving, our current coordinates configuration are based on what our
         //converged velocity field ended up being equal to.
         //If the update fails we want to exit.
         MFEM_VERIFY(newton_solver.GetConverged(), "Newton Solver did not converge.");
      }// end of upscaling process
   }// end of 1/4 reduction case
   
   //Reset our max number of iterations to our original desired value.
   newton_solver.SetMaxIter(newton_iter);
}

// compute: y = H(x,p)
void NonlinearMechOperator::Mult(const Vector &k, Vector &y) const
{
   //Wanted to put this in the mechanics_solver.cpp file, but I would have needed to update
   //Solver class to use the NonlinearMechOperator instead of Operator class.
   //We now update our end coordinates based on the solved for velocity.
   UpdateEndCoords(k);
   // Apply the nonlinear form
   if(umat_used){
//      const ParFiniteElementSpace *fes = GetFESpace();
      //I really don't like this. It feels so hacky and
      //potentially dangerous to have these methods just
      //lying around.
      ParGridFunction* end_crds = model->GetEndCoords();
//      ParGridFunction* beg_crds = model->GetBegCoords();
//      ParMesh* pmesh = model->GetPMesh();
      Vector temp;
      temp.SetSize(k.Size());
      end_crds->GetTrueDofs(temp);
      //Creating a new vector that's going to be used for our
      //UMAT custorm Hform->Mult
      const Vector crd(temp.GetData(), temp.Size());
      model -> calc_incr_end_def_grad(crd);
      //The Mult expects our mesh to have the beg. time step
      //nodes
      //I should probably do the whole
      /*if(!model->GetEndCoordsMesh()){
       model->SwapMeshNodes();
       }*/
      //thing here as well...
//      model->SwapMeshNodes();
//      Hform->Mult(crd, y, pmesh, end_crds, beg_crds, k);
//      Hform -> Mult(k, y);
      //We need to swap back to the current time step nodes
      //here
//      model->SwapMeshNodes();
   }//else{
      //Without the umat things become nice and simple
   Hform->Mult(k, y);
//   }
}
//Update the end coords used in our model
void NonlinearMechOperator::UpdateEndCoords(const Vector& vel) const {
   model->UpdateEndCoords(vel);
}

// Compute the Jacobian from the nonlinear form
Operator &NonlinearMechOperator::GetGradient(const Vector &x) const
{
   Jacobian = &Hform->GetGradient(x);
   return *Jacobian;
}

void NonlinearMechOperator::ComputeVolAvgTensor(const ParFiniteElementSpace* fes,
                                                const QuadratureFunction* qf,
                                                Vector& tensor, int size){
   
   const FiniteElement *fe;
   const IntegrationRule *ir;
   double* qf_data = qf->GetData();
   int qf_offset = qf->GetVDim(); // offset at each integration point
   QuadratureSpace* qspace = qf->GetSpace();
   
   double el_vol = 0.0;
   double temp_wts = 0.0;
   double incr = 0.0;
   
   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   
   // loop over elements
   for (int i = 0; i < fes->GetNE(); ++i)
   {
      // get element transformation for the ith element
      ElementTransformation* Ttr = fes->GetElementTransformation(i);
      fe = fes->GetFE(i);
      ir = &(qspace->GetElementIntRule(i));
      int elem_offset = qf_offset * ir->GetNPoints();
      // loop over element quadrature points
      for (int j = 0; j < ir->GetNPoints(); ++j)
      {
         const IntegrationPoint &ip = ir->IntPoint(j);
         Ttr->SetIntPoint(&ip);
         //Here we're setting the integration for the average value
         temp_wts = ip.weight * Ttr->Weight();
         //This tells us the element volume
         el_vol += temp_wts;
         incr += 1.0;
         int k = 0;
         for (int m = 0; m < size; ++m)
         {
            tensor[m] += temp_wts * qf_data[i * elem_offset + j * qf_offset + k];
            ++k;
         }
      }
   }
   
   double data[size];
   
   for(int i = 0; i < size; i++){
      data[i] = tensor[i];
   }
   
   MPI_Allreduce(&data, tensor.GetData(), size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   
   double temp = el_vol;
   
   //Here we find what el_vol should be equal to
   MPI_Allreduce(&temp, &el_vol, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   
   //We meed to multiple by 1/V by our tensor values to get the appropriate
   //average value for the tensor in the end.
   double inv_vol = 1.0/el_vol;
   
   for (int m = 0; m < size; m++) {
      tensor[m] *= inv_vol;
   }
   
}

void NonlinearMechOperator::UpdateModel(const Vector &x)
{
   const ParFiniteElementSpace *fes = GetFESpace();
   const FiniteElement *fe;
   const IntegrationRule *ir;
   
   if(umat_used){
      
      //I really don't like this. It feels so hacky and
      //potentially dangerous to have these methods just
      //lying around.
//      ParGridFunction* end_crds = model->GetEndCoords();
//      ParGridFunction* beg_crds = model->GetBegCoords();
//      ParMesh* pmesh = model->GetPMesh();
//      Vector temp;
//      temp.SetSize(x.Size());
//      end_crds->GetTrueDofs(temp);
      //Creating a new vector that's going to be used for our
      //UMAT custorm Hform->Mult
//      const Vector crd(temp.GetData(), temp.Size());
      //As pointed out earlier I should probably check here again that we're
      //doing what we expect here aka swap the nodes to beg time step before
      //swapping back to the end time step coords
//      model->SwapMeshNodes();
      model->UpdateModelVars();
//      model->SwapMeshNodes();
   }
   else{
      model->UpdateModelVars();
   }
   
   //Everything is the same here no matter if we're using a UMAT
   //or not...
   //update state variables on a ExaModel
   for (int i = 0; i < fes->GetNE(); ++i)
   {
      fe = fes->GetFE(i);
      ir = &(IntRules.Get(fe->GetGeomType(), 2*fe->GetOrder() + 1));
      
      // loop over element quadrature points
      for (int j = 0; j < ir->GetNPoints(); ++j)
      {
         // update the beginning step stress
         model->UpdateStress(i, j);
         
         // compute von Mises stress
         model->ComputeVonMises(i, j);
         
         // update the beginning step state variables
         if (model->numStateVars > 0)
         {
            model->UpdateStateVars(i, j);
         }
      }
   }
   
   
   //Here we're getting the average stress value
   Vector stress;
   int size = 6;
   
   stress.SetSize(size);
   
   stress = 0.0;
   
   QuadratureVectorFunctionCoefficient* qstress = model->GetStress0();
   
   const QuadratureFunction* qf = qstress->GetQuadFunction();
   
   ComputeVolAvgTensor(fes, qf, stress, size);
   
   cout.setf(ios::fixed);
   cout.setf(ios::showpoint);
   cout.precision(8);
   
   int my_id;
   MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
   //Now we're going to save off the average stress tensor to a file
   if(my_id == 0){
      std::ofstream file;
      
      file.open("avg_stress.txt", std::ios_base::app);
      
      stress.Print(file, 6);
   }
   
   qstress = NULL;
   qf = NULL;
   
   //Here we're computing the average deformation gradient
   Vector defgrad;
   size = 9;
   
   defgrad.SetSize(size);
   
   defgrad = 0.0;
   
   QuadratureVectorFunctionCoefficient* qdefgrad = model->GetDefGrad0();
   
   const QuadratureFunction* qf1 = qdefgrad->GetQuadFunction();
   
   ComputeVolAvgTensor(fes, qf1, defgrad, size);
   
   //We're now saving the average def grad off to a file
   if(my_id == 0){
      std::ofstream file;
      
      file.open("avg_dgrad.txt", std::ios_base::app);
      
      defgrad.Print(file, 9);
   }
   
   qf1 = NULL;
   qdefgrad = NULL;
   
   fes = NULL;
   fe = NULL;
   ir = NULL;
   
}

//This is probably wrong and we need to make this more in line with what
//the ProjectVonMisesStress is doing
void NonlinearMechOperator::ProjectModelStress(ParGridFunction &s)
{
   QuadratureVectorFunctionCoefficient *stress;
   stress = model->GetStress0();
   s.ProjectCoefficient(*stress);
   
   stress = NULL;
   
   return;
}

void NonlinearMechOperator::ProjectVonMisesStress(ParGridFunction &vm)
{
   QuadratureFunctionCoefficient *vonMisesStress;
   vonMisesStress = model->GetVonMises();
   vm.ProjectDiscCoefficient(*vonMisesStress, mfem::GridFunction::ARITHMETIC);
   
   vonMisesStress = NULL;
   
   return;
}

void NonlinearMechOperator::SetTime(const double t)
{
   solVars.SetTime(t);
   model->SetModelTime(t);
   return;
}

void NonlinearMechOperator::SetDt(const double dt)
{
   solVars.SetDt(dt);
   model->SetModelDt(dt);
   return;
}

void NonlinearMechOperator::SetModelDebugFlg(const bool dbg)
{
   model->debug = dbg;
}

void NonlinearMechOperator::DebugPrintModelVars(int procID, double time)
{
   // print material properties vector on the model
   Vector *props = model->GetMatProps();
   ostringstream props_name;
   props_name << "props." << setfill('0') << setw(6) << procID << "_" << time;
   ofstream props_ofs(props_name.str().c_str());
   props_ofs.precision(8);
   props->Print(props_ofs);
   
   // print the beginning step material state variables quadrature function
   QuadratureVectorFunctionCoefficient *mv0 = model->GetMatVars0();
   ostringstream mv_name;
   mv_name << "matVars." << setfill('0') << setw(6) << procID << "_" << time;
   ofstream mv_ofs(mv_name.str().c_str());
   mv_ofs.precision(8);
   
   QuadratureFunction *matVars0 = mv0->GetQuadFunction();
   matVars0->Print(mv_ofs);
   
   matVars0 = NULL;
   props = NULL;
   
   return;
   
}
//A generic test function that we can add whatever unit tests to and then have them be tested
void NonlinearMechOperator::testFuncs(const Vector &x0, ParFiniteElementSpace *fes){
   model->test_def_grad_func(fes, x0);
}

NonlinearMechOperator::~NonlinearMechOperator()
{
   delete J_solver;
   if (J_prec != NULL) {
      delete J_prec;
   }
   delete model;
   delete Hform;
}
