#include<vector>
#include <fstream>
#include <limits>
#include <cstdlib>
#include "mfem.hpp"
#include <math.h>

using namespace mfem;
using namespace std;

#include "utilities.cpp"

/////////////////////////////////////////////////////
/////////////// FE_EVOLUTION CLASS //////////////////
/////////////////////////////////////////////////////
class FE_Evolution 
{
public:
  FE_Evolution(FiniteElementSpace &fes, 
	       bool &precondition_c_matrices, 
	       int &PROBLEM_TYPE, 
	       int &verbosity);

  double xFlux(const double& u_vel, const double& soln);
  double yFlux(const double& v_vel, const double& soln);
  double compute_dij(const double& solni, const double& solnj,
		     const double& u_veli, const double& v_veli,
		     const double& u_velj, const double& v_velj,
		     const double& Cx, const double& Cy,
		     const double& CTx, const double& CTy);
  void get_matrices();
  void assemble_mass_matrix();
  void assemble_C_matrices();
  void compute_solution(GridFunction &unp1, GridFunction &un);
  void evolve_to_time(const double &final_time, const double &runCFL);
  void setup();
  virtual ~FE_Evolution();

  // finite element space
  FiniteElementSpace *fespace;

  // Generic
  int PROBLEM_TYPE;
  int verbosity;
  int numDOFs, dofs_per_cell, nElements;

  // for time evolution
  double max_edge_based_cfl;
  int num_time_steps;
  double fixed_dt;
  double dt;
  double time;

  // mass matrix
  SparseMatrix mass_matrix;
  Vector lumped_mass_matrix;
  DSmoother mass_matrix_prec;
  CGSolver mass_matrix_solver;
  int skip_zeros;

  // Other matrices
  std::vector<double> dLowElemi;
  
  // C-matrices
  bool precondition_c_matrices;
  SparseMatrix Cx_matrix, Cy_matrix, CTx_matrix, CTy_matrix;
  std::vector<double> CxElem, CyElem, CTxElem, CTyElem;
  std::vector<double> PrCxElem, PrCyElem, PrCTxElem, PrCTyElem;

  // Flux vectors
  std::vector<double> lowOrderFluxTerm, lowOrderDissipativeTerm;

  // solution 
  GridFunction un, unp1;
};

FE_Evolution::FE_Evolution(FiniteElementSpace &fes, 
			   bool &precondition_c_matrices, 
			   int &PROBLEM_TYPE, 
			   int &verbosity)
  : un(&fes), unp1(&fes),
    fespace(&fes),
    precondition_c_matrices(precondition_c_matrices),
    PROBLEM_TYPE(PROBLEM_TYPE),
    verbosity(verbosity),
    Cx_matrix(fespace->GetVSize(),fespace->GetVSize()),
    Cy_matrix(fespace->GetVSize(),fespace->GetVSize()),
    CTx_matrix(fespace->GetVSize(),fespace->GetVSize()),
    CTy_matrix(fespace->GetVSize(),fespace->GetVSize()),
    mass_matrix(fespace->GetVSize(),fespace->GetVSize())
{}

FE_Evolution::~FE_Evolution(){}

double FE_Evolution::xFlux(const double& u_vel, const double& soln)
{
  return u_vel*soln;
}

double FE_Evolution::yFlux(const double& v_vel, const double& soln)
{
  return v_vel*soln;
}

double FE_Evolution::compute_dij(const double& solni, const double& solnj,
				 const double& u_veli, const double& v_veli,
				 const double& u_velj, const double& v_velj,
				 const double& Cx, const double& Cy,
				 const double& CTx, const double& CTy)
{
  if (PROBLEM_TYPE==0)
    if (true)// USE_DISCRETE_UPWINDING==1)
      {
	return fmax(0.0,fmax((Cx*u_velj + Cy*v_velj),
			     (CTx*u_veli+ CTy*v_veli)));
      }
    else
      {
	return fmax(fmax(fabs(Cx*u_veli + Cy*v_veli),
			 fabs(Cx*u_velj + Cy*v_velj)),
		    fmax(fabs(CTx*u_veli + CTy*v_veli),
			 fabs(CTx*u_velj + CTy*v_velj)));
      }
  else if (PROBLEM_TYPE==1)
    {
      double lambda = fmax(fabs(solni),fabs(solnj));
      return fmax(fabs(Cx+Cy), fabs(CTx+CTy))*lambda;
    }
  else
    {
      double lambda = 1;
      return fmax(fabs(Cx+Cy), fabs(CTx+CTy))*lambda;
    }
}

void FE_Evolution::get_matrices()
{
  // Element C-matrices
  CxElem.resize(nElements*dofs_per_cell*dofs_per_cell,0);
  CyElem.resize(nElements*dofs_per_cell*dofs_per_cell,0);
  CTxElem.resize(nElements*dofs_per_cell*dofs_per_cell,0);
  CTyElem.resize(nElements*dofs_per_cell*dofs_per_cell,0);
  
  // Preconditioned Element C-matrices
  PrCxElem.resize(nElements*dofs_per_cell*dofs_per_cell,0);
  PrCyElem.resize(nElements*dofs_per_cell*dofs_per_cell,0);
  PrCTxElem.resize(nElements*dofs_per_cell*dofs_per_cell,0);
  PrCTyElem.resize(nElements*dofs_per_cell*dofs_per_cell,0);
  
  lumped_mass_matrix.SetSize(numDOFs);

  if (verbosity>0)
    cout << "Assemble mass matrix" << std::endl;
  assemble_mass_matrix();

  if (verbosity>0)
    cout << "Assemble C matrix" << std::endl;
  assemble_C_matrices();

  dLowElemi.resize(numDOFs);
}

void FE_Evolution::assemble_mass_matrix()
{
  lumped_mass_matrix = 0.0;
  mass_matrix = 0.0;

  int dim = fespace->GetMesh()->Dimension();

  const FiniteElement &fe0 = *fespace->GetFE(0);
  int order = 2*fe0.GetOrder();

  Array<int> local_dofs_indices;
  DenseMatrix cell_mass_matrix(dofs_per_cell,dofs_per_cell);
  Vector cell_lumped_mass_matrix(dofs_per_cell);
  Vector shape;
  shape.SetSize(dofs_per_cell);

  // loop over the cells
  for (int k=0; k < nElements; ++k)
    {
      fespace->GetElementDofs(k,local_dofs_indices);

      const FiniteElement &fe = *fespace->GetFE(k);
      ElementTransformation &eltrans = *fespace->GetElementTransformation(k);
      const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), order);

      cell_mass_matrix = 0.0;
      cell_lumped_mass_matrix = 0.0;
      for (int q=0; q < ir->GetNPoints(); ++q)
	{
	  const IntegrationPoint &ip = ir->IntPoint(q);
	  fe.CalcShape(ip, shape);
	  
	  eltrans.SetIntPoint (&ip);
	  double w = eltrans.Weight() * ip.weight;
	  
	  AddMult_a_VVt(w, shape, cell_mass_matrix);
	  cell_lumped_mass_matrix.Add(w,shape);
	}
      
      // distribute
      mass_matrix.AddSubMatrix(local_dofs_indices,local_dofs_indices,cell_mass_matrix,skip_zeros);
      lumped_mass_matrix.AddElementVector(local_dofs_indices,cell_lumped_mass_matrix);      
    }
  mass_matrix.Finalize(skip_zeros);
}

void FE_Evolution::assemble_C_matrices()
{
  Cx_matrix=0.0;
  Cy_matrix=0.0;
  CTx_matrix=0.0;
  CTy_matrix=0.0;
  int dim = fespace->GetMesh()->Dimension();

  const FiniteElement &fe0 = *fespace->GetFE(0);
  int order = 2*fe0.GetOrder();

  Array<int> local_dofs_indices;
  DenseMatrix cell_Cx(dofs_per_cell), cell_Cy(dofs_per_cell);
  DenseMatrix cell_CTx(dofs_per_cell), cell_CTy(dofs_per_cell);
  // to precondition the C-matrices
  DenseMatrix cell_prCx(dofs_per_cell), cell_prCy(dofs_per_cell);
  DenseMatrix cell_prCTx(dofs_per_cell), cell_prCTy(dofs_per_cell);
  DenseMatrix cell_MC(dofs_per_cell), cell_ML(dofs_per_cell);
  DenseMatrix inv_cell_MC(dofs_per_cell), inv_cell_ML(dofs_per_cell);
  DenseMatrix leftPreconditioner(dofs_per_cell), rightPreconditioner(dofs_per_cell);

  DenseMatrix dshape(dofs_per_cell,dim), adjJ(dim);
  Vector shape(dofs_per_cell), vec1(dim), vec2(dim), dshape_times_velocity(dofs_per_cell);
  DenseMatrix dshape_times_JxW(dofs_per_cell,dim);

  DenseMatrix grad;
  grad.SetSize(dofs_per_cell,dim);

  // loop over the cells
  for (int eN=0; eN<nElements; ++eN)
    {
      fespace->GetElementDofs(eN,local_dofs_indices);
      const FiniteElement &fe = *fespace->GetFE(eN);
      ElementTransformation &eltrans = *fespace->GetElementTransformation(eN);
      const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), order);

      inv_cell_MC = 0.0;
      inv_cell_ML = 0.0;
      cell_MC = 0.0;
      cell_ML = 0.0;
      leftPreconditioner = 0.0;
      rightPreconditioner = 0.0;
	
      cell_Cx = 0.0;
      cell_Cy = 0.0;
      cell_CTx = 0.0;
      cell_CTy = 0.0;

      cell_prCx = 0.0;
      cell_prCy = 0.0;
      cell_prCTx = 0.0;
      cell_prCTy = 0.0;
      
      for (int q=0; q < ir->GetNPoints(); ++q)
	{
	  const IntegrationPoint &ip = ir->IntPoint(q);
	  fe.CalcDShape(ip, dshape);
	  fe.CalcShape(ip, shape);
	  eltrans.SetIntPoint (&ip);

	  // get dshape_times_JxW
	  CalcAdjugate(eltrans.Jacobian(), adjJ);
	  mfem::Mult(dshape,adjJ,grad);
	  double w = ip.weight;
	  double detJ = eltrans.Weight();

	  for (int i=0; i<dofs_per_cell; ++i)
	    {
	      for (int j=0; j<dofs_per_cell; ++j)
		{
		  cell_MC(i,j) += shape[i]*shape[j]*detJ*w;
		  cell_Cx(i,j) += grad(j,0)*shape[i]*w;
		  cell_Cy(i,j) += dim == 2 ? grad(j,1)*shape[i]*w : 0.0;
		  // Transport C matrices
		  cell_CTx(i,j) += grad(i,0)*shape[j]*w;
		  cell_CTy(i,j) += dim == 2 ? grad(i,1)*shape[j]*w : 0.0;
		}
	    } 
	}
      // distribute
      Cx_matrix.AddSubMatrix(local_dofs_indices,local_dofs_indices,cell_Cx,skip_zeros);
      Cy_matrix.AddSubMatrix(local_dofs_indices,local_dofs_indices,cell_Cy,skip_zeros);
      CTx_matrix.AddSubMatrix(local_dofs_indices,local_dofs_indices,cell_CTx,skip_zeros);
      CTy_matrix.AddSubMatrix(local_dofs_indices,local_dofs_indices,cell_CTy,skip_zeros);

      ///////////////////////////////////////////////
      // COMPUTE PRECONDITIONED ELEMENT C-MATRICES //
      ///////////////////////////////////////////////
      for (unsigned int i=0; i<dofs_per_cell; ++i)
	{
	  double sum_j_cell_matrix_ij = 0.;
	  for (unsigned int j=0; j<dofs_per_cell; ++j)
	    sum_j_cell_matrix_ij += cell_MC(i,j);
	  
	  cell_ML(i,i) = sum_j_cell_matrix_ij;
	  inv_cell_ML(i,i) = 1.0/sum_j_cell_matrix_ij;
	}
      
      if (verbosity>1)
	cout << "Preconditioned C-matrix, element eN: " << eN << std::endl;

      inv_cell_MC = cell_MC;
      inv_cell_MC.Invert();
      // left preconditioner
      mfem::Mult(cell_ML,inv_cell_MC,leftPreconditioner);      
      // right preconditioner
      mfem::Mult(inv_cell_MC,cell_ML,rightPreconditioner);

      // compute preconditioned C matrices
      mfem::Mult(leftPreconditioner,cell_Cx,cell_prCx);
      mfem::Mult(leftPreconditioner,cell_Cy,cell_prCy);
      mfem::Mult(cell_CTx,rightPreconditioner,cell_prCTx);
      mfem::Mult(cell_CTy,rightPreconditioner,cell_prCTy);
      
      // save C matrices
      for (int i=0; i<dofs_per_cell; ++i)
	{
	  int eN_i = eN*dofs_per_cell+i;
	  for (int j=0; j<dofs_per_cell; ++j)
	    {
	      int eN_i_j = eN_i*dofs_per_cell+j;
	      CxElem[eN_i_j] =  cell_Cx(i,j);
	      CyElem[eN_i_j] =  cell_Cy(i,j);
	      CTxElem[eN_i_j] = cell_CTx(i,j);
	      CTyElem[eN_i_j] = cell_CTy(i,j);

	      if (precondition_c_matrices==false)
		{
		  PrCxElem[eN_i_j]  =  cell_Cx(i,j);
		  PrCyElem[eN_i_j]  =  cell_Cy(i,j);
		  PrCTxElem[eN_i_j] =  cell_CTx(i,j);
		  PrCTyElem[eN_i_j] =  cell_CTy(i,j);
		}
	      else
		{
		  PrCxElem[eN_i_j]  =  cell_prCx(i,j);
		  PrCyElem[eN_i_j]  =  cell_prCy(i,j);
		  PrCTxElem[eN_i_j] =  cell_prCTx(i,j);
		  PrCTyElem[eN_i_j] =  cell_prCTy(i,j);
		}
	    }
	}
    }
  Cx_matrix.Finalize();
  Cy_matrix.Finalize();
  CTx_matrix.Finalize();
  CTy_matrix.Finalize();
}

void FE_Evolution::compute_solution(GridFunction &unp1, GridFunction &un)
{
  max_edge_based_cfl = 0.;
  Array<int> local_dofs_indices;

  // zero out vectors
  for (int i=0; i<numDOFs; i++)
    {
      lowOrderFluxTerm[i] = 0.0;
      lowOrderDissipativeTerm[i] = 0.0;
    }

  //loop in cells
  for (int eN=0; eN<nElements; ++eN)
    {
      fespace->GetElementDofs(eN,local_dofs_indices);
      
      for (unsigned int i=0; i<dofs_per_cell; ++i)
	{
	  int eN_i=eN*dofs_per_cell+i;
	  int gi = local_dofs_indices[i];
	  double solni=un[gi];
	  double u_veli=1.0;
	  double v_veli=0.0;
	  
	  // for edge_based_cfl
	  double dLowElemii = 0.;
	  for (unsigned int j=0; j < dofs_per_cell; ++j)
	    {
	      int eN_i_j = eN_i*dofs_per_cell+j;
	      int gj = local_dofs_indices[j];
	      
	      double solnj = un[gj];
	      double u_velj=1.0;
	      double v_velj=0.0;
	      
	      double fxj = xFlux(u_velj,solnj);
	      double fyj = yFlux(v_velj,solnj);
	      lowOrderFluxTerm[gi] += (PrCxElem[eN_i_j]*fxj + PrCyElem[eN_i_j]*fyj);
	      
	      // compute low-order dissipative flux
	      double dLowElemij = 0;
	      if (i!=j)
		{
		  dLowElemij = compute_dij(solni,solnj,
					   u_veli,v_veli,
					   u_velj,v_velj,
					   PrCxElem[eN_i_j], PrCyElem[eN_i_j],
					   PrCTxElem[eN_i_j], PrCTyElem[eN_i_j]);
		  dLowElemii -= dLowElemij;
		  lowOrderDissipativeTerm[gi] += dLowElemij*(solnj-solni);
		}
	    } //j
	  dLowElemi[gi] = dLowElemii;
	} //i
    } //elements

  //loop in DOFs
  for (int gi=0; gi<numDOFs; gi++)
    {
      double solni = un[gi];
      double mi = lumped_mass_matrix[gi];
      
      // compute edge based cfl	
      max_edge_based_cfl = fmax(max_edge_based_cfl,2*fabs(dLowElemi[gi])/mi);

      // compute low order solution
      unp1[gi] = solni - dt/mi * (lowOrderFluxTerm[gi]
				  -lowOrderDissipativeTerm[gi]);

    }  
}

void FE_Evolution::evolve_to_time(const double &final_time, const double &runCFL)
{
  bool final_step = false;
  while (time<final_time)
    {
      num_time_steps++;
      if (verbosity>1)
	{
	  cout << "Time step " << num_time_steps << std::endl;
	  cout << "   Current time = " << time << std::endl;
	  cout << "   Current dt = "  << dt << std::endl;
	}

      // First Forward Euler Step //
      if (verbosity>1)
	cout << "   First SSP stage" << std::endl;

      // compute solution 
      if (verbosity>1)
	cout << "      updating solution" << std::endl;
      compute_solution(unp1,un);
      
      fixed_dt=0; // if 0: None
      // update old solution
      un = unp1;
      //locally_relevant_un_solution = unp1_solution;
      if (final_step==true)
	break;
      else
	{
	  // update time
	  time += dt; // time at the recently taken step
	  // compute next time step
	  dt = (fixed_dt > 0 ? fixed_dt : runCFL/max_edge_based_cfl);

	  if (time+dt >= final_time)
	    {
	      final_step=true;
	      dt = final_time-time;
	    }
	}
    }
  time+=dt;
}

void FE_Evolution::setup()
{
  skip_zeros=0;
  const FiniteElement &fe0 = *fespace->GetFE(0);
  dofs_per_cell = fe0.GetDof();
  nElements = fespace->GetNE();
  numDOFs=fespace->GetVSize();

  // init vectors
  lowOrderFluxTerm.resize(numDOFs,0.0);
  lowOrderDissipativeTerm.resize(numDOFs,0.);
}

int main(int argc, char *argv[])
{
  OptionsParser args(argc, argv);

  ////////////////////////////
  // ***** PARAMETERS ***** //
  ////////////////////////////
  const char *mesh_file = "../data/star.mesh";
  int ref_levels = 0;
  int order = 1;
  double final_time = 0.2;
  double output_time = 0.1;
  bool vis_autopause = false;
  int verbosity = 2;
  bool precondition_c_matrices = true;

  // other parameters //
  double runCFL=0.25;
  int PROBLEM_TYPE=0;
  bool project_initial_condition = false;
  int precision = 8;
  bool visualization = true;
  int vis_steps = 5;
  const char *vishost = "localhost";
  int visport = 19916;

  
  args.AddOption(&mesh_file, "-m", "--mesh",
		 "Mesh file to use.");
  args.AddOption(&ref_levels, "-r", "--refine",
		 "Number of times to refine the mesh uniformly.");
  args.AddOption(&order, "-o", "--order",
		 "Order (degree) of the finite elements.");
  args.AddOption(&final_time, "-tf", "--t-final",
		 "Final time; start time is 0.");
  args.AddOption(&vis_autopause, "-va", "--visualization-autopause", "-vr",
		 "--visualization-run",
		 "Auto-pause or not the GLVis visualization.");
  args.AddOption(&verbosity, "-v", "--verbosity",
		 "Level of verbosity.");
  args.AddOption(&precondition_c_matrices, 
		 "-pr", "--precondition",
		 "-npr", "--no-precondition",
		 "Precondition (or not) the C-matrices.");
  args.Parse();
  
  if (!args.Good())
    {
      args.PrintUsage(cout);
      return 1;
    }
  cout.precision(precision);
  
  if (verbosity>0)
    args.PrintOptions(cout);
  
  ///////////////////////////
  // ***** LOAD MESH ***** //
  ///////////////////////////
  Mesh *mesh;
  {
    ifstream imesh(mesh_file);
    if (!imesh)
      {
	cout << "Can not open mesh: " << mesh_file << endl;
	return 2;
      }
    mesh = new Mesh(imesh, 1, 1);
  }
  
  for (int lev = 0; lev < ref_levels; lev++)
    mesh->UniformRefinement();
  
  if (verbosity>0)
    {
      cout << '\n';
      mesh->PrintCharacteristics();
    }
  
  int dim = mesh->Dimension();
  bool save_mesh = true;
  if (save_mesh)
    {
      const char out_mesh_file[] = "dg-advection.mesh";
      if (verbosity>0)
	{
	  cout << "Saving the mesh to '" << out_mesh_file << "'\n" << endl;
	}
      ofstream omesh(out_mesh_file);
      omesh.precision(precision);
      mesh->Print(omesh);
    }
  
  //////////////////////////////////////
  // ***** FINITE ELEMENT SPACE ***** //
  //////////////////////////////////////
  // CG FINITE ELEMENT SPACE
  H1_FECollection fec(order, dim, 2); //positive basis
  FiniteElementSpace fes(mesh, &fec);
  
  ////////////////////////
  // ***** SOLVER ***** //
  ////////////////////////
  FE_Evolution adv(fes, precondition_c_matrices, PROBLEM_TYPE, verbosity);
  VectorFunctionCoefficient velocity(dim, velocity_function);
  FunctionCoefficient u0Mat1(u0Mat1_function);
  
  ///////////////////////////////////
  // ***** INITIAL CONDITION ***** //
  ///////////////////////////////////
  // use piecewise constant projection?
  if (!project_initial_condition)
    {
      adv.un.ProjectCoefficient(u0Mat1);
      adv.unp1.ProjectCoefficient(u0Mat1);
    }
  else
    {
      // project the initial condition to the finite element space
      BilinearForm M(&fes);
      LinearForm b(&fes);
      M.AddDomainIntegrator(new MassIntegrator);
      b.AddDomainIntegrator(new DomainLFIntegrator(u0Mat1));
      
      M.Assemble();
      M.Finalize();
      b.Assemble();
      
      DSmoother M_prec;
      CGSolver M_solver;
      M_solver.SetPreconditioner(M_prec);
      M_solver.SetOperator(M.SpMat());
      M_solver.iterative_mode = false;
      M_solver.SetRelTol(1e-9);
      M_solver.SetAbsTol(0.0);
      M_solver.SetMaxIter(2000);
      M_solver.Mult(b,adv.un);
      M_solver.Mult(b,adv.unp1);
    }
  //
  bool save_solution = true;
  if (save_solution)
    {
      const char init_solution_file[] = "dg-advection-init.sol";
      if (verbosity>0)
	{
	  cout << "Saving the initial solution to '" << init_solution_file
	       << "'\n" << endl;
	}
      ofstream osol(init_solution_file);
      osol.precision(precision);
      adv.unp1.Save(osol);
    }
  
  ///////////////////////////////
  // ***** INITIAL ERROR ***** //
  ///////////////////////////////
  bool print_errors = true;
  Array<const IntegrationRule *> irs(Geometry::NumGeom);
  // assuming same geometry for all elements
  int geom = mesh->GetElementBaseGeometry(0);
  int error_integ_order = 2 * order + 3;
  irs[geom] = &IntRules.Get(geom, error_integ_order);
  double init_errors[3];
  if (print_errors)
    {
      init_errors[0] = adv.unp1.ComputeL1Error(u0Mat1, irs);
      init_errors[1] = adv.unp1.ComputeL2Error(u0Mat1, irs);
      init_errors[2] = adv.unp1.ComputeMaxError(u0Mat1, irs);
    }

  ///////////////////////////////
  // ***** VISUALIZATION ***** //
  ///////////////////////////////
  socketstream sout;
  if (visualization)
    {
      sout.open(vishost, visport);
      if (!sout)
	{
	  cout << "Unable to connect to GLVis visualization at "
	       << vishost << ':' << visport << endl;
	  visualization = false;
	}
      else
	{
	  sout.precision(precision);
	  sout << "solution\n";
	  mesh->Print(sout);
	  adv.unp1.Save(sout);
	  if (vis_autopause)
	    sout << "autopause on\n";
	  sout << flush;
	}
    }
  
  ///////////////////////
  // ***** SETUP ***** //
  ///////////////////////
  adv.setup();
  
  //////////////////////////////
  // ***** GET MATRICES ***** // mass matrix and C-matrices
  //////////////////////////////
  adv.get_matrices();
  
  ///////////////////////////
  // ***** TIME LOOP ***** //
  ///////////////////////////
  adv.time = 0.0;
  if (verbosity>0)
    {
      cout << "*******************************" << std::endl;
      cout << "********** TIME LOOP **********" << std::endl;
      cout << "*******************************" << std::endl;
    }
  adv.dt = 1E-6; // initial time step
  // get tnList. E.g.: [0, 0.1, 0.2, ...]
  int nOut = int(final_time/output_time)+1;
  nOut += (nOut-1)*output_time < final_time ? 1 : 0;
  std::vector<double> tnList(nOut);
  for (int i=0; i<nOut; i++)
    tnList[i] = i*output_time <= final_time ? i*output_time : final_time;
  
  // loop over tnList
  for (int time_interval = 1; time_interval<nOut; time_interval++)
    {
      if (verbosity>0)
	{
	  cout << "***** Evolve solution from time " 
	       << tnList[time_interval-1]
	       << " to time "
	       << tnList[time_interval]
	       << " *****"
	       << std::endl;
	}
      adv.evolve_to_time(tnList[time_interval],runCFL);
      // output solution 
      sout << "solution\n";
      mesh->Print(sout);
      adv.unp1.Save(sout);
      sout << flush;
    }

  if (verbosity>0)
    cout << "Final time achieved. Time=" << adv.time << std::endl;
  
  ///////////////////////////////////////////////////////
  // ***** FINAL VISUALIZATION AND SAVE SOLUTION ***** //
  ///////////////////////////////////////////////////////
  if (visualization)
    {
      sout << "solution\n";
      mesh->Print(sout);
      adv.unp1.Save(sout);
      sout << flush;
    }  
  
  if (save_solution)
    {
      const char final_solution_file[] = "dg-advection-final.sol";
      if (verbosity>0)
	{
	  cout << "Saving the final solution to '" << final_solution_file
	       << "'\n" << endl;
	}
      
      ofstream osol(final_solution_file);
      osol.precision(precision);
      adv.unp1.Save(osol);
    }
  
  //////////////////////////////
  // ***** FINAL ERRORS ***** //
  //////////////////////////////
  if (verbosity>0)
    {
      cout << "****************************" << std::endl;
      cout << "********** ERRORS **********" << std::endl;
      cout << "****************************" << std::endl;
    }
  double final_errors[3];
  if (print_errors)
    {
      final_errors[0] = adv.unp1.ComputeL1Error(u0Mat1, irs);
      final_errors[1] = adv.unp1.ComputeL2Error(u0Mat1, irs);
      final_errors[2] = adv.unp1.ComputeMaxError(u0Mat1, irs);
      
      cout.precision(precision);
      cout.setf(ios::scientific);
      int s = precision + 10;
      if (verbosity>0)
	cout << "Errors:" << setw(s-2) << "initial" << setw(s) << "final" << '\n'
	     << "L1:  "
	     << setw(s) << init_errors[0]
	     << setw(s) << final_errors[0] << '\n'
	     << "L2:  "
	     << setw(s) << init_errors[1]
	     << setw(s) << final_errors[1] << '\n'    
	     << "Max: "
	     << setw(s) << init_errors[2]
	     << setw(s) << final_errors[2] << '\n' << endl;
    }
  delete mesh;
  return 0;
}
