#include<vector>
#include <fstream>
#include <limits>
#include <cstdlib>
#include "mfem.hpp"
#include <math.h>

using namespace mfem;
using namespace std;

#include "utilities.cpp"

#define ENTROPY_POWER 1.0
#define TEST_FLUX_STAR 0
#define USE_Q1_STENCIL 1
#define USE_DISCRETE_UPWINDING 1 // Only for linear problems
#define METHOD 2
// METHOD 4
// 0: low-order
// 1: high-order non-limited
// 2: limiting without gamma indicator

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
  double ENTROPY(const double& soln);
  double DENTROPY(const double& soln);
  double xFlux(const double& u_vel, const double& soln);
  double yFlux(const double& v_vel, const double& soln);
  double xFluxij(const double& u_veli, const double& u_velj,
		 const double& solni, const double& solnj);
  double yFluxij(const double& v_veli, const double& v_velj,
		 const double& solni, const double& solnj);
  double xEntFlux(const double& u_vel, const double& soln);
  double yEntFlux(const double& v_vel, const double& soln);
  double xEntPot(const double& u_vel, const double& soln);
  double yEntPot(const double& u_vel, const double& soln);
  double compute_dij(const double& solni, const double& solnj,
		     const double& u_veli, const double& v_veli,
		     const double& u_velj, const double& v_velj,
		     const double& Cx, const double& Cy,
		     const double& CTx, const double& CTy);
  void get_matrices_and_init_data_structures();
  void get_inv_element_ML_minus_MC_Q1();
  void get_inv_element_ML_minus_MC_Q1_v2();
  void assemble_mass_matrix();
  void assemble_C_matrices();
  void compute_EV_residual(GridFunction &un);
  void compute_uDot(GridFunction &un);
  void compute_solution(GridFunction &unp1, GridFunction &un);  
  void evolve_to_time(const double &final_time, const double &runCFL);
  void setup();
  void update_velocity();
  void setCoord(DenseMatrix coord){this->coord=coord;};  
  void set_cE(double cE){this->cE=cE;};
  void set_SSP_order(int SSP_order){this->SSP_order=SSP_order;};
  virtual ~FE_Evolution();

  // finite element space
  FiniteElementSpace *fespace;

  // Generic
  int PROBLEM_TYPE;
  int verbosity;
  int numDOFs, dofs_per_cell, nElements, nNonZeroEntries, dim, quad_order;

  // for time evolution
  double max_edge_based_cfl;
  int num_time_steps;
  double fixed_dt;
  double dt;
  double time;
  int SSP_order;

  // mass matrix
  SparseMatrix mass_matrix;
  Vector lumped_mass_matrix;
  DSmoother mass_matrix_prec;
  CGSolver mass_matrix_solver;

  // Other matrices
  std::vector<double> dLowElemi;
  
  // C-matrices
  bool precondition_c_matrices;
  SparseMatrix Cx_matrix, Cy_matrix, CTx_matrix, CTy_matrix;
  std::vector<double> CxElem, CyElem, CTxElem, CTyElem;
  std::vector<double> PrCxElem, PrCyElem, PrCTxElem, PrCTyElem;
  std::vector<double> Cx, Cy;
  int skip_zeros;

  // Flux vectors
  std::vector<double> lowOrderFluxTerm, lowOrderDissipativeTerm;
  std::vector<double> u_vel_dofs, v_vel_dofs;

  // for low-order solution 
  std::vector<double> lowOrderSolution;

  // for high-order method
  double cE;
  std::vector<double> EntVisc;
  std::vector<double> element_flux_qij;
  std::vector<double> fluxStar;
  Vector uDot; // Vector since this passes throught he MFEM's solver 

  // bounds 
  std::vector<double> umin, umax;

  // limiters
  std::vector<double> wBarij, wBarji;
  std::vector<double> dLowElem;
  std::vector<double> element_flux_i;
  std::vector<double> vVector;
  
  // solution 
  GridFunction un, unp1, uSt1, uSt2;

  // for sparsity pattern
  std::vector<int> rowptr;
  std::vector<int> colind;

  // for local stencil
  DenseMatrix element_MC_Q1;
  DenseMatrix inv_element_ML_minus_MC_Q1;
  DenseMatrix element_Q1_sparsity;

  // for computation of errors
  DenseMatrix coord;

  // for entropy stability
  std::vector<double> element_dijMin;
};

FE_Evolution::FE_Evolution(FiniteElementSpace &fes, 
			   bool &precondition_c_matrices, 
			   int &PROBLEM_TYPE, 
			   int &verbosity)
  : un(&fes), unp1(&fes), uSt1(&fes), uSt2(&fes),
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

double FE_Evolution::ENTROPY(const double& soln)
{
  return 1./2.*std::pow(soln,2.);
}

double FE_Evolution::DENTROPY(const double& soln)
{
  return soln;
}

double FE_Evolution::xEntFlux(const double& u_vel, const double& soln)
{
  if (PROBLEM_TYPE==0)
    return u_vel*ENTROPY(soln);
  else if (PROBLEM_TYPE==1)
    return soln*soln*soln/3.0;
  else if (PROBLEM_TYPE==2)
    //return std::exp(soln)*(std::sin(soln)+std::cos(soln))/2.0;
    return soln*std::sin(soln)+std::cos(soln);
  else
    return (2*soln-1)/(4*soln*soln-4*soln+2);
}

double FE_Evolution::yEntFlux(const double& v_vel, const double& soln)
{
  if (PROBLEM_TYPE==0)
    return v_vel*ENTROPY(soln);
  else if (PROBLEM_TYPE==1)
    return soln*soln*soln/3.0;
  else if (PROBLEM_TYPE==2)
    //return -std::exp(soln)*(std::sin(soln)-std::cos(soln))/2.0;
    return soln*std::cos(soln)-std::sin(soln);
  else
    {
      double u = soln;
      return 1.0/12*(-20*u*u*u + 15*u*u - (9*u+6)/(2*u*u-2*u+1)
		     -3*std::log(2*u*u-2*u+1) - 15*std::atan(1-2*u));
    }
}

double FE_Evolution::xFlux(const double& u_vel, const double& soln)
{
  if (PROBLEM_TYPE==0)
    return u_vel*soln;
  else if (PROBLEM_TYPE==1)
    return 0.5*soln*soln;
  else if (PROBLEM_TYPE==2)
    return std::sin(soln);
  else
    {
      double u =soln;
      return u*u/(u*u+std::pow(1-u,2.0));
    }
}

double FE_Evolution::xFluxij(const double& u_veli, const double& u_velj, 
			     const double& solni, const double& solnj)
{
  if (PROBLEM_TYPE==0)
    return 0.5*(u_veli*solni+u_velj*solnj);    
  else if (PROBLEM_TYPE==1)
    return 0.5*std::pow(0.5*(solni+solnj),2.0);
  else if (PROBLEM_TYPE==2)
    return std::sin(0.5*(solni+solnj));
  else
    {
      double u = 0.5*(solni+solnj);
      return u*u/(u*u+std::pow(1-u,2.0));
    }
}

double FE_Evolution::yFlux(const double& v_vel, const double& soln)
{
  if (PROBLEM_TYPE==0)
    return v_vel*soln;
  else if (PROBLEM_TYPE==1)
    return 0.5*soln*soln;
  else if (PROBLEM_TYPE==2)
    return std::cos(soln);
  else
    {
      double u=soln;
      double fx = u*u/(u*u+std::pow(1-u,2.0));
      return fx*(1-5*std::pow(1-u,2.0));
    }
}

double FE_Evolution::yFluxij(const double& v_veli, const double& v_velj, 
			     const double& solni, const double& solnj)
{
  if (PROBLEM_TYPE==0)
    return 0.5*(v_veli*solni+v_velj*solnj);
  else if (PROBLEM_TYPE==1)
    return 0.5*std::pow(0.5*(solni+solnj),2.0);
  else if (PROBLEM_TYPE==2)
    return std::cos(0.5*(solni+solnj));
  else
    {
      double u = 0.5*(solni+solnj);
      double fx = u*u/(u*u+std::pow(1-u,2.0));
      return fx*(1-5*std::pow(1-u,2.0));
    }
}

double FE_Evolution::xEntPot(const double& u_vel, const double& soln)
{
  double w = DENTROPY(soln);
  double flux = xFlux(u_vel,soln);
  double entFlux = xEntFlux(u_vel,soln);
  return w*flux-entFlux;
}

double FE_Evolution::yEntPot(const double& u_vel, const double& soln)
{
  double w = DENTROPY(soln);
  double flux = yFlux(u_vel,soln);
  double entFlux = yEntFlux(u_vel,soln);
  return w*flux-entFlux;
}

double FE_Evolution::compute_dij(const double& solni, const double& solnj,
				 const double& u_veli, const double& v_veli,
				 const double& u_velj, const double& v_velj,
				 const double& Cx, const double& Cy,
				 const double& CTx, const double& CTy)
{
  if (PROBLEM_TYPE==0)
    if (USE_DISCRETE_UPWINDING==1)
      {
	return fmax(0.0,fmax((Cx*u_velj + Cy*v_velj),
			     (CTx*u_veli+ CTy*v_veli)));
      }
    else // low-order by Guermond and Popov
      {
	return fmax(fmax(fabs(Cx*u_veli + Cy*v_veli),
			 fabs(Cx*u_velj + Cy*v_velj)),
		    fmax(fabs(CTx*u_veli + CTy*v_veli),
			 fabs(CTx*u_velj + CTy*v_velj)));
      }
  else if (PROBLEM_TYPE==1)
    {
      // burgers
      double lambda = fmax(fabs(solni),fabs(solnj));
      return fmax(std::sqrt(Cx*Cx+Cy*Cy), std::sqrt(CTx*CTx+CTy*CTy))*lambda;
    }
  else
    {
      double lambda = 1;
      return fmax(std::sqrt(Cx*Cx+Cy*Cy), std::sqrt(CTx*CTx+CTy*CTy))*lambda;
      //return fmax(fabs(Cx)+fabs(Cy), fabs(CTx)+fabs(CTy))*lambda;
    }
}

void FE_Evolution::get_matrices_and_init_data_structures()
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

  // ******************************** //
  // ***** get sparsity pattern ***** //
  // ******************************** //
  rowptr.resize(numDOFs+1,0);
  int *I = mass_matrix.GetI();
  int *J = mass_matrix.GetJ();
  for (unsigned int i=0; i<numDOFs+1; ++i)
    rowptr[i] = I[i];
 
  // get Cx-matrices in CSR format 
  nNonZeroEntries = rowptr[numDOFs];
  colind.resize(nNonZeroEntries,0);
  Cx.resize(nNonZeroEntries,0.0);
  Cy.resize(nNonZeroEntries,0.0);
  double* tmp_Cx = Cx_matrix.GetData();
  double* tmp_Cy = Cy_matrix.GetData();
  for (unsigned int i=0; i<nNonZeroEntries; i++)
    {
      colind[i] = J[i];
      Cx[i] = tmp_Cx[i];
      Cy[i] = tmp_Cy[i];
    }

  // init some vectors 
  // bounds 
  umax.resize(numDOFs,0.0);
  umin.resize(numDOFs,0.0);

  // low-order 
  lowOrderFluxTerm.resize(numDOFs,0.0);
  lowOrderDissipativeTerm.resize(numDOFs,0.);
  lowOrderSolution.resize(numDOFs,0.0);

  // velocity 
  u_vel_dofs.resize(numDOFs,0.0);
  v_vel_dofs.resize(numDOFs,0.0);

  // high-order vectors
  uDot.SetSize(numDOFs); // MFEM's vector  
  EntVisc.resize(numDOFs,0.0);
  fluxStar.resize(numDOFs,0.0);
  element_flux_qij.resize(nElements*dofs_per_cell*dofs_per_cell,0);

  // limiters
  wBarij.resize(nElements*dofs_per_cell*dofs_per_cell,0);
  wBarji.resize(nElements*dofs_per_cell*dofs_per_cell,0);
  dLowElem.resize(nElements*dofs_per_cell*dofs_per_cell,0);
  element_flux_i.resize(nElements*dofs_per_cell,0);
  vVector.resize(nElements*dofs_per_cell,0);

  // for local stencil
  element_MC_Q1.SetSize(dofs_per_cell,dofs_per_cell);
  inv_element_ML_minus_MC_Q1.SetSize(dofs_per_cell,dofs_per_cell);
  element_Q1_sparsity.SetSize(dofs_per_cell,dofs_per_cell);
  get_inv_element_ML_minus_MC_Q1_v2();

  // for entropy stability
  element_dijMin.resize(nElements*dofs_per_cell*dofs_per_cell,0);
}

void FE_Evolution::get_inv_element_ML_minus_MC_Q1()
{
  double hRef = 1.0/fespace->GetOrder(0); // h at reference element
  double tol = 1e-2*hRef;
  double dist_neighbour = hRef*sqrt(dim) + tol; // give some tolerance
  double dist=0;

  DenseMatrix element_ML_minus_MC_Q1(dofs_per_cell);
  Vector element_ML_Q1(dofs_per_cell);

  element_MC_Q1 = 0.;
  element_ML_Q1 = 0.;
  element_ML_minus_MC_Q1 = 0.;
  element_Q1_sparsity = 0.;

  // get coordinates in the reference element
  DenseMatrix DOFs_coord(dim,dofs_per_cell);
  DOFs_coord = 0.;
  const FiniteElement &fe = *fespace->GetFE(0); //element 0
  const IntegrationRule &ir = fe.GetNodes();
  ElementTransformation *el_trans = fespace->GetElementTransformation(0);
  for (int i=0; i<dofs_per_cell; i++)
    {
      DOFs_coord(0,i) = ir.IntPoint(i).x;
      if (dim==2)
	DOFs_coord(1,i) = ir.IntPoint(i).y;
    }

  // get the pseudo element lumped and consistent mass matrix
  for (int i=0; i<dofs_per_cell; i++)
    {
      for (int j=0; j<dofs_per_cell; j++)
	{
	  // get the distance from node i to node j
	  if (dim==1) 
	    dist = abs(DOFs_coord(0,i)-DOFs_coord(0,j));
	  else // dim=2
	    dist = sqrt(pow(DOFs_coord(0,i)-DOFs_coord(0,j),2)+
			pow(DOFs_coord(1,i)-DOFs_coord(1,j),2));

	  // check if node j is in the local stencil
	  if (i==j)
	    {
	      element_Q1_sparsity(i,j)=1;
	      // check if node i is at the boundary
	      element_MC_Q1(i,j) = 1.0;
	    }
	  else if (dist <= dist_neighbour)
	    {
	      element_Q1_sparsity(i,j)=1;
	      element_MC_Q1(i,j) = 0.5; 
	    }
	  else
	    {
	      element_Q1_sparsity(i,j)=0;
	      element_MC_Q1(i,j) = 0.0;
	    }

	  // compute lumped mass
	  element_ML_Q1(i) += element_MC_Q1(i,j);
	}
      
      for (int j=0; j<dofs_per_cell; j++)
	{
	  if (i==j)
	    element_ML_minus_MC_Q1(i,j) = element_ML_Q1(i) - element_MC_Q1(i,j);
	  else
	    element_ML_minus_MC_Q1(i,j) = - element_MC_Q1(i,j);
	}
    }
  //element_MC_Q1.Print();
  //element_ML_Q1.Print();
  //element_ML_minus_MC_Q1.Print();
  //abort();

  // remove the constant space from the null space 
  element_ML_minus_MC_Q1.SetRow(0,0.0);
  element_ML_minus_MC_Q1(0,0)=1.0;
  inv_element_ML_minus_MC_Q1 = element_ML_minus_MC_Q1;
  inv_element_ML_minus_MC_Q1.Invert();

  // check dissipative matrix ML_minus_MC
  DenseMatrix transpose_matrix(dofs_per_cell);
  Vector row(dofs_per_cell); 
  for (int r=1; r<dofs_per_cell; r++)
    {
      element_ML_minus_MC_Q1.GetRow(r,row);
      if (fabs(row.Sum()) >= 1E-10)
	{
	  std::cout << "matrix ML-MC does not have zero row sum" << std::endl;
	  abort();
	}
      
      element_ML_minus_MC_Q1.Transpose(transpose_matrix);
      //transpose_matrix -= element_ML_minus_MC_Q1;
      double max_norm = transpose_matrix.MaxMaxNorm();
      if (max_norm >= 1E-10)
	{
	  std::cout << "Matrix ML-MC is not symmetric" << std::endl;
	  abort();
	}
    }
}

void FE_Evolution::get_inv_element_ML_minus_MC_Q1_v2()
{
  double hRef = 1.0/fespace->GetOrder(0); // h at reference element
  double tol = 1e-2*hRef;
  double distLayer1 = hRef + tol; // give some tolerance
  double distLayer2 = hRef*sqrt(dim) + tol; // give some tolerance
  double dist=0;

  DenseMatrix element_ML_minus_MC_Q1(dofs_per_cell);
  Vector element_ML_Q1(dofs_per_cell);

  element_MC_Q1 = 0.;
  element_ML_Q1 = 0.;
  element_ML_minus_MC_Q1 = 0.;
  element_Q1_sparsity = 0.;

  // get coordinates in the reference element
  DenseMatrix DOFs_coord(dim,dofs_per_cell);
  DOFs_coord = 0.;
  const FiniteElement &fe = *fespace->GetFE(0); //element 0
  const IntegrationRule &ir = fe.GetNodes();
  ElementTransformation *el_trans = fespace->GetElementTransformation(0);
  for (int i=0; i<dofs_per_cell; i++)
    {
      DOFs_coord(0,i) = ir.IntPoint(i).x;
      if (dim==2)
	DOFs_coord(1,i) = ir.IntPoint(i).y;
    }

  // get the pseudo element lumped and consistent mass matrix
  for (int i=0; i<dofs_per_cell; i++)
    {
      for (int j=0; j<dofs_per_cell; j++)
	{
	  // get the distance from node i to node j
	  if (dim==1) 
	    dist = abs(DOFs_coord(0,i)-DOFs_coord(0,j));
	  else // dim=2
	    dist = sqrt(pow(DOFs_coord(0,i)-DOFs_coord(0,j),2)+
			pow(DOFs_coord(1,i)-DOFs_coord(1,j),2));
	  
	  // check if node j is in the local stencil
	  if (i==j)
	    {
	      element_Q1_sparsity(i,j)=1;
	      element_MC_Q1(i,j) = 4.0/36.0;
	    }
	  else if (dist <= distLayer1)
	    {
	      element_Q1_sparsity(i,j)=1;
	      element_MC_Q1(i,j) = 2.0/36.0; 
	    }
	  else if (dist <= distLayer2)
	    {
	      element_Q1_sparsity(i,j)=1;
	      element_MC_Q1(i,j) = 1.0/36.0;
	    }
	  else
	    {
	      element_Q1_sparsity(i,j)=0;
	      element_MC_Q1(i,j) = 0.0;
	    }
	  
	  // compute lumped mass
	  element_ML_Q1(i) += element_MC_Q1(i,j);
	}
      
      for (int j=0; j<dofs_per_cell; j++)
	{
	  if (i==j)
	    element_ML_minus_MC_Q1(i,j) = element_ML_Q1(i) - element_MC_Q1(i,j);
	  else
	    element_ML_minus_MC_Q1(i,j) = - element_MC_Q1(i,j);
	}
    }
  //element_MC_Q1.Print();
  //element_ML_Q1.Print();
  //element_ML_minus_MC_Q1.Print();
  //abort();

  // remove the constant space from the null space 
  element_ML_minus_MC_Q1.SetRow(0,0.0);
  element_ML_minus_MC_Q1(0,0)=1.0;
  inv_element_ML_minus_MC_Q1 = element_ML_minus_MC_Q1;
  inv_element_ML_minus_MC_Q1.Invert();

  //element_ML_minus_MC_Q1.Print();
  //std::cout << "**********" << std::endl;
  //inv_element_ML_minus_MC_Q1.Print();
  //abort();
  // check dissipative matrix ML_minus_MC
  DenseMatrix transpose_matrix(dofs_per_cell);
  Vector row(dofs_per_cell); 
  for (int r=1; r<dofs_per_cell; r++)
    {
      element_ML_minus_MC_Q1.GetRow(r,row);
      if (fabs(row.Sum()) >= 1E-10)
	{
	  std::cout << "matrix ML-MC does not have zero row sum" << std::endl;
	  abort();
	}
      
      element_ML_minus_MC_Q1.Transpose(transpose_matrix);
      //transpose_matrix -= element_ML_minus_MC_Q1;
      double max_norm = transpose_matrix.MaxMaxNorm();
      if (max_norm >= 1E-10)
	{
	  std::cout << "Matrix ML-MC is not symmetric" << std::endl;
	  abort();
	}
    }
}

void FE_Evolution::assemble_mass_matrix()
{
  lumped_mass_matrix = 0.0;
  mass_matrix = 0.0;

  Array<int> local_dofs_indices;
  DenseMatrix cell_mass_matrix(dofs_per_cell,dofs_per_cell);
  Vector cell_lumped_mass_matrix(dofs_per_cell);
  Vector shape;
  shape.SetSize(dofs_per_cell);

  // loop over the cells
  for (int eN=0; eN < nElements; ++eN)
    {
      // get local to global dofs, fe space, etc
      fespace->GetElementDofs(eN,local_dofs_indices);
      const FiniteElement &fe = *fespace->GetFE(eN);
      ElementTransformation &eltrans = *fespace->GetElementTransformation(eN);
      const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), quad_order);

      cell_mass_matrix = 0.0;
      cell_lumped_mass_matrix = 0.0;
      for (int q=0; q < ir->GetNPoints(); ++q)
	{
	  const IntegrationPoint &ip = ir->IntPoint(q);
	  //shape function and grad(shape) at reference element 
	  fe.CalcShape(ip, shape);
	  
	  // related to the transformation: detJ, invJ and weight
	  eltrans.SetIntPoint (&ip);
	  double detJxW = eltrans.Weight() * ip.weight;
	  
	  AddMult_a_VVt(detJxW, shape, cell_mass_matrix);
	  cell_lumped_mass_matrix.Add(detJxW,shape);
	}
      // distribute
      mass_matrix.AddSubMatrix(local_dofs_indices,local_dofs_indices,cell_mass_matrix,skip_zeros);
      lumped_mass_matrix.AddElementVector(local_dofs_indices,cell_lumped_mass_matrix);      
    }
  mass_matrix.Finalize(skip_zeros);

  // setup the linear solver 
  mass_matrix_solver.SetPreconditioner(mass_matrix_prec);
  mass_matrix_solver.SetOperator(mass_matrix);
  mass_matrix_solver.iterative_mode = false;
  mass_matrix_solver.SetRelTol(1e-9);
  mass_matrix_solver.SetAbsTol(0.0);
  mass_matrix_solver.SetMaxIter(2000);
}

void FE_Evolution::assemble_C_matrices()
{
  Cx_matrix=0.0;
  Cy_matrix=0.0;
  CTx_matrix=0.0;
  CTy_matrix=0.0;

  Array<int> local_dofs_indices;
  DenseMatrix cell_Cx(dofs_per_cell), cell_Cy(dofs_per_cell);
  DenseMatrix cell_CTx(dofs_per_cell), cell_CTy(dofs_per_cell);
  // to precondition the C-matrices
  DenseMatrix cell_prCx(dofs_per_cell), cell_prCy(dofs_per_cell);
  DenseMatrix cell_prCTx(dofs_per_cell), cell_prCTy(dofs_per_cell);
  DenseMatrix cell_MC(dofs_per_cell), cell_ML(dofs_per_cell);
  DenseMatrix inv_cell_MC(dofs_per_cell), inv_cell_ML(dofs_per_cell);
  DenseMatrix leftPreconditioner(dofs_per_cell), rightPreconditioner(dofs_per_cell);

  DenseMatrix dshape(dofs_per_cell,dim), invJ(dim);
  Vector shape(dofs_per_cell), vec1(dim), vec2(dim), dshape_times_velocity(dofs_per_cell);
  DenseMatrix dshape_times_JxW(dofs_per_cell,dim);

  DenseMatrix grad_shape;
  grad_shape.SetSize(dofs_per_cell,dim);

  // loop over the cells
  for (int eN=0; eN<nElements; ++eN)
    {
      fespace->GetElementDofs(eN,local_dofs_indices);
      const FiniteElement &fe = *fespace->GetFE(eN);
      ElementTransformation &eltrans = *fespace->GetElementTransformation(eN);
      const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), quad_order);

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
	  //shape function and grad(shape) at reference element
	  fe.CalcDShape(ip, dshape);
	  fe.CalcShape(ip, shape);

	  // related to the transformation: detJ, invJ and weight
	  eltrans.SetIntPoint (&ip);
	  double detJ = eltrans.Weight();
	  double w = ip.weight;
	  double detJxW = detJ * w;
	  CalcAdjugate(eltrans.Jacobian(), invJ);
	  invJ *= 1.0/detJ;
	  mfem::Mult(dshape,invJ,grad_shape);	  

	  for (int i=0; i<dofs_per_cell; ++i)
	    {
	      for (int j=0; j<dofs_per_cell; ++j)
		{
		  cell_MC(i,j) += shape[i]*shape[j]*detJ*w;
		  cell_Cx(i,j) += grad_shape(j,0)*shape[i]*detJ*w;
		  cell_Cy(i,j) += dim == 2 ? grad_shape(j,1)*shape[i]*detJ*w : 0.0;
		  // Transport C matrices
		  cell_CTx(i,j) += grad_shape(i,0)*shape[j]*detJ*w;
		  cell_CTy(i,j) += dim == 2 ? grad_shape(i,1)*shape[j]*detJ*w : 0.0;
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
  Cx_matrix.Finalize(skip_zeros);
  Cy_matrix.Finalize(skip_zeros);
  CTx_matrix.Finalize(skip_zeros);
  CTy_matrix.Finalize(skip_zeros);
}

void FE_Evolution::compute_EV_residual(GridFunction &un)
{  
  int ij=0;
  for (int gi=0; gi<numDOFs; gi++)
    {
      double solni = un[gi];
      double DEnti = DENTROPY(solni);

      double DenEntViscPart1 = 0.;
      double DenEntViscPart2 = 0.;
      double ith_NumEntVisc = 0;
      double ith_DenEntVisc = 0.;
      
      // loop on the support of ith shape function
      for (int offset=rowptr[gi]; offset<rowptr[gi+1]; offset++)
	{
	  int gj = colind[offset];
	  // get solution, velocity and fluxes at node j
	  double solnj = un[gj];
	  double u_velj = u_vel_dofs[gj];
	  double v_velj = v_vel_dofs[gj];
	  
	  double fxj = xFlux(u_velj,solnj);
	  double fyj = yFlux(v_velj,solnj);
	  
	  // For entropy viscosity //
	  double x_EntFluxj=xEntFlux(u_velj,solnj);// - xEntFlux(u_veli,solni);
	  double y_EntFluxj=yEntFlux(v_velj,solnj);// - yEntFlux(v_veli,solni);
	  ith_NumEntVisc += (Cx[ij]*(x_EntFluxj-DEnti*fxj) + Cy[ij]*(y_EntFluxj-DEnti*fyj));
	
	  // aux parts to compute DenEntVisc
	  DenEntViscPart1 += Cx[ij]*x_EntFluxj + Cy[ij]*y_EntFluxj;
	  DenEntViscPart2 += Cx[ij]*fxj + Cy[ij]*fyj;
	  
	  // update ij
	  ij+=1;
	} //gj
      ith_DenEntVisc = (fabs(DenEntViscPart1) + fabs(DEnti)*fabs(DenEntViscPart2)+1E-10);
      EntVisc[gi] = std::pow(fabs(ith_NumEntVisc)/ith_DenEntVisc, ENTROPY_POWER);
    }//gi
}

void FE_Evolution::compute_uDot(GridFunction &un)
{
  Vector rhs_uDot;
  rhs_uDot.SetSize(numDOFs);
  rhs_uDot=0.;
  uDot=0.;

  DenseMatrix dshape(dofs_per_cell,dim), invJ(dim), grad_shape(dofs_per_cell,dim);
  Vector  cell_rhs_uDot(dofs_per_cell), shape(dofs_per_cell), 
    un_dofs(dofs_per_cell), grad_un_at_ref(dim), grad_un(dim);
  double uhn;
  Array<int> local_dofs_indices;

  // loop in cells
  for (int eN=0; eN<nElements; ++eN)
    {
      // get local to global dofs, fe space, etc
      fespace->GetElementDofs(eN,local_dofs_indices);
      const FiniteElement &fe = *fespace->GetFE(eN);
      ElementTransformation &eltrans = *fespace->GetElementTransformation(eN);
      const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), quad_order);
      
      // local dofs of un
      un.GetSubVector(local_dofs_indices,un_dofs);

      cell_rhs_uDot = 0.0;
      for (int q=0; q<ir->GetNPoints(); ++q)
	{
	  const IntegrationPoint &ip = ir->IntPoint(q);
	  //shape function and grad(shape) at reference element 
	  fe.CalcShape(ip, shape);
	  fe.CalcDShape(ip, dshape);

	  // related to the transformation: detJ, invJ and weight
	  eltrans.SetIntPoint (&ip);
	  double detJ = eltrans.Weight();
	  double w = ip.weight;
	  double detJxW = detJ * w;
	  CalcAdjugate(eltrans.Jacobian(), invJ);
	  invJ *= 1.0/detJ;

	  // get solution uhn at quad points
	  double uhn = un_dofs*shape;

	  // get grad(un) at quad points
	  dshape.MultTranspose(un_dofs,grad_un_at_ref);
	  invJ.Mult(grad_un_at_ref,grad_un);
	  
	  Vector vel(2), X(2);
	  X[0] = ip.x; X[1] = (dim==2 ? ip.y : 0.0);
	  velocity_function(X,vel);

	  double flux = 0;
	  if (PROBLEM_TYPE==0)
	    {
	      flux = vel[0] * grad_un[0]; 
	      if (dim==2)
		flux += vel[1] * grad_un[1]; 
	    }
	  else if (PROBLEM_TYPE==1) // Burgers
	    flux = uhn*(grad_un[0]+grad_un[1]);
	  else if (PROBLEM_TYPE==2) // KPP
	    flux = std::cos(uhn)-std::sin(uhn);
	  else
	    {
	      std::cout << "BL problem not yet implemented... aborting!" << std::endl;
	      abort();
	    }
	  for (unsigned int i=0; i<dofs_per_cell; ++i)
	    cell_rhs_uDot[i] -= flux * shape[i] * detJxW;
	}      
      // distribute
      rhs_uDot.AddElementVector(local_dofs_indices,cell_rhs_uDot);
    } //elements  
  mass_matrix_solver.Mult(rhs_uDot,uDot);
}

void FE_Evolution::update_velocity()
{
  Vector coord_ith_node(dim);  
  Vector vel(dim);
  for (int gi=0; gi<numDOFs; gi++)
    {
      coord.GetRow(gi,coord_ith_node);
      velocity_function(coord_ith_node, vel);
      u_vel_dofs[gi] = vel[0];
      v_vel_dofs[gi] = dim==2 ? vel[1] : 0;
    }
}

void FE_Evolution::compute_solution(GridFunction &unp1, GridFunction &un)
{
  std::vector<double> tmp_to_test_fluxStar;
  tmp_to_test_fluxStar.resize(numDOFs);

  max_edge_based_cfl = 0.;
  Array<int> local_dofs_indices;

  update_velocity();
  ////////////////////////////////////////////////////
  // ********** COMPUTE ENTROPY RESIDUAL ********** //
  ////////////////////////////////////////////////////
  compute_EV_residual(un);

  ////////////////////////////////////////
  // ********** COMPUTE uDot ********** //
  ////////////////////////////////////////
  compute_uDot(un);
  
  ///////////////////////////////////////////////////////////////
  // ********** ZERO OUT VECTORS AND COMPUTE BOUNDS ********** //
  ///////////////////////////////////////////////////////////////
  int ij=0;
  for (int gi=0; gi<numDOFs; gi++)
    {
      lowOrderFluxTerm[gi] = 0.0;
      lowOrderDissipativeTerm[gi] = 0.0;
      fluxStar[gi] = 0.0;
      tmp_to_test_fluxStar[gi] = 0.0;
      umax[gi] = -1E10;
      umin[gi] = 1E10;
    }

  for (int i=0; i<nElements*dofs_per_cell; i++)
    {
      element_flux_i[i] = 0;
      vVector[i] = 0;
    }

  //////////////////////////////////////////////////
  // ********** FIRST LOOP ON ELEMENTS ********** //
  //////////////////////////////////////////////////
  //* compute umax and umin
  //* compute lowOrderFluxTerm, lowOrderDissipativeTerm
  //* compute matrix dLowElem
  //* compute qi and store it in element_flux_i
  //* compute the first part of element_flux_qij: which is dij*(ui-uj)
  //* compute wBarij and wBarji
  //* compute dLowElemi, which is used later to compute the cfl
  DenseMatrix dshape(dofs_per_cell,dim), invJ(dim), grad_shape(dofs_per_cell,dim);
  Vector un_dofs(dofs_per_cell), uDot_dofs(dofs_per_cell), shape(dofs_per_cell), grad_shape_i(dim);
  Vector elementMass(dofs_per_cell), elementMassCorrection(dofs_per_cell), elementFluxCorrection(dofs_per_cell);
  for (int eN=0; eN<nElements; ++eN)
    {
      // get local to global dofs, fe space, etc
      fespace->GetElementDofs(eN,local_dofs_indices);
      const FiniteElement &fe = *fespace->GetFE(eN);
      ElementTransformation &eltrans = *fespace->GetElementTransformation(eN);
      const IntegrationRule *ir = &IntRules.Get(fe.GetGeomType(), quad_order);

      // zero out element vectors
      elementMass = 0.;
      elementMassCorrection = 0.;
      elementFluxCorrection = 0.;

      // local dofs of un and uDot
      un.GetSubVector(local_dofs_indices,un_dofs);
      uDot.GetSubVector(local_dofs_indices,uDot_dofs);

      // loop over quad points: get the integral-based components of qi
      for (int q=0; q < ir->GetNPoints(); ++q)
	{
	  const IntegrationPoint &ip = ir->IntPoint(q);
	  //shape function and grad(shape) at reference element 
	  fe.CalcShape(ip, shape);
	  fe.CalcDShape(ip, dshape);

	  // related to the transformation: detJ, invJ and weight
	  eltrans.SetIntPoint (&ip);
	  double detJ = eltrans.Weight();
	  double w = ip.weight;
	  double detJxW = detJ * w;
	  CalcAdjugate(eltrans.Jacobian(), invJ);
	  invJ *= 1.0/detJ;
	  mfem::Mult(dshape,invJ,grad_shape);

	  // get un at quad points
	  double uhn = un_dofs*shape;
	  double uhDot = uDot_dofs*shape;

	  // get flux
	  Vector vel(2), X(2);
	  X[0] = ip.x; X[1]= dim==2 ? ip.y : 0.0;
	  velocity_function(X,vel);
	  Vector flux(dim);
	  flux[0] = xFlux(vel[0],uhn);
	  if (dim==2)
	    flux[1] = yFlux(vel[1],uhn);

	  for (unsigned int i=0; i<dofs_per_cell; ++i)
	    {
	      grad_shape.GetRow(i,grad_shape_i);
	      elementMass[i] += shape[i]*detJxW;
	      elementMassCorrection[i] += -uhDot*shape[i]*detJxW;
	      elementFluxCorrection[i] += (flux*grad_shape_i)*detJxW;
	    }	  
	}
      
      // loop on local dofs
      for (unsigned int i=0; i<dofs_per_cell; ++i)
	{
	  int eN_i=eN*dofs_per_cell+i;
	  int gi = local_dofs_indices[i];

	  // solution, velocity and fluxes at node i (within the element eN)
	  double solni=un[gi];
	  double u_veli=u_vel_dofs[gi];
	  double v_veli=v_vel_dofs[gi];

	  // min and max in the given cell
	  double umaxi = solni;
	  double umini = solni;
	  
	  // about fluxes
	  double fxi = xFlux(u_veli,solni);
	  double fyi = yFlux(v_veli,solni);
	  
	  // about entropy
	  double etai = ENTROPY(solni);
	  double wi = DENTROPY(solni);
	  double xEntPoti = xEntPot(u_veli,solni);
	  double yEntPoti = yEntPot(v_veli,solni);
	  
	  // first part of vector qi
	  elementMassCorrection[i] += elementMass[i]*uDot[gi];
	  double qi = elementMassCorrection[i] + elementFluxCorrection[i];

	  // for edge_based_cfl
	  double dLowElemii = 0.;

	  // loop in j
	  for (unsigned int j=0; j < dofs_per_cell; ++j)
	    {
	      int eN_i_j = eN_i*dofs_per_cell+j;
	      int gj = local_dofs_indices[j];

	      // solution, velocity and fluxes at node j (within the element eN)
	      double solnj = un[gj];
	      double u_velj=u_vel_dofs[gj];
	      double v_velj=v_vel_dofs[gj];
	      
	      // min and max in the given cell
	      if (USE_Q1_STENCIL==1)
		{
		  if (element_Q1_sparsity(i,j)>0)
		    {
		      umaxi = fmax(solnj, umaxi);
		      umini = fmin(solnj, umini);
		    }
		}
	      else
		{
		  umaxi = fmax(solnj, umaxi);
		  umini = fmin(solnj, umini);
		}
	      
	      // about fluxes
	      double fxj = xFlux(u_velj,solnj);
	      double fyj = yFlux(v_velj,solnj);
	      double fxij = xFluxij(u_veli,u_velj,solni,solnj);
	      double fyij = yFluxij(v_veli,v_velj,solni,solnj);

	      // about entropy
	      double etaj = ENTROPY(solnj);
	      double wj = DENTROPY(solnj);
	      double xEntPotj = xEntPot(u_velj,solnj);
	      double yEntPotj = yEntPot(v_velj,solnj);
	      
	      // matrices
	      double Cx = CxElem[eN_i_j];
	      double Cy = CyElem[eN_i_j];
	      double CTx = CTxElem[eN_i_j];
	      double CTy = CTyElem[eN_i_j];
	      double PrCx = PrCxElem[eN_i_j];
	      double PrCy = PrCyElem[eN_i_j];
	      double PrCTx = PrCTxElem[eN_i_j];
	      double PrCTy = PrCTyElem[eN_i_j];

	      // low-order flux part of qi 
	      qi -= (CTx*fxj + CTy*fyj);
	      qi -= ((Cx-PrCx)*fxj + (Cy-PrCy)*fyj);
	      
	      // compute low order flux term
	      lowOrderFluxTerm[gi] += (PrCx*fxj + PrCy*fyj);
	      
	      // compute low-order dissipative flux
	      double dLowElemij = 0;
	      if (i!=j)
		{
		  dLowElemij = compute_dij(solni,solnj,
					   u_veli,v_veli,
					   u_velj,v_velj,
					   PrCx, PrCy,
					   PrCTx, PrCTy);
		  dLowElemii -= dLowElemij;
		  
		  // ************************** //
		  // ***** compute dijMin ***** // for entropy stability
		  // ************************** //
		  double qTildeij = -0.5*(wi-wj)*((PrCx-Cx)*(fxj-fxi)+(PrCy-Cy)*(fyj-fyi));
		  double qTildeji = -0.5*(wj-wi)*((PrCTx-CTx)*(fxi-fxj)+(PrCTy-CTy)*(fyi-fyj));
		  double Qij = (PrCx*(xEntPotj-xEntPoti+0.5*(wi-wj)*(fxj+fxi)) + 
				PrCy*(yEntPotj-yEntPoti+0.5*(wi-wj)*(fyj+fyi)));// + qTildeij;
		  double Qji = (PrCTx*(xEntPoti-xEntPotj+0.5*(wj-wi)*(fxj+fxi)) + 
				PrCTy*(yEntPoti-yEntPotj+0.5*(wj-wi)*(fyj+fyi)));// + qTildeji;
		  double dijMin = (solnj==solni ? 0 :
				   2*fmin(0.,fmin(Qij,Qji))/((solnj-solni)*(wi-wj)));
		  if (dijMin < 0) 
		    {
		      std::cout << "dijMin is negative... aborting" << std::endl;
		      abort();
		    }
		  dijMin = fmin(dijMin,dLowElemij);
		  element_dijMin[eN_i_j] = dijMin;
		  double nuij = 
		    fmax(0.,fmax((Cx*(fxj+fxi-2*fxij)+Cy*(fyj+fyi-2*fyij))/(wj-wi+1E-15),
				 (CTx*(fxj+fxi-2*fxij)+CTy*(fyj+fyi-2*fyij))/(wi-wj+1E-15)));
		  // tmp hack
		  //element_flux_qij[eN_i_j] += (dijMin-dLowElemij)*(solnj-solni) + nuij*(wj-wi);
		  // end of computation of dijMin //
		  
		  // save anti-dissipative flux into element_flux_qij
		  element_flux_qij[eN_i_j] = (cE*fmax(EntVisc[gi],EntVisc[gj])-1.0)*dLowElemij*(solnj-solni);
		  tmp_to_test_fluxStar[gi] += (cE*fmax(EntVisc[gi],EntVisc[gj])-1.0)*dLowElemij*(solnj-solni);

		  // compute low-order dissipative term 
		  lowOrderDissipativeTerm[gi] += dLowElemij*(solnj-solni);

		  // compute wBar states (wBar = 2*dij*uBar)
		  wBarij[eN_i_j] = (2.0*dLowElemij*(solnj+solni)/2.0
				    -(PrCx*(fxj-fxi) + PrCy*(fyj-fyi)));
		  wBarji[eN_i_j] = (2.0*dLowElemij*(solnj+solni)/2.0
				    -(PrCTx*(fxi-fxj) + PrCTy*(fyi-fyj)));
		  
		  // save low order matrix
		  dLowElem[eN_i_j] = dLowElemij;
		}
	      else
		{
		  dLowElem[eN_i_j] = 0.0; // not true but irrelevant since we know that fii=0
		  element_flux_qij[eN_i_j] = 0.0;
		  wBarij[eN_i_j] = 0.0;
		  wBarji[eN_i_j] = 0.0;
		  element_dijMin[eN_i_j] = 0.0;
		}
	    } //j
	  element_flux_i[eN_i] += qi;
	  dLowElemi[gi] = dLowElemii;

	  // min and max 
	  umax[gi] = fmax(umax[gi],umaxi);
	  umin[gi] = fmin(umin[gi],umini);
	} //i
    } //elements

  ///////////////////////////////////////////////////
  // ********** SECOND LOOP IN ELEMENTS ********** //
  ///////////////////////////////////////////////////
  // * compute vVector = inv(element_ML_minus_MC)*element_flux_i
  for(int eN=0;eN<nElements;eN++) //loop in cells
    {
      // compute element vector v //
      double mean = 0;
      for(int i=0;i<dofs_per_cell;i++)
	{
	  int eN_i = eN*dofs_per_cell+i;
	  for(int j=0;j<dofs_per_cell;j++)
	    {
	      int eN_j = eN*dofs_per_cell+j;
	      int eN_i_j = eN_i*dofs_per_cell+j;
	      vVector[eN_i] += inv_element_ML_minus_MC_Q1(i,j)*element_flux_i[eN_j];
	      mean += inv_element_ML_minus_MC_Q1(i,j)*element_flux_i[eN_j];
	    }
	}
      mean /= dofs_per_cell;
      
      // substract mean
      for(int i=0;i<dofs_per_cell;i++)
	{
	  int eN_i = eN*dofs_per_cell+i;
	  vVector[eN_i] -= mean;
	}
      // end of computation of the vector v //
    }//elements
 
  //////////////////////////////////////////////////
  // ********** THIRD LOOP IN ELEMENTS ********** //
  //////////////////////////////////////////////////
  // *compute element_flux_qij
  for(int eN=0;eN<nElements;eN++) //loop in cells
    {
      for(int i=0;i<dofs_per_cell;i++)
	{
	  int eN_i = eN*dofs_per_cell+i;
	  for(int j=0;j<dofs_per_cell;j++)
	    {
	      int eN_j = eN*dofs_per_cell+j;
	      int eN_i_j = eN_i*dofs_per_cell+j;
	      element_flux_qij[eN_i_j] += element_MC_Q1(i,j)*(vVector[eN_i]-vVector[eN_j]);	      
	    }
	}
      // print element_flux_qij for element 0... for debugging
      //DenseMatrix test_element_flux_qij(dofs_per_cell);
      //for(int i=0;i<dofs_per_cell;i++)
      //{
      //int eN_i = eN*dofs_per_cell+i;
      //for(int j=0;j<dofs_per_cell;j++)
      //  {
      //    int eN_i_j = eN_i*dofs_per_cell+j;
      //    test_element_flux_qij(i,j) = element_flux_qij[eN_i_j];
      //  }
      //}
      //test_element_flux_qij.Print();
      //abort();
    }
  
  ///////////////////////////////////////////////////
  // ********** FOURTH LOOP IN ELEMENTS ********** //
  ///////////////////////////////////////////////////
  // NOTE: merge this loop with the previous ones 
  // *compute fluxStar
  for (unsigned int eN=0; eN<nElements; ++eN)
    {
      // get local to global dofs, fe space, etc
      fespace->GetElementDofs(eN,local_dofs_indices);

      // loop in local dofs
      for (unsigned int i=0; i<dofs_per_cell; ++i)
	{
	  int eN_i=eN*dofs_per_cell+i;
	  int gi = local_dofs_indices[i];
	  
	  // get bounds at node i
	  double umini = umin[gi];
	  double umaxi = umax[gi];

	  //TMP: for debugging
	  tmp_to_test_fluxStar[gi] += element_flux_i[eN_i]; 

	  // loop in j
	  for(unsigned int j=0; j<dofs_per_cell; ++j)
	    {
	      int eN_i_j = eN_i*dofs_per_cell+j;
	      int gj = local_dofs_indices[j];

	      // get bounds at node j
	      double uminj = umin[gj];
	      double umaxj = umax[gj];

	      // compute element flux star
	      double fij = element_flux_qij[eN_i_j]; // high-order re-distributed flux
	      double dij = dLowElem[eN_i_j];
	      double wij = wBarij[eN_i_j];
	      double wji = wBarji[eN_i_j];
	      
	      if (i!=j)
		{
		  if (fij > 0)
		    {
		      if (METHOD==0)
			fluxStar[gi] += 0; // low-order method
		      else if (METHOD==1)
			fluxStar[gi] += fij; // high-order without flux limiting 
		      else if (METHOD==2)
			{
			  if (USE_DISCRETE_UPWINDING==1 && PROBLEM_TYPE==0)
			    {
			      fluxStar[gi] += fmin(fij,fmax(0.,fmin(2*dij*umaxi-wij, wji-2*dij*uminj)));
			    }
			  else
			    fluxStar[gi] += fmin(fij,fmin(2*dij*umaxi-wij,wji-2*dij*uminj));
			}
		    }
		  else // fij<0
		    {
		      if (METHOD==0)
			fluxStar[gi] = 0; // low-order method
		      else if (METHOD==1)
			fluxStar[gi] += fij; // high-order without flux limiting
		      else if (METHOD==2)
			{
			  if (USE_DISCRETE_UPWINDING==1 && PROBLEM_TYPE==0)
			    fluxStar[gi] += fmax(fij,fmin(0.,fmax(2*dij*umini-wij, wji-2*dij*umaxj)));
			  else
			    fluxStar[gi] += fmax(fij,fmax(2*dij*umini-wij,wji-2*dij*umaxj));
			}
		    }
		} //i!=j
	    } //j
	} //i
    } //eN

  // tmp to test flux star //
  if (TEST_FLUX_STAR==1)
    {
      for (unsigned int gi=0; gi<numDOFs; gi++)
	{
	  if (fabs(tmp_to_test_fluxStar[gi] - fluxStar[gi]) > 1E-12)
	    {
	      std::cout << "test of fluxStar did not work" << std::endl;
	      std::cout << tmp_to_test_fluxStar[gi] << "\t" 
			<< fluxStar[gi] << "\t"
			<< tmp_to_test_fluxStar[gi] - fluxStar[gi] 
			<< std::endl;
	      abort();
	    }
	}
    }

  //////////////////////////////////////////////
  // ********** FIRST LOOP IN DOFs ********** //
  //////////////////////////////////////////////
  // *compute edge_based_cfl = 2|dLowii|/mi
  // *compute low-order solution
  // *compute the solution unp1
  for (unsigned int gi=0; gi<numDOFs; gi++)
    {
      double mi = lumped_mass_matrix[gi];
      double solni = un[gi];
      
      // compute edge based cfl	
      max_edge_based_cfl = fmax(max_edge_based_cfl,2*fabs(dLowElemi[gi])/mi);

      // compute low order solution
      lowOrderSolution[gi] = solni - dt/mi * (lowOrderFluxTerm[gi]
					      -lowOrderDissipativeTerm[gi]
					      );

      // compute update: solution at tnp1
      unp1[gi] = lowOrderSolution[gi] + dt/mi*fluxStar[gi];

      //std::cout << uDot[gi] << std::endl;
      //unp1[gi] = solni + dt*uDot[gi];
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

      // ************************ //
      // ***** SSP-RK method **** //
      // ************************ //
      if (SSP_order==1)
	{
	  // first stage //
	  if (verbosity>1)
	    cout << "   ...First SSP stage" << std::endl;
	  compute_solution(unp1,un);
	}
      else if (SSP_order==2)
	{
	  // first stage //
	  if (verbosity>1)
	    cout << "   ...First SSP stage" << std::endl;	  
	  compute_solution(uSt1,un);

	  // second stage //
	  if (verbosity>1)
	    cout << "   ...Second SSP stage" << std::endl;
	  compute_solution(unp1,uSt1);
	  unp1*=0.5;
	  unp1.Add(0.5,un);
	}
      else // SSP_order=3
	{
	  // first stage //
	  if (verbosity>1)
	    cout << "   ...First SSP stage" << std::endl;	  
	  compute_solution(uSt1,un);

	  // second stage //
	  if (verbosity>1)
	    cout << "   ...Second SSP stage" << std::endl;
	  compute_solution(uSt2,uSt1);
	  uSt2*=0.25;
	  uSt2.Add(0.75,un);

	  // third stage //
	  if (verbosity>1)
	    cout << "   ...Third SSP stage" << std::endl;
	  compute_solution(unp1,uSt2);
	  unp1*=2.0/3.0;
	  unp1.Add(1.0/3.0,un);
	}

      // update old solution
      un = unp1;

      // check if this is last step or update time
      fixed_dt=0; // if 0: None
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
  // init some parameters 
  cE=1.0;
  SSP_order=3;

  skip_zeros=0; //This must be 0 to preserve the sparsity between the operators: mass matrix, Cx, etc
  const FiniteElement &fe0 = *fespace->GetFE(0);
  dofs_per_cell = fe0.GetDof();
  nElements = fespace->GetNE();
  numDOFs=fespace->GetVSize();
  dim = (*fespace->GetMesh()).Dimension();
  quad_order = 2*fe0.GetOrder();
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
  int verbosity = 1;
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
  H1_FECollection fec(order, dim, mfem::BasisType::Positive); //positive basis
  FiniteElementSpace fes(mesh, &fec);
  
  ////////////////////////
  // ***** SOLVER ***** //
  ////////////////////////
  FE_Evolution adv(fes, precondition_c_matrices, PROBLEM_TYPE, verbosity);
  FunctionCoefficient u0Mat1(u0Mat1_function);
  
  //////////////////////////////////////////////
  // ***** GET COORDINATES OF THE NODES ***** //
  //////////////////////////////////////////////
  FiniteElementSpace fesVSpace(mesh, &fec, dim);
  GridFunction nodes(&fesVSpace);
  mesh->GetNodes(nodes);
  const int nNodes = nodes.Size() / dim;
  DenseMatrix coord(nNodes,dim);
  for (int i = 0; i < nNodes; ++i) 
    {
      for (int j = 0; j < dim; ++j) 
	{
	  coord(i,j) = nodes(j * nNodes + i); 
	}   
    }
  adv.setCoord(coord);

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

	  if (PROBLEM_TYPE==2)
	    {
	      string ProblemName = "ProblemName";
	      sout << "window_title '" << ProblemName << "'\n"
		   << "window_geometry "
		   << 0 << " " << 0 << " " << 1080/2 << " " << 1080/2 << "\n"
		   << "autoscale on\n"
		   << "keys cpppppppppp8884444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444444";
	      bool vec = false;
	      if ( vec ) { sout << "vvv"; }
	      sout << endl;
	    }
	  
	}
    }
  
  ///////////////////////
  // ***** SETUP ***** //
  ///////////////////////
  adv.setup();
  cout << "Number of DOFs: " << adv.numDOFs << std::endl;

  //////////////////////////////
  // ***** GET MATRICES ***** // mass matrix and C-matrices
  //////////////////////////////
  adv.get_matrices_and_init_data_structures();
  
  ///////////////////////////
  // ***** TIME LOOP ***** //
  ///////////////////////////
  adv.time = 0.0;
  adv.num_time_steps=0;
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
