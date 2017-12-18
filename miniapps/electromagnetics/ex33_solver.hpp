// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.
// asad anees technology University of Clausthal 
// Feel free to if you have any questions
// asad.anees@tu-clausthal.de
// asad.anees@gmail.com   //   https://github.com/Asadanees
#ifndef MFEM_EX33_SOLVER
#define MFEM_Ex33_SOLVER

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "mfem.hpp"


using namespace std;
using namespace mfem;
/// Global variable for convergence 
	static const double Rel_toL=1e-8;
	
	static const double Abs_toL=0.0;
	
	static const int Max_Iter=1000;

namespace mfem
{

namespace electromagnetics
{


/**   
After spatial discretization, the Maxwell's equation can be written
   as a system of ODEs:

     
      dE/dt       = (Nspmat1)^{-1}*(+ MB^T*B+J))   
      
      dB/dt       = (curl (E))                      
      curl is a discrete curl
      Nspmat1= (M1(sigma) + dt S1(1/mu))
      M1 is mass matrix obtained by VectorFEMassIntegrator
      S1 is matric Curl matrix obtained by CurlIntegrator
      
      
   where

    
     E is the 1-form electric field (see the end of the file),
     B is the 2-form magnetic field (see the end of the file),
     Nspmat is the mass matrix discretized by nedelec space with weight permittivity,
     Rspmat is the mass matrix discretize dby raviart space with weight permeability,
     permittivity is a electric permittivity, 
     permeability is magnetic permeability,
     conductivity is the electrical conductivity,
     J is given Electric current density, (see the end of the file),
     <curl u, v>E was descretized by mixed finite element method,
     MB is Curl matrix obtained and discretized by Mixed finite element method,
     <curl u, v>E= MB,  (MixedBilinearForm is use to get MB with VectorFECurlIntegrator),
     <u, Curl v>B= MB^T (MB^T is a transpose of MB),
     <u, v>_{conductivity}= Conduct sparse matix with weight electric conductivity
**/

class Maxwell :public TimeDependentOperator
{
protected: 
	FiniteElementSpace &Nedelec;
	
	FiniteElementSpace &Raviart;
	
	mutable Array<int> ess_bdr;
	
	BilinearForm *N, *N1;
	
	BilinearForm *R;
	
	
	BilinearForm *Conduct;
	
	MixedBilinearForm *MB, *weakCurl;
	
	DiscreteLinearOperator  *curl;
	
	LinearForm *Current_Density;
	
	SparseMatrix *Nspmat, *Rspmat, *Conductspmat, *MBspmat, *Nspmat1;
	

	
	SparseMatrix *MBTmat;
	
	/// precondition for the mass Matrix Nspmat1 and Pspmat
	
	
	mutable DSmoother Nmat_prec;

	mutable DSmoother Rmat_prec;

	/// krylov solver to converting the mass matrix Nspmat and Rspmat
	
	mutable CGSolver *N_Solver;
	
	mutable CGSolver *R_Solver;

	 Vector *X0,*X1, *B01v;
	
	 
	 GridFunction *E_right, *B0;
	 
	 GridFunction *Current;
	 
	 double dt_A;
	
	double permittivity, permeability, conductivity, muinv;
	
	void BuildCurl(double muInv);
	
	void buildN1( double premittivi, double muinv, double dt);
	double current_dt;
	
	
	
	const int E_size, B_size;
	
	/// dim of mesh
	
	int Mesh_dim;
	
	 void   (*Jcurrent_src )(const Vector&, double, Vector&);
	 
	 VectorCoefficient *JCoef_;      // Time dependent current density coefficiet
	
public:

Maxwell (FiniteElementSpace & nedelec,FiniteElementSpace & raviart,  
    double premittivity, double conductivity, double premability, int mesh_dim, Array<int>&ess_bdr, void   (*Jcurrent_src_arg  )(const Vector&, double, Vector&));

virtual void ImplicitSolve(const double dt, const Vector &EB,  Vector &dEB_t ) ;

//virtual void Mult(const Vector &EB, Vector &dEB_t) const;


void Init(VectorCoefficient & electric_coef, VectorCoefficient & magnetic_coeff );

virtual ~Maxwell(); 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
};

}// namespae electromagnetics
} // namespace mfem

#endif // MFEM_USE_MPI

#endif // MFEM_MAXWELL_SOLVER
