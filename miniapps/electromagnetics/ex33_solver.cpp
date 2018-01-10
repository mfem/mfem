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

#include "../../config/config.hpp"

#ifdef MFEM_USE_MPI

#include "ex33_solver.hpp"

using namespace std;

namespace mfem
{



namespace electromagnetics
{



Maxwell
::Maxwell(FiniteElementSpace &nedelec, FiniteElementSpace &raviart,
                           double permittivity_arg, double conductivity_arg, double permeability_arg, int mesh_dim, Array<int>&ess_bdr_arg, void   (*Jcurrent_src_arg  )(const Vector&, double, Vector&))
   : TimeDependentOperator(nedelec.GetVSize()+raviart.GetVSize(), 0.0), 
   Nedelec(nedelec), 
   Raviart(raviart),
  
     N(NULL), 
     R(NULL), 
     Conduct(NULL), 
     MB(NULL), 
     Current_Density(NULL),
     N1(NULL), 
     weakCurl(NULL),
     E_size(nedelec.GetVSize()), 
     B_size(raviart.GetVSize()), 
     Mesh_dim(mesh_dim),
     N_Solver(NULL), 
     R_Solver(NULL),
     Nspmat(NULL), 
     Rspmat(NULL), 
     Conductspmat(NULL), 
     MBspmat(NULL), 
     Nspmat1(NULL),
     curl(NULL), 
     //Current(curr_arg),
     dt_A(-1.0),
     E_right(NULL),
     Jcurrent_src(Jcurrent_src_arg), 
     JCoef_(NULL)
     
     
{
	cout<<"Initialize of constructor Maxwell operator"<<endl;
	ess_bdr.SetSize(ess_bdr_arg.Size());
	for(int i=0; i<ess_bdr_arg.Size(); i++)
	{
		ess_bdr[i]=ess_bdr_arg[i];
	}
	E_right= new GridFunction(&Nedelec);
	
	B0=new  GridFunction(&Raviart);
	
	// declare the vector
	
	X0= new Vector;

	B01v=new Vector;
	
	X1=new Vector;
	
	
	
	Nspmat= new SparseMatrix;
	
	Nspmat1= new SparseMatrix;
	
	Rspmat= new SparseMatrix;
	
	
	// Here we are setting permittivity constant
	// permeability, conductivity and muinv. 
	 
	permittivity=permittivity_arg;
	
	permeability=permeability_arg;
	
	conductivity=conductivity_arg;
	
    muinv= 1/permeability;
   
   cout<<"muInv="<<muinv<<endl;
  
	this->BuildCurl(muinv);
	
	ConstantCoefficient Permittivity(permittivity);
	
	ConstantCoefficient Permeability(permeability);
	
	ConstantCoefficient Muinv(muinv);
	
    ConstantCoefficient Conductivity(conductivity);
	
	// Here we shall find the mass Matrix Nspmat  for Mult member funciton 
	// But we did not implemet Mult in this code
	
	N= new BilinearForm(&Nedelec);
	
	N->AddDomainIntegrator(new VectorFEMassIntegrator(Permittivity));
	
	N->AddDomainIntegrator(new CurlCurlIntegrator(Muinv));
	
	N->Assemble();
	
	N->FormSystemMatrix(ess_bdr, *Nspmat);
	

    
    
    
    
    Conduct=new BilinearForm(&Nedelec);
    
    Conduct->AddDomainIntegrator(new VectorFEMassIntegrator(Conductivity));
	
	Conduct->Assemble();
	
	//Conduct->Finalize();
	

	
	
	
	// Here we shall find the mass Matrix R descritzed by Raviart thomas finite element method
	
	R= new BilinearForm(&Raviart);
	
	R->AddDomainIntegrator(new VectorFEMassIntegrator(Permeability));
	
	
	R->Assemble();
	
	R->Finalize();


	
	
	
	// now we shall use the Mixed Method Matix to solve <Curl u/mu, v>E and <u Curl V>B
	
	MB= new MixedBilinearForm(&Nedelec, &Raviart);
	
	MB->AddDomainIntegrator(new VectorFECurlIntegrator(Muinv));
	
	MB->Assemble();
	MB->Finalize();
	
	JCoef_ = new VectorFunctionCoefficient(3, Jcurrent_src);
	
	// Find the vector for given Electric_Current see the ex33.cpp file
	Current_Density= new LinearForm(&Nedelec);
    Current_Density->AddDomainIntegrator(new VectorFEDomainLFIntegrator(*JCoef_));
    Current_Density->Assemble();
    
   
}


/**
This is computational code that computes dEB/dt implicitly
where EB is the block vector containing E and B, for detail see block vector class in linalg

        
	  dE/dt       = - (Nspma1))^{-1}*(-Conduct*E + MB^T*B+J))
      dB/dt       = - {-1}*(Curl (E))

where W is the Joule heating.

Boundary conditions are applied to E.
boundary conditions are not applied to B.

**/
void Maxwell::ImplicitSolve(const double dt, const Vector &EB,  Vector &dEB_t ) 

{
	if(Nspmat1==NULL || fabs(dt-dt_A) > 1.0e-12*dt) 

	{
		cout<<"in the implicit solve"<<endl;
		cout<<"gettime="<<dt<<endl;
		this->buildN1(permittivity, 0.666667, dt);
	}

		dEB_t=0.0;

		int Vsize_nd= Nedelec.GetTrueVSize();


		int Vsize_rt= Raviart.GetTrueVSize();	
		
		Array<int> block_offsets(3);
        
        block_offsets[0] = 0;
        
        block_offsets[1] = block_offsets[0]+Vsize_nd;
		
		block_offsets[2] = block_offsets[1]+Vsize_rt;
         
        Vector* ehptr  = (Vector*) &EB;
          
        GridFunction E, B;
          
        E.MakeRef(&Nedelec,   *ehptr,block_offsets[0]);
           
		B.MakeRef(&Raviart, *ehptr,block_offsets[1]);
	
		   
		GridFunction dE, dB;
		   
		dE.MakeRef(&Nedelec, dEB_t, block_offsets[0]);
		   
		dB.MakeRef(&Raviart, dEB_t, block_offsets[1]);
		
		// solve the first equations 
		weakCurl->MultTranspose(B, *E_right);
		
		
		*E_right+=*Current_Density;
      
   
		 if(N_Solver==NULL)
		 {
		 
		 cout<<"in N_Solver"<<endl;
		 
		 N_Solver= new CGSolver ;
		 
		 N_Solver->iterative_mode = false;
	
		 N_Solver->SetRelTol(Rel_toL);
	
		 N_Solver->SetAbsTol(Abs_toL);
	
		 N_Solver->SetMaxIter(Max_Iter);
	
		 N_Solver->SetPrintLevel(0);
	
		 N_Solver->SetPreconditioner(Nmat_prec);
		 
		 N_Solver->SetOperator(*Nspmat1);
		
		}
		N_Solver->Mult(*E_right, dE);
		// Here I find the value of E_riht 
		// than use E_ringt to solve the second equation dB/dt = - curl E
		*E_right=dE;
		curl->Mult(*E_right, dB);
		dB*=-1.0;
		
		
		
	}
void Maxwell::Init(VectorCoefficient & electric_coeff, VectorCoefficient & magnetic_coeff )
{
	int Vsize_nd= Nedelec.GetTrueVSize();
	
	int Vsize_rt= Raviart.GetTrueVSize();	
	
	Array<int> block_offsets(3);
        
        block_offsets[0] = 0;
        
        block_offsets[1] = block_offsets[0]+Vsize_nd;
		
		block_offsets[2] = block_offsets[1]+Vsize_rt;
		BlockVector EB(block_offsets);
         
        Vector* ehptr  = (Vector*) &EB;
          
        GridFunction E, B;
          
        E.MakeRef(&Nedelec,   *ehptr,block_offsets[0]);
           
		B.MakeRef(&Raviart, *ehptr,block_offsets[1]);
		
		// Initialize electric field E 
         E.ProjectCoefficient(electric_coeff);
       
       // Initialze magnetic flux B
        B.ProjectCoefficient(magnetic_coeff);
		
		
	
     
}

void Maxwell::BuildCurl(double muinv)
{
   if ( curl != NULL ) { delete curl; }
  

   curl = new DiscreteLinearOperator(&Nedelec, &Raviart);
   curl->AddDomainInterpolator(new CurlInterpolator);
   curl->Assemble();
 if ( weakCurl != NULL ) { delete weakCurl; }
 cout<<"value of muinv in build curl:="<<muinv<<endl;
   ConstantCoefficient MuInv(muinv);
   weakCurl = new MixedBilinearForm(&Nedelec, &Raviart);
   weakCurl->AddDomainIntegrator(new VectorFECurlIntegrator(MuInv));
   weakCurl->Assemble();
 
   
}
void Maxwell::buildN1( double permittivity, double muinv, double dt)
{

	
	
	ConstantCoefficient Permittivity1(permittivity);
	
    ConstantCoefficient MuInv_dt(muinv*dt);
	
	// Here we shall find the mass Matrix Nspmat1 
	
	N1= new BilinearForm(&Nedelec);
	
	
	N1->AddDomainIntegrator(new VectorFEMassIntegrator(Permittivity1));
	N1->AddDomainIntegrator(new CurlCurlIntegrator(MuInv_dt));
	N1->Assemble();
	N1->FormSystemMatrix(ess_bdr, *Nspmat1);
	//N1->Finalize();
	dt_A=dt;
}
Maxwell::~Maxwell()
{
	
	if(N!=NULL)
	{
	
	delete N;
	}
	if(N1!=NULL)
	{
	
	delete N1;
	}
	if(R!=NULL)
	{
	
	delete R;
	}
	if(Conduct!=NULL)
	{
	
	delete Conduct;
	}
	if(Current_Density!=NULL)
	{
		delete Current_Density;
	}
	if(JCoef_!=NULL)
	{  
		
		delete JCoef_;
	}
	if(MB!=NULL)
	{
	;
	delete MB;
	}
	if(N_Solver!=NULL)
	{
	
	delete N_Solver;
	}
	if(R_Solver!=NULL)
	{
	;
	delete R_Solver;
	}
	if(E_right!=NULL)
	{
	
	delete E_right;
	}
	
	if(B0!=NULL)
	{
	
	delete B0;
	}
	/// delete the vector
	
	if(X0!=NULL)
	{
	
	delete X0;
	}

	if(X1!=NULL)
	{
	
	delete X1;
	}
	if(B01v!=NULL)
	{
	
	delete B01v;
	}
	
	
	if (Nspmat!=NULL)
	{
	
	delete Nspmat;
	}
	if(Nspmat1!=NULL)
	{
	
	delete Nspmat1;
	}
	if (Rspmat!=NULL)
	{
	
	delete Rspmat;
	}
	if (Conductspmat!=NULL)
	{
	
	delete Conductspmat;
	}
	if (MBspmat!=NULL)
	{
	
	delete MBspmat;
	}
	
	if ( curl != NULL ) { delete curl; }
	if(weakCurl !=NULL) {  delete weakCurl;}
}

}// namespace electromagnetics

} // namespace mfem

#endif // MFEM_USE_MPI
