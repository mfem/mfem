

///Here I mentioned five points. Please check carefully five points
// (1)it is convege very small time step problem
// (2) have you any exact solution of linear Maxwell equations (bench mark problem)? 
// (3) Please provide the exact solution for value of permitticity, permeability and conductivity if you have
// (4) please check care fully electric field visulation and Implicitsolve
// (5) Plesae check the solution different value of permittivity, conductivity and permeability for example
//         double Permittivity=0.2;
//         double Conductivity=0;
//          double Permeability=2.5
// One Important this that we cheched carefully, Trivial Soltion of this problem is also satisfied
// if we take both ElectricField and MagneticField zero than trivial solution is exist 
/// ABOVE are 5 issued. Could you please check carefully.
//                         example 26
//                    Compile with make ex26
//                           Run ./ex26
// Description:  This example code solves a simple 3D  Maxwell's problem
//               
//                                 permittivity * dE/dt= -conductivity E + Curl H + Electric_Current
//                                 permeability * dH/dt= - Curl E 
//               permittivity, permittivity and conductivity are constant and in command line  oarameter
//               with essential boundary condition nxE = 0.
// 				 Boundary condition are applied only to E but 
//               boundary condition is not appplied  to H.
//               Electric_Current is given electric field at the end of this file.
//               Here, we use a given exact solution (E,H) and compute the
//               corresponding l.h.s. (E,H).  We discretize with Nedelec 
//               finite elements (electric field E) and Raviart-Thomas finite  
//               element (magnetic field H).
//
//                              boundary attribute 3
//                            +---------------------+
//               boundary --->|                     | boundary
//               attribute 1  |                     | attribute 2
//                            +---------------------+
//
// The E-field boundary condition specifies the essential BC (n cross E) on
// attribute 1 (front) and 2 (rear) given by function ElectricField_BC at bottom of this
// file. The E-field can be set on attribute 3 also.
// This problem work both beam Tetrahedron and Escher Mesh.
// You can use this commad in section 1.
// const char *mesh_file = "../data/beam-tet.mesh";
// const char *mesh_file = "../data/escher.mesh";


#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <memory>
#include <cmath>



using namespace std;

using namespace mfem;
/// Global variable for convergence of  
	static const double Rel_toL=1e-8;
	
	static const double Abs_toL=0.0;
	
	static const int Max_Iter=1000;
/**   
After spatial discretization, the Maxwell's equation can be written
   as a system of ODEs:

     
      dE/dt       = - (Nspmat(permittivity))^{-1}*(-Conduct*E + MB^T*H+J))
      dH/dt       = - (Rspmat(permeability))^{-1}*(MB*(E))
      
   where

    
     E is the 1-form electric field (see the end of the file),
     H is the 2-form magnetic field (see the end of the file),
     Nspmat is the mass matrix discretized by nedelec space with weight permittivity,
     Rspmat is the mass matrix discretize dby raviart space with weight permeability,
     permittivity is a electric permittivity, 
     permeability is magnetic permeability,
     conductivity is the electrical conductivity,
     J is given Electric current density, (see the end of the file),
     <curl u, v>E was descretized by mixed finite element method,
     MB is Curl matrix obtained and discretized by Mixed finite element method,
     <curl u, v>E= MB,  (MixedBilinearForm is use to get MB with VectorFECurlIntegrator),
     <u, Curl v>H= MB^T (MB^T is a transpose of MB),
     <u, v>_{conductivity}= Conduct sparse matix with weight electric conductivity
**/

class Maxwell :public TimeDependentOperator
{
protected: 
	FiniteElementSpace &Nedelec;
	FiniteElementSpace &Raviart;
	
	/// this is remain empty for the pure Neumann b.c
	
	mutable Array<int> ess_bdr;
	
	BilinearForm *N;
	
	BilinearForm *R;
	
	BilinearForm *Conduct;
	
	
	MixedBilinearForm *MB;
	
	LinearForm *Current_density;
	
	SparseMatrix *Nspmat, *Rspmat, *Conductspmat, *MBspmat;
	
	SparseMatrix *MBTmat;
	
	/// precondition for the mass Matrix N and R
	
	
	mutable DSmoother Nmat_prec;

	mutable DSmoother Rmat_prec;

	/// krylov solver to converting the mass matrix Nmat and Rmat
	
	mutable CGSolver *N_Solver;
	
	mutable CGSolver *R_Solver;

	 Vector *X0,*X1, *B0;
	 
	 GridFunction *E0, *H0;
	
	double permittivity, permeability, conductivity;
	
	double current_dt;
	
	
	
	const int E_size, H_size;
	
	/// dim of mesh
	
	int Mesh_dim;
	
public:
Maxwell (FiniteElementSpace & nedelec,FiniteElementSpace & raviart,  
    double premittivity, double conductivity, double premability, int mesh_dim, Array<int>&ess_bdr);
//virtual void Mult(const Vector &EH,  Vector &dEH_t ) const;
virtual void ImplicitSolve(const double dt, const Vector &EH,  Vector &dEH_t ) ;


// update the bilinearFoem K using the given true-dof u.
//void SetParameters(double _dt, const Vector *_E, const Vector *_H);

 

virtual ~Maxwell(); 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
};
/// exact solution of electrix field

	void  ElectricField(const Vector & x, const double t, Vector & E);

/// The exact solution of Magnectic Field

	void MagneticField(const Vector & x,const double t, Vector & H);

/// The electric current density is given

	void Electric_Current(const Vector & x, const double t, Vector &J);
	
/// The boundary condition for E


void ElectricField_BC(const Vector &x,  Vector &E);
	
	

/**
This is computational code that computes dEH/dt implicitly
where EH is the block vector containing E and H, for detail see block vector class in linalg

        
	  dE/dt       = - (Nspmat(permittivity))^{-1}*(-Conduct*E + MB^T*H+J))
      dH/dt       = - (Rspmat(permeability))^{-1}*(MB*(E))

where W is the Joule heating.

Boundary conditions are applied to E.
boundary conditions are not applied to H.

**/
void Maxwell::ImplicitSolve(const double dt, const Vector &EH,  Vector &dEH_t ) 
{
	//cout<<"in ImplicitSolve"<<endl;
	// Current is a vector that is corresponing exact value of J
	// compute dE_t=N{-1}(-Conductmat*E+BTmat*H+ Current) first Maxwell equation
		
		dEH_t=0.0;
/**     The big BlockVector stores the fields as follows
         E field
         H field
**/
		int Vsize_nd= Nedelec.GetTrueVSize();


		int Vsize_rt= Raviart.GetTrueVSize();	
		
		Array<int> block_offsets(3);
        
        block_offsets[0] = 0;
        
        block_offsets[1] = block_offsets[0]+Vsize_nd;
		
		block_offsets[2] = block_offsets[1]+Vsize_rt;
         
        Vector* ehptr  = (Vector*) &EH;
          
        GridFunction E, H;
          
        E.MakeRef(&Nedelec,   *ehptr,block_offsets[0]);
           
		H.MakeRef(&Raviart, *ehptr,block_offsets[1]);
		   
		GridFunction dE, dH;
		   
		dE.MakeRef(&Nedelec, dEH_t, block_offsets[0]);
		   
		dH.MakeRef(&Raviart, dEH_t, block_offsets[1]);
   
       
/**     Set up the linear form b(.) which corresponds to the Electric_Current J
        of the FEM linear system, which in this case is (Electric_Current, phi_i) where Electric_Current is
        given by the function f_exact and phi_i are the basis functions in the nedelec finite element fespace.
	
**/
        VectorFunctionCoefficient J(3, Electric_Current);
        
		Current_density=new LinearForm(&Nedelec);
   
		Current_density->AddDomainIntegrator(new VectorFEDomainLFIntegrator(J));
   
		Current_density->Assemble();
		
		// to check Current_density , uncomment the below line
		//Current_density->Print(); 
		//casting of Current_density to the GridFunction *curr
		
		GridFunction *curr= (GridFunction*)Current_density;
		*curr*=.01;
		// You can confirm that value of curr and Current_density is same, please uncomment the curr->Print();
		//curr->Print();
		
		Conduct->AddMult(E, *curr, -1.0);
		// curr+= MB *H;
		MB->AddMultTranspose(H, *curr, 1);
		//curr->Print();
		// E0=< v, curl u> H
		///MB->MultTranspose(H, *E0);
		// E0=< v, curl u> H - Conduct * E
		///Conduct->AddMult(E, *E0, -1.0);	
		
		GridFunction e_gf(&Nedelec);
        VectorFunctionCoefficient e_bc(3, ElectricField_BC);
        e_gf=0.0;
        e_gf.ProjectCoefficient(e_bc);
        e_gf.ProjectBdrCoefficientTangent(e_bc, ess_bdr);
     
        Array<int>ess_tdof_list;
        Nedelec.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
        N->FormLinearSystem(ess_tdof_list, e_gf,*curr, *Nspmat, *X0, *B0); 
         
         /// use the Krylov solver to intervating the mass matrix Nspmat
         if(N_Solver==NULL)
		 {
		 N_Solver= new CGSolver ;
		 
		 N_Solver->iterative_mode = false;
	
		 N_Solver->SetRelTol(Rel_toL);
	
		 N_Solver->SetAbsTol(Abs_toL);
	
		 N_Solver->SetMaxIter(Max_Iter);
	
		 N_Solver->SetPrintLevel(0);
	
		 N_Solver->SetPreconditioner(Nmat_prec);
	
		 N_Solver->SetOperator(*Nspmat);
		 }
/**          Mult operation is a solve
             X0 = A0^-1 * B0
 **/
		 N_Solver->Mult(*B0, *X0);
		 N->RecoverFEMSolution(*X0, *curr,E); 
		 dE=0.0;
		 GridFunction h_gf(&Raviart);
		 VectorFunctionCoefficient magnetic_exact(3, MagneticField);
         h_gf.ProjectCoefficient(magnetic_exact);
         //h_gf=0.0;
		
		 MB->Mult(E, h_gf);
		 h_gf*=-1;
		 
		 // use the Krylov solver to intervating the mass matrix R->SpMat()
		 if(R_Solver==NULL)
		 {
		
		 R_Solver= new CGSolver;
			  
		R_Solver->iterative_mode = false;
	
		R_Solver->SetRelTol(Rel_toL);
	
		R_Solver->SetAbsTol(Abs_toL);
	
		R_Solver->SetMaxIter(Max_Iter);
	
		R_Solver->SetPrintLevel(0);
	
		R_Solver->SetPreconditioner(Rmat_prec);
	   
		 R_Solver->SetOperator(R->SpMat()); 
		   
		}
		R_Solver->Mult(h_gf, dH);
		
		
		
	}
	


int main(int argc, char *argv[])
{
	
   /// 1. Parse command-line options
  
  const char *mesh_file = "../data/beam-tet.mesh";
  
   //const char *mesh_file = "../data/escher.mesh";
   int ref_levels = 2;
   int order = 2;
   int ode_solver_type = 22;
   double t_final = 12;
   double dt = 0.15;
   /// please check different value of permittivity , Conductivity and Permeability.
   /**double Permittivity=0.2;
   double Conductivity=0;
   double Permeability=2.5;**/
   bool visualization = true;
   bool visit=false;
   int vis_steps = 1;
   
    double Permittivity=100;
    double Conductivity=2;
    double Permeability=150;
   
   
   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "            11 - Forward Euler, 12 - RK2,\n\t"
                  "            13 - RK3 SSP, 14 - RK4.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&Permittivity, "-v", "--Permittivity",
                  "permittivity coefficient.");
   args.AddOption(&Permeability, "-permeability", "--permeability ",
                  "permeability coefficient.");
   args.AddOption(&Conductivity, "-conductivity", "--conductivity",
                  "conducitivity coefficient.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&visit, "-visit", "--visit-datafiles", "-no-visit",
                  "--no-visit-datafiles",
                  "Save data files for VisIt (visit.llnl.gov) visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);
   
/**   2. Read the mesh from the given mesh file. We can handle triangular,
         quadrilateral, tetrahedral and hexahedral meshes with the same code.
**/ 
      Mesh *mesh = new Mesh(mesh_file, 1, 1);
      
      int dim = mesh->Dimension();
      
/**  3.  Define the ODE solver used for time integration. Several implicit
         singly diagonal implicit Runge-Kutta (SDIRK) methods, as well as
         explicit Runge-Kutta methods are available.
**/
   ODESolver *ode_solver;
   switch (ode_solver_type)
   {
      // Implicit L-stable methods
      case 1: ode_solver = new BackwardEulerSolver; break;
      case 2: ode_solver = new SDIRK23Solver(2); break;
      case 3: ode_solver = new SDIRK33Solver; break;
      // Implicit A-stable methods (not L-stable)
      case 22: ode_solver = new ImplicitMidpointSolver; break;
      case 23: ode_solver = new SDIRK23Solver; break;
      case 24: ode_solver = new SDIRK34Solver; break;
      default:
         cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         return 3;
   }

/**    4.  Refine the mesh to increase the resolution. In this example we do
           ref_levels' of uniform refinement, where 'ref_levels' is a
           command-line parameter.
**/
        for (int lev = 0; lev < ref_levels; lev++)
        {
			
          mesh->UniformRefinement();
        
        }
        
        mesh->ReorientTetMesh();
/**		5.	Determine the list of true (i.e. conforming) essential boundary dofs.
			In this example, the boundary conditions are defined by marking all
			the boundary attributes from the mesh as essential (Dirichlet) and
			converting them to a list of true dofs.
			Array<int> ess_tdof_list;
	        if (mesh->bdr_attributes.Size())
            {
            Array<int> ess_bdr(mesh->bdr_attributes.Max());
            ess_bdr = 1;
            //fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
             }
			
**/
      
          Array<int>ess_bdr(mesh->attributes.Max());
	      ess_bdr=1;
		  ess_bdr[0]=1;
		  ess_bdr[1]=1;
		  //ess_bdr[2]=1;
		  cout<<"boundary attributes"<<"*****************"<<endl;
		  ess_bdr.Print();
		  cout<<"boundary attributes"<<"*****************"<<endl;
		  
      
   
/**    5.  Define the vector finite element spaces representing the 
           electric field E, the magnetic field H.c
**/   
   
		FiniteElementCollection *HDiv_Coll(new RT_FECollection(order, dim));
		
		FiniteElementCollection *HCurl_Coll(new ND_FECollection(order, dim));
		
	/// choose the Nedele FiniteElementSpace  Curl conforming finite elment space 
		
		FiniteElementSpace *NEDELEC_space(new FiniteElementSpace (mesh, HCurl_Coll));
		
	/// choose the Raviart Thomas FiniteElementSpace div conforning finite element space
		
		FiniteElementSpace *RAVIART_space(new FiniteElementSpace(mesh, HDiv_Coll));
	
/**    6. Define the BlockStructure of the problem, i.e. define the array of
          offsets for each variable. The last component of the Array is the sum
          of the dimensions of each block.
**/
      ///     number of variables + 1
      /// If numner of unknown is 4 the Array<int>block_offsets(5)
         Array<int> block_offsets(3);
         block_offsets[0] = 0;
         block_offsets[1] = NEDELEC_space->GetTrueVSize();
         block_offsets[2] = RAVIART_space->GetTrueVSize();
         block_offsets.PartialSum();
    
        std::cout << "***********************************************************\n";
        std::cout << "dim(Nedelec) = " << block_offsets[1] - block_offsets[0] << "\n";
        std::cout << "dim(Raviart) = " << block_offsets[2] - block_offsets[1] << "\n";
        std::cout << "dim(Nedelec+Raviart) = " << block_offsets.Last() << "\n";
        std::cout << "***********************************************************\n";
   // we tested for the double check .
   
        std::cout << "***********************************************************\n";
        std::cout << "Nedelec number of degrees of freedom. = "       << NEDELEC_space->GetNDofs()<< "\n";
        std::cout << "Raviart space number of degrees of freedom.= " << RAVIART_space->GetNDofs() << "\n";
        std::cout << "***********************************************************\n";
    
    
    
  /**   7. Set up the linear form b(.) which corresponds to exact solution Electric_Current
            the FEM linear system, which in this case is (J,phi_i) where J is
            given by the function Electric_Current and phi_i are the basis functions in the
             finite element fespace.
   **/
   
   // you can also find in the implicit solve 
   // we find the Current_density vetor to confirm our result here
   VectorFunctionCoefficient J(3, Electric_Current);
   
   LinearForm *Current_density=new LinearForm(NEDELEC_space);
   
   Current_density->AddDomainIntegrator(new VectorFEDomainLFIntegrator(J));
   
   Current_density->Assemble();
   // to print the Current_density vector
   //Current_density->Print();
   cout<<"size of vector Current_density="<<Current_density->Size()<<endl;
   
	
/**     8. Since E and H are integrate in time so we group them together in block vector EH,  
           with offsets given  by the block_offsetsarray. for detial see the block vector class in linalg
**/
        BlockVector EH(block_offsets);
    
   
    
/**     9. Define a solutions Vectors E (electric field )and H (magnetic field) as a Finite element Grid functions
           corresponding to Nedelec_space (finiete element space) and Raviart_space (finite element space) respectively.

**/   
        GridFunction E, H;
        
///      we can use both syntax to find the E and H
/**      E.MakeRef(NEDELEC_space, EH.GetBlock(0), 0);
        
         H.MakeRef(RAVIART_space, EH.GetBlock(1), 0);
         
**/
        E.MakeRef(NEDELEC_space, EH, block_offsets[0]);
        
       
        H.MakeRef(RAVIART_space, EH, block_offsets[1]);
        
     
        
/**    10. Set the initial conditions for E (electric field) and  H (magnetics field).
 *		   Initialize E (electric field) by projecting the Exact solution of Electric field.
		   Similarly initialze H (magnetic field ) by projecting thee Exact soluton of Magnetic field.
            and the boundary conditions on a beam-like mesh (see description above).
**/
        /**  VectorFunctionCoefficient electric_coeff(dim, ElectricField);
             E.ProjectCoefficient(electric_coeff);
       **/
        VectorFunctionCoefficient electric_coeff(3, ElectricField);
        E.ProjectCoefficient(electric_coeff);
       
      
       /// if we want to initialize with E=0.0 then uncomment the below line and comments the above line E.ProjectCoefficeint.
        //electric_coeff.SetTime(0.0);
        //E.Print();
         
         /// to check the initial value of E uncomment the below line E.Print()
         //E.Print();
         cout<<"size of E GridFunction is "<<E.Size()<<endl;
   
        /**  VectorFunctionCoefficient magnetic_coeff(dim, MagneticField);
              H.ProjectCoefficient(magnetic_coeff);
        **/
        VectorFunctionCoefficient magnetic_coeff(3, MagneticField);
        H.ProjectCoefficient(magnetic_coeff);
		
        
        /// if we want to initialize with H=0.0 then uncomment the below line and comments the above line H.ProjectCoefficeint.
        //magnetic_coeff.SetTime(0.0);
        /// to check the initial value of H uncomment the below line H.Print()
       // H.Print();
       cout<<"size of H GridFunctions is "<<H.Size()<<endl;
      
/**   11.  Initialize the Maxwell Constcutor and visulaization.
 * 
 
**/      ///	 	Initialize the maxwell constructor
		 
		Maxwell maxwell(*NEDELEC_space, *RAVIART_space, Permittivity, Conductivity, Permeability, dim, ess_bdr) ; 
		
		{
			
		ofstream mesh_ofs("ex26.mesh");
		mesh_ofs.precision(8);
		mesh->Print(mesh_ofs);
		
		ofstream E_ofs("sol_E26.gf");
		
		E_ofs.precision(8);
		
		E.Save(E_ofs);
		
		
		ofstream H_ofs("sol_H26.gf");
		
		H_ofs.precision(8);
		
		H.Save(H_ofs);
		
		
		
		
		}
		VisItDataCollection visit_dc("Example26, mesh");
		visit_dc.RegisterField("Electric intensities", &E);
		
		visit_dc.RegisterField("Magnetic field", &H);
		
		if(visit)
		{
			
			visit_dc.SetCycle(0);
			visit_dc.SetTime(0.0);
			visit_dc.Save();
		}
		
   
   
       socketstream e_sock, h_sock;
       if(visualization)
       {
	   
	   char vishost[]= "localhost";
	   
	   int visport= 19916;
	   
	   e_sock.open(vishost, visport);
	   
	   e_sock.precision(8);
	   
	   e_sock<<"solution\n"<<*mesh<<E<<"window_title 'Electric Field'" << endl;
	   
	   h_sock.open(vishost, visport);
	   
	   h_sock.precision(8);
	   
	   h_sock<<"solution\n"<<*mesh<<H<<"window_title 'Magnetic Field'" << endl;
       }
		
		
		
		
		ode_solver->Init(maxwell);
        
        double t = 0.0;
  
         
  
        bool last_step = false;
        
        for (int ti = 1; !last_step; ti++)
         {
		   
           if (t + dt >= t_final - dt/2)
             {
                 last_step = true;
             }

            ode_solver->Step(EH, t, dt);
      

      if (last_step || (ti % vis_steps) == 0)
      {
		  
         cout << "step " << ti << ", t = " << t <<endl;   
         
         if(visualization)
         {
			 e_sock<<"solution\n"<<*mesh<<E<<flush;
			
			 h_sock<<"solution\n"<<*mesh<<H<<flush;
			 
		 }
      if(visit)
      {
		  visit_dc.SetCycle(ti);
		  
		  visit_dc.SetTime(t);
		  
		  visit_dc.Save();
		 
	  }
      
     }
     
  } 
  
  
   
  
   /// 12. Grid functions E and H. Compute the L2 error norms.

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   //cout << "Geometry::NumGeom is the edges of tetrahedon  = " << Geometry::NumGeom << endl;
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double err_e  = E.ComputeL2Error(electric_coeff, irs);
  
   double norm_e = ComputeLpNorm(2., electric_coeff, *mesh, irs);
   double err_h  = H.ComputeL2Error(magnetic_coeff, irs);
   double norm_h = ComputeLpNorm(2., magnetic_coeff, *mesh, irs);

   std::cout << "|| E_h - E_ex ||  = " << err_e  << "\n";
   std::cout << "|| H_h - H_ex ||  = " << err_h  << "\n";
    
		if(ode_solver!=NULL)
		{
		cout<<"delete ode_solver"<<endl;
		delete ode_solver;
		}
		if(HDiv_Coll!=NULL)
		{
		cout<<"delete HDiv_Coll"<<endl;
		delete HDiv_Coll;
		}
		if(HCurl_Coll!=NULL)
		{
		cout<<"delete HCurl_Coll"<<endl;
		delete HCurl_Coll;
		}
		if(NEDELEC_space!=NULL)
		{
		cout<<"delete NEDELEC_space"<<endl;
		delete NEDELEC_space;
		}
		if(RAVIART_space!=NULL)
		{
		cout<<"delete RAVIART_space"<<endl;
		delete RAVIART_space;
		}
		
		if(mesh!=NULL)
		{
		cout<<"delete mesh"<<endl;
		delete mesh;
		}
		if(Current_density!=NULL)
		{
		cout<<"delete Current_density"<<endl;
		delete Current_density;
	}
		 
   return 0;
}
// *Nmat, *Rmat, *Conductmat, *MBmat;

Maxwell::Maxwell(FiniteElementSpace &nedelec, FiniteElementSpace &raviart,
                           double premittivi, double conductivi, double premabili, int mesh_dim, Array<int>&ess_bdr_arg)
   : TimeDependentOperator(nedelec.GetVSize()+raviart.GetVSize(), 0.0), Nedelec(nedelec), Raviart(raviart),
     N(NULL), R(NULL), Conduct(NULL), MB(NULL), 
     E_size(nedelec.GetVSize()), H_size(raviart.GetVSize()), Mesh_dim(mesh_dim),
     N_Solver(NULL), R_Solver(NULL),
     Nspmat(NULL), Rspmat(NULL), Conductspmat(NULL), MBspmat(NULL)
    
     
{
	
	ess_bdr.SetSize(ess_bdr_arg.Size());
	for(int i=0; i<ess_bdr_arg.Size(); i++)
	{
		ess_bdr[i]=ess_bdr_arg[i];
	}
	E0= new GridFunction(&Nedelec);
	
	H0=new  GridFunction(&Raviart);
	
	// declare the vector
	
	X0= new Vector;
	
	B0=new Vector;
	
	X1=new Vector;
	
	
	
	Nspmat= new SparseMatrix;
	
	Rspmat= new SparseMatrix;
	cout<<"Initialize of constructor Maxwell operator"<<endl;
	
	// Here we are setting Permittivity constant
	
	permittivity=premittivi;
	
	ConstantCoefficient Permittivity(permittivity);
	 
	
	// Here we shall find the mass Matrix Nmat 
	
	N= new BilinearForm(&Nedelec);
	
	
	N->AddDomainIntegrator(new VectorFEMassIntegrator(Permittivity));
	
	N->Assemble();
	
	N->Finalize();
	//N->Print();
	
	
	/// to find the height and width of the mass matrix N, below both lines
	cout << "\nHeight of the N Mass Martirx:="<<N->Height()<<endl;
	cout << "\nWidth of the N Mass Martirx:="<<N->Width()<<endl;

    
    conductivity=conductivi;
    
    ConstantCoefficient Conductivity(conductivity);
    
    
    Conduct=new BilinearForm(&Nedelec);
    
    Conduct->AddDomainIntegrator(new VectorFEMassIntegrator(Conductivity));
	
	Conduct->Assemble();
	
	Conduct->Finalize();
	
	/// to find the height and width of the conduct matrix, below both lines
	cout << "\nHeight of the Conductivity dependent (Sigma E) Martirx:="<<Conduct->Height()<<endl;
    cout << "\nWidth of the Conductitivity dependent (Sigma E) Martirx:="<<Conduct->Width()<<endl;
	
	
	
	// Set the Permeability const 
	
	permeability=premabili;
	
	ConstantCoefficient Permeability(permeability);
	
	// Current density 
		
		/*VectorFunctionCoefficient J(3, Electric_Current);
        
		Current_density=new LinearForm(&Nedelec);
   
		Current_density->AddDomainIntegrator(new VectorFEDomainLFIntegrator(J));
   
		Current_density->Assemble();*/
	
	// Here we shall find the mass Matrix R descritzed by Raviart thomas finite element method
	
	R= new BilinearForm(&Raviart);
	
	R->AddDomainIntegrator(new VectorFEMassIntegrator( Permeability));
	
	
	R->Assemble();
	
	R->Finalize();

	//cout << "\nHeight of the R permeability dependent Mass Martirx:="<<R->Height()<<endl;
    
    //cout << "\nWidth of the R permeability dependent Mass Martirx:="<<R->Width()<<endl;
	
	
	
	// now we shall use the Mixed Method Matix to solve <Curl u, V>E and <u Curl V>H
	
	MB= new MixedBilinearForm(&Nedelec, &Raviart);
	
	MB->AddDomainIntegrator(new VectorFECurlIntegrator);
	
	MB->Assemble();
	MB->Finalize();
	cout << "\nHeight of the Mixed (Curl E, H) Matrix=:"<<MB->Height()<<endl;
    
    cout << "\nWidth of the Mixed (CurE, H) Matrix =:"<<MB->Width()<<endl;
	
	
}

Maxwell::~Maxwell()
{
	if(Current_density!=NULL)
	{
	cout<<"delete Linear form Current_density"<<endl;
	delete Current_density;
	}
	if(N!=NULL)
	{
	cout<<"delete Bilinear Form N"<<endl;
	delete N;
	}
	if(R!=NULL)
	{
	cout<<"delete Bilinear Form R"<<endl;
	delete R;
	}
	if(Conduct!=NULL)
	{
	cout<<"delete Bilinear Form Conduct"<<endl;
	delete Conduct;
	}
	if(MB!=NULL)
	{
	cout<<"delete MixedBilinear Form MB"<<endl;
	delete MB;
	}
	if(N_Solver!=NULL)
	{
	cout<<"delete N_Solver to inverting the mass matrix N"<<endl;
	delete N_Solver;
	}
	if(R_Solver!=NULL)
	{
	cout<<"delete R_Solver to inverting the mass matrix R"<<endl;
	delete R_Solver;
	}
	if(E0!=NULL)
	{
	cout<<"delete GridFunction E0"<<endl;
	delete E0;
	}
	
	if(H0!=NULL)
	{
	cout<<"delete GridFunction H0"<<endl;
	delete H0;
	}
	// delete the vector
	
	if(X0!=NULL)
	{
	cout<<"delete Vector X0"<<endl;
	delete X0;
	}

	if(X1!=NULL)
	{
	cout<<"delete Vector X1"<<endl;
	delete X1;
	}
	if(B0!=NULL)
	{
	cout<<"delete Vector B0"<<endl;
	delete B0;
	}
	// delete of sparse Matrix 
	
	if (Nspmat!=NULL)
	{
	cout<<"delete Nspmat "<<endl;
	delete Nspmat;
	}
	if (Rspmat!=NULL)
	{
	cout<<"delete Rspmat "<<endl;
	delete Rspmat;
	}
	if (Conductspmat!=NULL)
	{
	cout<<"delete Conductspmat"<<endl;
	delete Conductspmat;
	}
	if (MBspmat!=NULL)
	{
	cout<<"delete MBspmat "<<endl;
	delete MBspmat;
	}
}
void ElectricField_BC(const Vector &x, Vector &E)
{
	E=0.0;
}
	
 void  ElectricField(const Vector & x, const double t, Vector & E)
{
	
	E(0) = sin(2*t-3*x(2));
    E(1) = sin(2*t-3*x(0));
    E(2) = sin(2*t-3*x(1));
   
   
   /*
   E[0] = 0.0;
   E[1] = 0.0;
   E[2] = 0.0;
   */
   
}
void MagneticField(const Vector & x, const double t, Vector & H)
{

	  H(0) = 0.4*sin(2*t-3*x(1));
      H(1) = 1.4*sin(2*t-3*x(2));
      H(2) = 0.4*sin(2*t-3*x(1));
      
     /*
      H[0] = 0.0;
	  H[1] = 0.0;
	  H[2] = 0.0;
	  */
	  
      
      
 
}

void Electric_Current(const Vector & x, const double t, Vector &J)
{
	    J(0)= 1.4*cos(2*t -x(2));
	
		J(1)= 1.4*cos(2*t -x(0));
		
		J(2)=1.4*cos(2*t  -x(1));
	
}   
