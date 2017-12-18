
 
//   void  ElectricField(const Vector & x, const double t, Vector & E)
//   {
//	
//	  E(0) = sin(2*t - 3*x(2));
//    E(1) = sin(2*t - 3*x(0));
//    E(2) = sin(2*t - 3*x(1)); 
//   }
//    void MagneticField(const Vector & x, const double t, Vector & B)
//   {
//      B(0) = 1.5*sin(2*t - 3*x(1));
//      B(1) = 1.5*sin(2*t - 3*x(2));
//      B(2) = 1.5*sin(2*t - 3*x(0));
//
//}
//    void Electric_Current(const Vector & x, const double t, Vector &J)
//    {
//	    J(0)= cos(2*t - 3*x(2));
//	    J(1)= cos(2*t - 3*x(0));
//		J(2)= cos(2*t - 3*x(1));
//    }   
//                        Maxwelll equations when exact solution is known
//                         we find the absolute error 
//                         
//                         || E_h - E_ex ||  = C * t
//                         || B_h - B_ex ||  = C * t   C is a independent of t 
// Description:  This example code solves a simple 3D  Maxwell's problem
//               
//                                 permittivity * dE/dt =  Curl B/mu + Electric_Current
//                                                dB/dt = - Curl E 
//                                permeability =mu     1/mu = muInv
// 
//                   Abouve is the exacet soluiotn for electric feilds and magnetics flux and current density
//                  Material properties  
//                  permitttivity=2   pereability= mu= 1.5  
//                  Muinv= 1/1.5 = 0.66666667
// 
//         
//                          
//               permittivity, permability and muInv are constant and in command line  parameter
//               with essential boundary condition nxE = given tangential field
// 
// 				 Boundary condition are applied only to E but 
//               boundary condition is not appplied  to B.
//               Electric_Current is given electric field at the end of this file.
//               Here, we use a given exact solution (E,B) and compute the
//               corresponding l.h.s. (E,B).  We discretize with Nedelec 
//               finite elements (electric field E) and Raviart-Thomas finite  
//               element (magnetic field B).
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
#include "ex33_solver.hpp"

using namespace std;
using namespace mfem;

using namespace mfem::electromagnetics;


/// exact solution of electrix field

	void  ElectricField(const Vector & x, const double t, Vector & E);

/// The exact solution of Magnectic Field

	void MagneticField(const Vector & x,const double t, Vector & B);

/// The electric current density is given

	void Electric_Current(const Vector & x, const double t, Vector &J);
	
/// The boundary condition for E

    void ElectricField_BC(const Vector &x,  Vector &E);

int main(int argc, char *argv[])
{
   /// 1. Parse command-line options
  
     const char *mesh_file = "../../data/beam-tet.mesh";
    //const char *mesh_file = "../../data/escher.mesh";
     //const char *mesh_file = "../../data/fichera.mesh";
   int ref_levels = 2;
   int order = 2;
   int ode_solver_type = 1;
 
   double t_final = 1;
   // double dt = 0.0625; //t=1/16
   double dt= 0.03125; // t=32
   
   // For small tine step 
//double t_final = 2.5e-7;
// double dt= 1e-9;; // t=250
 //double dt= 2e-9;; // t=125
 
  
   double Permittivity=2;
   double Conductivity=0.0;
   double Permeability=1.5;//1.5
   double MuInv=1/Permeability;
   bool visualization = true;
   bool visit=false;
   int vis_steps = 1;
   OptionsParser args(argc, argv);
  args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&ref_levels, "-r", "--refine",
                  "Number of times to refine the mesh uniformly.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Backward Euler, 2 - SDIRK2, 3 - SDIRK3,\n\t"
                  "\t   11 - Forward Euler, 12 - RK2, 13 - RK3 SSP, 14 - RK4.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step.");
   args.AddOption(&Permittivity, "-v", "--Permittivity",
                  "permittivity coefficient.");
   args.AddOption(&Permeability, "-per", "--Permeability",
                  "permeability coefficient.");
   
   args.AddOption(&MuInv, "-muinv", "--permeabilityInverse ",
                  "permeabilityinv coefficient.");
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
   
/**    5.  Define the vector finite element spaces representing the 
           electric field E, the magnetic field B.
**/   
   
		FiniteElementCollection *HDiv_Coll(new RT_FECollection(order, dim));
		
		FiniteElementCollection *HCurl_Coll(new ND_FECollection(order, dim));
		
	/// choose the Nedele FiniteElementSpace  Curl conforming finite elment space 
		
		FiniteElementSpace *NEDELEC_space(new FiniteElementSpace (mesh, HCurl_Coll));
		
	/// choose the Raviart Thomas FiniteElementSpace div conforning finite element space
		
		FiniteElementSpace *RAVIART_space(new FiniteElementSpace(mesh, HDiv_Coll));

/**		5.	Determine the list of true (i.e. conforming) essential boundary dofs.
			In this example, the boundary conditions are defined by marking all
			the boundary attributes from the mesh as essential (Dirichlet) and
			converting them to a list of true dofs.
**/
		
		Array<int>ess_tdof_list;
		 Array<int>ess_bdr(mesh->bdr_attributes.Max());
	      ess_bdr=1;
		  ess_bdr[0]=1;
		  ess_bdr[1]=1;
		 
		  if(mesh->bdr_attributes.Size())
	{
		Array<int>ess_bdr(mesh->bdr_attributes.Max());
		ess_bdr=1;
		NEDELEC_space->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
		cout<<"ess_tdof_list"<<endl;
		
	cout<<"********************************************"<<endl;
	}
	
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
   VectorFunctionCoefficient jcoeff(3, Electric_Current);
   
   LinearForm *Current_density=new LinearForm(NEDELEC_space);
   
   Current_density->AddDomainIntegrator(new VectorFEDomainLFIntegrator(jcoeff));
   
   Current_density->Assemble();
   
   
	
/**     7. Since E and B are integrate in time so we group them together in block vector EB,  
           with offsets given  by the block_offsetsarray.
**/
        BlockVector EB(block_offsets);
    
   
    
/**   8. Define a solutions Vectors E (electric field )and B (magnetic field) as a Finite element Grid functions
          corresponding to Nedelec_space (finiete element space) and Raviart_space (finite element space) respectively.

**/    
        GridFunction E, B;
        
    
///      we can use both syntax to find the E and B
/**      E.MakeRef(NEDELEC_space, EB.GetBlock(0), 0);
        
         B.MakeRef(RAVIART_space, EB.GetBlock(1), 0);
         
**/
        E.MakeRef(NEDELEC_space, EB, block_offsets[0]);
        
       
        B.MakeRef(RAVIART_space, EB, block_offsets[1]);
        
     
        
/** 9.   Set the initial conditions for E (electric field) and  B (magnetics field).
 *		 Initialize E (electric field) by projecting the Exact solution of Electric field.
		 Similarly initialze B (magnetic field ) by projecting thee Exact soluton of Magnetic field.
         and the boundary conditions on a beam-like mesh (see description above).
**/
        VectorFunctionCoefficient electric_coeff(3, ElectricField);
        E.ProjectCoefficient(electric_coeff);
       
      
       /// if we want to initialize with E=0.0 then uncomment the below line and comments the above line E.ProjectCoefficeint.
        //electric_coeff.SetTime(0.0);
      
  
        VectorFunctionCoefficient magnetic_coeff(3, MagneticField);
        B.ProjectCoefficient(magnetic_coeff);
      
       
      
/**   10.  Initialize the Maxwell Constrcutor and visulaization.
 * 
 
**/
///	 	Initialize the maxwell constructor
		 
		Maxwell maxwell(*NEDELEC_space, *RAVIART_space, Permittivity, Conductivity, Permeability, dim, ess_bdr, &Electric_Current) ; 
		maxwell.Init( electric_coeff, magnetic_coeff);
		{
			
		ofstream mesh_ofs("ex33.mesh");
		mesh_ofs.precision(8);
		mesh->Print(mesh_ofs);
		
		ofstream E_ofs("sol_E33.gf");
		
		E_ofs.precision(8);
		
		E.Save(E_ofs);
		
		
		ofstream B_ofs("sol_B33.gf");
		
		B_ofs.precision(8);
		
		B.Save(B_ofs);
		
		
		
		
		}
		VisItDataCollection visit_dc("Example33, mesh");
		visit_dc.RegisterField("Electric intensities", &E);
		
		visit_dc.RegisterField("Magnetic field", &B);
		
		if(visit)
		{
			
			visit_dc.SetCycle(0);
			visit_dc.SetTime(0.0);
			visit_dc.Save();
		}
		
   
   
		socketstream e_sock, b_sock;
		if(visualization)
		{
	   
	    char vishost[]= "localhost";
	   
	    int visport= 19916;
	   
	    e_sock.open(vishost, visport);
	   
	    e_sock.precision(8);
	   
	    e_sock<<"solution\n"<<*mesh<<E<<"window_title 'Electric Field'" << endl;
	   
	    b_sock.open(vishost, visport);
	   
	    b_sock.precision(8);
	   
	    b_sock<<"solution\n"<<*mesh<<B<<"window_title 'Magnetic Field'" << endl;
        
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

            ode_solver->Step(EB, t, dt);
            
      

        if (last_step || (ti % vis_steps) == 0)
        {
		  
         cout << "step " << ti << ", t = " << t <<endl;
         if(visualization)
         {
			 e_sock<<"solution\n"<<*mesh<<E<<flush;
			
			 b_sock<<"solution\n"<<*mesh<<B<<flush;
			 
		 }
        if(visit)
        {
		  visit_dc.SetCycle(ti);
		  
		  visit_dc.SetTime(t);
		  
		  visit_dc.Save();
		 
	    }
      
     }
     
  } 
  
  
   
  
   /// 12.GridFunction E and B. Compute the L2 error norms.

   int order_quad = max(2, 2*order+1);
   const IntegrationRule *irs[Geometry::NumGeom];
   
   for (int i=0; i < Geometry::NumGeom; ++i)
   {
      irs[i] = &(IntRules.Get(i, order_quad));
   }

   double err_e  = E.ComputeL2Error(electric_coeff, irs);
  
   double norm_e = ComputeLpNorm(2., electric_coeff, *mesh, irs);
   double err_b  = B.ComputeL2Error(magnetic_coeff, irs);
   double norm_b = ComputeLpNorm(2., magnetic_coeff, *mesh, irs);
   
    std::cout << "|| E_h - E_ex ||  = " << err_e  << "\n";
    std::cout << "|| B_h - B_ex ||  = " << err_b  << "\n";

  
    
		if(ode_solver!=NULL)
		{
	
		delete ode_solver;
		}
		if(HDiv_Coll!=NULL)
		{
	
		delete HDiv_Coll;
		}
		if(HCurl_Coll!=NULL)
		{
		
		delete HCurl_Coll;
		}
		if(NEDELEC_space!=NULL)
		{
		
		delete NEDELEC_space;
		}
		if(RAVIART_space!=NULL)
		{
		
		delete RAVIART_space;
		}
		
		if(mesh!=NULL)
		{
	
		delete mesh;
		}
		if(Current_density!=NULL)
		{
		
		delete Current_density;
	}
		 
   return 0;
}

void ElectricField_BC(const Vector &x, Vector &E)
{
	E=0.0;
}
	
void  ElectricField(const Vector & x, const double t, Vector & E)
{
	
	E(0) = sin(2*t - 3*x(2));
    E(1) = sin(2*t - 3*x(0));
    E(2) = sin(2*t - 3*x(1));
  
   
}
void MagneticField(const Vector & x, const double t, Vector & B)
{

      B(0) = 1.5*sin(2*t - 3*x(1));
      B(1) = 1.5*sin(2*t - 3*x(2));
      B(2) = 1.5*sin(2*t - 3*x(0));

}
void Electric_Current(const Vector & x, const double t, Vector &J)
{
	    J(0)= cos(2*t - 3*x(2));
	    J(1)= cos(2*t - 3*x(0));
		J(2)= cos(2*t - 3*x(1));
}   
