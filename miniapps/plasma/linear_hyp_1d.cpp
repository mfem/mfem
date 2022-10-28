//                       MFEM Example 18 - Parallel Version
//
// Compile with: make ex18
//
// Sample runs:
//
//       mpirun -np 4 ex18p -p 1 -rs 2 -rp 1 -o 1 -s 3
//       mpirun -np 4 ex18p -p 1 -rs 1 -rp 1 -o 3 -s 4
//       mpirun -np 4 ex18p -p 1 -rs 1 -rp 1 -o 5 -s 6
//       mpirun -np 4 ex18p -p 2 -rs 1 -rp 1 -o 1 -s 3
//       mpirun -np 4 ex18p -p 2 -rs 1 -rp 1 -o 3 -s 3
//
// Description:  This example code solves the compressible Euler system of
//               equations, a model nonlinear hyperbolic PDE, with a
//               discontinuous Galerkin (DG) formulation.
//
//               Specifically, it solves for an exact solution of the equations
//               whereby a vortex is transported by a uniform flow. Since all
//               boundaries are periodic here, the method's accuracy can be
//               assessed by measuring the difference between the solution and
//               the initial condition at a later time when the vortex returns
//               to its initial location.
//
//               Note that as the order of the spatial discretization increases,
//               the timestep must become smaller. This example currently uses a
//               simple estimate derived by Cockburn and Shu for the 1D RKDG
//               method. An additional factor can be tuned by passing the --cfl
//               (or -c shorter) flag.
//
//               The example demonstrates user-defined bilinear and nonlinear
//               form integrators for systems of equations that are defined with
//               block vectors, and how these are used with an operator for
//               explicit time integrators. In this case the system also
//               involves an external approximate Riemann solver for the DG
//               interface flux. It also demonstrates how to use GLVis for
//               in-situ visualization of vector grid functions.
//
//               We recommend viewing examples 9, 14 and 17 before viewing this
//               example.

#include "mfem.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;
using namespace mfem;

// Choice for the problem setup. See InitialCondition in ex18.hpp.
int problem;

// Equation constant parameters.
const int num_equation = 2;
//const double specific_heat_ratio = 1.4;
//const double gas_constant = 1.0;

// Maximum characteristic speed (updated by integrators)
double max_char_speed;

// Classes FE_Evolution, RiemannSolver, and FaceIntegrator
// shared between the serial and parallel version of the example.
//#include "ex18.hpp"

// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form.
class FE_Evolution : public TimeDependentOperator
{
private:
   const int dim;

   FiniteElementSpace &vfes;
   ParBlockNonlinearForm &A;
   DenseTensor Me_inv;

   mutable Vector state;
   mutable DenseMatrix f;
   mutable DenseTensor flux;
   mutable Vector z;

   void GetFlux(const DenseMatrix &state_, DenseTensor &flux_) const;

public:
   FE_Evolution(FiniteElementSpace &vfes_,
                ParBlockNonlinearForm &A_);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~FE_Evolution() { }
};

// Implements a simple Rusanov flux
class RiemannSolver
{
private:
   Vector flux1;
   Vector flux2;

public:
   RiemannSolver();
   double Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux);
};

// Interior face term: <F.n(u),[w]>
class FaceIntegrator : public NonlinearFormIntegrator
{
private:
   RiemannSolver rsolver;
   Vector shape1;
   Vector shape2;
   Vector funval1;
   Vector funval2;
   Vector nor;
   Vector fluxN;

public:
   FaceIntegrator(RiemannSolver &rsolver_, const int dim);

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
};

class MyParBlockNonlinearForm : public ParBlockNonlinearForm
{
protected:
   mutable Array<ParGridFunction*> X, Y;
  
public:
  MyParBlockNonlinearForm(Array<ParFiniteElementSpace *> &pf);

   void Mult(const Vector &x, Vector &y) const;
};

class LinearHyp1DIntegrator : public BlockNonlinearFormIntegrator
{
public:
  LinearHyp1DIntegrator() {}

  void AssembleElementVector(const Array<const FiniteElement *> &el,
			     ElementTransformation &Tr,
			     const Array<const Vector *> &elfun,
			     const Array<Vector *> &elvec);

  void AssembleFaceVector(const Array<const FiniteElement *> &el1,
			  const Array<const FiniteElement *> &el2,
			  FaceElementTransformations &Tr,
			  const Array<const Vector *> &elfun,
			  const Array<Vector *> &elvect);
};

//class LinearHypFaceIntegrator : public BlockNonlinearformIntegrator
//{};

// Implementation of class FE_Evolution
FE_Evolution::FE_Evolution(FiniteElementSpace &vfes_,
			   ParBlockNonlinearForm &A_)
   : TimeDependentOperator(A_.Height()),
     dim(vfes_.GetFE(0)->GetDim()),
     vfes(vfes_),
     A(A_),
     Me_inv(vfes.GetFE(0)->GetDof(), vfes.GetFE(0)->GetDof(), vfes.GetNE()),
     state(num_equation),
     // f(num_equation, dim),
     // flux(A_.ParFESpace(0)->GetNDofs(), dim, num_equation),
     z(A.Height())
{
   // Standard local assembly and inversion for energy mass matrices.
   const int dof = vfes.GetFE(0)->GetDof();
   DenseMatrix Me(dof);
   DenseMatrixInverse inv(&Me);
   MassIntegrator mi;
   for (int i = 0; i < vfes.GetNE(); i++)
   {
      mi.AssembleElementMatrix(*vfes.GetFE(i), *vfes.GetElementTransformation(i), Me);
      inv.Factor();
      inv.GetInverseMatrix(Me_inv(i));
   }
}

void FE_Evolution::Mult(const Vector &x, Vector &y) const
{
   // 0. Reset wavespeed computation before operator application.
   max_char_speed = 0.;

   // 1. Create the vector z with the face terms -<F.n(u), [w]>.
   A.Mult(x, z);

   // 2. Add the element terms.
   // i.  computing the flux approximately as a grid function by interpolating
   //     at the solution nodes.
   // ii. multiplying this grid function by a (constant) mixed bilinear form for
   //     each of the num_equation, computing (F(u), grad(w)) for each equation.
   /*
   DenseMatrix xmat(x.GetData(), vfes.GetNDofs(), num_equation);
   GetFlux(xmat, flux);

   for (int k = 0; k < num_equation; k++)
   {
      Vector fk(flux(k).GetData(), dim * vfes.GetNDofs());
      Vector zk(z.GetData() + k * vfes.GetNDofs(), vfes.GetNDofs());
      Aflux.AddMult(fk, zk);
   }
   */
   // 3. Multiply element-wise by the inverse mass matrices.
   Vector zval;
   Array<int> vdofs;
   const int dof = vfes.GetFE(0)->GetDof();
   DenseMatrix zmat, ymat(dof, num_equation);

   for (int i = 0; i < vfes.GetNE(); i++)
   {
      // Return the vdofs ordered byNODES
      vfes.GetElementVDofs(i, vdofs);
      z.GetSubVector(vdofs, zval);
      zmat.UseExternalData(zval.GetData(), dof, num_equation);
      mfem::Mult(Me_inv(i), zmat, ymat);
      y.SetSubVector(vdofs, ymat.GetData());
   }
}

// Physicality check (at end)
//bool StateIsPhysical(const Vector &state, const int dim);

// Pressure (EOS) computation
/*
inline double ComputePressure(const Vector &state, int dim)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   double den_vel2 = 0;
   for (int d = 0; d < dim; d++) { den_vel2 += den_vel(d) * den_vel(d); }
   den_vel2 /= den;

   return (specific_heat_ratio - 1.0) * (den_energy - 0.5 * den_vel2);
}
*/

// Compute the vector flux F(u)
void ComputeFlux(const Vector &state, int dim, DenseMatrix &flux)
{
   const double n = state(0);
   const double q = state(1);

   flux(0, 0) = q;
   flux(1, 0) = n;

   for (int d = 1; d < dim; d++)
   {
      flux(0, d) = 0.0;
      flux(1, d) = 0.0;
   }
}

// Compute the scalar F(u).n
void ComputeFluxDotN(const Vector &state, const Vector &nor,
                     Vector &fluxN)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();
   const double n = state(0);
   const double q = state(1);

   Vector b(dim); b = 0.0; b(0) = 1.0;
   
   double bN = 0;
   for (int d = 0; d < dim; d++) { bN += b(d) * nor(d); }

   fluxN(0) = q * bN;
   fluxN(1) = n * bN;
}

// Compute the maximum characteristic speed.
inline double ComputeMaxCharSpeed(const Vector &state, const int dim)
{
   return 1.0;
}

// Compute the flux at solution nodes.
void FE_Evolution::GetFlux(const DenseMatrix &x_, DenseTensor &flux_) const
{
   const int flux_dof = flux_.SizeI();
   const int flux_dim = flux_.SizeJ();

   for (int i = 0; i < flux_dof; i++)
   {
      for (int k = 0; k < num_equation; k++) { state(k) = x_(i, k); }
      ComputeFlux(state, flux_dim, f);

      for (int d = 0; d < flux_dim; d++)
      {
         for (int k = 0; k < num_equation; k++)
         {
            flux_(i, d, k) = f(k, d);
         }
      }

      // Update max char speed
      const double mcs = ComputeMaxCharSpeed(state, flux_dim);
      if (mcs > max_char_speed) { max_char_speed = mcs; }
   }
}

// Implementation of class RiemannSolver
RiemannSolver::RiemannSolver() :
   flux1(num_equation),
   flux2(num_equation) { }

double RiemannSolver::Eval(const Vector &state1, const Vector &state2,
                           const Vector &nor, Vector &flux)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();

   // MFEM_ASSERT(StateIsPhysical(state1, dim), "");
   // MFEM_ASSERT(StateIsPhysical(state2, dim), "");

   const double maxE1 = ComputeMaxCharSpeed(state1, dim);
   const double maxE2 = ComputeMaxCharSpeed(state2, dim);

   const double maxE = max(maxE1, maxE2);

   ComputeFluxDotN(state1, nor, flux1);
   ComputeFluxDotN(state2, nor, flux2);

   double normag = 0;
   for (int i = 0; i < dim; i++)
   {
      normag += nor(i) * nor(i);
   }
   normag = sqrt(normag);

   for (int i = 0; i < num_equation; i++)
   {
      flux(i) = 0.5 * (flux1(i) + flux2(i))
                - 0.5 * maxE * (state2(i) - state1(i)) * normag;
   }

   return maxE;
}

// Implementation of class FaceIntegrator
FaceIntegrator::FaceIntegrator(RiemannSolver &rsolver_, const int dim) :
   rsolver(rsolver_),
   funval1(num_equation),
   funval2(num_equation),
   nor(dim),
   fluxN(num_equation) { }

void FaceIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Tr,
                                        const Vector &elfun, Vector &elvect)
{
   int dim = el1.GetDim();
  
   // Compute the term <F.n(u),[w]> on the interior faces.
   const int dof1 = el1.GetDof();
   const int dof2 = el2.GetDof();

   shape1.SetSize(dof1);
   shape2.SetSize(dof2);

   elvect.SetSize((dof1 + dof2) * num_equation);
   elvect = 0.0;

   DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equation);
   DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equation, dof2,
                          num_equation);

   DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equation);
   DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equation, dof2,
                           num_equation);

   // Integration order calculation from DGTraceIntegrator
   int intorder;
   if (Tr.Elem2No >= 0)
      intorder = (min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) +
                  2*max(el1.GetOrder(), el2.GetOrder()));
   else
   {
      intorder = Tr.Elem1->OrderW() + 2*el1.GetOrder();
   }
   if (el1.Space() == FunctionSpace::Pk)
   {
      intorder++;
   }
   const IntegrationRule *ir = &IntRules.Get(Tr.GetGeometryType(), intorder);

   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip); // set face and element int. points

      const IntegrationPoint &eip1 = Tr.GetElement1IntPoint();
      
      // Calculate basis functions on both elements at the face
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
      el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1, funval1);
      elfun2_mat.MultTranspose(shape2, funval2);

      // Get the normal vector and the flux on the face
      if (dim == 1)
      {
         nor(0) = 2*eip1.x - 1.0;
      }
      else
      {
	CalcOrtho(Tr.Jacobian(), nor);
      }
      const double mcs = rsolver.Eval(funval1, funval2, nor, fluxN);

      // Update max char speed
      if (mcs > max_char_speed) { max_char_speed = mcs; }

      fluxN *= ip.weight;
      for (int k = 0; k < num_equation; k++)
      {
         for (int s = 0; s < dof1; s++)
         {
            elvect1_mat(s, k) -= fluxN(k) * shape1(s);
         }
         for (int s = 0; s < dof2; s++)
         {
            elvect2_mat(s, k) += fluxN(k) * shape2(s);
         }
      }
   }
}

MyParBlockNonlinearForm::MyParBlockNonlinearForm(Array<ParFiniteElementSpace *> &pf)
  : ParBlockNonlinearForm(pf), X(pf.Size()), Y(pf.Size())
{
  for (int s=0; s<pf.Size(); s++)
    {
      X[s] = new ParGridFunction;
      Y[s] = new ParGridFunction;

      X[s]->MakeRef(pf[s], NULL);
      Y[s]->MakeRef(pf[s], NULL);
    }
}

void MyParBlockNonlinearForm::Mult(const Vector &x, Vector &y) const
{
   // xs_true is not modified, so const_cast is okay
   xs_true.Update(const_cast<Vector &>(x), block_trueOffsets);
   ys_true.Update(y, block_trueOffsets);
   xs.Update(block_offsets);
   ys.Update(block_offsets);

   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->Mult(
         xs_true.GetBlock(s), xs.GetBlock(s));
   }

   BlockNonlinearForm::MultBlocked(xs, ys);

   if (fnfi.Size() > 0)
   {
      // MFEM_ABORT("TODO: assemble contributions from shared face terms");

      // MFEM_VERIFY(!NonlinearForm::ext, "Not implemented (extensions + faces");
     cout << 0 << endl;
      // Terms over shared interior faces in parallel.
      const ParFiniteElementSpace *pfes = ParFESpace(0);
      ParMesh *pmesh = pfes->GetParMesh();
      FaceElementTransformations *tr;
      //const FiniteElement *fe1, *fe2;
      Array<const FiniteElement*> fe1(fes.Size());
      Array<const FiniteElement*> fe2(fes.Size());
      
      Array<Array<int> *> vdofs1(fes.Size());
      Array<Array<int> *> vdofs2(fes.Size());
      Array<Vector*> el_x(fes.Size()), el_y(fes.Size());
     cout << 1 << endl;
      for (int i=0; i<fes.Size(); ++i)
      {
	el_x[i] = new Vector();
	el_y[i] = new Vector();
	vdofs1[i] = new Array<int>;
	vdofs2[i] = new Array<int>;
      }
     cout << 2 << endl;
      aux1.HostReadWrite();
           cout << 3 << endl;
      for (int i=0; i<fes.Size(); ++i)
      {
	cout << X[i] << endl;
	X[i]->MakeRef(aux1.GetBlock(i), 0); // aux1 contains P.x
           cout << "3a" << endl;
	   X[i]->ExchangeFaceNbrData();
           cout << "3b" << endl;
	Y[i]->MakeRef(aux2.GetBlock(i), 0); // aux2 contains P.y
      }
      cout << 4 << endl;
      const int n_shared_faces = pmesh->GetNSharedFaces();
      for (int i = 0; i < n_shared_faces; i++)
      {
	 tr = pmesh->GetSharedFaceTransformations(i, true);
	 int Elem2NbrNo = tr->Elem2No - pmesh->GetNE();

	 for (int s=0; s<fes.Size(); s++)
	 {
 	    fe1[s] = ParFESpace(s)->GetFE(tr->Elem1No);
	    fe2[s] = ParFESpace(s)->GetFaceNbrFE(Elem2NbrNo);

	    ParFESpace(s)->GetElementVDofs(tr->Elem1No, *vdofs1[s]);
	    ParFESpace(s)->GetFaceNbrElementVDofs(Elem2NbrNo, *vdofs2[s]);

	    el_x[s]->SetSize(vdofs1[s]->Size() + vdofs2[s]->Size());
	    // el_y[s]->SetSize(vdofs1[s]->Size() + vdofs2[s]->Size());

	    X[s]->GetSubVector(*vdofs1[s], el_x[s]->GetData());
	    X[s]->FaceNbrData().GetSubVector(*vdofs2[s],
					     el_x[s]->GetData() + vdofs1[s]->Size());
	 }

	 for (int k = 0; k < fnfi.Size(); k++)
	 {
	    fnfi[k]->AssembleFaceVector(fe1, fe2, *tr, el_x, el_y);
	    // aux2.AddElementVector(vdofs1, el_y.GetData());
	    for (int s=0; s<fes.Size(); s++)
	    {
	      Y[s]->AddElementVector(*vdofs1[s], el_y[s]->GetData());
	    }
	 }
      }
   }

   for (int s=0; s<fes.Size(); ++s)
   {
      fes[s]->GetProlongationMatrix()->MultTranspose(
         ys.GetBlock(s), ys_true.GetBlock(s));

      ys_true.GetBlock(s).SetSubVector(*ess_tdofs[s], 0.0);
   }

   ys_true.SyncFromBlocks();
   y.SyncMemory(ys_true);
}

void LinearHyp1DIntegrator::AssembleElementVector(
			    const Array<const FiniteElement *> &el,
			    ElementTransformation &Tr,
			    const Array<const Vector *> &elfun,
			    const Array<Vector *> &elvec)
{

}

void LinearHyp1DIntegrator::AssembleFaceVector(
			    const Array<const FiniteElement *> &el1,
			    const Array<const FiniteElement *> &el2,
			    FaceElementTransformations &Tr,
			    const Array<const Vector *> &elfun,
			    const Array<Vector *> &elvect)
{

}

// Check that the state is physical - enabled in debug mode
/*
bool StateIsPhysical(const Vector &state, const int dim)
{
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, dim);
   const double den_energy = state(1 + dim);

   if (den < 0)
   {
      cout << "Negative density: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }
   if (den_energy <= 0)
   {
      cout << "Negative energy: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }

   double den_vel2 = 0;
   for (int i = 0; i < dim; i++) { den_vel2 += den_vel(i) * den_vel(i); }
   den_vel2 /= den;

   const double pres = (specific_heat_ratio - 1.0) * (den_energy - 0.5 * den_vel2);

   if (pres <= 0)
   {
      cout << "Negative pressure: " << pres << ", state: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }
   return true;
}
*/
// Initial condition
void InitialCondition(const Vector &x, Vector &y)
{
   MFEM_ASSERT(x.Size() >= 1, "");
   MFEM_ASSERT(y.Size() >= 2, "");

   const double xc = 0.5;

   double px = x(0) - xc;
   
   y(0) = 1.0;
   y(1) = exp(-50.0 * px * px);
}


int main(int argc, char *argv[])
{
   // 1. Initialize MPI and HYPRE.
   Mpi::Init(argc, argv);
   Hypre::Init();

   // 2. Parse command-line options.
   problem = 1;
   const char *mesh_file = "../../data/periodic-segment.mesh";
   int ser_ref_levels = 0;
   int par_ref_levels = 1;
   int order = 3;
   int ode_solver_type = 4;
   double t_final = 2.0;
   double dt = -0.01;
   double cfl = 0.3;
   bool visualization = true;
   int vis_steps = 50;

   int precision = 8;
   cout.precision(precision);

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&problem, "-p", "--problem",
                  "Problem setup to use. See options in velocity_function().");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly before parallel"
                  " partitioning, -1 for auto.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly after parallel"
                  " partitioning.");
   args.AddOption(&order, "-o", "--order",
                  "Order (degree) of the finite elements.");
   args.AddOption(&ode_solver_type, "-s", "--ode-solver",
                  "ODE solver: 1 - Forward Euler,\n\t"
                  "            2 - RK2 SSP, 3 - RK3 SSP, 4 - RK4, 6 - RK6.");
   args.AddOption(&t_final, "-tf", "--t-final",
                  "Final time; start time is 0.");
   args.AddOption(&dt, "-dt", "--time-step",
                  "Time step. Positive number skips CFL timestep calculation.");
   args.AddOption(&cfl, "-c", "--cfl-number",
                  "CFL number for timestep calculation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&vis_steps, "-vs", "--visualization-steps",
                  "Visualize every n-th timestep.");

   args.Parse();
   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(cout); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(cout); }

   // 3. Read the mesh from the given mesh file. This example requires a 2D
   //    periodic mesh, such as ../data/periodic-square.mesh.
   Mesh mesh(mesh_file, 1, 1);
   const int dim = mesh.Dimension();

   // MFEM_ASSERT(dim == 2, "Need a two-dimensional mesh for the problem definition");

   // 4. Define the ODE solver used for time integration. Several explicit
   //    Runge-Kutta methods are available.
   ODESolver *ode_solver = NULL;
   switch (ode_solver_type)
   {
      case 1: ode_solver = new ForwardEulerSolver; break;
      case 2: ode_solver = new RK2Solver(1.0); break;
      case 3: ode_solver = new RK3SSPSolver; break;
      case 4: ode_solver = new RK4Solver; break;
      case 6: ode_solver = new RK6Solver; break;
      default:
         if (Mpi::Root())
         {
            cout << "Unknown ODE solver type: " << ode_solver_type << '\n';
         }
         return 3;
   }

   // 5. Refine the mesh in serial to increase the resolution. In this example
   //    we do 'ser_ref_levels' of uniform refinement, where 'ser_ref_levels' is
   //    a command-line parameter.
   for (int lev = 0; lev < ser_ref_levels; lev++)
   {
      mesh.UniformRefinement();
   }

   // 6. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   ParMesh pmesh(MPI_COMM_WORLD, mesh);
   mesh.Clear();
   for (int lev = 0; lev < par_ref_levels; lev++)
   {
      pmesh.UniformRefinement();
   }

   // 7. Define the discontinuous DG finite element space of the given
   //    polynomial order on the refined mesh.
   DG_FECollection fec(order, dim);
   // Finite element space for a scalar (thermodynamic quantity)
   ParFiniteElementSpace fes(&pmesh, &fec);
   // Finite element space for a mesh-dim vector quantity (momentum)
   // ParFiniteElementSpace dfes(&pmesh, &fec, dim, Ordering::byNODES);
   // Finite element space for all variables together (total thermodynamic state)
   ParFiniteElementSpace vfes(&pmesh, &fec, num_equation, Ordering::byNODES);

   Array<ParFiniteElementSpace*> afes(num_equation);
   afes = &fes;
   
   // This example depends on this ordering of the space.
   MFEM_ASSERT(fes.GetOrdering() == Ordering::byNODES, "");

   HYPRE_BigInt glob_size = vfes.GlobalTrueVSize();
   if (Mpi::Root())
   {
      cout << "Number of unknowns: " << glob_size << endl;
   }

   // 8. Define the initial conditions, save the corresponding mesh and grid
   //    functions to a file. This can be opened with GLVis with the -gc option.

   // The solution u has components {density, x-momentum, y-momentum, energy}.
   // These are stored contiguously in the BlockVector u_block.
   Array<int> offsets(num_equation + 1);
   for (int k = 0; k <= num_equation; k++) { offsets[k] = k * fes.GetNDofs(); }
   BlockVector u_block(offsets);

   // Momentum grid function on dfes for visualization.
   ParGridFunction n(&fes, u_block.GetData() + offsets[0]);
   ParGridFunction q(&fes, u_block.GetData() + offsets[1]);

   // Initialize the state.
   VectorFunctionCoefficient u0(num_equation, InitialCondition);
   ParGridFunction sol(&vfes, u_block.GetData());
   sol.ProjectCoefficient(u0);

   // Output the initial solution.
   {
      ostringstream mesh_name;
      mesh_name << "lin-hyp-mesh." << setfill('0')
                << setw(6) << Mpi::WorldRank();
      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(precision);
      mesh_ofs << pmesh;

      for (int k = 0; k < num_equation; k++)
      {
         ParGridFunction uk(&fes, u_block.GetBlock(k));
         ostringstream sol_name;
         sol_name << "lin-hyp-" << k << "-init."
                  << setfill('0') << setw(6) << Mpi::WorldRank();
         ofstream sol_ofs(sol_name.str().c_str());
         sol_ofs.precision(precision);
         sol_ofs << uk;
      }
   }

   // 9. Set up the nonlinear form corresponding to the DG discretization of the
   //    flux divergence, and assemble the corresponding mass matrix.
   /*
   MixedBilinearForm Aflux(&fes, &fes);
   Aflux.AddDomainIntegrator(new TransposeIntegrator(new GradientIntegrator()));
   Aflux.Assemble();

   ParNonlinearForm A(&vfes);
   RiemannSolver rsolver;
   A.AddInteriorFaceIntegrator(new FaceIntegrator(rsolver, dim));
   */
   
   MyParBlockNonlinearForm A(afes);
   A.AddDomainIntegrator(new LinearHyp1DIntegrator());
   A.AddInteriorFaceIntegrator(new LinearHyp1DIntegrator());
   
   // 10. Define the time-dependent evolution operator describing the ODE
   //     right-hand side, and perform time-integration (looping over the time
   //     iterations, ti, with a time-step dt).
   // FE_Evolution lin_hyp(vfes, A, Aflux.SpMat());
   FE_Evolution lin_hyp(vfes, A);

   // Visualize the density
   socketstream nout, qout;
   if (visualization)
   {
      char vishost[] = "localhost";
      int  visport   = 19916;

      MPI_Barrier(pmesh.GetComm());
      nout.open(vishost, visport);
      if (!nout)
      {
         if (Mpi::Root())
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         }
         visualization = false;
         if (Mpi::Root())
         {
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         nout << "parallel " << Mpi::WorldSize()
              << " " << Mpi::WorldRank() << "\n";
         nout.precision(precision);
         nout << "solution\n" << pmesh << n;
	 nout << "window_title 'n'\n";
	 nout << "window_geometry 0 0 400 400\n";
         nout << "pause\n";
         nout << flush;
         if (Mpi::Root())
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
      }
      MPI_Barrier(pmesh.GetComm());
      qout.open(vishost, visport);
      if (!qout)
      {
         if (Mpi::Root())
         {
            cout << "Unable to connect to GLVis server at "
                 << vishost << ':' << visport << endl;
         }
         visualization = false;
         if (Mpi::Root())
         {
            cout << "GLVis visualization disabled.\n";
         }
      }
      else
      {
         qout << "parallel " << Mpi::WorldSize()
              << " " << Mpi::WorldRank() << "\n";
         qout.precision(precision);
         qout << "solution\n" << pmesh << q;
	 qout << "window_title 'q'\n";
	 qout << "window_geometry 400 0 400 400\n";
         qout << "pause\n";
         qout << flush;
         if (Mpi::Root())
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }
      }
   }

   // Determine the minimum element size.
   double hmin;
   if (cfl > 0)
   {
      double my_hmin = pmesh.GetElementSize(0, 1);
      for (int i = 1; i < pmesh.GetNE(); i++)
      {
         my_hmin = min(pmesh.GetElementSize(i, 1), my_hmin);
      }
      // Reduce to find the global minimum element size
      MPI_Allreduce(&my_hmin, &hmin, 1, MPI_DOUBLE, MPI_MIN, pmesh.GetComm());
   }

   // Start the timer.
   tic_toc.Clear();
   tic_toc.Start();

   double t = 0.0;
   lin_hyp.SetTime(t);
   ode_solver->Init(lin_hyp);

   if (cfl > 0)
   {
      // Find a safe dt, using a temporary vector. Calling Mult() computes the
      // maximum char speed at all quadrature points on all faces.
      max_char_speed = 0.;
      Vector z(sol.Size());
      A.Mult(sol, z);
      // Reduce to find the global maximum wave speed
      {
         double all_max_char_speed;
         MPI_Allreduce(&max_char_speed, &all_max_char_speed,
                       1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
         max_char_speed = all_max_char_speed;
      }
      dt = cfl * hmin / max_char_speed / (2*order+1);
   }

   // Integrate in time.
   bool done = false;
   for (int ti = 0; !done; )
   {
      double dt_real = min(dt, t_final - t);

      ode_solver->Step(sol, t, dt_real);
      if (cfl > 0)
      {
         // Reduce to find the global maximum wave speed
         {
            double all_max_char_speed;
            MPI_Allreduce(&max_char_speed, &all_max_char_speed,
                          1, MPI_DOUBLE, MPI_MAX, pmesh.GetComm());
            max_char_speed = all_max_char_speed;
         }
         dt = cfl * hmin / max_char_speed / (2*order+1);
      }
      ti++;

      done = (t >= t_final - 1e-8*dt);
      if (done || ti % vis_steps == 0)
      {
         if (Mpi::Root())
         {
            cout << "time step: " << ti << ", time: " << t << endl;
         }
         if (visualization)
         {
            MPI_Barrier(pmesh.GetComm());
            nout << "parallel " << Mpi::WorldSize()
                 << " " << Mpi::WorldRank() << "\n";
            nout << "solution\n" << pmesh << n << flush;

	    MPI_Barrier(pmesh.GetComm());
            qout << "parallel " << Mpi::WorldSize()
                 << " " << Mpi::WorldRank() << "\n";
            qout << "solution\n" << pmesh << q << flush;
         }
      }
   }

   tic_toc.Stop();
   if (Mpi::Root())
   {
      cout << " done, " << tic_toc.RealTime() << "s." << endl;
   }

   // 11. Save the final solution. This output can be viewed later using GLVis:
   //     "glvis -np 4 -m vortex-mesh -g vortex-1-final".
   for (int k = 0; k < num_equation; k++)
   {
      ParGridFunction uk(&fes, u_block.GetBlock(k));
      ostringstream sol_name;
      sol_name << "lin-hyp-" << k << "-final."
               << setfill('0') << setw(6) << Mpi::WorldRank();
      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(precision);
      sol_ofs << uk;
   }

   // 12. Compute the L2 solution error summed for all components.
   if ((t_final == 2.0 &&
        strcmp(mesh_file, "../data/periodic-square.mesh") == 0) ||
       (t_final == 3.0 &&
        strcmp(mesh_file, "../data/periodic-hexagon.mesh") == 0))
   {
      const double error = sol.ComputeLpError(2, u0);
      if (Mpi::Root())
      {
         cout << "Solution error: " << error << endl;
      }
   }

   // Free the used memory.
   delete ode_solver;

   return 0;
}
