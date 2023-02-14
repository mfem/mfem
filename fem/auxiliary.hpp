//                  MFEM Example 18 - Serial/Parallel Shared Code

#include "mfem.hpp"
using namespace std;
using namespace mfem;

// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form.
class EulerSystem : public TimeDependentOperator
{
private:
   const int dim;

   FiniteElementSpace * vfes;
   Operator &A;
   SparseMatrix &Aflux;
   std::vector<DenseMatrix> Me_inv;

   double specific_heat_ratio;
   int num_equation;

   mutable Vector state;
   mutable DenseMatrix f;
   mutable DenseTensor flux;
   mutable Vector z;

   void GetFlux(const DenseMatrix &state_, DenseTensor &flux_) const;

public:
   EulerSystem(FiniteElementSpace &vfes_,
               Operator &A_, SparseMatrix &Aflux_,
               double specific_heat_ratio_, int num_equation_);

   float GetMaxWavespeed(const Vector &x);

   void GetDensityBounds(GridFunction &solution, int dim, double gamma, int num_equation,
                         Vector &avgs, Vector &d_min, Vector &d_max);
   void ApplyLimiter(GridFunction &solution, int dim, double gamma, int num_equation,
                     Vector &avgs, Vector &d_min, Vector &d_max);

   virtual void Mult(const Vector &x, Vector &y) const;

   virtual ~EulerSystem() {};
};

// Implements a simple Rusanov flux
class RiemannSolver
{
public:
   Vector flux1;
   Vector flux2;

   int num_equation;
   double specific_heat_ratio;
   RiemannSolver(double specific_heat_ratio_, int num_equation_);
   double Eval(const Vector &state1, const Vector &state2,
               const Vector &nor, Vector &flux);
};

// Interior face term: <F.n(u),[w]>
class FaceIntegrator : public NonlinearFormIntegrator
{
public:
   RiemannSolver rsolver;
   int num_equation;
   Vector shape1;
   Vector shape2;
   Vector funval1;
   Vector funval2;
   Vector nor;
   Vector fluxN;

   FaceIntegrator(RiemannSolver &rsolver_, const int dim, double num_equation_);

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
};

// Boundary face term: <F.n(u),[w]> (Neumann BCs)
class NeumannBCFaceIntegrator : public NonlinearFormIntegrator
{
public:
   RiemannSolver rsolver;
   int num_equation;
   Vector shape;
   Vector funval;
   Vector nor;
   Vector fluxN;

   NeumannBCFaceIntegrator(RiemannSolver &rsolver_, const int dim,
                           double num_equation_);

   virtual void AssembleFaceVector(const FiniteElement &el1,
                                   const FiniteElement &el2,
                                   FaceElementTransformations &Tr,
                                   const Vector &elfun, Vector &elvect);
};

// Implementation of class FE_Evolution
EulerSystem::EulerSystem(FiniteElementSpace &vfes_,
                         Operator &A_, SparseMatrix &Aflux_,
                         double specific_heat_ratio_,
                         int num_equation_)
   : TimeDependentOperator(A_.Height()),
     dim(vfes_.GetFE(0)->GetDim()),
     vfes(&vfes_),
     A(A_),
     Aflux(Aflux_),
     specific_heat_ratio(specific_heat_ratio_),
     num_equation(num_equation_),
     state(num_equation),
     f(num_equation, dim),
     flux(vfes->GetNDofs(), dim, num_equation),
     z(A.Height())
{
   MassIntegrator mi;

   for (int i = 0; i < vfes->GetNE(); i++)
   {
      // Standard local assembly and inversion for energy mass matrices.
      int dof = vfes->GetFE(i)->GetDof();
      DenseMatrix Me(dof);
      DenseMatrixInverse inv(&Me);

      DenseMatrix inv_mi = DenseMatrix(vfes->GetFE(i)->GetDof(),
                                       vfes->GetFE(i)->GetDof());

      mi.AssembleElementMatrix(*vfes->GetFE(i), *vfes->GetElementTransformation(i),
                               Me);
      inv.Factor();
      inv.GetInverseMatrix(inv_mi);

      Me_inv.push_back(inv_mi);
   }
}

void EulerSystem::Mult(const Vector &x, Vector &y) const
{
   // 1. Create the vector z with the face terms -<F.n(u), [w]>.
   A.Mult(x, z);

   // 2. Add the element terms.
   // i.  computing the flux approximately as a grid function by interpolating
   //     at the solution nodes.
   // ii. multiplying this grid function by a (constant) mixed bilinear form for
   //     each of the num_equation, computing (F(u), grad(w)) for each equation.

   DenseMatrix xmat(x.GetData(), vfes->GetNDofs(), num_equation);
   GetFlux(xmat, flux);

   for (int k = 0; k < num_equation; k++)
   {
      Vector fk(flux(k).GetData(), dim * vfes->GetNDofs());
      Vector zk(z.GetData() + k * vfes->GetNDofs(), vfes->GetNDofs());
      Aflux.AddMult(fk, zk);
   }

   // 3. Multiply element-wise by the inverse mass matrices.
   Vector zval;
   Array<int> vdofs;

   for (int i = 0; i < vfes->GetNE(); i++)
   {
      int dof = vfes->GetFE(i)->GetDof();
      DenseMatrix zmat, ymat(dof, num_equation);

      // Return the vdofs ordered byNODES
      vfes->GetElementVDofs(i, vdofs);
      z.GetSubVector(vdofs, zval);
      zmat.UseExternalData(zval.GetData(), dof, num_equation);
      mfem::Mult(Me_inv[i], zmat, ymat);
      y.SetSubVector(vdofs, ymat.GetData());
   }
}

// Physicality check (at end)
bool StateIsPhysical(const Vector &state, const int dim);

// Pressure (EOS) computation
inline double ComputePressure(const Vector &state, int num_equation,
                              double specific_heat_ratio)
{
   const int udim = num_equation - 2;
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, udim);
   const double den_energy = state(num_equation - 1);

   double den_vel2 = 0;
   for (int d = 0; d < udim; d++)
   {
      den_vel2 += den_vel(d)*den_vel(d);
   }
   den_vel2 /= den;

   return (specific_heat_ratio-1.0)*(den_energy - 0.5*den_vel2);
}

// Compute the vector flux F(u)
void ComputeFlux(const Vector &state, int dim, DenseMatrix &flux,
                 double specific_heat_ratio, int num_equation)
{
   const int udim = num_equation - 2;
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, udim);
   const double den_energy = state(num_equation - 1);

   MFEM_ASSERT(StateIsPhysical(state, dim), "");

   const double pres = ComputePressure(state, num_equation, specific_heat_ratio);
   const double H = (den_energy + pres)/den;

   // Hard-code quasi-1D cases
   if (num_equation == 3)
   {
      // Set x-flux
      flux(0, 0) = den_vel(0);
      flux(1, 0) = den_vel(0)*den_vel(0)/den + pres;
      flux(2, 0) = den_vel(0)*H;

      // Zero other components
      for (int d = 1; d < dim; d++)
      {
         for (int eq = 0; eq < num_equation; eq++)
         {
            flux(eq, d) = 0.0;
         }
      }
   }
   else
   {
      MFEM_ASSERT(num_equation == dim + 2, "2D/3D solutions must be of size dim+2.")
      for (int d = 0; d < dim; d++)
      {
         flux(0, d) = den_vel(d);
         for (int i = 0; i < dim; i++)
         {
            flux(1+i, d) = den_vel(i) * den_vel(d) / den;
         }
         flux(1+d, d) += pres;
      }
      for (int d = 0; d < dim; d++)
      {
         flux(1+dim, d) = den_vel(d) * H;
      }
   }
}

// Compute the scalar F(u).n
void ComputeFluxDotN(const Vector &state, const Vector &nor, Vector &fluxN,
                     double specific_heat_ratio, int num_equation)
{
   // const int udim = num_equation - 2; // unused
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();
   MFEM_ASSERT(StateIsPhysical(state, dim), "");

   DenseMatrix flux = DenseMatrix(num_equation, dim);
   ComputeFlux(state, dim, flux, specific_heat_ratio, num_equation);

   for (int i = 0; i < num_equation; i++)
   {
      fluxN(i) = 0.0;
      for (int d = 0; d < dim; d++)
      {
         fluxN(i) += nor(d)*flux(i,d);
      }
   }
}

// Compute the maximum characteristic speed.
inline double ComputeMaxCharSpeed(const Vector &state, const int dim,
                                  double specific_heat_ratio, int num_equation)
{
   const int udim = num_equation - 2;
   const double den = state(0);
   const Vector den_vel(state.GetData() + 1, udim);

   double den_vel2 = 0;
   for (int d = 0; d < udim; d++)
   {
      den_vel2 += den_vel(d)*den_vel(d);
   }
   den_vel2 /= den;

   const double pres = ComputePressure(state, num_equation, specific_heat_ratio);
   const double sound = sqrt(specific_heat_ratio*pres/den);
   const double vel = sqrt(den_vel2/den);

   return vel + sound;
}

// Compute the flux at solution nodes.
void EulerSystem::GetFlux(const DenseMatrix &x_, DenseTensor &flux_) const
{
   const int flux_dof = flux_.SizeI();
   const int flux_dim = flux_.SizeJ();

   for (int i = 0; i < flux_dof; i++)
   {
      for (int k = 0; k < num_equation; k++)
      {
         state(k) = x_(i, k);
      }

      ComputeFlux(state, flux_dim, f, specific_heat_ratio, num_equation);

      for (int d = 0; d < flux_dim; d++)
      {
         for (int k = 0; k < num_equation; k++)
         {
            flux_(i, d, k) = f(k, d);
         }
      }
   }
}

float EulerSystem::GetMaxWavespeed(const Vector &u)
{
   const int flux_dof = flux.SizeI();
   double lambda = 0.0;

   DenseMatrix umat(u.GetData(), vfes->GetNDofs(), num_equation);

   for (int i = 0; i < flux_dof; i++)
   {
      for (int k = 0; k < num_equation; k++)
      {
         state(k) = umat(i, k);
      }

      // Update maximum wavespeed
      lambda = max(lambda, ComputeMaxCharSpeed(state, dim, specific_heat_ratio,
                                               num_equation));
   }

   return lambda;
}


// Implementation of class RiemannSolver
RiemannSolver::RiemannSolver(double specific_heat_ratio_, int num_equation_) :
   flux1(num_equation_),
   flux2(num_equation_),
   num_equation(num_equation_),
   specific_heat_ratio(specific_heat_ratio_)
{ }

double RiemannSolver::Eval(const Vector &state1, const Vector &state2,
                           const Vector &nor, Vector &flux)
{
   // NOTE: nor in general is not a unit normal
   const int dim = nor.Size();

   MFEM_ASSERT(StateIsPhysical(state1, dim), "");
   MFEM_ASSERT(StateIsPhysical(state2, dim), "");

   const double maxE1 = ComputeMaxCharSpeed(state1, dim, specific_heat_ratio,
                                            num_equation);
   const double maxE2 = ComputeMaxCharSpeed(state2, dim, specific_heat_ratio,
                                            num_equation);

   const double maxE = max(maxE1, maxE2);

   ComputeFluxDotN(state1, nor, flux1, specific_heat_ratio, num_equation);
   ComputeFluxDotN(state2, nor, flux2, specific_heat_ratio, num_equation);

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
FaceIntegrator::FaceIntegrator(RiemannSolver &rsolver_, const int dim,
                               double num_equation_) :
   rsolver(rsolver_),
   num_equation(num_equation_),
   funval1(num_equation),
   funval2(num_equation),
   nor(dim),
   fluxN(num_equation) { }

void FaceIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Tr,
                                        const Vector &elfun, Vector &elvect)
{
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

      // Calculate basis functions on both elements at the face
      el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
      el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

      // Interpolate elfun at the point
      elfun1_mat.MultTranspose(shape1, funval1);
      elfun2_mat.MultTranspose(shape2, funval2);

      // Get the normal vector and the flux on the face
      CalcOrtho(Tr.Jacobian(), nor);
      const double mcs = rsolver.Eval(funval1, funval2, nor, fluxN);

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

// Implementation of class NeumannBCFaceIntegrator
NeumannBCFaceIntegrator::NeumannBCFaceIntegrator(RiemannSolver &rsolver_,
                                                 const int dim, double num_equation_) :
   rsolver(rsolver_),
   num_equation(num_equation_),
   funval(num_equation),
   nor(dim),
   fluxN(num_equation) { }

void NeumannBCFaceIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                                 const FiniteElement &el2,
                                                 FaceElementTransformations &Tr,
                                                 const Vector &elfun, Vector &elvect)
{
   // Compute the term <F.n(u),[w]> on the interior faces.
   const int dof = el1.GetDof();

   shape.SetSize(dof);

   elvect.SetSize(dof * num_equation);
   elvect = 0.0;

   DenseMatrix elfun_mat(elfun.GetData(), dof, num_equation);
   DenseMatrix elvect_mat(elvect.GetData(), dof, num_equation);

   // Integration order calculation from DGTraceIntegrator
   int intorder = Tr.Elem1->OrderW() + 2*el1.GetOrder();


   const IntegrationRule *ir = &IntRules.Get(Tr.GetGeometryType(), intorder);
   for (int i = 0; i < ir->GetNPoints(); i++)
   {
      const IntegrationPoint &ip = ir->IntPoint(i);

      Tr.SetAllIntPoints(&ip); // set face and element int. points

      // Calculate basis functions on both elements at the face
      el1.CalcShape(Tr.GetElement1IntPoint(), shape);

      // Interpolate elfun at the point
      elfun_mat.MultTranspose(shape, funval);

      // Get the normal vector and the flux on the face
      CalcOrtho(Tr.Jacobian(), nor);
      const double mcs = rsolver.Eval(funval, funval, nor, fluxN);

      fluxN *= ip.weight;
      for (int k = 0; k < num_equation; k++)
      {
         for (int s = 0; s < dof; s++)
         {
            elvect_mat(s, k) -= fluxN(k) * shape(s);
         }
      }
   }
}

// Check that the state is physical - enabled in debug mode
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

   const double int_energy = den_energy - 0.5 * den_vel2;

   if (int_energy <= 0)
   {
      cout << "Negative internal energy: " << int_energy << ", state: ";
      for (int i = 0; i < state.Size(); i++)
      {
         cout << state(i) << " ";
      }
      cout << endl;
      return false;
   }
   return true;
}

void EulerSystem::GetDensityBounds(GridFunction &solution, int dim, double gamma, int num_equation,
                                   Vector &avgs, Vector &d_min, Vector &d_max) {

   Vector dofs = Vector();
   // Gridfunction ordered by [rho_0, rho_1, ..., rhou[0], rhou[1], ...]
   solution.GetTrueDofs(dofs);

   int nelems = solution.FESpace()->GetMesh()->GetNE();

   // Constant order elements only
   int p = solution.FESpace()->GetElementOrder(0);
   int elem_ndofs = (p+1)*(p+1);

   Vector sol_l = Vector(num_equation);
   Vector sol_r = Vector(num_equation);
   Vector f_intl = Vector(dim);
   Vector f_intr = Vector(dim);

   // Tolerance and minimum value for density bounds
   double tol = pow(10, -6.0);

   // Get element neighbor map
   Table etable = solution.FESpace()->GetMesh()->ElementToElementTable();

   for (int i = 0; i < nelems; i++) {
      d_min(i) = std::numeric_limits<double>::max();
      d_max(i) = 0.0;

      // Compute min/max density within element
      for (int j = 0; j < elem_ndofs; j++) {
         d_min(i) = min(d_min(i), dofs(i*elem_ndofs + j));
         d_max(i) = max(d_max(i), dofs(i*elem_ndofs + j));
      }

      // Get average solution within element
      for (int j = 0; j < num_equation; j++) {
         sol_l(j) = avgs(i + j*nelems);
      }

      // Get average density flux within element
      for (int j = 0; j < dim; j++) {
         f_intl(j) = sol_l(1 + j);
      }

      // Get element center and average max wavespeed
      double lam_l = ComputeMaxCharSpeed(sol_l, dim, specific_heat_ratio, num_equation);

      // Get and loop through face-adjacent elements
      Array<int> adj_elems;
      etable.GetRow(i, adj_elems);
      for (int k : adj_elems) {
         // Get neighbor element average solution and density flux
         for (int j = 0; j < num_equation; j++) {
            sol_r(j) = avgs(k + j*nelems);
         }
         for (int j = 0; j < dim; j++) {
            f_intr(j) = sol_r(1 + j);
         }

         // Get max wavespeed for the Riemann problem between the two states (using Davis estimate)
         double lam_r = ComputeMaxCharSpeed(sol_r, dim, specific_heat_ratio, num_equation);
         double lam_max = max(lam_l, lam_r);

         // Compute Riemann-averaged density over the Riemann fan
         double dfdotn = 0.0;
         for (int j = 0; j < dim; j++) {
            dfdotn += (f_intr(j) - f_intl(j))*(f_intr(j) - f_intl(j));
         }
         dfdotn = sqrt(dfdotn);

         double d_bar = 0.5*(sol_l(0) + sol_r(0)) - 0.5*dfdotn/lam_max;

         // Update density bounds
         d_min(i) = min(d_min(i), d_bar);
         d_max(i) = max(d_max(i), d_bar);

         d_bar = 0.5*(sol_l(0) + sol_r(0)) + 0.5*dfdotn/lam_max;

         // Update density bounds
         d_min(i) = min(d_min(i), d_bar);
         d_max(i) = max(d_max(i), d_bar);
      }

   // Add tolerances and minimum bounds
   d_min(i) -= tol;
   d_max(i) += tol;
   d_min(i) = max(d_min(i), tol);
   }
}


void EulerSystem::ApplyLimiter(GridFunction &solution, int dim, double gamma, int num_equation,
                               Vector &avgs, Vector &d_min, Vector &d_max) {
   Vector dofs = Vector();
   // Gridfunction ordered by [rho_0, rho_1, ..., rhou[0], rhou[1], ...]
   solution.GetTrueDofs(dofs);

   int nelems = solution.FESpace()->GetMesh()->GetNE();

   // Constant order elements only
   int p = solution.FESpace()->GetElementOrder(0);
   int elem_ndofs = (p+1)*(p+1);

   double ptol = pow(10, -6.0);
   double rtol = pow(10, -12.0);

   Vector sol = Vector(num_equation);
   Vector sol_avg = Vector(num_equation);
   for (int i = 0; i < nelems; i++) {
      // Get min/max density and min pressure within element
      double elem_d_min = std::numeric_limits<double>::max();
      double elem_d_max = 0.0;
      double elem_p_min = std::numeric_limits<double>::max();
      int min_p_idx = 0;

      for (int j = 0; j < elem_ndofs; j++) {
         // Get solution at nodal point within element
         for (int k = 0; k < num_equation; k++) {
            sol(k) = dofs(i*elem_ndofs + j + k*elem_ndofs*nelems);
         }
         elem_d_min = min(elem_d_min, sol(0));
         elem_d_max = max(elem_d_max, sol(0));

         double p = ComputePressure(sol, num_equation, specific_heat_ratio);
         if (p < elem_p_min) {
            elem_p_min = min(elem_p_min, p);
            min_p_idx = j;
         }
      }

      // Get average solution within element
      for (int j = 0; j < num_equation; j++) {
         sol_avg(j) = avgs(i + j*nelems);
      }

      // Compute limiting factor for density
      double theta1 = 1.0;
      if (elem_d_min < d_min(i) || elem_d_max < d_max(i)) {
         double d_avg = avgs(i);

         double a1 = (d_min(i) - d_avg)/(elem_d_min - d_avg + rtol);
         double a2 = (d_max(i) - d_avg)/(elem_d_max - d_avg + rtol);

         theta1 = min(1.0, min(a1, a2));
      }

      // Compute limiting factor for pressure
      double theta2 = 1.0;
      if (elem_p_min < ptol) {
         /*
         Solve the equation:
         (r + a*dr)*(E + a*dE) - 0.5*((ru + a*dru)^2 + (rv + a*drv)^2) = ptol*(r + a*dr)/(gamma - 1)

         =>
         c1 = dr*dE - 0.5*dru^2 - 0.5*drv^2
         c2 = dr*E + dE*r - ru*dru - rv*drv - ptol*dr/(g-1)
         c3 = r*E - 0.5*ru^2 - 0.5*rv^2 - ptol*r/(g-1)

         Solve c1*a^2 + c2*a + c3 = 0
         */

         // Get nodal solution at minimum pressure point
         for (int j = 0; j < num_equation; j++) {
            sol(j) = dofs(i*elem_ndofs + min_p_idx + j*elem_ndofs*nelems);
         }

         double c1, c2, c3;
         if (dim == 1) {
            double r = sol(0);
            double ru = sol(1);
            double E = sol(2);
            double dr = sol_avg(0) - sol(0);
            double dru = sol_avg(1) - sol(1);
            double dE = sol_avg(2) - sol(2);

            c1 = dr*dE - 0.5*dru*dru;
            c2 = dr*E + dE*r - dru*ru - ptol*dr/(gamma-1);
            c3 = r*E -0.5*ru*ru - ptol*r/(gamma-1);
         }
         else if (dim == 2) {
            double r = sol(0);
            double ru = sol(1);
            double rv = sol(2);
            double E = sol(3);
            double dr = sol_avg(0) - sol(0);
            double dru = sol_avg(1) - sol(1);
            double drv = sol_avg(2) - sol(2);
            double dE = sol_avg(3) - sol(3);

            c1 = dr*dE - 0.5*dru*dru - 0.5*drv*drv;
            c2 = dr*E + dE*r - dru*ru - drv*rv - ptol*dr/(gamma-1);
            c3 = r*E - 0.5*ru*ru - 0.5*rv*rv - ptol*r/(gamma-1);
         }

         double a1 = (-c2 + sqrt(c2*c2 - 4*c1*c3))/(2*c1);
         double a2 = (-c2 - sqrt(c2*c2 - 4*c1*c3))/(2*c1);

         theta2 = 1 - max(a1, a2);
      }

      // Taking limiting factor as the stronger of the two
      double theta = min(theta1, theta2);

      theta = min(1.0, max(0.0, theta));

      // Apply limiting if necessary
      if (theta != 1.0) {
         for (int j = 0; j < elem_ndofs; j++) {
            for (int k = 0; k < num_equation; k++) {
               dofs(i*elem_ndofs + j + k*elem_ndofs*nelems) =
                  sol_avg(k) + theta*(dofs(i*elem_ndofs + j + k*elem_ndofs*nelems) - sol_avg(k));
            }
         }
      }
   }

   solution.SetFromTrueDofs(dofs);
}
