//                  MFEM Example 18 - Serial/Parallel Shared Code

#include "mfem.hpp"

using namespace std;
using namespace mfem;

// Problem definition
// extern int problem;

// Maximum characteristic speed (updated by integrators)
extern double max_char_speed;

// Base Hyperbolic conservation law class.
// This contains all methods needed except the flux function evaluation.
class HyperbolicConservationLaws : public TimeDependentOperator {
 private:
  const int dim;
  const int num_equation;
  // Vector finite element space containing conserved variables
  FiniteElementSpace &vfes;
  // Face integration form. Should contain ComputeFluxDotN and Riemann Solver
  NonlinearForm &faceForm;
  // Element integration form, (u, grad V) where u is scalar, V is vector
  MixedBilinearForm &divA;
  std::vector<DenseMatrix> Me_inv;  // element-wise inverse mass matrix

  // auxiliary variables
  mutable Vector state;
  mutable DenseMatrix f;
  mutable DenseTensor flux;
  mutable Vector z;

  // Get flux value for given states for all elements
  void GetFlux(const DenseMatrix &state_, DenseTensor &flux_) const;
  // Compute element-wise inverse mass matrix
  void ComputeInvMass();

 protected:
  // Compute flux for given states at a node
  // WARNING: This should be implemented in the sub-class
  virtual double ComputeFlux(const Vector &state, const int dim,
                             DenseMatrix &flux) const = 0;

 public:
  // Constructor
  HyperbolicConservationLaws(FiniteElementSpace &vfes_, MixedBilinearForm &divA,
                             NonlinearForm &faceForm_, const int num_equation_);
  // Apply M\(DIV F(U) + JUMP HAT{F}(U))
  virtual void Mult(const Vector &x, Vector &y) const;
  // Update operators when mesh and finite element spaces are updated
  void Update();

  virtual ~HyperbolicConservationLaws() {}
};

// Abstract Numerical flux hat{F}(u-, u+).
// Eval: state (u-, u+), flux (Fu-, Fu+), speed, face normal |-> flux
class NumericalFlux {
 public:
  NumericalFlux(){};
  virtual void Eval(const Vector &state1, const Vector &state2,
                    const Vector &flux1, const Vector &flux2, const double maxE,
                    const Vector &nor, Vector &flux) {
    mfem_error("Not Implemented");
  }
};

// Interior face term: <hat{F}.n,[w]>
// where hat{F}.n is determined by NumericalFlux rsolver.
class FaceIntegrator : public NonlinearFormIntegrator {
 private:
  const int num_equation;
  NumericalFlux *rsolver;
  Vector shape1;
  Vector shape2;
  Vector funval1;
  Vector funval2;
  Vector flux1;
  Vector flux2;
  Vector nor;
  Vector fluxN;

 protected:
  virtual double ComputeFluxDotN(const Vector &state, const Vector &nor,
                                 Vector &flux) = 0;

 public:
  FaceIntegrator(NumericalFlux *rsolver_, const int dim,
                 const int num_equation_)
      : num_equation(num_equation_),
        rsolver(rsolver_),
        funval1(num_equation_),
        funval2(num_equation_),
        flux1(num_equation_),
        flux2(num_equation_),
        nor(dim),
        fluxN(num_equation_){};

  virtual void AssembleFaceVector(const FiniteElement &el1,
                                  const FiniteElement &el2,
                                  FaceElementTransformations &Tr,
                                  const Vector &elfun, Vector &elvect);
};

//////////////////////////////////////////////////////////////////
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class HyperbolicConservationLaws
HyperbolicConservationLaws::HyperbolicConservationLaws(
    FiniteElementSpace &vfes_, MixedBilinearForm &divA_,
    NonlinearForm &faceForm_, const int num_equation_)
    : TimeDependentOperator(faceForm_.Height()),
      dim(vfes_.GetFE(0)->GetDim()),
      num_equation(num_equation_),
      vfes(vfes_),
      faceForm(faceForm_),
      divA(divA_),
      Me_inv(0),
      state(num_equation),
      f(num_equation, dim),
      flux(vfes.GetNDofs(), dim, num_equation),
      z(faceForm.Height()) {
  // Standard local assembly and inversion for energy mass matrices.
  divA.Assemble();
  ComputeInvMass();
}

void HyperbolicConservationLaws::ComputeInvMass() {
  DenseMatrix Me;
  MassIntegrator mi;
  Me_inv.resize(vfes.GetNE());
  for (int i = 0; i < vfes.GetNE(); i++) {
    Me.SetSize(vfes.GetFE(i)->GetDof());
    mi.AssembleElementMatrix(*vfes.GetFE(i), *vfes.GetElementTransformation(i),
                             Me);
    DenseMatrixInverse inv(&Me);
    inv.Factor();
    inv.GetInverseMatrix(Me_inv[i]);
  }
}

void HyperbolicConservationLaws::Update() {
  faceForm.Update();
  divA.Update();
  divA.Assemble();
  ComputeInvMass();

  width = faceForm.Width();
  height = faceForm.Height();
  flux.SetSize(vfes.GetNDofs(), dim, num_equation);
  z.SetSize(height);
}

void HyperbolicConservationLaws::Mult(const Vector &x, Vector &y) const {
  // 0. Reset wavespeed computation before operator application.
  max_char_speed = 0.;

  // 1. Create the vector z with the face terms -<F.n(u), [w]>.
  faceForm.Mult(x, z);

  // 2. Add the element terms.
  // i.  computing the flux approximately as a grid function by interpolating
  //     at the solution nodes.
  // ii. multiplying this grid function by a (constant) mixed bilinear form for
  //     each of the num_equation, computing (F(u), grad(w)) for each equation.

  DenseMatrix xmat(x.GetData(), vfes.GetNDofs(), num_equation);
  GetFlux(xmat, flux);

  for (int k = 0; k < num_equation; k++) {
    Vector fk(flux(k).GetData(), dim * vfes.GetNDofs());
    Vector zk(z.GetData() + k * vfes.GetNDofs(), vfes.GetNDofs());
    divA.SpMat().AddMult(fk, zk);
  }

  // 3. Multiply element-wise by the inverse mass matrices.
  Vector zval;
  Array<int> vdofs;
  //   const int dof = vfes.GetFE(0)->GetDof();
  DenseMatrix zmat, ymat;

  for (int i = 0; i < vfes.GetNE(); i++) {
    // Return the vdofs ordered byNODES
    vfes.GetElementVDofs(i, vdofs);
    z.GetSubVector(vdofs, zval);
    zmat.UseExternalData(zval.GetData(), vfes.GetFE(i)->GetDof(), num_equation);
    ymat.SetSize(Me_inv[i].Height(), num_equation);
    mfem::Mult(Me_inv[i], zmat, ymat);
    y.SetSubVector(vdofs, ymat.GetData());
  }
}

// Compute the flux at solution nodes.
void HyperbolicConservationLaws::GetFlux(const DenseMatrix &x_,
                                         DenseTensor &flux_) const {
  const int flux_dof = flux_.SizeI();
  const int flux_dim = flux_.SizeJ();

  for (int i = 0; i < flux_dof; i++) {
    for (int k = 0; k < num_equation; k++) {
      state(k) = x_(i, k);
    }
    const double mcs = ComputeFlux(state, flux_dim, f);

    for (int d = 0; d < flux_dim; d++) {
      for (int k = 0; k < num_equation; k++) {
        flux_(i, d, k) = f(k, d);
      }
    }

    // Update max char speed
    // const double mcs = ComputeMaxCharSpeed(state, flux_dim);
    if (mcs > max_char_speed) {
      max_char_speed = mcs;
    }
  }
}

//////////////////////////////////////////////////////////////////
///                       FACE INTEGRATOR                      ///
//////////////////////////////////////////////////////////////////

void FaceIntegrator::AssembleFaceVector(const FiniteElement &el1,
                                        const FiniteElement &el2,
                                        FaceElementTransformations &Tr,
                                        const Vector &elfun, Vector &elvect) {
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
  if (Tr.Elem2No >= 0) {
    intorder = (min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) +
                2 * max(el1.GetOrder(), el2.GetOrder()));
  } else {
    intorder = Tr.Elem1->OrderW() + 2 * el1.GetOrder();
  }
  if (el1.Space() == FunctionSpace::Pk) {
    intorder++;
  }
  const IntegrationRule *ir = &IntRules.Get(Tr.GetGeometryType(), intorder);

  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);

    Tr.SetAllIntPoints(&ip);  // set face and element int. points

    // Calculate basis functions on both elements at the face
    el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
    el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

    // Interpolate elfun at the point
    elfun1_mat.MultTranspose(shape1, funval1);
    elfun2_mat.MultTranspose(shape2, funval2);

    // Get the normal vector and the flux on the face
    if (nor.Size() == 1)
      nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;
    else
      CalcOrtho(Tr.Jacobian(), nor);

    const double mcs = max(ComputeFluxDotN(funval1, nor, flux1),
                           ComputeFluxDotN(funval2, nor, flux2));
    rsolver->Eval(funval1, funval2, flux1, flux2, mcs, nor, fluxN);

    // Update max char speed
    if (mcs > max_char_speed) {
      max_char_speed = mcs;
    }

    fluxN *= ip.weight;
    for (int k = 0; k < num_equation; k++) {
      for (int s = 0; s < dof1; s++) {
        elvect1_mat(s, k) -= fluxN(k) * shape1(s);
      }
      for (int s = 0; s < dof2; s++) {
        elvect2_mat(s, k) += fluxN(k) * shape2(s);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////
///                      NUMERICAL FLUXES                      ///
//////////////////////////////////////////////////////////////////

// Rusanov Flux
class RusanovFlux : public NumericalFlux {
 public:
  void Eval(const Vector &state1, const Vector &state2, const Vector &flux1,
            const Vector &flux2, const double maxE, const Vector &nor,
            Vector &flux) {
    // NOTE: nor in general is not a unit normal

    flux = 0.0;
    flux += state1;
    flux -= state2;
    flux *= maxE * sqrt(nor * nor);
    flux += flux1;
    flux += flux2;
    flux *= 0.5;
  }
};

// Upwind Flux, Not Yet Implemented
class UpwindFlux : public NumericalFlux {
 public:
  void Eval(const Vector &state1, const Vector &state2, const Vector &flux1,
            const Vector &flux2, const double maxE, const Vector &nor,
            Vector &flux) {
    // NOTE: nor in general is not a unit normal
    mfem_error("Not Implemented");
  }
};