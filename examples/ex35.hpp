//                  MFEM Example 18 - Serial/Parallel Shared Code

#include "mfem.hpp"

using namespace std;
using namespace mfem;

// Problem definition
extern int problem;

// Maximum characteristic speed (updated by integrators)
extern double max_char_speed;

extern const int num_equation;
extern const double specific_heat_ratio;
extern const double gas_constant;

// Time-dependent operator for the right-hand side of the ODE representing the
// DG weak form.
class HyperbolicConservationLaws : public TimeDependentOperator {
 private:
  const int dim;

  FiniteElementSpace &vfes;
  Operator &faceForm;
  SparseMatrix &Aflux;
  DenseTensor Me_inv;

  mutable Vector state;
  mutable DenseMatrix f;
  mutable DenseTensor flux;
  mutable Vector z;

  void GetFlux(const DenseMatrix &state_, DenseTensor &flux_) const;

 protected:
  virtual double ComputeFlux(const Vector &state, const int dim,
                             DenseMatrix &flux) const = 0;

 public:
  HyperbolicConservationLaws(FiniteElementSpace &vfes_, Operator &A_, SparseMatrix &Aflux_);

  virtual void Mult(const Vector &x, Vector &y) const;

  virtual ~HyperbolicConservationLaws() {}
};

class EulerSystem : public HyperbolicConservationLaws {
 private:
  double ComputeFlux(const Vector &state, const int dim,
                     DenseMatrix &flux) const;

 public:
  EulerSystem(FiniteElementSpace &vfes_, Operator &A_, SparseMatrix &Aflux_)
      : HyperbolicConservationLaws(vfes_, A_, Aflux_){};
};

class BurgersEquation : public HyperbolicConservationLaws {
 private:
  double ComputeFlux(const Vector &state, const int dim,
                     DenseMatrix &flux) const;

 public:
  BurgersEquation(FiniteElementSpace &vfes_, Operator &A_, SparseMatrix &Aflux_)
      : HyperbolicConservationLaws(vfes_, A_, Aflux_){};
};

// Implements a simple numerical flux
class NumericalFlux {
 private:
  Vector flux1;
  Vector flux2;

 public:
  NumericalFlux();
  virtual void Eval(const Vector &state1, const Vector &state2,
                    const Vector &flux1, const Vector &flux2, const double maxE,
                    const Vector &nor, Vector &flux) {
    mfem_error("Not Implemented");
  }
};

class UpwindFlux : public NumericalFlux {
 public:
  void Eval(const Vector &state1, const Vector &state2, const Vector &flux1,
            const Vector &flux2, const double maxE, const Vector &nor,
            Vector &flux);
};

class RusanovFlux : public NumericalFlux {
 public:
  void Eval(const Vector &state1, const Vector &state2, const Vector &flux1,
            const Vector &flux2, const double maxE, const Vector &nor,
            Vector &flux);
};

// Interior face term: <F.n(u),[w]>
class FaceIntegrator : public NonlinearFormIntegrator {
 private:
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
  FaceIntegrator(NumericalFlux *rsolver_, const int dim);

  virtual void AssembleFaceVector(const FiniteElement &el1,
                                  const FiniteElement &el2,
                                  FaceElementTransformations &Tr,
                                  const Vector &elfun, Vector &elvect);
};

class EulerFaceIntegrator : public FaceIntegrator {
 private:
  double ComputeFluxDotN(const Vector &state, const Vector &nor, Vector &flux);

 public:
  EulerFaceIntegrator(NumericalFlux *rsolver_, const int dim)
      : FaceIntegrator(rsolver_, dim){};
};

class BurgersFaceIntegrator : public FaceIntegrator {
 private:
  double ComputeFluxDotN(const Vector &state, const Vector &nor, Vector &flux);

 public:
  BurgersFaceIntegrator(NumericalFlux *rsolver_, const int dim)
      : FaceIntegrator(rsolver_, dim){};
};

// Implementation of class HyperbolicConservationLaws
HyperbolicConservationLaws::HyperbolicConservationLaws(FiniteElementSpace &vfes_, Operator &A_,
                           SparseMatrix &Aflux_)
    : TimeDependentOperator(A_.Height()),
      dim(vfes_.GetFE(0)->GetDim()),
      vfes(vfes_),
      faceForm(A_),
      Aflux(Aflux_),
      Me_inv(vfes.GetFE(0)->GetDof(), vfes.GetFE(0)->GetDof(), vfes.GetNE()),
      state(num_equation),
      f(num_equation, dim),
      flux(vfes.GetNDofs(), dim, num_equation),
      z(faceForm.Height()) {
  // Standard local assembly and inversion for energy mass matrices.
  const int dof = vfes.GetFE(0)->GetDof();
  DenseMatrix Me(dof);
  DenseMatrixInverse inv(&Me);
  MassIntegrator mi;
  for (int i = 0; i < vfes.GetNE(); i++) {
    mi.AssembleElementMatrix(*vfes.GetFE(i), *vfes.GetElementTransformation(i),
                             Me);
    inv.Factor();
    inv.GetInverseMatrix(Me_inv(i));
  }
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
    Aflux.AddMult(fk, zk);
  }

  // 3. Multiply element-wise by the inverse mass matrices.
  Vector zval;
  Array<int> vdofs;
  const int dof = vfes.GetFE(0)->GetDof();
  DenseMatrix zmat, ymat(dof, num_equation);

  for (int i = 0; i < vfes.GetNE(); i++) {
    // Return the vdofs ordered byNODES
    vfes.GetElementVDofs(i, vdofs);
    z.GetSubVector(vdofs, zval);
    zmat.UseExternalData(zval.GetData(), dof, num_equation);
    mfem::Mult(Me_inv(i), zmat, ymat);
    y.SetSubVector(vdofs, ymat.GetData());
  }
}

// Physicality check (at end)
bool StateIsPhysical(const Vector &state, const int dim);

// Pressure (EOS) computation
inline double ComputePressure(const Vector &state, int dim) {
  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, dim);
  const double den_energy = state(1 + dim);

  double den_vel2 = 0;
  for (int d = 0; d < dim; d++) {
    den_vel2 += den_vel(d) * den_vel(d);
  }
  den_vel2 /= den;

  return (specific_heat_ratio - 1.0) * (den_energy - 0.5 * den_vel2);
}

// Compute the vector flux F(u)
double EulerSystem::ComputeFlux(const Vector &state, const int dim,
                                DenseMatrix &flux) const {
  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, dim);
  const double den_energy = state(1 + dim);

  MFEM_ASSERT(StateIsPhysical(state, dim), "");

  const double pres = ComputePressure(state, dim);

  for (int d = 0; d < dim; d++) {
    flux(0, d) = den_vel(d);
    for (int i = 0; i < dim; i++) {
      flux(1 + i, d) = den_vel(i) * den_vel(d) / den;
    }
    flux(1 + d, d) += pres;
  }

  const double H = (den_energy + pres) / den;
  for (int d = 0; d < dim; d++) {
    flux(1 + dim, d) = den_vel(d) * H;
  }

  const double sound = sqrt(specific_heat_ratio * pres / den);
  const double vel = sqrt(den_vel * den_vel) / den;

  return vel + sound;
}

// Compute the scalar F(u).n

double EulerFaceIntegrator::ComputeFluxDotN(const Vector &state,
                                            const Vector &nor, Vector &fluxN) {
  // NOTE: nor in general is not a unit normal
  const int dim = nor.Size();
  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, dim);
  const double den_energy = state(1 + dim);

  MFEM_ASSERT(StateIsPhysical(state, dim), "");

  const double pres = ComputePressure(state, dim);

  double den_velN = 0;
  for (int d = 0; d < dim; d++) {
    den_velN += den_vel(d) * nor(d);
  }

  fluxN(0) = den_velN;
  for (int d = 0; d < dim; d++) {
    fluxN(1 + d) = den_velN * den_vel(d) / den + pres * nor(d);
  }

  const double H = (den_energy + pres) / den;
  fluxN(1 + dim) = den_velN * H;

  const double sound = sqrt(specific_heat_ratio * pres / den);
  const double vel = sqrt(den_vel * den_vel) / den;

  return vel + sound;
}
// Compute the vector flux F(u)
double BurgersEquation::ComputeFlux(const Vector &state, const int dim,
                                    DenseMatrix &flux) const {
  flux = state * state * 0.5;
  return abs(state(0));
}

double BurgersFaceIntegrator::ComputeFluxDotN(const Vector &state,
                                              const Vector &nor,
                                              Vector &fluxN) {
  fluxN = nor.Sum() * (state * state) * 0.5;
  return abs(state(0));
}

// // Compute the maximum characteristic speed.
// inline double ComputeMaxCharSpeed(const Vector &state, const int dim) {
//   const double den = state(0);
//   const Vector den_vel(state.GetData() + 1, dim);

//   double den_vel2 = 0;
//   for (int d = 0; d < dim; d++) {
//     den_vel2 += den_vel(d) * den_vel(d);
//   }
//   den_vel2 /= den;

//   const double pres = ComputePressure(state, dim);
//   const double sound = sqrt(specific_heat_ratio * pres / den);
//   const double vel = sqrt(den_vel2 / den);

//   return vel + sound;
// }

// Compute the flux at solution nodes.
void HyperbolicConservationLaws::GetFlux(const DenseMatrix &x_, DenseTensor &flux_) const {
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

// Implementation of class NumericalFlux
NumericalFlux::NumericalFlux() : flux1(num_equation), flux2(num_equation) {}

void UpwindFlux::Eval(const Vector &state1, const Vector &state2,
                      const Vector &flux1, const Vector &flux2,
                      const double maxE, const Vector &nor, Vector &flux) {
  // NOTE: nor in general is not a unit normal

  mfem_error("Not Implemented");
}

void RusanovFlux::Eval(const Vector &state1, const Vector &state2,
                       const Vector &flux1, const Vector &flux2,
                       const double maxE, const Vector &nor, Vector &flux) {
  // NOTE: nor in general is not a unit normal

  flux = 0.0;
  flux += state1;
  flux -= state2;
  flux *= maxE * sqrt(nor * nor);
  flux += flux1;
  flux += flux2;
  flux *= 0.5;
}

// Implementation of class FaceIntegrator
FaceIntegrator::FaceIntegrator(NumericalFlux *rsolver_, const int dim)
    : rsolver(rsolver_),
      funval1(num_equation),
      funval2(num_equation),
      flux1(num_equation),
      flux2(num_equation),
      nor(dim),
      fluxN(num_equation) {}

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
  if (Tr.Elem2No >= 0)
    intorder = (min(Tr.Elem1->OrderW(), Tr.Elem2->OrderW()) +
                2 * max(el1.GetOrder(), el2.GetOrder()));
  else {
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
      nor = 1.0;
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

// Check that the state is physical - enabled in debug mode
bool StateIsPhysical(const Vector &state, const int dim) {
  const double den = state(0);
  const Vector den_vel(state.GetData() + 1, dim);
  const double den_energy = state(1 + dim);

  if (den < 0) {
    cout << "Negative density: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
    return false;
  }
  if (den_energy <= 0) {
    cout << "Negative energy: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
    return false;
  }

  double den_vel2 = 0;
  for (int i = 0; i < dim; i++) {
    den_vel2 += den_vel(i) * den_vel(i);
  }
  den_vel2 /= den;

  const double pres =
      (specific_heat_ratio - 1.0) * (den_energy - 0.5 * den_vel2);

  if (pres <= 0) {
    cout << "Negative pressure: " << pres << ", state: ";
    for (int i = 0; i < state.Size(); i++) {
      cout << state(i) << " ";
    }
    cout << endl;
    return false;
  }
  return true;
}

// Initial condition
void EulerInitialCondition(const Vector &x, Vector &y) {
  MFEM_ASSERT(x.Size() == 2, "");
  if (problem < 3) {
    double radius = 0, Minf = 0, beta = 0;
    if (problem == 1) {
      // "Fast vortex"
      radius = 0.2;
      Minf = 0.5;
      beta = 1. / 5.;
    } else if (problem == 2) {
      // "Slow vortex"
      radius = 0.2;
      Minf = 0.05;
      beta = 1. / 50.;
    } else {
      mfem_error(
          "Cannot recognize problem."
          "Options are: 1 - fast vortex, 2 - slow vortex");
    }

    const double xc = 0.0, yc = 0.0;

    // Nice units
    const double vel_inf = 1.;
    const double den_inf = 1.;

    // Derive remainder of background state from this and Minf
    const double pres_inf =
        (den_inf / specific_heat_ratio) * (vel_inf / Minf) * (vel_inf / Minf);
    const double temp_inf = pres_inf / (den_inf * gas_constant);

    double r2rad = 0.0;
    r2rad += (x(0) - xc) * (x(0) - xc);
    r2rad += (x(1) - yc) * (x(1) - yc);
    r2rad /= (radius * radius);

    const double shrinv1 = 1.0 / (specific_heat_ratio - 1.);

    const double velX =
        vel_inf * (1 - beta * (x(1) - yc) / radius * exp(-0.5 * r2rad));
    const double velY =
        vel_inf * beta * (x(0) - xc) / radius * exp(-0.5 * r2rad);
    const double vel2 = velX * velX + velY * velY;

    const double specific_heat = gas_constant * specific_heat_ratio * shrinv1;
    const double temp = temp_inf - 0.5 * (vel_inf * beta) * (vel_inf * beta) /
                                       specific_heat * exp(-r2rad);

    const double den = den_inf * pow(temp / temp_inf, shrinv1);
    const double pres = den * gas_constant * temp;
    const double energy = shrinv1 * pres / den + 0.5 * vel2;

    y(0) = den;
    y(1) = den * velX;
    y(2) = den * velY;
    y(3) = den * energy;
  } else if (problem == 3) {
    // std::cout << "2D Accuracy Test." << std::endl;
    // std::cout << "domain = (-1, 1) x (-1, 1)" << std::endl;
    const double density = 1.0 + 0.2 * __sinpi(x(0) + x(1));
    const double velocity_x = 0.7;
    const double velocity_y = 0.3;
    const double pressure = 1.0;
    const double energy =
        pressure / (1.4 - 1.0) +
        density * 0.5 * (velocity_x * velocity_x + velocity_y * velocity_y);

    y(0) = density;
    y(1) = density * velocity_x;
    y(2) = density * velocity_y;
    y(3) = energy;
  } else {
    mfem_error("Invalid problem.");
  }
}

// Initial condition
void BurgersInitialCondition(const Vector &x, Vector &y) {
  if (problem == 1) {
    y(0) = __sinpi(x(0) * 2 + 1);
  } else if (problem == 2) {
    y(0) = __sinpi(x.Sum());
  } else if (problem == 3) {
    y = 0.0;
    y(0) = x(0) < 0.5 ? 1.0 : 2.0;
  } else {
    mfem_error("Invalid problem.");
  }
}
