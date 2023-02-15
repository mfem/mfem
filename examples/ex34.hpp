//                  MFEM Hyperbolic Conservation Laws - Serial/Parallel Shared
//                  Code
#include "mfem.hpp"
using namespace std;
using namespace mfem;

enum FluxType { RUSANOV };

typedef std::function<double(const Vector &, const Vector &, Vector &)>
    NormalFlux;
typedef std::function<double(const Vector &, DenseMatrix &)> Flux;

class HyperbolicConservationLaws;
class DivFlux;
class NumericalFlux;
class RusanovFlux;
class EulerSystem;

class HyperbolicConservationLaws : public TimeDependentOperator {
 private:
  const int dim;
  const int num_equations;
  FiniteElementSpace &vfes;  // vector space, size: The number of equations
  FiniteElementSpace &dfes;  // vector space, size: Spatial dimension
  FiniteElementSpace &sfes;  // scalar space
  MixedBilinearForm Adiv; // -(F(u), grad phi)
  Flux F;
  NormalFlux FudotN;
  NumericalFlux *riemann_solver = NULL;

 public:
  HyperbolicConservationLaws(const int num_equations_, const int dim_,
                             FiniteElementSpace &sfes_,
                             FiniteElementSpace &dfes_,
                             FiniteElementSpace &vfes_, Flux F_,
                             NormalFlux FudotN_,
                             FluxType fluxname = FluxType::RUSANOV);
  virtual ~HyperbolicConservationLaws() = default;
};

class DivFlux : public NonlinearFormIntegrator {
 private:
  Flux F;
  Vector shape;        // placeholder for shape function eval
  Vector funval;       // placeholder for solution value
  DenseMatrix dshape;  // placeholder for reference gradient of shape function
  DenseMatrix gshape;  // placeholder for physical gradient of shape function
  DenseMatrix Fmat;    // placeholder for flux eval
  DenseMatrix Jadj;    // placeholder for Jacobian adjoint

 public:
  double max_char_speed;
  DivFlux(Flux F_) : F(F_){};

  void AssembleElementVector(const FiniteElement &elem,
                             ElementTransformation &trans, const Vector &elfun,
                             Vector &elvect);
};

class NumericalFlux : public NonlinearFormIntegrator {
 protected:
  NormalFlux FdotN;
  virtual double riemannSolver(const Vector &state1, const Vector &state2,
                               const Vector &nor, Vector &fluxN) = 0;

 public:
  double max_char_speed;
  NumericalFlux(NormalFlux FdotN_) : FdotN(FdotN_){};
  ~NumericalFlux() = default;

  void AssembleFaceVector(const FiniteElement &el1, const FiniteElement &el2,
                          FaceElementTransformations &Tr, const Vector &elfun,
                          Vector &elvect);
};

class RusanovFlux : public NumericalFlux {
 private:
  Vector flux1, flux2;
  double riemannSolver(const Vector &state1, const Vector &state2,
                       const Vector &nor, Vector &fluxN);

 public:
  /// @brief Compute Rusanov flux for the given flux function and edge by F* =
  /// 0.5(F(s₁)n + F(s₂)n) - 0.5λ(F(s₁)n + F(s₂)n)
  /// @param FdotN_ flux normal function: F(s, n, Fsn)
  /// @param num_equations the number of equation
  RusanovFlux(NormalFlux FdotN_, const int num_equations)
      : NumericalFlux(FdotN_), flux1(num_equations), flux2(num_equations){};
};

double RusanovFlux::riemannSolver(const Vector &state1, const Vector &state2,
                                  const Vector &nor, Vector &fluxN) {
  const double mcs1 = FdotN(state1, nor, flux1);
  const double mcs2 = FdotN(state2, nor, flux2);
  const double maxE = max(mcs1, mcs2);
  const double scale = nor.Norml2();
  fluxN = 0.0;
  fluxN.Add(0.5, flux1);
  fluxN.Add(0.5, flux2);
  fluxN.Add(-0.5 * maxE * scale, flux1);
  fluxN.Add(+0.5 * maxE * scale, flux2);
  return maxE;
}

void DivFlux::AssembleElementVector(const FiniteElement &elem,
                                    ElementTransformation &trans,
                                    const Vector &elfun, Vector &elvect) {
  const int dof = elem.GetDof();
  const int dim = elem.GetDim();
  const int num_equations = elfun.Size() / dof;

  shape.SetSize(dof);
  funval.SetSize(dof);
  dshape.SetSize(dof, dim);
  gshape.SetSize(dof, dim);
  Jadj.SetSize(dim, dim);

  elvect.SetSize(dof * num_equations);
  elvect = 0.0;

  // Solution coefficient elfun_mat(i,j) = i'th basis coefficient of j's
  // variable
  DenseMatrix elfun_mat(elfun.GetData(), dof, num_equations);
  // After apply operation, elvect_mat(i,j) = i'th trial function on j's
  // equation = -(Fⱼ(u), grad(phiᵢ))
  DenseMatrix elvect_mat(elvect.GetData(), dof, num_equations);

  int order = 2 * elem.GetOrder();
  if (order > 0) order += 3;  //

  const IntegrationRule *ir = &(IntRules.Get(elem.GetGeomType(), order));

  for (int i = 0; i < ir->GetNPoints(); i++) {
    // Element and integration point information
    const IntegrationPoint &ip = ir->IntPoint(i);
    trans.SetIntPoint(&ip);
    CalcAdjugate(trans.Jacobian(), Jadj);

    // Basis function and its derivative
    elem.CalcShape(trans.GetIntPoint(), shape);
    elem.CalcDShape(trans.GetIntPoint(), dshape);
    Mult(dshape, Jadj, gshape);
    // Current state from coefficient and shape function
    elfun_mat.MultTranspose(shape, funval);
    // maximum characteristic speed evaluated from flux
    const double mcs = F(funval, Fmat);
    AddMult_a_ABt(1.0, gshape, Fmat, elvect_mat);
  }
}

void NumericalFlux::AssembleFaceVector(const FiniteElement &el1,
                                       const FiniteElement &el2,
                                       FaceElementTransformations &Tr,
                                       const Vector &elfun, Vector &elvect) {
  // Compute the term <F.n(u),[w]> on the interior faces.
  const int dof1 = el1.GetDof();
  const int dof2 = el2.GetDof();
  const int num_equations = elfun.Size() / (dof1 + dof2);
  Vector nor(el1.GetDim());

  elvect.SetSize((dof1 + dof2) * num_equations);
  elvect = 0.0;

  DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equations);
  DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equations, dof2,
                         num_equations);

  DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equations);
  DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equations, dof2,
                          num_equations);

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

  Vector shape1(dof1), shape2(dof2);
  Vector funval1(num_equations), funval2(num_equations);
  Vector fluxN(num_equations);
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
    CalcOrtho(Tr.Jacobian(), nor);
    const double mcs = riemannSolver(funval1, funval2, nor, fluxN);

    // Update max char speed
    if (mcs > max_char_speed) {
      max_char_speed = mcs;
    }

    fluxN *= ip.weight;
    for (int k = 0; k < num_equations; k++) {
      for (int s = 0; s < dof1; s++) {
        elvect1_mat(s, k) -= fluxN(k) * shape1(s);
      }
      for (int s = 0; s < dof2; s++) {
        elvect2_mat(s, k) += fluxN(k) * shape2(s);
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////////
// Advection
Flux getAdvectionF(const Vector &b) {
  return [&](const Vector &state, DenseMatrix &flux) {
    const int dim = b.Size();
    const int vdim = state.Size();
    for (int j = 0; j < dim; j++) {
      for (int i = 0; i < vdim; i++) {
        flux(i, j) = b(j) * state(i);
      }
    }
    return b.Norml2();
  };
}
NormalFlux getAdvectionFdotN(const Vector &b) {
  return [&](const Vector &state, const Vector &nor, Vector &flux) {
    flux = 0.0;
    flux.Add(nor * b, state);
    return b.Norml2();
  };
}

class AdvectionEquation : public HyperbolicConservationLaws {
 public:
  AdvectionEquation(const int dim, const Vector &b_, FiniteElementSpace &sfes_,
                    FiniteElementSpace &dfes_, FiniteElementSpace &vfes_)
      : HyperbolicConservationLaws(dim + 2, dim, sfes_, dfes_, vfes_,
                                   getAdvectionF(b_), getAdvectionFdotN(b_)){};
};
/////////////////////////////////////////////////////////////////////////////
// Shallow Water
// Todo: Make flux for shallow water
// Flux getShallowWaterF() {}
// NormalFlux getShallowWaterFdotN() {}
/////////////////////////////////////////////////////////////////////////////
// Euler

/// @brief Compute Euler Flux
/// @param state current state s = (ρ, u, E)
/// @param flux flux, F(s) = [ρu, ρuuᵀ+pI, u(E+p)]ᵀ
/// @return characteristic speed, |u| + (γp/ρ)^(1/2)
Flux getEulerF(const double gamma_euler = 1.4) {
  return [=](const Vector &state, DenseMatrix &flux) {
    // Parse current state
    const int dim = state.Size() - 2;
    const double den = state(0);                 // density
    const Vector mom(state.GetData() + 1, dim);  // momentum
    const double E = state(1 + dim);             // energy

    // Pressure = (γ - 1)(E - ρ|u|^2/2)
    const double p = (gamma_euler - 1) * (E - mom.Norml2() / (2 * den));
    // Enthalpy
    const double H = (E + p) / den;
    for (int i = 0; i < dim; i++) {
      flux(0, i) = mom(i);
      const double vel_i = mom(i) / den;  // compute i'th velocity
      for (int j = 0; j < dim; i++) {
        flux(1 + j, i) = mom(j) * vel_i;
      }
      flux(1 + i, i) += p;
      flux(1 + dim, i) = mom(i) * H;
    }
    return mom.Norml2() / den + sqrt(gamma_euler * p / den);
  };
}

/// @brief Compute Euler flux dot normal
/// @param state current state s = (ρ, u, E)
/// @param nor outer normal (generally not a unit vector)
/// @param fluxN normal flux, F(s)n = [ρu⋅n, ρuᵀu⋅n+pn, u⋅n(E+p)]
/// @return characteristic speed, |u| + (γp/ρ)^(1/2)
NormalFlux getEulerFdotN(const double gamma_euler = 1.4) {
  return [=](const Vector &state, const Vector &nor, Vector &fluxN) {
    const int dim = state.Size() - 2;
    const double den = state(0);
    const Vector mom(state.GetData() + 1, dim);
    const double E = state(1 + dim);

    const double p = (gamma_euler - 1) * (E - mom.Norml2() / (2 * den));

    fluxN(0) = mom * nor;
    const double normal_vel = fluxN(0) / den;

    for (int i = 0; i < dim; i++) {
      fluxN(1 + i) = mom(i) * normal_vel + p * nor(i);
    }
    fluxN(dim) = normal_vel * (E + p);

    return mom.Norml2() / den + gamma_euler * p / den;
  };
}

HyperbolicConservationLaws::HyperbolicConservationLaws(
    const int num_equations_, const int dim_, FiniteElementSpace &sfes_,
    FiniteElementSpace &dfes_, FiniteElementSpace &vfes_, Flux F_,
    NormalFlux FudotN_, FluxType fluxname)
    : TimeDependentOperator(0),
      num_equations(num_equations_),
      dim(dim_),
      sfes(sfes_),
      dfes(dfes_),
      vfes(vfes_),
      FudotN(FudotN_),
      F(F_),
      Adiv(&dfes, &sfes) {
  switch (fluxname) {
    case FluxType::RUSANOV: {
      riemann_solver = new RusanovFlux(FudotN, num_equations);
      break;
    }
  }
}

class EulerSystem : public HyperbolicConservationLaws {
 public:
  EulerSystem(const int dim, FiniteElementSpace &sfes_,
              FiniteElementSpace &dfes_, FiniteElementSpace &vfes_,
              FluxType fluxname = FluxType::RUSANOV)
      : HyperbolicConservationLaws(dim + 2, dim, sfes_, dfes_, vfes_,
                                   getEulerF(), getEulerFdotN(), fluxname){};
};