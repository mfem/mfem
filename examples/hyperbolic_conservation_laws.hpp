//                  MFEM Example 18 - Serial/Parallel Shared Code

#include "mfem.hpp"

using namespace std;
using namespace mfem;

// Problem definition
// extern int problem;

// Maximum characteristic speed (updated by integrators)

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

// Element term: (F(u), grad v)
class HyperbolicElementFormIntegrator : public NonlinearFormIntegrator {
 private:
  const int num_equations;
  double *max_char_speed;
  const int IntOrderOffset;
  Vector shape;
  Vector state;
  DenseMatrix flux;
  DenseMatrix dshape;

 protected:
  virtual double ComputeFlux(const Vector &state, DenseMatrix &flux) = 0;

 public:
  HyperbolicElementFormIntegrator(const int dim, const int num_equations_,
                                  const int IntOrderOffset_ = 3)
      : NonlinearFormIntegrator(),
        num_equations(num_equations_),
        IntOrderOffset(IntOrderOffset_),
        state(num_equations_),
        flux(num_equations_, dim){};
  HyperbolicElementFormIntegrator(const int dim, const int num_equations_,
                                  const IntegrationRule *ir)
      : NonlinearFormIntegrator(ir),
        num_equations(num_equations_),
        IntOrderOffset(0),
        state(num_equations_),
        flux(num_equations_, dim){};

  const IntegrationRule &GetRule(const FiniteElement &el) {
    int order;
    order = 2 * el.GetOrder() + IntOrderOffset;
    return IntRules.Get(el.GetGeomType(), order);
  }
  void setMaxCharSpeed(double &max_char_speed_) {
    max_char_speed = &max_char_speed_;
  }

  virtual void AssembleElementVector(const FiniteElement &el,
                                     ElementTransformation &Tr,
                                     const Vector &elfun, Vector &elvect);
  virtual ~HyperbolicElementFormIntegrator() {}
};

// Interior face term: <hat{F}.n,[w]>
// where hat{F}.n is determined by NumericalFlux rsolver.
class HyperbolicFaceFormIntegrator : public NonlinearFormIntegrator {
 private:
  const int num_equations;
  double *max_char_speed;
  const int IntOrderOffset;
  NumericalFlux *rsolver;
  Vector shape1;
  Vector shape2;
  Vector state1;
  Vector state2;
  Vector flux1;
  Vector flux2;
  Vector nor;
  Vector fluxN;

 protected:
  virtual double ComputeFluxDotN(const Vector &state, const Vector &nor,
                                 Vector &flux) = 0;

 public:
  HyperbolicFaceFormIntegrator(NumericalFlux *rsolver_, const int dim,
                               const int num_equations_,
                               const int IntOrderOffset_ = 3)
      : NonlinearFormIntegrator(),
        num_equations(num_equations_),
        IntOrderOffset(IntOrderOffset_),
        rsolver(rsolver_),
        state1(num_equations_),
        state2(num_equations_),
        flux1(num_equations_),
        flux2(num_equations_),
        nor(dim),
        fluxN(num_equations_){};
  HyperbolicFaceFormIntegrator(NumericalFlux *rsolver_, const int dim,
                               const int num_equations_,
                               const IntegrationRule *ir)
      : NonlinearFormIntegrator(ir),
        num_equations(num_equations_),
        max_char_speed(),
        IntOrderOffset(0),
        rsolver(rsolver_),
        state1(num_equations_),
        state2(num_equations_),
        flux1(num_equations_),
        flux2(num_equations_),
        nor(dim),
        fluxN(num_equations_){};

  const IntegrationRule &GetRule(const FiniteElement &trial_fe,
                                 const FiniteElement &test_fe) {
    int order;
    order = trial_fe.GetOrder() + test_fe.GetOrder() + IntOrderOffset;
    return IntRules.Get(trial_fe.GetGeomType(), order);
  }
  void setMaxCharSpeed(double &max_char_speed_) {
    max_char_speed = &max_char_speed_;
  }

  virtual void AssembleFaceVector(const FiniteElement &el1,
                                  const FiniteElement &el2,
                                  FaceElementTransformations &Tr,
                                  const Vector &elfun, Vector &elvect);
  virtual ~HyperbolicFaceFormIntegrator() {}
};

// Base Hyperbolic conservation law class.
// This contains all methods needed except the flux function.
class DGHyperbolicConservationLaws : public TimeDependentOperator {
 private:
  const int dim;
  const int num_equations;
  // Vector finite element space containing conserved variables
  FiniteElementSpace &vfes;
  // Element integration form. Should contain ComputeFlux
  HyperbolicElementFormIntegrator &elementFormIntegrator;
  // Face integration form. Should contain ComputeFluxDotN and Riemann Solver
  HyperbolicFaceFormIntegrator &faceFormIntegrator;
  // Base Nonlinear Form
  NonlinearForm nonlinearForm;
  // element-wise inverse mass matrix
  std::vector<DenseMatrix> Me_inv;
  // global maximum characteristic speed. Updated by form integrators
  mutable double max_char_speed;
  mutable Vector z;

  // Compute element-wise inverse mass matrix
  void ComputeInvMass();

 public:
  // Constructor
  DGHyperbolicConservationLaws(
      FiniteElementSpace &vfes_,
      HyperbolicElementFormIntegrator &elementFormIntegrator_,
      HyperbolicFaceFormIntegrator &faceFormIntegrator_,
      const int num_equations_);
  // Apply M\(DIV F(U) + JUMP HAT{F}(U))
  virtual void Mult(const Vector &x, Vector &y) const;
  // Update operators when mesh and finite element spaces are updated
  void Update();
  inline double getMaxCharSpeed() { return max_char_speed; }
#ifdef MFEM_USE_MPI
  double getParMaxCharSpeed() {
    int myid = Mpi::WorldRank();
    MPI_Allreduce(&myid, &max_char_speed, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);
    return max_char_speed;
  }
#endif

  virtual ~DGHyperbolicConservationLaws() {}
};

//////////////////////////////////////////////////////////////////
///        HYPERBOLIC CONSERVATION LAWS IMPLEMENTATION         ///
//////////////////////////////////////////////////////////////////

// Implementation of class DGHyperbolicConservationLaws
DGHyperbolicConservationLaws::DGHyperbolicConservationLaws(
    FiniteElementSpace &vfes_,
    HyperbolicElementFormIntegrator &elementFormIntegrator_,
    HyperbolicFaceFormIntegrator &faceFormIntegrator_, const int num_equations_)
    : TimeDependentOperator(vfes_.GetNDofs() * num_equations_),
      dim(vfes_.GetFE(0)->GetDim()),
      num_equations(num_equations_),
      vfes(vfes_),
      elementFormIntegrator(elementFormIntegrator_),
      faceFormIntegrator(faceFormIntegrator_),
      nonlinearForm(&vfes),
      Me_inv(0),
      z(vfes_.GetNDofs() * num_equations_) {
  // Standard local assembly and inversion for energy mass matrices.
  ComputeInvMass();
  elementFormIntegrator.setMaxCharSpeed(max_char_speed);
  faceFormIntegrator.setMaxCharSpeed(max_char_speed);

  nonlinearForm.AddDomainIntegrator(&elementFormIntegrator);
  nonlinearForm.AddInteriorFaceIntegrator(&faceFormIntegrator);

  height = z.Size();
  width = z.Size();
}

void DGHyperbolicConservationLaws::ComputeInvMass() {
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

void DGHyperbolicConservationLaws::Update() {
  nonlinearForm.Update();
  ComputeInvMass();

  width = nonlinearForm.Width();
  height = nonlinearForm.Height();
  z.SetSize(height);
}

void DGHyperbolicConservationLaws::Mult(const Vector &x, Vector &y) const {
  // 0. Reset wavespeed computation before operator application.
  max_char_speed = 0.;
  // 1. Create the vector z with the face terms (F(u), grad v) - <F.n(u), [w]>.
  nonlinearForm.Mult(x, z);

  // 3. Multiply element-wise by the inverse mass matrices.
  Vector zval;
  Array<int> vdofs;
  //   const int dof = vfes.GetFE(0)->GetDof();
  DenseMatrix zmat, ymat;

  for (int i = 0; i < vfes.GetNE(); i++) {
    // Return the vdofs ordered byNODES
    vfes.GetElementVDofs(i, vdofs);
    z.GetSubVector(vdofs, zval);
    zmat.UseExternalData(zval.GetData(), vfes.GetFE(i)->GetDof(),
                         num_equations);
    ymat.SetSize(Me_inv[i].Height(), num_equations);
    mfem::Mult(Me_inv[i], zmat, ymat);
    y.SetSubVector(vdofs, ymat.GetData());
  }
}

//////////////////////////////////////////////////////////////////
///                      ELEMENT INTEGRATOR                    ///
//////////////////////////////////////////////////////////////////
void HyperbolicElementFormIntegrator::AssembleElementVector(
    const FiniteElement &el, ElementTransformation &Tr, const Vector &elfun,
    Vector &elvect) {
  const int dof = el.GetDof();

  shape.SetSize(dof);
  dshape.SetSize(dof, el.GetDim());

  elvect.SetSize(dof * num_equations);
  elvect = 0.0;

  const DenseMatrix elfun_mat(elfun.GetData(), dof, num_equations);
  DenseMatrix elvect_mat(elvect.GetData(), dof, num_equations);

  const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el);
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);
    Tr.SetIntPoint(&ip);

    el.CalcShape(ip, shape);
    el.CalcPhysDShape(Tr, dshape);

    elfun_mat.MultTranspose(shape, state);
    const double mcs = ComputeFlux(state, flux);
    *max_char_speed = mcs > *max_char_speed ? mcs : *max_char_speed;
    AddMult_a_ABt(ip.weight * Tr.Weight(), dshape, flux, elvect_mat);
  }
}
//////////////////////////////////////////////////////////////////
///                       FACE INTEGRATOR                      ///
//////////////////////////////////////////////////////////////////

void HyperbolicFaceFormIntegrator::AssembleFaceVector(
    const FiniteElement &el1, const FiniteElement &el2,
    FaceElementTransformations &Tr, const Vector &elfun, Vector &elvect) {
  // Compute the term <F.n(u),[w]> on the interior faces.
  const int dof1 = el1.GetDof();
  const int dof2 = el2.GetDof();

  shape1.SetSize(dof1);
  shape2.SetSize(dof2);

  elvect.SetSize((dof1 + dof2) * num_equations);
  elvect = 0.0;

  const DenseMatrix elfun1_mat(elfun.GetData(), dof1, num_equations);
  const DenseMatrix elfun2_mat(elfun.GetData() + dof1 * num_equations, dof2,
                               num_equations);

  DenseMatrix elvect1_mat(elvect.GetData(), dof1, num_equations);
  DenseMatrix elvect2_mat(elvect.GetData() + dof1 * num_equations, dof2,
                          num_equations);

  const IntegrationRule *ir = IntRule ? IntRule : &GetRule(el1, el2);
  for (int i = 0; i < ir->GetNPoints(); i++) {
    const IntegrationPoint &ip = ir->IntPoint(i);

    Tr.SetAllIntPoints(&ip);  // set face and element int. points

    // Calculate basis functions on both elements at the face
    el1.CalcShape(Tr.GetElement1IntPoint(), shape1);
    el2.CalcShape(Tr.GetElement2IntPoint(), shape2);

    // Interpolate elfun at the point
    elfun1_mat.MultTranspose(shape1, state1);
    elfun2_mat.MultTranspose(shape2, state2);

    // Get the normal vector and the flux on the face
    if (nor.Size() == 1)
      nor(0) = (Tr.GetElement1IntPoint().x - 0.5) * 2.0;
    else
      CalcOrtho(Tr.Jacobian(), nor);

    const double mcs = max(ComputeFluxDotN(state1, nor, flux1),
                           ComputeFluxDotN(state2, nor, flux2));
    rsolver->Eval(state1, state2, flux1, flux2, mcs, nor, fluxN);

    // Update max char speed
    *max_char_speed = mcs > *max_char_speed ? mcs : *max_char_speed;

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

//////////////////////////////////////////////////////////////////
///                        EULER SYSTEM                        ///
//////////////////////////////////////////////////////////////////
class EulerElementFormIntegrator : public HyperbolicElementFormIntegrator {
 private:
  const double specific_heat_ratio;
  const double gas_constant;
  double ComputeFlux(const Vector &state, DenseMatrix &flux) {
    const int dim = state.Size() - 2;
    const double den = state(0);
    const Vector den_vel(state.GetData() + 1, dim);
    const double den_energy = state(1 + dim);

    const double pres = (specific_heat_ratio - 1.0) *
                        (den_energy - 0.5 * (den_vel * den_vel) / den);

    MFEM_ASSERT(den >= 0, "Negative Density");
    MFEM_ASSERT(pres >= 0, "Negative Pressure");
    MFEM_ASSERT(den_energy >= 0, "Negative Energy");

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

 public:
  EulerElementFormIntegrator(const int dim, const int IntOrderOffset_ = 3,
                             const double specific_heat_ratio_ = 1.4,
                             const double gas_constant_ = 1.0)
      : HyperbolicElementFormIntegrator(dim, dim + 2, IntOrderOffset_),
        specific_heat_ratio(specific_heat_ratio_),
        gas_constant(gas_constant_) {}

  EulerElementFormIntegrator(const int dim, const IntegrationRule *ir,
                             const double specific_heat_ratio_ = 1.4,
                             const double gas_constant_ = 1.0)
      : HyperbolicElementFormIntegrator(dim, dim + 2, ir),
        specific_heat_ratio(specific_heat_ratio_),
        gas_constant(gas_constant_) {}
};

// Euler System face integration. Overload ComputeFluxDotN
class EulerFaceFormIntegrator : public HyperbolicFaceFormIntegrator {
 private:
  const double specific_heat_ratio;
  const double gas_constant;
  double ComputeFluxDotN(const Vector &state, const Vector &nor,
                         Vector &fluxN) {
    // NOTE: nor in general is not a unit normal
    const int dim = nor.Size();
    const double den = state(0);
    const Vector den_vel(state.GetData() + 1, dim);
    const double den_energy = state(1 + dim);

    const double pres = (specific_heat_ratio - 1.0) *
                        (den_energy - 0.5 * (den_vel * den_vel) / den);

    MFEM_ASSERT(den >= 0, "Negative Density");
    MFEM_ASSERT(pres >= 0, "Negative Pressure");
    MFEM_ASSERT(den_energy >= 0, "Negative Energy");

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

 public:
  EulerFaceFormIntegrator(NumericalFlux *rsolver_, const int dim,
                          const int IntOrderOffset_ = 3,
                          const double specific_heat_ratio_ = 1.4,
                          const double gas_constant_ = 1.0)
      : HyperbolicFaceFormIntegrator(rsolver_, dim, dim + 2, IntOrderOffset_),
        specific_heat_ratio(specific_heat_ratio_),
        gas_constant(gas_constant_){};
  EulerFaceFormIntegrator(NumericalFlux *rsolver_, const int dim,
                          const int num_equations_, const IntegrationRule *ir,
                          const double specific_heat_ratio_ = 1.4,
                          const double gas_constant_ = 1.0)
      : HyperbolicFaceFormIntegrator(rsolver_, dim, dim + 2, ir),
        specific_heat_ratio(specific_heat_ratio_),
        gas_constant(gas_constant_){};
};

//////////////////////////////////////////////////////////////////
///                      BURGERS EQUATION                      ///
//////////////////////////////////////////////////////////////////

// Burgers equation main class. Overload ComputeFlux
class BurgersElementFormIntegrator : public HyperbolicElementFormIntegrator {
 private:
  double ComputeFlux(const Vector &state, DenseMatrix &flux) {
    flux = state * state * 0.5;
    return abs(state(0));
  };

 public:
  BurgersElementFormIntegrator(const int dim, const int IntOrderOffset_ = 3)
      : HyperbolicElementFormIntegrator(dim, 1, IntOrderOffset_){};
  BurgersElementFormIntegrator(const int dim, const IntegrationRule *ir)
      : HyperbolicElementFormIntegrator(dim, 1, ir){};
};

// Burgers equation face integration. Overload ComputeFluxDotN
class BurgersFaceFormIntegrator : public HyperbolicFaceFormIntegrator {
 private:
  double ComputeFluxDotN(const Vector &state, const Vector &nor,
                         Vector &fluxN) {
    fluxN = nor.Sum() * (state * state) * 0.5;
    return abs(state(0));
  };

 public:
  BurgersFaceFormIntegrator(NumericalFlux *rsolver_, const int dim,
                            const int IntOrderOffset_ = 3)
      : HyperbolicFaceFormIntegrator(rsolver_, dim, 1, IntOrderOffset_){};
  BurgersFaceFormIntegrator(NumericalFlux *rsolver_, const int dim,
                            const IntegrationRule *ir)
      : HyperbolicFaceFormIntegrator(rsolver_, dim, 1, ir){};
};

// //////////////////////////////////////////////////////////////////
// ///                        SHALLOW WATER                       ///
// //////////////////////////////////////////////////////////////////

// // Burgers equation main class. Overload ComputeFlux
// class ShallowWater : public DGHyperbolicConservationLaws {
//  private:
//   const double g;
//   double ComputeFlux(const Vector &state, const int dim,
//                      DenseMatrix &flux) const {
//     const double height = state(0);
//     const Vector h_vel(state.GetData() + 1, dim);

//     const double energy = 0.5 * g * (height * height);

//     MFEM_ASSERT(height >= 0, "Negative Height");

//     for (int d = 0; d < dim; d++) {
//       flux(0, d) = h_vel(d);
//       for (int i = 0; i < dim; i++) {
//         flux(1 + i, d) = h_vel(i) * h_vel(d) / height;
//       }
//       flux(1 + d, d) += energy;
//     }

//     const double sound = sqrt(g * height);
//     const double vel = sqrt(h_vel * h_vel) / height;

//     return vel + sound;
//   };

//  public:
//   ShallowWater(FiniteElementSpace &vfes_, MixedBilinearForm &divA_,
//                HyperbolicFaceFormIntegrator &faceForm_, const double g_
//                = 9.81)
//       : DGHyperbolicConservationLaws(vfes_, divA_, faceForm_,
//                                      1 + vfes_.GetFE(0)->GetDim()),
//         g(g_){};
// };

// // Burgers equation face integration. Overload ComputeFluxDotN
// class ShallowWaterFaceIntegrator : public HyperbolicFaceFormIntegrator {
//  private:
//   const double g;
//   double ComputeFluxDotN(const Vector &state, const Vector &nor,
//                          Vector &fluxN) {
//     const int dim = nor.Size();
//     const double height = state(0);
//     const Vector h_vel(state.GetData() + 1, dim);

//     const double energy = 0.5 * g * (height * height);

//     MFEM_ASSERT(height >= 0, "Negative Height");
//     fluxN(0) = h_vel * nor;
//     const double normal_vel = fluxN(0) / height;
//     for (int i = 0; i < dim; i++) {
//       fluxN(1 + i) = normal_vel * h_vel(i) + energy * nor(i);
//     }

//     const double sound = sqrt(g * height);
//     const double vel = sqrt(h_vel * h_vel) / height;

//     return vel + sound;
//   };

//  public:
//   ShallowWaterFaceIntegrator(NumericalFlux *rsolver_, const int dim_,
//                              const double g_ = 9.81)
//       : HyperbolicFaceFormIntegrator(rsolver_, dim_, 1 + dim_), g(g_){};
// };