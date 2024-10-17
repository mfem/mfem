#ifndef TOPOPT_HPP
#define TOPOPT_HPP

#include "mfem.hpp"
#include "funs.hpp"
#include "linear_solver.hpp"

namespace mfem {

class GLVis {
    Array<socketstream *> sockets;
    Array<GridFunction *> gfs;
    Array<Mesh *> meshes;
    bool parallel;
    const char *hostname;
    const int port;
    bool secure;

public:
#ifdef MFEM_USE_GNUTLS
    static const bool secure_default = true;
#else
    static const bool secure_default = false;
#endif
    GLVis(const char hostname[], int port, bool parallel,
          bool secure = secure_default)
        : sockets(0), gfs(0), meshes(0), parallel(parallel), hostname(hostname),
          port(port), secure(secure_default) {}

    ~GLVis() {
        for (socketstream *socket : sockets) {
            if (socket) {
                delete socket;
            }
        }
    }

    void Append(GridFunction &gf, const char window_title[] = nullptr,
                const char keys[] = nullptr);
    void Update();
    socketstream &GetSocket(int i) {
        return *sockets[i];
    }
};

void ProjectCoefficient(GridFunction &x, Coefficient &coeff, int attribute);

// Hooke's Law
// -kx\cdot d = (f_ext + sigma n)\cdot d
class DirectionalHookesLawBdrIntegrator : public BilinearFormIntegrator {
    // properties
private:
    VectorCoefficient *direction;
    real_t k;
    mutable Vector shape;

protected:
public:
    // methods
private:
protected:
public:
    DirectionalHookesLawBdrIntegrator(const real_t k,
                                      VectorCoefficient *direction)
        : k(k), direction(direction) {}
    void AssembleFaceMatrix(const FiniteElement &el1, const FiniteElement &el2,
                            FaceElementTransformations &Trans,
                            DenseMatrix &elmat) override;
};

class DesignDensity {
private:
    FiniteElementSpace &fes_control;
    const real_t tot_vol;
    const real_t min_vol;
    const real_t max_vol;
    int void_attr_id;
    int solid_attr_id;
    LegendreEntropy *entropy;
    std::unique_ptr<GridFunction> zero;

public:
    DesignDensity(FiniteElementSpace &fes_control, const real_t tot_vol,
                  const real_t min_vol, const real_t max_vol,
                  LegendreEntropy *entropy = nullptr);

    void SetSolidAttr(int attr) {
        solid_attr_id = attr;
    }
    void SetVoidAttr(int attr) {
        void_attr_id = attr;
    }

    real_t ApplyVolumeProjection(GridFunction &x, bool use_entropy);
    real_t ComputeVolume(GridFunction &x);
    bool hasEntropy() {
        return entropy ? true : false;
    }
};

class StrainEnergyDensityCoefficient : public Coefficient {
protected:
    Coefficient &lambda;
    Coefficient &mu;
    Coefficient &der_simp_cf;
    GridFunction &state_gf;    // displacement
    GridFunction *adjstate_gf; // adjoint displacement
    DenseMatrix grad;          // auxiliary matrix, used in Eval
    DenseMatrix adjgrad;       // auxiliary matrix, used in Eval
    Array<Coefficient*> owned_coeffs;
    std::unique_ptr<ElasticityIntegrator> energy;

public:
    StrainEnergyDensityCoefficient(Coefficient &lambda, Coefficient &mu,
                                   Coefficient &der_simp_cf,
                                   GridFunction &state_gf,
                                   GridFunction *adju_gf = nullptr)
        : lambda(lambda), mu(mu), der_simp_cf(der_simp_cf), state_gf(state_gf),
    adjstate_gf(adju_gf) {
    auto neg_der_simp_cf = new ProductCoefficient(-1.0, der_simp_cf);
    auto neg_der_simp_lambda_cf = new ProductCoefficient(*neg_der_simp_cf, lambda);
    auto neg_der_simp_mu_cf = new ProductCoefficient(*neg_der_simp_cf, mu);
    energy.reset(new ElasticityIntegrator(*neg_der_simp_lambda_cf, *neg_der_simp_mu_cf));
    owned_coeffs.Append(neg_der_simp_cf);
    owned_coeffs.Append(neg_der_simp_lambda_cf);
    owned_coeffs.Append(neg_der_simp_mu_cf);
  }
  ~StrainEnergyDensityCoefficient()
  {
    for(auto cf : owned_coeffs) { delete cf; }
  }

    void SetAdjState(GridFunction &adj) {
        adjstate_gf = &adj;
    }

    real_t Eval(ElementTransformation &T, const IntegrationPoint &ip) override;
};

class DensityBasedTopOpt {
private:
    DesignDensity &density;
    GridFunction &control_gf;
    GridFunction &grad_control;
    HelmholtzFilter &filter;
    GridFunction &filter_gf;
    GridFunction &grad_filter;
    GridFunctionCoefficient grad_filter_cf;
    ElasticityProblem &elasticity;
    GridFunction &state_gf;
    std::unique_ptr<GridFunction> adj_state_gf;
    LinearForm &obj;

    std::unique_ptr<L2Projection> L2projector;
    real_t objval;
    real_t current_volume;

   bool enforce_volume_constraint;

public:
    DensityBasedTopOpt(DesignDensity &density, GridFunction &gf_control,
                       GridFunction &grad_control, HelmholtzFilter &filter,
                       GridFunction &gf_filter, GridFunction &grad_filter,
                       ElasticityProblem &elasticity, GridFunction &gf_state,
                       bool enforce_volume_constraint=true);

    real_t GetCurrentVolume() {
        return current_volume;
    }
    real_t GetCurrentObjectValue() {
        return objval;
    }
    GridFunction &GetAdjState() {
        return *adj_state_gf;
    }

    real_t Eval();

    void UpdateGradient();
};

} // end of namespace mfem
#endif