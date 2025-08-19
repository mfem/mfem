#ifndef EX41_HPP
#define EX41_HPP

#include <cmath>
#include "mfem.hpp"

using namespace mfem;

/// This class defines two analytic reference solutions for poroelasticity
/// problems in (0,1)^2 or (0,1)^3. The first problem is designed to test
/// the convergence with h refinement, while the second problem is designed
/// to test the convergence regarding the time step size. The solutions are
/// defined by the following equations:
/// (P1) u_j(x,t) = sin(M_PI * t) * Π_(i=1)^d (x_i * (1 - x_i))
/// (P1) p(x,t) = t * Π_(i=1)^d sin(2 * M_PI * x_i)
/// (P2) u_j(x,t) = t^12 * Π_(i=1)^d (x_i * (1 - x_i))
/// (P2) p(x,t) = t^12 * Π_(i=1)^d sin(2 * M_PI * x_i)
/// where d is the dimension of the problem (2 or 3).
/// All further quantities are derived from these equations, i.e., the darcy
/// velocity and the right hand side of the equations. They are computed with
/// (2D) K = ((3,1), (1,2)) and
/// (3D) K = ((3,1,0), (1,2,0), (0,0,1)).
/// Note that the solutions are chosen to be compatible with the boundary
/// conditions u = 0 and p = 0. The examples (u,p) are taken from the paper:
/// "Wheeler, M., Xue, G. & Yotov, I. Coupling multipoint flux mixed finite
// element methods with continuous Galerkin methods for poroelasticity.
/// Comput Geosci 18, 57–75 (2014). https://doi.org/10.1007/s10596-013-9382-y"
/// with z,f,s computed with sympy.
/// The paper uses the following parameters fore testing the convergence:
/// (P1) E = 1.0, nu = 0.2, alpha = 1.0, c0 = 0.1
/// (P2) E = 1.0, nu = 0.2, alpha = 10.0, c0 = 100.0
class PoroelasticityReferenceSolution {
 public:
  /// @brief Constructor
  /// @param dim Dimension of the problem (2 or 3)
  /// @param alpha Biot-coefficient
  /// @param nu Poisson's ratio
  /// @param E Young's modulus
  /// @param c0 Storage coefficient
  /// @param problem Problem number (1 or 2)
  PoroelasticityReferenceSolution(int dim, double alpha, double nu, double E,
                                  double c0, int problem)
      : dim_(dim), alpha_(alpha), nu_(nu), E_(E), c0_(c0), problem_(problem) {
    if (problem_ != 1 && problem_ != 2) {
      MFEM_ABORT("Invalid problem number. Must be 1 or 2.");
    }
  }

  // Analytic solution for the displacement field u(x,t)
  void AnalyticDisplacementSolution(const Vector &x, real_t t, Vector &u) {
    if (problem_ == 1) {
      AnalyticDisplacementSolutionP1(x, t, u);
    } else {
      AnalyticDisplacementSolutionP2(x, t, u);
    }
  };

  // Analytic solution for the Darcy velocity field z(x,t)
  void AnalyticDarcySolution(const Vector &x, real_t t, Vector &u) {
    if (problem_ == 1) {
      AnalyticDarcySolutionP1(x, t, u);
    } else {
      AnalyticDarcySolutionP2(x, t, u);
    }
  };

  // Analytic solution for the pressure field p(x,t)
  real_t AnalyticPressureSolution(const Vector &x, real_t t) {
    if (problem_ == 1) {
      return AnalyticPressureSolutionP1(x, t);
    } else {
      return AnalyticPressureSolutionP2(x, t);
    }
  };

  // Volume force f(x,t) in the displacement equation
  void VolumeForce(const Vector &x, real_t t, Vector &u) {
    if (problem_ == 1) {
      VolumeForceP1(x, t, u);
    } else {
      VolumeForceP2(x, t, u);
    }
  };

  // Source term s(x,t) in the pressure equation
  real_t SourceTerm(const Vector &x, real_t t) {
    if (problem_ == 1) {
      return SourcesAndSinksP1(x, t);
    } else {
      return SourcesAndSinksP2(x, t);
    }
  };

 private:
  int dim_;       // Dimension of the problem
  double alpha_;  // Biot-Wheeler coefficient
  double nu_;     // Poisson's ratio
  double E_;      // Young's modulus
  double c0_;     // Source term coefficient
  int problem_;   // Problem number (1 or 2)

  /// @brief Analytic solution for the displacement field u(x,t)
  /// @param x x in D = [0,1]^2
  /// @param t time parameter t
  /// @param u u in R^2 (solution vector field)
  void AnalyticDisplacementSolutionP1(const Vector &x, real_t t, Vector &u) {
    if (dim_ == 2) {
      u(0) = u1_P1(x(0), x(1), t);
      u(1) = u2_P1(x(0), x(1), t);
    }
    if (dim_ == 3) {
      u(0) = u1_P1_3D(x(0), x(1), x(2), t);
      u(1) = u2_P1_3D(x(0), x(1), x(2), t);
      u(2) = u3_P1_3D(x(0), x(1), x(2), t);
    }
  };
  void AnalyticDisplacementSolutionP2(const Vector &x, real_t t, Vector &u) {
    if (dim_ == 2) {
      u(0) = u1_P2(x(0), x(1), t);
      u(1) = u2_P2(x(0), x(1), t);
    }
    if (dim_ == 3) {
      u(0) = u1_P2_3D(x(0), x(1), x(2), t);
      u(1) = u2_P2_3D(x(0), x(1), x(2), t);
      u(2) = u3_P2_3D(x(0), x(1), x(2), t);
    }
  };

  /// @brief Analytic solution for the Darcy velocity field z(x,t)
  /// @param x x in D = [0,1]^2
  /// @param t time parameter t
  /// @param u z in R^2 (solution vector field)
  void AnalyticDarcySolutionP1(const Vector &x, real_t t, Vector &u) {
    if (dim_ == 2) {
      u(0) = z1_P1(x(0), x(1), t);
      u(1) = z2_P1(x(0), x(1), t);
    }
    if (dim_ == 3) {
      u(0) = z1_P1_3D(x(0), x(1), x(2), t);
      u(1) = z2_P1_3D(x(0), x(1), x(2), t);
      u(2) = z3_P1_3D(x(0), x(1), x(2), t);
    }
  };
  void AnalyticDarcySolutionP2(const Vector &x, real_t t, Vector &u) {
    if (dim_ == 2) {
      u(0) = z1_P2(x(0), x(1), t);
      u(1) = z2_P2(x(0), x(1), t);
    }
    if (dim_ == 3) {
      u(0) = z1_P2_3D(x(0), x(1), x(2), t);
      u(1) = z2_P2_3D(x(0), x(1), x(2), t);
      u(2) = z3_P2_3D(x(0), x(1), x(2), t);
    }
  };

  /// @brief Analytic solution for the pressure field p(x,t)
  /// @param x x in D = [0,1]^2
  /// @param t time parameter t
  /// @return p p in R (solution scalar field)
  real_t AnalyticPressureSolutionP1(const Vector &x, real_t t) {
    if (dim_ == 2) {
      return p_P1(x(0), x(1), t);
    }
    if (dim_ == 3) {
      return p_P1_3D(x(0), x(1), x(2), t);
    }
    return 0.0;
  };
  real_t AnalyticPressureSolutionP2(const Vector &x, real_t t) {
    if (dim_ == 2) {
      return p_P2(x(0), x(1), t);
    }
    if (dim_ == 3) {
      return p_P2_3D(x(0), x(1), x(2), t);
    }
    return 0.0;
  };

  /// @brief Volume force f(x,t) in the displacement equation
  /// @param x x in D = [0,1]^2
  /// @param t time parameter t
  /// @param u f in R^2 (force vector field)
  void VolumeForceP1(const Vector &x, real_t t, Vector &u) {
    if (dim_ == 2) {
      u(0) = f1_P1(x(0), x(1), t);
      u(1) = f2_P1(x(0), x(1), t);
    }
    if (dim_ == 3) {
      u(0) = f1_P1_3D(x(0), x(1), x(2), t);
      u(1) = f2_P1_3D(x(0), x(1), x(2), t);
      u(2) = f3_P1_3D(x(0), x(1), x(2), t);
    }
  };
  void VolumeForceP2(const Vector &x, real_t t, Vector &u) {
    if (dim_ == 2) {
      u(0) = f1_P2(x(0), x(1), t);
      u(1) = f2_P2(x(0), x(1), t);
    }
    if (dim_ == 3) {
      u(0) = f1_P2_3D(x(0), x(1), x(2), t);
      u(1) = f2_P2_3D(x(0), x(1), x(2), t);
      u(2) = f3_P2_3D(x(0), x(1), x(2), t);
    }
  };

  /// @brief Source term s(x,t) in the pressure equation
  /// @param x x in D = [0,1]^2
  /// @param t time parameter t
  /// @return s s in R (source scalar field)
  real_t SourcesAndSinksP1(const Vector &x, real_t t) {
    if (dim_ == 2) {
      return s_func_P1(x(0), x(1), t);
    }
    if (dim_ == 3) {
      return s_func_P1_3D(x(0), x(1), x(2), t);
    }
    return 0.0;
  };
  real_t SourcesAndSinksP2(const Vector &x, real_t t) {
    if (dim_ == 2) {
      return s_func_P2(x(0), x(1), t);
    }
    if (dim_ == 3) {
      return s_func_P2_3D(x(0), x(1), x(2), t);
    }
    return 0.0;
  };

  /// -----------------------------------------------------------------------
  /// @brief Analytic solutions
  /// -----------------------------------------------------------------------

  double z1_P1(double x, double y, double t) {
    return 2 * M_PI * t *
           (sin(M_PI * (2 * x - 2 * y)) - 2 * sin(M_PI * (2 * x + 2 * y)));
  }

  double z2_P1(double x, double y, double t) {
    return -M_PI * t *
           (sin(M_PI * (2 * x - 2 * y)) + 3 * sin(M_PI * (2 * x + 2 * y)));
  }

  double u1_P1(double x, double y, double t) {
    return x * y * (1 - x) * (1 - y) * sin(M_PI * t);
  }

  double u2_P1(double x, double y, double t) {
    return x * y * (1 - x) * (1 - y) * sin(M_PI * t);
  }

  double p_P1(double x, double y, double t) {
    return t * sin(2 * M_PI * x) * sin(2 * M_PI * y);
  }

  double f1_P1(double x, double y, double t) {
    return (E_ * nu_ *
                (x * y + x * (y - 1) + y * (x - 1) + 2 * y * (y - 1) +
                 (x - 1) * (y - 1)) *
                sin(M_PI * t) -
            1.0 / 2.0 * E_ * (2 * nu_ - 1) *
                (x * y + 2 * x * (x - 1) + x * (y - 1) + y * (x - 1) +
                 4 * y * (y - 1) + (x - 1) * (y - 1)) *
                sin(M_PI * t) +
            2 * M_PI * alpha_ * t * (nu_ + 1) * (2 * nu_ - 1) *
                sin(2 * M_PI * y) * cos(2 * M_PI * x)) /
           ((nu_ + 1) * (2 * nu_ - 1));
  }

  double f2_P1(double x, double y, double t) {
    return (E_ * nu_ *
                (x * y + 2 * x * (x - 1) + x * (y - 1) + y * (x - 1) +
                 (x - 1) * (y - 1)) *
                sin(M_PI * t) -
            1.0 / 2.0 * E_ * (2 * nu_ - 1) *
                (x * y + 4 * x * (x - 1) + x * (y - 1) + y * (x - 1) +
                 2 * y * (y - 1) + (x - 1) * (y - 1)) *
                sin(M_PI * t) +
            2 * M_PI * alpha_ * t * (nu_ + 1) * (2 * nu_ - 1) *
                sin(2 * M_PI * x) * cos(2 * M_PI * y)) /
           ((nu_ + 1) * (2 * nu_ - 1));
  }

  double s_func_P1(double x, double y, double t) {
    return M_PI * alpha_ *
               (x * y * (x - 1) + x * y * (y - 1) + x * (x - 1) * (y - 1) +
                y * (x - 1) * (y - 1)) *
               cos(M_PI * t) +
           c0_ * sin(2 * M_PI * x) * sin(2 * M_PI * y) +
           20 * pow(M_PI, 2) * t * sin(2 * M_PI * x) * sin(2 * M_PI * y) -
           8 * pow(M_PI, 2) * t * cos(2 * M_PI * x) * cos(2 * M_PI * y);
  }

  double z1_P2(double x, double y, double t) {
    return 2 * M_PI * pow(t, 12) *
           (sin(M_PI * (2 * x - 2 * y)) - 2 * sin(M_PI * (2 * x + 2 * y)));
  }

  double z2_P2(double x, double y, double t) {
    return -M_PI * pow(t, 12) *
           (sin(M_PI * (2 * x - 2 * y)) + 3 * sin(M_PI * (2 * x + 2 * y)));
  }

  double u1_P2(double x, double y, double t) {
    return pow(t, 12) * x * y * (1 - x) * (1 - y);
  }

  double u2_P2(double x, double y, double t) {
    return pow(t, 12) * x * y * (1 - x) * (1 - y);
  }

  double p_P2(double x, double y, double t) {
    return pow(t, 12) * sin(2 * M_PI * x) * sin(2 * M_PI * y);
  }

  double f1_P2(double x, double y, double t) {
    return pow(t, 12) *
           (E_ * nu_ *
                (x * y + x * (y - 1) + y * (x - 1) + 2 * y * (y - 1) +
                 (x - 1) * (y - 1)) -
            1.0 / 2.0 * E_ * (2 * nu_ - 1) *
                (x * y + 2 * x * (x - 1) + x * (y - 1) + y * (x - 1) +
                 4 * y * (y - 1) + (x - 1) * (y - 1)) +
            2 * M_PI * alpha_ * (nu_ + 1) * (2 * nu_ - 1) * sin(2 * M_PI * y) *
                cos(2 * M_PI * x)) /
           ((nu_ + 1) * (2 * nu_ - 1));
  }

  double f2_P2(double x, double y, double t) {
    return pow(t, 12) *
           (E_ * nu_ *
                (x * y + 2 * x * (x - 1) + x * (y - 1) + y * (x - 1) +
                 (x - 1) * (y - 1)) -
            1.0 / 2.0 * E_ * (2 * nu_ - 1) *
                (x * y + 4 * x * (x - 1) + x * (y - 1) + y * (x - 1) +
                 2 * y * (y - 1) + (x - 1) * (y - 1)) +
            2 * M_PI * alpha_ * (nu_ + 1) * (2 * nu_ - 1) * sin(2 * M_PI * x) *
                cos(2 * M_PI * y)) /
           ((nu_ + 1) * (2 * nu_ - 1));
  }

  double s_func_P2(double x, double y, double t) {
    return pow(t, 11) *
           (24 * alpha_ * pow(x, 2) * y - 12 * alpha_ * pow(x, 2) +
            24 * alpha_ * x * pow(y, 2) - 48 * alpha_ * x * y +
            12 * alpha_ * x - 12 * alpha_ * pow(y, 2) + 12 * alpha_ * y +
            6 * c0_ * cos(M_PI * (2 * x - 2 * y)) -
            6 * c0_ * cos(M_PI * (2 * x + 2 * y)) +
            6 * pow(M_PI, 2) * t * cos(M_PI * (2 * x - 2 * y)) -
            14 * pow(M_PI, 2) * t * cos(M_PI * (2 * x + 2 * y)));
  }

  double z1_P1_3D(double x, double y, double z, double t) {
    return 2 * M_PI * t *
           (sin(M_PI * (2 * x - 2 * y)) - 2 * sin(M_PI * (2 * x + 2 * y))) *
           sin(2 * M_PI * z);
  }

  double z2_P1_3D(double x, double y, double z, double t) {
    return -M_PI * t *
           (sin(M_PI * (2 * x - 2 * y)) + 3 * sin(M_PI * (2 * x + 2 * y))) *
           sin(2 * M_PI * z);
  }

  double z3_P1_3D(double x, double y, double z, double t) {
    return -2 * M_PI * t * sin(2 * M_PI * x) * sin(2 * M_PI * y) *
           cos(2 * M_PI * z);
  }

  double u1_P1_3D(double x, double y, double z, double t) {
    return z * x * y * (1 - z) * (1 - x) * (1 - y) * sin(M_PI * t);
  }

  double u2_P1_3D(double x, double y, double z, double t) {
    return z * x * y * (1 - z) * (1 - x) * (1 - y) * sin(M_PI * t);
  }

  double u3_P1_3D(double x, double y, double z, double t) {
    return z * x * y * (1 - z) * (1 - x) * (1 - y) * sin(M_PI * t);
  }

  double p_P1_3D(double x, double y, double z, double t) {
    return t * sin(2 * M_PI * z) * sin(2 * M_PI * x) * sin(2 * M_PI * y);
  }

  double f1_P1_3D(double x, double y, double z, double t) {
    return (-E_ * nu_ *
                (z * x * y * (z - 1) + z * x * y * (y - 1) +
                 z * x * (z - 1) * (y - 1) + z * y * (z - 1) * (x - 1) +
                 2 * z * y * (z - 1) * (y - 1) + z * y * (x - 1) * (y - 1) +
                 z * (z - 1) * (x - 1) * (y - 1) + x * y * (z - 1) * (y - 1) +
                 y * (z - 1) * (x - 1) * (y - 1)) *
                sin(M_PI * t) +
            (1.0 / 2.0) * E_ * (2 * nu_ - 1) *
                (4 * z * y * (z - 1) * (y - 1) +
                 z * (z - 1) *
                     (x * y + 2 * x * (x - 1) + x * (y - 1) + y * (x - 1) +
                      (x - 1) * (y - 1)) +
                 y * (y - 1) *
                     (z * x + z * (x - 1) + x * (z - 1) + 2 * x * (x - 1) +
                      (z - 1) * (x - 1))) *
                sin(M_PI * t) +
            2 * M_PI * alpha_ * t * (nu_ + 1) * (2 * nu_ - 1) *
                sin(2 * M_PI * z) * sin(2 * M_PI * y) * cos(2 * M_PI * x)) /
           ((nu_ + 1) * (2 * nu_ - 1));
  }

  double f2_P1_3D(double x, double y, double z, double t) {
    return (-E_ * nu_ *
                (z * x * y * (z - 1) + z * x * y * (x - 1) +
                 2 * z * x * (z - 1) * (x - 1) + z * x * (z - 1) * (y - 1) +
                 z * x * (x - 1) * (y - 1) + z * y * (z - 1) * (x - 1) +
                 z * (z - 1) * (x - 1) * (y - 1) + x * y * (z - 1) * (x - 1) +
                 x * (z - 1) * (x - 1) * (y - 1)) *
                sin(M_PI * t) +
            (1.0 / 2.0) * E_ * (2 * nu_ - 1) *
                (4 * z * x * (z - 1) * (x - 1) +
                 z * (z - 1) *
                     (x * y + x * (y - 1) + y * (x - 1) + 2 * y * (y - 1) +
                      (x - 1) * (y - 1)) +
                 x * (x - 1) *
                     (z * y + z * (y - 1) + y * (z - 1) + 2 * y * (y - 1) +
                      (z - 1) * (y - 1))) *
                sin(M_PI * t) +
            2 * M_PI * alpha_ * t * (nu_ + 1) * (2 * nu_ - 1) *
                sin(2 * M_PI * z) * sin(2 * M_PI * x) * cos(2 * M_PI * y)) /
           ((nu_ + 1) * (2 * nu_ - 1));
  }

  double f3_P1_3D(double x, double y, double z, double t) {
    return (-E_ * nu_ *
                (z * x * y * (x - 1) + z * x * y * (y - 1) +
                 z * x * (x - 1) * (y - 1) + z * y * (x - 1) * (y - 1) +
                 x * y * (z - 1) * (x - 1) + x * y * (z - 1) * (y - 1) +
                 2 * x * y * (x - 1) * (y - 1) +
                 x * (z - 1) * (x - 1) * (y - 1) +
                 y * (z - 1) * (x - 1) * (y - 1)) *
                sin(M_PI * t) +
            (1.0 / 2.0) * E_ * (2 * nu_ - 1) *
                (4 * x * y * (x - 1) * (y - 1) +
                 x * (x - 1) *
                     (z * y + 2 * z * (z - 1) + z * (y - 1) + y * (z - 1) +
                      (z - 1) * (y - 1)) +
                 y * (y - 1) *
                     (z * x + 2 * z * (z - 1) + z * (x - 1) + x * (z - 1) +
                      (z - 1) * (x - 1))) *
                sin(M_PI * t) +
            2 * M_PI * alpha_ * t * (nu_ + 1) * (2 * nu_ - 1) *
                sin(2 * M_PI * x) * sin(2 * M_PI * y) * cos(2 * M_PI * z)) /
           ((nu_ + 1) * (2 * nu_ - 1));
  }

  double s_func_P1_3D(double x, double y, double z, double t) {
    return -M_PI * alpha_ *
               (z * x * y * (z - 1) * (x - 1) + z * x * y * (z - 1) * (y - 1) +
                z * x * y * (x - 1) * (y - 1) +
                z * x * (z - 1) * (x - 1) * (y - 1) +
                z * y * (z - 1) * (x - 1) * (y - 1) +
                x * y * (z - 1) * (x - 1) * (y - 1)) *
               cos(M_PI * t) +
           c0_ * sin(2 * M_PI * z) * sin(2 * M_PI * x) * sin(2 * M_PI * y) +
           24 * pow(M_PI, 2) * t * sin(2 * M_PI * z) * sin(2 * M_PI * x) *
               sin(2 * M_PI * y) -
           8 * pow(M_PI, 2) * t * sin(2 * M_PI * z) * cos(2 * M_PI * x) *
               cos(2 * M_PI * y);
  }

  double z1_P2_3D(double x, double y, double z, double t) {
    return 2 * M_PI * pow(t, 12) *
           (sin(M_PI * (2 * x - 2 * y)) - 2 * sin(M_PI * (2 * x + 2 * y))) *
           sin(2 * M_PI * z);
  }

  double z2_P2_3D(double x, double y, double z, double t) {
    return -M_PI * pow(t, 12) *
           (sin(M_PI * (2 * x - 2 * y)) + 3 * sin(M_PI * (2 * x + 2 * y))) *
           sin(2 * M_PI * z);
  }

  double z3_P2_3D(double x, double y, double z, double t) {
    return -2 * M_PI * pow(t, 12) * sin(2 * M_PI * x) * sin(2 * M_PI * y) *
           cos(2 * M_PI * z);
  }

  double u1_P2_3D(double x, double y, double z, double t) {
    return z * pow(t, 12) * x * y * (1 - z) * (1 - x) * (1 - y);
  }

  double u2_P2_3D(double x, double y, double z, double t) {
    return z * pow(t, 12) * x * y * (1 - z) * (1 - x) * (1 - y);
  }

  double u3_P2_3D(double x, double y, double z, double t) {
    return z * pow(t, 12) * x * y * (1 - z) * (1 - x) * (1 - y);
  }

  double p_P2_3D(double x, double y, double z, double t) {
    return pow(t, 12) * sin(2 * M_PI * z) * sin(2 * M_PI * x) *
           sin(2 * M_PI * y);
  }

  double f1_P2_3D(double x, double y, double z, double t) {
    return pow(t, 12) *
           (-E_ * nu_ *
                (z * x * y * (z - 1) + z * x * y * (y - 1) +
                 z * x * (z - 1) * (y - 1) + z * y * (z - 1) * (x - 1) +
                 2 * z * y * (z - 1) * (y - 1) + z * y * (x - 1) * (y - 1) +
                 z * (z - 1) * (x - 1) * (y - 1) + x * y * (z - 1) * (y - 1) +
                 y * (z - 1) * (x - 1) * (y - 1)) +
            (1.0 / 2.0) * E_ * (2 * nu_ - 1) *
                (4 * z * y * (z - 1) * (y - 1) +
                 z * (z - 1) *
                     (x * y + 2 * x * (x - 1) + x * (y - 1) + y * (x - 1) +
                      (x - 1) * (y - 1)) +
                 y * (y - 1) *
                     (z * x + z * (x - 1) + x * (z - 1) + 2 * x * (x - 1) +
                      (z - 1) * (x - 1))) +
            2 * M_PI * alpha_ * (nu_ + 1) * (2 * nu_ - 1) * sin(2 * M_PI * z) *
                sin(2 * M_PI * y) * cos(2 * M_PI * x)) /
           ((nu_ + 1) * (2 * nu_ - 1));
  }

  double f2_P2_3D(double x, double y, double z, double t) {
    return pow(t, 12) *
           (-E_ * nu_ *
                (z * x * y * (z - 1) + z * x * y * (x - 1) +
                 2 * z * x * (z - 1) * (x - 1) + z * x * (z - 1) * (y - 1) +
                 z * x * (x - 1) * (y - 1) + z * y * (z - 1) * (x - 1) +
                 z * (z - 1) * (x - 1) * (y - 1) + x * y * (z - 1) * (x - 1) +
                 x * (z - 1) * (x - 1) * (y - 1)) +
            (1.0 / 2.0) * E_ * (2 * nu_ - 1) *
                (4 * z * x * (z - 1) * (x - 1) +
                 z * (z - 1) *
                     (x * y + x * (y - 1) + y * (x - 1) + 2 * y * (y - 1) +
                      (x - 1) * (y - 1)) +
                 x * (x - 1) *
                     (z * y + z * (y - 1) + y * (z - 1) + 2 * y * (y - 1) +
                      (z - 1) * (y - 1))) +
            2 * M_PI * alpha_ * (nu_ + 1) * (2 * nu_ - 1) * sin(2 * M_PI * z) *
                sin(2 * M_PI * x) * cos(2 * M_PI * y)) /
           ((nu_ + 1) * (2 * nu_ - 1));
  }

  double f3_P2_3D(double x, double y, double z, double t) {
    return pow(t, 12) *
           (-E_ * nu_ *
                (z * x * y * (x - 1) + z * x * y * (y - 1) +
                 z * x * (x - 1) * (y - 1) + z * y * (x - 1) * (y - 1) +
                 x * y * (z - 1) * (x - 1) + x * y * (z - 1) * (y - 1) +
                 2 * x * y * (x - 1) * (y - 1) +
                 x * (z - 1) * (x - 1) * (y - 1) +
                 y * (z - 1) * (x - 1) * (y - 1)) +
            (1.0 / 2.0) * E_ * (2 * nu_ - 1) *
                (4 * x * y * (x - 1) * (y - 1) +
                 x * (x - 1) *
                     (z * y + 2 * z * (z - 1) + z * (y - 1) + y * (z - 1) +
                      (z - 1) * (y - 1)) +
                 y * (y - 1) *
                     (z * x + 2 * z * (z - 1) + z * (x - 1) + x * (z - 1) +
                      (z - 1) * (x - 1))) +
            2 * M_PI * alpha_ * (nu_ + 1) * (2 * nu_ - 1) * sin(2 * M_PI * x) *
                sin(2 * M_PI * y) * cos(2 * M_PI * z)) /
           ((nu_ + 1) * (2 * nu_ - 1));
  }

  double s_func_P2_3D(double x, double y, double z, double t) {
    return pow(t, 11) *
           (-12 * alpha_ *
                (z * x * y * (z - 1) * (x - 1) + z * x * y * (z - 1) * (y - 1) +
                 z * x * y * (x - 1) * (y - 1) +
                 z * x * (z - 1) * (x - 1) * (y - 1) +
                 z * y * (z - 1) * (x - 1) * (y - 1) +
                 x * y * (z - 1) * (x - 1) * (y - 1)) +
            12 * c0_ * sin(2 * M_PI * z) * sin(2 * M_PI * x) *
                sin(2 * M_PI * y) +
            24 * pow(M_PI, 2) * t * sin(2 * M_PI * z) * sin(2 * M_PI * x) *
                sin(2 * M_PI * y) -
            8 * pow(M_PI, 2) * t * sin(2 * M_PI * z) * cos(2 * M_PI * x) *
                cos(2 * M_PI * y));
  }
};

#endif  // EX41_HPP