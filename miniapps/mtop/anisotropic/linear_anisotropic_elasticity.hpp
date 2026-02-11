#ifndef LINEAR_ANISOTROPIC_ELASTICITY_HPP
#define LINEAR_ANISOTROPIC_ELASTICITY_HPP

#include "mfem.hpp"

namespace mfem
{

namespace voigt
{
using mfem::future::tensor;

//------------------------------------------------------------------------------
// Stress: tensor -> Voigt (ordering 11,22,33,23,13,12)
// NOTE: no factor-of-2 scaling is applied to shear stress.
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,6>
StressTensorToVoigt(const tensor<T,3,3> &sigma)
{
   tensor<T,6> v;
   v(0) = sigma(0,0); // 11
   v(1) = sigma(1,1); // 22
   v(2) = sigma(2,2); // 33
   v(3) = sigma(1,2); // 23
   v(4) = sigma(0,2); // 13
   v(5) = sigma(0,1); // 12
   return v;
}

//------------------------------------------------------------------------------
// Stress: Voigt -> tensor (ordering 11,22,33,23,13,12)
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
VoigtToStressTensor(const tensor<T,6> &v)
{
   tensor<T,3,3> sigma;

   sigma(0,0) = v(0);
   sigma(1,1) = v(1);
   sigma(2,2) = v(2);

   sigma(1,2) = v(3); sigma(2,1) = v(3); // 23
   sigma(0,2) = v(4); sigma(2,0) = v(4); // 13
   sigma(0,1) = v(5); sigma(1,0) = v(5); // 12

   return sigma;
}

//------------------------------------------------------------------------------
// Strain: tensor -> *engineering* Voigt (ordering 11,22,33,23,13,12)
// Engineering shear strains: gamma_ij = 2 * epsilon_ij (i != j)
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,6>
StrainTensorToEngVoigt(const tensor<T,3,3> &eps)
{
   tensor<T,6> v;
   const T two = T(2);

   v(0) = eps(0,0);       // 11
   v(1) = eps(1,1);       // 22
   v(2) = eps(2,2);       // 33
   v(3) = two * eps(1,2); // gamma_23
   v(4) = two * eps(0,2); // gamma_13
   v(5) = two * eps(0,1); // gamma_12

   return v;
}

//------------------------------------------------------------------------------
// Strain: *engineering* Voigt -> tensor (ordering 11,22,33,23,13,12)
// epsilon_ij = gamma_ij / 2 for i != j
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
EngVoigtToStrainTensor(const tensor<T,6> &v)
{
   tensor<T,3,3> eps;
   const T half = T(0.5); // assumes floating/scalar type

   eps(0,0) = v(0);
   eps(1,1) = v(1);
   eps(2,2) = v(2);

   eps(1,2) = half * v(3); eps(2,1) = half * v(3); // 23
   eps(0,2) = half * v(4); eps(2,0) = half * v(4); // 13
   eps(0,1) = half * v(5); eps(1,0) = half * v(5); // 12

   return eps;
}

// ============================================================================
// 2D (compact) Voigt: [11, 22, 12]
// ============================================================================

//------------------------------------------------------------------------------
// Stress (2D): tensor -> Voigt3 (ordering 11,22,12)
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3>
StressTensorToVoigt(const tensor<T,2,2> &sigma)
{
   tensor<T,3> v;
   v(0) = sigma(0,0); // 11
   v(1) = sigma(1,1); // 22
   v(2) = sigma(0,1); // 12
   return v;
}

//------------------------------------------------------------------------------
// Stress (2D): Voigt3 -> tensor (ordering 11,22,12)
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,2,2>
VoigtToStressTensor(const tensor<T,3> &v)
{
   tensor<T,2,2> sigma;
   sigma(0,0) = v(0);
   sigma(1,1) = v(1);
   sigma(0,1) = v(2);
   sigma(1,0) = v(2);
   return sigma;
}

//------------------------------------------------------------------------------
// Strain (2D): tensor -> engineering Voigt3 (ordering 11,22,12)
// gamma_12 = 2 * eps_12
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3>
StrainTensorToEngVoigt(const tensor<T,2,2> &eps)
{
   tensor<T,3> v;
   const T two = T(2);

   v(0) = eps(0,0);       // 11
   v(1) = eps(1,1);       // 22
   v(2) = two * eps(0,1); // gamma_12
   return v;
}

//------------------------------------------------------------------------------
// Strain (2D): engineering Voigt3 -> tensor (ordering 11,22,12)
// eps_12 = gamma_12 / 2
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,2,2>
EngVoigtToStrainTensor(const tensor<T,3> &v)
{
   tensor<T,2,2> eps;
   const T half = T(1) / T(2);

   eps(0,0) = v(0);
   eps(1,1) = v(1);

   eps(0,1) = half * v(2);
   eps(1,0) = half * v(2);

   return eps;
}

// ============================================================================
// OPTIONAL: 2D -> 3D-Voigt6 embedding (ordering 11,22,33,23,13,12)
// Useful if you want to reuse 3D Voigt6 operators with plane stress/strain.
// Missing shear components 23,13 are set to 0.
// You can optionally supply the out-of-plane normal (33).
// ============================================================================

//------------------------------------------------------------------------------
// Stress (2D): tensor -> Voigt6 (11,22,33,23,13,12), with optional sigma_33
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,6>
StressTensor2DToVoigt6(const tensor<T,2,2> &sigma, const T sigma33 = T(0))
{
   tensor<T,6> v;
   v(0) = sigma(0,0); // 11
   v(1) = sigma(1,1); // 22
   v(2) = sigma33;    // 33
   v(3) = T(0);       // 23
   v(4) = T(0);       // 13
   v(5) = sigma(0,1); // 12
   return v;
}

//------------------------------------------------------------------------------
// Stress (2D): Voigt6 -> tensor2D, discarding 33/23/13 (keeps 11,22,12)
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,2,2>
Voigt6ToStressTensor2D(const tensor<T,6> &v6)
{
   tensor<T,2,2> sigma;
   sigma(0,0) = v6(0);
   sigma(1,1) = v6(1);
   sigma(0,1) = v6(5);
   sigma(1,0) = v6(5);
   return sigma;
}

//------------------------------------------------------------------------------
// Strain (2D): tensor -> engineering Voigt6 with optional eps_33
// gamma_23 and gamma_13 are set to 0.
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,6>
StrainTensor2DToEngVoigt6(const tensor<T,2,2> &eps, const T eps33 = T(0))
{
   tensor<T,6> v;
   const T two = T(2);

   v(0) = eps(0,0);       // 11
   v(1) = eps(1,1);       // 22
   v(2) = eps33;          // 33
   v(3) = T(0);           // gamma_23
   v(4) = T(0);           // gamma_13
   v(5) = two * eps(0,1); // gamma_12
   return v;
}

//------------------------------------------------------------------------------
// Strain (2D): engineering Voigt6 -> tensor2D (uses gamma_12 only)
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,2,2>
EngVoigt6ToStrainTensor2D(const tensor<T,6> &v6)
{
   tensor<T,2,2> eps;
   const T half = T(1) / T(2);

   eps(0,0) = v6(0);
   eps(1,1) = v6(1);

   eps(0,1) = half * v6(5);
   eps(1,0) = half * v6(5);

   return eps;
}

template <typename T, int N, int M>
MFEM_HOST_DEVICE inline void SetZero(tensor<T,N,M> &A)
{
   for (int i = 0; i < N; i++)
      for (int j = 0; j < M; j++)
      {
         A(i,j) = T(0);
      }
}

template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3> Invert3x3(const tensor<T,3,3> &A)
{
   const T a00 = A(0,0), a01 = A(0,1), a02 = A(0,2);
   const T a10 = A(1,0), a11 = A(1,1), a12 = A(1,2);
   const T a20 = A(2,0), a21 = A(2,1), a22 = A(2,2);

   const T det =
      a00*(a11*a22 - a12*a21)
      - a01*(a10*a22 - a12*a20)
      + a02*(a10*a21 - a11*a20);
   const T inv_det = T(1.0) / det;

   tensor<T,3,3> inv;
   inv(0,0) =  (a11*a22 - a12*a21) * inv_det;
   inv(0,1) =  (a02*a21 - a01*a22) * inv_det;
   inv(0,2) =  (a01*a12 - a02*a11) * inv_det;

   inv(1,0) =  (a12*a20 - a10*a22) * inv_det;
   inv(1,1) =  (a00*a22 - a02*a20) * inv_det;
   inv(1,2) =  (a02*a10 - a00*a12) * inv_det;

   inv(2,0) =  (a10*a21 - a11*a20) * inv_det;
   inv(2,1) =  (a01*a20 - a00*a21) * inv_det;
   inv(2,2) =  (a00*a11 - a01*a10) * inv_det;

   return inv;
}

template <typename T>
MFEM_HOST_DEVICE inline tensor<T,2,2> Invert2x2(const tensor<T,2,2> &A)
{
   const T a00 = A(0,0), a01 = A(0,1);
   const T a10 = A(1,0), a11 = A(1,1);

   const T det = a00*a11 - a01*a10;
   const T det_inv = T(1.0) / det;

   tensor<T,2,2> inv;

   inv(0,0) =  a11 * det_inv;
   inv(0,1) = -a01 * det_inv;
   inv(1,0) = -a10 * det_inv;
   inv(1,1) =  a00 * det_inv;

   return inv;
}


// ============================================================================
// 3D ISOTROPIC (engineering Voigt, order 11,22,33,23,13,12)
// Inputs: Young's modulus E, Poisson ratio nu
// ============================================================================
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,6,6> IsotropicStiffness3D(const T E,
                                                           const T nu)
{
   const T one = T(1);
   const T two = T(2);

   const T mu     = E / (two * (one + nu));                 // shear modulus G
   const T lambda = (E * nu) / ((one + nu) * (one - two*nu));

   tensor<T,6,6> C;
   SetZero(C);

   const T a = lambda + two*mu;

   // normal-normal
   C(0,0) = a;      C(0,1) = lambda; C(0,2) = lambda;
   C(1,0) = lambda; C(1,1) = a;      C(1,2) = lambda;
   C(2,0) = lambda; C(2,1) = lambda; C(2,2) = a;

   // shear (engineering shear strains -> diagonal is mu)
   C(3,3) = mu;  // 23
   C(4,4) = mu;  // 13
   C(5,5) = mu;  // 12

   return C;
}


// ============================================================================
// 2D ISOTROPIC
// Compact engineering Voigt: [11,22,12], strain uses gamma_12
// Provide both plane stress and plane strain.
// ============================================================================
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3> IsotropicStiffness2D_PlaneStress(
   const T E,
   const T nu)
{
   const T one = T(1);
   const T two = T(2);

   const T fac = E / (one - nu*nu);

   tensor<T,3,3> C;
   SetZero(C);

   C(0,0) = fac;
   C(1,1) = fac;
   C(0,1) = fac * nu;
   C(1,0) = fac * nu;

   // C66 = G = E/(2(1+nu)) when using engineering gamma12
   C(2,2) = E / (two * (one + nu));

   return C;
}

template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3> IsotropicStiffness2D_PlaneStrain(
   const T E,
   const T nu)
{
   const T one = T(1);
   const T two = T(2);

   const T mu     = E / (two * (one + nu));
   const T lambda = (E * nu) / ((one + nu) * (one - two*nu));

   tensor<T,3,3> C;
   SetZero(C);

   C(0,0) = lambda + two*mu;
   C(1,1) = lambda + two*mu;
   C(0,1) = lambda;
   C(1,0) = lambda;

   C(2,2) = mu; // engineering gamma12

   return C;
}

// ============================================================================
// 3D ORTHOTROPIC (engineering Voigt, order 11,22,33,23,13,12)
//
// Independent inputs (common choice):
//   E1,E2,E3, nu12,nu13,nu23, G12,G13,G23
// Reciprocity (enforced internally):
//   nu21 = nu12 * E2/E1
//   nu31 = nu13 * E3/E1
//   nu32 = nu23 * E3/E2
//
// Build compliance S (6x6), invert only the 3x3 normal block.
// Shear is diagonal: C44=G23, C55=G13, C66=G12 (engineering).
// ============================================================================
template <typename T>
struct Orthotropic3D
{
   T E1, E2, E3;
   T nu12, nu13, nu23;
   T G12, G13, G23;
};

template <typename T>
MFEM_HOST_DEVICE inline tensor<T,6,6> OrthotropicStiffness3D(
   const Orthotropic3D<T> &p)
{
   const T nu21 = p.nu12 * p.E2 / p.E1;
   const T nu31 = p.nu13 * p.E3 / p.E1;
   const T nu32 = p.nu23 * p.E3 / p.E2;

   // Normal compliance block Sn (3x3)
   tensor<T,3,3> Sn;
   Sn(0,0) = T(1)/p.E1;      Sn(0,1) = -nu21/p.E2;    Sn(0,2) = -nu31/p.E3;
   Sn(1,0) = -p.nu12/p.E1;   Sn(1,1) = T(1)/p.E2;     Sn(1,2) = -nu32/p.E3;
   Sn(2,0) = -p.nu13/p.E1;   Sn(2,1) = -p.nu23/p.E2;  Sn(2,2) = T(1)/p.E3;

   const tensor<T,3,3> Cn = Invert3x3(Sn);

   tensor<T,6,6> C;
   SetZero(C);

   // Fill normal block (11,22,33)
   for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
      {
         C(i,j) = Cn(i,j);
      }

   // Shear diagonal (engineering)
   C(3,3) = p.G23; // 23
   C(4,4) = p.G13; // 13
   C(5,5) = p.G12; // 12

   return C;
}

// ============================================================================
// 2D ORTHOTROPIC
//
// (A) Plane stress, *in-plane orthotropy only* (no E3/nu13/nu23 needed):
//     Uses E1,E2,nu12,G12 with reciprocity nu21=nu12*E2/E1.
//     Voigt2D: [11,22,12], strain uses gamma12.
//
// (B) Plane strain / plane stress reduced from full 3D orthotropy:
//     Uses Orthotropic3D and reduces the 3D normal block.
// ============================================================================

template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
OrthotropicStiffness2D_PlaneStress(const T E1, const T E2,
                                   const T nu12, const T G12)
{
   const T nu21  = nu12 * E2 / E1;
   const T denom = T(1) - nu12*nu21;

   tensor<T,3,3> C;
   SetZero(C);

   C(0,0) = E1 / denom;          // C11
   C(1,1) = E2 / denom;          // C22
   C(0,1) = (nu12*E2) / denom;   // C12
   C(1,0) = C(0,1);

   C(2,2) = G12;                 // C66 (engineering gamma12)

   return C;
}


/* Which 2D orthotropic function should you use?
   If you truly have a 2D orthotropic lamina law (only in-plane properties known/meaningful), use:
        - OrthotropicStiffness2D_PlaneStress(E1,E2,nu12,G12)

    If you have a full 3D orthotropic solid and are doing a 2D kinematic reduction:
        - Plane strain (ε33=0ε33​=0): OrthotropicStiffness2D_PlaneStrain_From3D(p)
        - Plane stress (σ33=0σ33​=0): OrthotropicStiffness2D_PlaneStress_From3D(p)
*/

// Plane strain (epsilon33 = 0): take in-plane submatrix from full 3D stiffness
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
OrthotropicStiffness2D_PlaneStrain_From3D(const Orthotropic3D<T> &p)
{
   const tensor<T,6,6> C3 = OrthotropicStiffness3D(p);

   tensor<T,3,3> C2;
   SetZero(C2);

   // normals: (11,22) from 3D stiffness (normal block)
   C2(0,0) = C3(0,0);
   C2(0,1) = C3(0,1);
   C2(1,0) = C3(1,0);
   C2(1,1) = C3(1,1);

   // in-plane shear: 12
   C2(2,2) = C3(5,5); // = G12

   return C2;
}

// Plane stress (sigma33 = 0): Schur complement reduction of the 3D normal block
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
OrthotropicStiffness2D_PlaneStress_From3D(const Orthotropic3D<T> &p)
{
   const tensor<T,6,6> C3 = OrthotropicStiffness3D(p);

   // Extract normal block indices 0,1,2 -> 11,22,33
   const T C11 = C3(0,0), C12 = C3(0,1), C13 = C3(0,2);
   const T C22 = C3(1,1), C23 = C3(1,2);
   const T C33 = C3(2,2);

   tensor<T,3,3> C2;
   SetZero(C2);

   // Reduced in-plane normal stiffness (Schur complement eliminating epsilon33)
   C2(0,0) = C11 - (C13*C13)/C33;
   C2(1,1) = C22 - (C23*C23)/C33;
   C2(0,1) = C12 - (C13*C23)/C33;
   C2(1,0) = C2(0,1);

   // In-plane shear 12
   C2(2,2) = C3(5,5); // = G12

   return C2;
}

// ============================================================================
// 3D ISOTROPIC COMPLIANCE (engineering Voigt, order 11,22,33,23,13,12)
// Inputs: Young's modulus E, Poisson ratio nu
// ============================================================================
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,6,6> IsotropicCompliance3D(const T E,
                                                            const T nu)
{
   const T one = T(1);
   const T two = T(2);

   const T invE = one / E;
   const T G    = E / (two * (one + nu)); // shear modulus
   const T invG = one / G;

   tensor<T,6,6> S;
   SetZero(S);

   // normal-normal
   S(0,0) = invE;   S(1,1) = invE;   S(2,2) = invE;

   const T off = -nu * invE;
   S(0,1) = off; S(0,2) = off;
   S(1,0) = off; S(1,2) = off;
   S(2,0) = off; S(2,1) = off;

   // shear (engineering): gamma_ij = tau_ij / G
   S(3,3) = invG; // 23
   S(4,4) = invG; // 13
   S(5,5) = invG; // 12

   return S;
}

// ============================================================================
// 2D ISOTROPIC COMPLIANCE (compact engineering Voigt: [11,22,12])
// Provide both plane stress and plane strain.
// ============================================================================
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3> IsotropicCompliance2D_PlaneStress(
   const T E,
   const T nu)
{
   const T one = T(1);
   const T two = T(2);

   const T invE = one / E;
   const T G    = E / (two * (one + nu));
   const T invG = one / G;

   tensor<T,3,3> S;
   SetZero(S);

   S(0,0) = invE;
   S(1,1) = invE;

   const T off = -nu * invE;
   S(0,1) = off;
   S(1,0) = off;

   S(2,2) = invG; // gamma12 = tau12 / G

   return S;
}

template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3> IsotropicCompliance2D_PlaneStrain(
   const T E,
   const T nu)
{
   const T one = T(1);
   const T two = T(2);

   const T invE = one / E;
   const T G    = E / (two * (one + nu));
   const T invG = one / G;

   // This is the inverse of the plane-strain stiffness you built:
   // C = [[lambda+2mu, lambda, 0],
   //      [lambda, lambda+2mu, 0],
   //      [0, 0, mu]]
   // expressed directly in (E,nu).
   tensor<T,3,3> S;
   SetZero(S);

   S(0,0) = (one - nu*nu) * invE;
   S(1,1) = (one - nu*nu) * invE;

   const T off = -nu * (one + nu) * invE;
   S(0,1) = off;
   S(1,0) = off;

   S(2,2) = invG; // gamma12 = tau12 / G

   return S;
}

// ============================================================================
// 3D ORTHOTROPIC COMPLIANCE (engineering Voigt, order 11,22,33,23,13,12)
// Inputs: E1,E2,E3, nu12,nu13,nu23, G12,G13,G23
// Reciprocity enforced:
//   nu21 = nu12 * E2/E1
//   nu31 = nu13 * E3/E1
//   nu32 = nu23 * E3/E2
// ============================================================================
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,6,6> OrthotropicCompliance3D(
   const Orthotropic3D<T> &p)
{
   const T one = T(1);

   const T nu21 = p.nu12 * p.E2 / p.E1;
   const T nu31 = p.nu13 * p.E3 / p.E1;
   const T nu32 = p.nu23 * p.E3 / p.E2;

   tensor<T,6,6> S;
   SetZero(S);

   // normal block
   S(0,0) = one / p.E1;
   S(1,1) = one / p.E2;
   S(2,2) = one / p.E3;

   S(0,1) = -nu21 / p.E2;   S(1,0) = -p.nu12 / p.E1;
   S(0,2) = -nu31 / p.E3;   S(2,0) = -p.nu13 / p.E1;
   S(1,2) = -nu32 / p.E3;   S(2,1) = -p.nu23 / p.E2;

   // shear (engineering)
   S(3,3) = one / p.G23; // 23
   S(4,4) = one / p.G13; // 13
   S(5,5) = one / p.G12; // 12

   return S;
}

// ============================================================================
// 2D ORTHOTROPIC COMPLIANCE (plane stress, in-plane orthotropy only)
// Compact engineering Voigt: [11,22,12]
// Inputs: E1,E2, nu12, G12 (nu21 enforced)
// ============================================================================
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
OrthotropicCompliance2D_PlaneStress(const T E1, const T E2,
                                    const T nu12, const T G12)
{
   const T one = T(1);

   const T nu21 = nu12 * E2 / E1;

   tensor<T,3,3> S;
   SetZero(S);

   S(0,0) = one / E1;
   S(1,1) = one / E2;

   // symmetric if reciprocity holds (we enforce nu21 from nu12,E1,E2)
   S(0,1) = -nu21 / E2;  // = -nu12/E1
   S(1,0) = -nu12 / E1;

   S(2,2) = one / G12;

   return S;
}

// Convenience: plane stress compliance from full 3D orthotropic inputs
// (in-plane compliance does not require E3,nu13,nu23 for sigma33=0)
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
OrthotropicCompliance2D_PlaneStress_From3D(const Orthotropic3D<T> &p)
{
   return OrthotropicCompliance2D_PlaneStress<T>(p.E1, p.E2, p.nu12, p.G12);
}

// ============================================================================
// 2D ORTHOTROPIC COMPLIANCE (plane strain, from full 3D orthotropy)
// Constraint: epsilon33 = 0 (sigma33 becomes whatever is needed)
// Effective in-plane compliance:
//   S_eff = Saa - Sab * (S33)^{-1} * Sba
// where b corresponds to the 33 component of the 3D normal block.
// ============================================================================
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
OrthotropicCompliance2D_PlaneStrain_From3D(const Orthotropic3D<T> &p)
{
   const T one = T(1);

   const T invE1 = one / p.E1;
   const T invE2 = one / p.E2;

   // reciprocity
   const T nu21 = p.nu12 * p.E2 / p.E1;
   const T nu31 = p.nu13 * p.E3 / p.E1;
   const T nu32 = p.nu23 * p.E3 / p.E2;

   tensor<T,3,3> S;
   SetZero(S);

   // Normal effective compliance under epsilon33 = 0:
   // S11 = 1/E1 - (S13*S31)/S33 = (1 - nu13*nu31)/E1
   // S22 = 1/E2 - (S23*S32)/S33 = (1 - nu23*nu32)/E2
   S(0,0) = invE1 * (one - p.nu13 * nu31);
   S(1,1) = invE2 * (one - p.nu23 * nu32);

   // S12 = -nu21/E2 - (S13*S32)/S33
   //     = -nu12/E1 - (nu13*nu23*E3)/(E1*E2)  (symmetric)
   const T cross = (p.nu13 * p.nu23 * p.E3) * (invE1 * invE2);
   S(0,1) = -p.nu12 * invE1 - cross;
   S(1,0) = S(0,1);

   // shear (engineering)
   S(2,2) = one / p.G12;

   return S;
}

//------------------------------------------------------------------------------
// Rotation matrix from quaternion q=(w,x,y,z).
// Works for non-unit q by internally normalizing via s = 2/||q||^2.
// If ||q||=0 -> returns Identity.
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
RotationMatrixFromQuaternion(const T w, const T x, const T y, const T z,
                             const T eps_t=T(1e-13))
{
   const T one = T(1);

   const T n2 = w*w + x*x + y*y + z*z;
   tensor<T,3,3> R;

   if (n2 < eps_t )
   {
      R(0,0)=one; R(0,1)=T(0); R(0,2)=T(0);
      R(1,0)=T(0); R(1,1)=one; R(1,2)=T(0);
      R(2,0)=T(0); R(2,1)=T(0); R(2,2)=one;
      return R;
   }

   // s = 2 / ||q||^2 (equals 2 for unit quaternion)
   const T s  = T(2) / n2;

   const T xx = x*x, yy = y*y, zz = z*z;
   const T xy = x*y, xz = x*z, yz = y*z;
   const T wx = w*x, wy = w*y, wz = w*z;

   R(0,0) = one - s*(yy + zz);
   R(0,1) =        s*(xy - wz);
   R(0,2) =        s*(xz + wy);

   R(1,0) =        s*(xy + wz);
   R(1,1) = one - s*(xx + zz);
   R(1,2) =        s*(yz - wx);

   R(2,0) =        s*(xz - wy);
   R(2,1) =        s*(yz + wx);
   R(2,2) = one - s*(xx + yy);

   return R;
}

template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
RotationMatrixFromQuaternion(const tensor<T,4>
                             &q /* q(0)=w,q(1)=x,q(2)=y,q(3)=z */, T esp_t=T(1e-13))
{
   return RotationMatrixFromQuaternion<T>(q(0), q(1), q(2), q(3));
}

//------------------------------------------------------------------------------
// Rotate stress tensor (active): sigma' = R * sigma * R^T
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
RotateStressTensor(const tensor<T,3,3> &sigma, const tensor<T,3,3> &R)
{
   tensor<T,3,3> tmp, out;

   // tmp = R * sigma
   for (int i = 0; i < 3; i++)
   {
      for (int j = 0; j < 3; j++)
      {
         T s = T(0);
         for (int k = 0; k < 3; k++) { s += R(i,k) * sigma(k,j); }
         tmp(i,j) = s;
      }
   }

   // out = tmp * R^T  => out(i,j) = sum_k tmp(i,k) * R(j,k)
   for (int i = 0; i < 3; i++)
   {
      for (int j = 0; j < 3; j++)
      {
         T s = T(0);
         for (int k = 0; k < 3; k++) { s += tmp(i,k) * R(j,k); }
         out(i,j) = s;
      }
   }

   return out;
}

//------------------------------------------------------------------------------
// 6x6 stress rotation operator in Voigt form (no factor-of-2 on shear stress).
// Ordering: [11,22,33,23,13,12]
// Such that: sigmaV' = Tσ * sigmaV
// where sigmaV = [σ11, σ22, σ33, σ23, σ13, σ12]^T
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,6,6>
StressVoigtRotationMatrix(const tensor<T,3,3> &R)
{
   const T r11 = R(0,0), r12 = R(0,1), r13 = R(0,2);
   const T r21 = R(1,0), r22 = R(1,1), r23 = R(1,2);
   const T r31 = R(2,0), r32 = R(2,1), r33 = R(2,2);

   const T two = T(2);

   tensor<T,6,6> Tm;

   // Row 0: σ'11
   Tm(0,0) = r11*r11;
   Tm(0,1) = r12*r12;
   Tm(0,2) = r13*r13;
   Tm(0,3) = two*r12*r13;
   Tm(0,4) = two*r11*r13;
   Tm(0,5) = two*r11*r12;

   // Row 1: σ'22
   Tm(1,0) = r21*r21;
   Tm(1,1) = r22*r22;
   Tm(1,2) = r23*r23;
   Tm(1,3) = two*r22*r23;
   Tm(1,4) = two*r21*r23;
   Tm(1,5) = two*r21*r22;

   // Row 2: σ'33
   Tm(2,0) = r31*r31;
   Tm(2,1) = r32*r32;
   Tm(2,2) = r33*r33;
   Tm(2,3) = two*r32*r33;
   Tm(2,4) = two*r31*r33;
   Tm(2,5) = two*r31*r32;

   // Row 3: σ'23
   Tm(3,0) = r21*r31;
   Tm(3,1) = r22*r32;
   Tm(3,2) = r23*r33;
   Tm(3,3) = (r22*r33 + r23*r32);
   Tm(3,4) = (r21*r33 + r23*r31);
   Tm(3,5) = (r21*r32 + r22*r31);

   // Row 4: σ'13
   Tm(4,0) = r11*r31;
   Tm(4,1) = r12*r32;
   Tm(4,2) = r13*r33;
   Tm(4,3) = (r12*r33 + r13*r32);
   Tm(4,4) = (r11*r33 + r13*r31);
   Tm(4,5) = (r11*r32 + r12*r31);

   // Row 5: σ'12
   Tm(5,0) = r11*r21;
   Tm(5,1) = r12*r22;
   Tm(5,2) = r13*r23;
   Tm(5,3) = (r12*r23 + r13*r22);
   Tm(5,4) = (r11*r23 + r13*r21);
   Tm(5,5) = (r11*r22 + r12*r21);

   return Tm;
}

template <typename T>
MFEM_HOST_DEVICE inline tensor<T,6,6>
StressVoigtRotationMatrixFromQuaternion(const tensor<T,4> &q)
{
   const tensor<T,3,3> R = RotationMatrixFromQuaternion<T>(q);
   return StressVoigtRotationMatrix<T>(R);
}

//------------------------------------------------------------------------------
// Rotate strain tensor (active): eps' = R * eps * R^T
// (Same code as your RotateStressTensor; strain is also 2nd-order tensor.)
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
RotateStrainTensor(const tensor<T,3,3> &eps, const tensor<T,3,3> &R)
{
   tensor<T,3,3> tmp, out;

   // tmp = R * eps
   for (int i = 0; i < 3; i++)
   {
      for (int j = 0; j < 3; j++)
      {
         T s = T(0);
         for (int k = 0; k < 3; k++) { s += R(i,k) * eps(k,j); }
         tmp(i,j) = s;
      }
   }

   // out = tmp * R^T  => out(i,j) = sum_k tmp(i,k) * R(j,k)
   for (int i = 0; i < 3; i++)
   {
      for (int j = 0; j < 3; j++)
      {
         T s = T(0);
         for (int k = 0; k < 3; k++) { s += tmp(i,k) * R(j,k); }
         out(i,j) = s;
      }
   }

   return out;
}

//------------------------------------------------------------------------------
// 6x6 strain rotation operator for *engineering* Voigt strain.
// Ordering: [11,22,33,23,13,12]
// Strain vector: [e11,e22,e33,g23,g13,g12]^T with g_ij = 2*e_ij
//
// Such that: epsV_eng' = Teps * epsV_eng
//------------------------------------------------------------------------------
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,6,6>
StrainEngVoigtRotationMatrix(const tensor<T,3,3> &R)
{
   const T r11 = R(0,0), r12 = R(0,1), r13 = R(0,2);
   const T r21 = R(1,0), r22 = R(1,1), r23 = R(1,2);
   const T r31 = R(2,0), r32 = R(2,1), r33 = R(2,2);

   const T two = T(2);

   tensor<T,6,6> Tm;

   // Row 0: e'11
   Tm(0,0) = r11*r11;
   Tm(0,1) = r12*r12;
   Tm(0,2) = r13*r13;
   Tm(0,3) = r12*r13;
   Tm(0,4) = r11*r13;
   Tm(0,5) = r11*r12;

   // Row 1: e'22
   Tm(1,0) = r21*r21;
   Tm(1,1) = r22*r22;
   Tm(1,2) = r23*r23;
   Tm(1,3) = r22*r23;
   Tm(1,4) = r21*r23;
   Tm(1,5) = r21*r22;

   // Row 2: e'33
   Tm(2,0) = r31*r31;
   Tm(2,1) = r32*r32;
   Tm(2,2) = r33*r33;
   Tm(2,3) = r32*r33;
   Tm(2,4) = r31*r33;
   Tm(2,5) = r31*r32;

   // Row 3: g'23 = 2*e'23
   Tm(3,0) = two*r21*r31;
   Tm(3,1) = two*r22*r32;
   Tm(3,2) = two*r23*r33;
   Tm(3,3) = (r22*r33 + r23*r32);
   Tm(3,4) = (r21*r33 + r23*r31);
   Tm(3,5) = (r21*r32 + r22*r31);

   // Row 4: g'13 = 2*e'13
   Tm(4,0) = two*r11*r31;
   Tm(4,1) = two*r12*r32;
   Tm(4,2) = two*r13*r33;
   Tm(4,3) = (r12*r33 + r13*r32);
   Tm(4,4) = (r11*r33 + r13*r31);
   Tm(4,5) = (r11*r32 + r12*r31);

   // Row 5: g'12 = 2*e'12
   Tm(5,0) = two*r11*r21;
   Tm(5,1) = two*r12*r22;
   Tm(5,2) = two*r13*r23;
   Tm(5,3) = (r12*r23 + r13*r22);
   Tm(5,4) = (r11*r23 + r13*r21);
   Tm(5,5) = (r11*r22 + r12*r21);

   return Tm;
}

template <typename T>
MFEM_HOST_DEVICE inline tensor<T,6,6>
StrainEngVoigtRotationMatrixFromQuaternion(const tensor<T,4> &q)
{
   const tensor<T,3,3> R = RotationMatrixFromQuaternion<T>(q);
   return StrainEngVoigtRotationMatrix<T>(R);
}

// ---------------- small dense ops (fixed size) ----------------

template <typename T, int N>
MFEM_HOST_DEVICE inline tensor<T,N,N> Transpose(const tensor<T,N,N> &A)
{
   tensor<T,N,N> AT;
   for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++)
      {
         AT(i,j) = A(j,i);
      }
   return AT;
}

template <typename T, int N>
MFEM_HOST_DEVICE inline tensor<T,N,N> MatMul(const tensor<T,N,N> &A,
                                             const tensor<T,N,N> &B)
{
   tensor<T,N,N> C;
   for (int i = 0; i < N; i++)
   {
      for (int j = 0; j < N; j++)
      {
         T s = T(0);
         for (int k = 0; k < N; k++) { s += A(i,k) * B(k,j); }
         C(i,j) = s;
      }
   }
   return C;
}

// C' = Tσ C Tσ^T  (equivalently Tσ C Tε^{-1})
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,6,6>
RotateStiffnessVoigt3D(const tensor<T,6,6> &C, const tensor<T,3,3> &R)
{
   const tensor<T,6,6> Ts  = StressVoigtRotationMatrix<T>(R);
   const tensor<T,6,6> TsT = Transpose<T,6>(Ts);
   return MatMul<T,6>(MatMul<T,6>(Ts, C), TsT);
}

// S' = Tε S Tε^T  (equivalently Tε S Tσ^{-1})
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,6,6>
RotateComplianceVoigt3D(const tensor<T,6,6> &S, const tensor<T,3,3> &R)
{
   const tensor<T,6,6> Te  = StrainEngVoigtRotationMatrix<T>(R);
   const tensor<T,6,6> TeT = Transpose<T,6>(Te);
   return MatMul<T,6>(MatMul<T,6>(Te, S), TeT);
}

// ============================================================================
// 2D: Voigt compact form
// Stress: [11,22,12]
// Strain (engineering): [e11,e22,g12]
// ============================================================================

// Build 2x2 rotation from angle if you want; otherwise supply R2 directly.
// Here we build the Voigt rotation operators directly from a 2x2 R.
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
StressVoigtRotationMatrix2D(const tensor<T,2,2> &R)
{
   const T r11 = R(0,0), r12 = R(0,1);
   const T r21 = R(1,0), r22 = R(1,1);
   const T two = T(2);

   // For a proper 2D rotation, you typically have r11=c, r12=-s, r21=s, r22=c,
   // but this formula works for any orthogonal 2x2 R.
   tensor<T,3,3> Ts;

   // σ'11
   Ts(0,0) = r11*r11;
   Ts(0,1) = r12*r12;
   Ts(0,2) = two*r11*r12;

   // σ'22
   Ts(1,0) = r21*r21;
   Ts(1,1) = r22*r22;
   Ts(1,2) = two*r21*r22;

   // σ'12
   Ts(2,0) = r11*r21;
   Ts(2,1) = r12*r22;
   Ts(2,2) = (r11*r22 + r12*r21);

   return Ts;
}

template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
StrainEngVoigtRotationMatrix2D(const tensor<T,2,2> &R)
{
   const T r11 = R(0,0), r12 = R(0,1);
   const T r21 = R(1,0), r22 = R(1,1);
   const T two = T(2);

   tensor<T,3,3> Te;

   // e'11
   Te(0,0) = r11*r11;
   Te(0,1) = r12*r12;
   Te(0,2) = r11*r12;

   // e'22
   Te(1,0) = r21*r21;
   Te(1,1) = r22*r22;
   Te(1,2) = r21*r22;

   // g'12 = 2 e'12
   Te(2,0) = two*r11*r21;
   Te(2,1) = two*r12*r22;
   Te(2,2) = (r11*r22 + r12*r21);

   return Te;
}

// Rotate 2D stiffness/compliance (compact Voigt)
template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
RotateStiffnessVoigt2D(const tensor<T,3,3> &C, const tensor<T,2,2> &R2)
{
   const tensor<T,3,3> Ts  = StressVoigtRotationMatrix2D<T>(R2);
   const tensor<T,3,3> TsT = Transpose<T,3>(Ts);
   return MatMul<T,3>(MatMul<T,3>(Ts, C), TsT);
}

template <typename T>
MFEM_HOST_DEVICE inline tensor<T,3,3>
RotateComplianceVoigt2D(const tensor<T,3,3> &S, const tensor<T,2,2> &R2)
{
   const tensor<T,3,3> Te  = StrainEngVoigtRotationMatrix2D<T>(R2);
   const tensor<T,3,3> TeT = Transpose<T,3>(Te);
   return MatMul<T,3>(MatMul<T,3>(Te, S), TeT);
}




};// namespace voigt

};




#endif
