// Copyright (c) 2010, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-443211. All Rights
// reserved. See file COPYRIGHT for details.
//
// This file is part of the MFEM library. For more information and source code
// availability see http://mfem.googlecode.com.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free
// Software Foundation) version 2.1 dated February 1999.

#ifndef MFEM_GRIDFUNC
#define MFEM_GRIDFUNC

/// Class for grid function - Vector with asociated FE space.
class GridFunction : public Vector
{
protected:
   /// FE space on which grid function lives.
   FiniteElementSpace *fes;

   /// Used when the grid function is read from a file
   FiniteElementCollection *fec;

   void SaveSTLTri(ostream &out, double p1[], double p2[], double p3[]);

   void GetVectorGradientHat(ElementTransformation &T, DenseMatrix &gh);

public:

   GridFunction() { fes = NULL; fec = NULL; }

   /// Creates grid function associated with *f.
   GridFunction(FiniteElementSpace *f) : Vector(f->GetVSize())
   { fes = f; fec = NULL; }

   GridFunction(Mesh *m, istream &input);

   GridFunction(Mesh *m, GridFunction *gf_array[], int num_pieces);

   /// Make the GridFunction the owner of 'fec' and 'fes'
   void MakeOwner(FiniteElementCollection *_fec)
   { fec = _fec; }

   FiniteElementCollection *OwnFEC() { return fec; }

   int VectorDim() const;

   /// Returns the values in the vertices of i'th element for dimension vdim.
   void GetNodalValues(int i, Array<double> &nval, int vdim = 1) const;

   double GetValue(int i, const IntegrationPoint &ip, int vdim = 1) const;

   void GetVectorValue(int i, const IntegrationPoint &ip, Vector &val) const;

   void GetValues(int i, const IntegrationRule &ir, Vector &vals,
                  DenseMatrix &tr, int vdim = 1) const;

   int GetFaceValues(int i, int side, const IntegrationRule &ir, Vector &vals,
                     DenseMatrix &tr, int vdim = 1) const;

   void GetVectorValues(int i, const IntegrationRule &ir,
                        DenseMatrix &vals, DenseMatrix &tr) const;

   int GetFaceVectorValues(int i, int side, const IntegrationRule &ir,
                           DenseMatrix &vals, DenseMatrix &tr) const;

   void GetValuesFrom(GridFunction &);

   void GetBdrValuesFrom(GridFunction &);

   void GetVectorFieldValues(int i, const IntegrationRule &ir,
                             DenseMatrix &vals,
                             DenseMatrix &tr, int comp = 0) const;

   /// For a vector grid function, makes sure that the ordering is byNODES.
   void ReorderByNodes();

   /// Return the values as a vector on mesh vertices for dimension vdim.
   void GetNodalValues(Vector &nval, int vdim = 1) const;

   void GetVectorFieldNodalValues(Vector &val, int comp) const;

   void ProjectVectorFieldOn(GridFunction &vec_field, int comp = 0);

   void GetDerivative(int comp, int der_comp, GridFunction &der);

   double GetDivergence(ElementTransformation &tr);

   void GetGradient(ElementTransformation &tr, Vector &grad);

   void GetGradients(const int elem, const IntegrationRule &ir,
                     DenseMatrix &grad);

   void GetVectorGradient(ElementTransformation &tr, DenseMatrix &grad);

   /** Compute \f$ (\int_{\Omega} (*this) \psi_i)/(\int_{\Omega} \psi_i) \f$,
       where \f$ \psi_i \f$ are the basis functions for the FE space of avgs.
       Both FE spaces should be scalar and on the same mesh. */
   void GetElementAverages(GridFunction &avgs);

   void ProjectCoefficient(Coefficient &coeff);

   // call fes -> BuildDofToArrays() before using this projection
   void ProjectCoefficient(Coefficient &coeff, Array<int> &dofs, int vd = 0);

   void ProjectCoefficient(VectorCoefficient &vcoeff);

   void ProjectCoefficient(Coefficient *coeff[]);

   void ProjectBdrCoefficient(Coefficient &coeff, Array<int> &attr)
   {
      Coefficient *coeff_p = &coeff;
      ProjectBdrCoefficient(&coeff_p, attr);
   }

   void ProjectBdrCoefficient(Coefficient *coeff[], Array<int> &attr);

   /** Project the normal component of the given VectorCoefficient on
       the boundary. Only boundary attributes that are marked in
       'bdr_attr' are projected. Assumes RT-type VectorFE GridFunction. */
   void ProjectBdrCoefficientNormal(VectorCoefficient &vcoeff,
                                    Array<int> &bdr_attr);

   /** Project the tangential components of the given VectorCoefficient on
       the boundary. Only boundary attributes that are marked in
       'bdr_attr' are projected. Assumes ND-type VectorFE GridFunction. */
   void ProjectBdrCoefficientTangent(VectorCoefficient &vcoeff,
                                     Array<int> &bdr_attr);

   double ComputeL2Error(Coefficient &exsol,
                         const IntegrationRule *irs[] = NULL) const
   {
      Coefficient *exsol_p = &exsol;
      return ComputeL2Error(&exsol_p, irs);
   }

   double ComputeL2Error(Coefficient *exsol[],
                         const IntegrationRule *irs[] = NULL) const;

   double ComputeL2Error(VectorCoefficient &exsol,
                         const IntegrationRule *irs[] = NULL,
                         Array<int> *elems = NULL) const;

   double ComputeH1Error(Coefficient *exsol, VectorCoefficient *exgrad,
                         Coefficient *ell_coef, double Nu,
                         int norm_type) const;

   double ComputeMaxError(Coefficient *exsol[],
                          const IntegrationRule *irs[] = NULL) const;

   double ComputeMaxError(VectorCoefficient &exsol,
                          const IntegrationRule *irs[] = NULL) const;

   double ComputeW11Error(Coefficient *exsol, VectorCoefficient *exgrad,
                          int norm_type, Array<int> *elems = NULL,
                          const IntegrationRule *irs[] = NULL) const;

   double ComputeL1Error(VectorCoefficient &exsol,
                         const IntegrationRule *irs[] = NULL) const;

   /// Redefine '=' for GridFunction = constant.
   GridFunction &operator=(double value);

   GridFunction &operator=(const Vector &v);

   GridFunction &operator=(const GridFunction &v);

   FiniteElementSpace *FESpace() { return fes; }

   void Update() { SetSize(fes->GetVSize()); }

   void Update(FiniteElementSpace *f);

   void Update(FiniteElementSpace *f, Vector &v, int v_offset);

   /// Save the GridFunction to an output stream.
   virtual void Save(ostream &out);

   /** Write the GridFunction in VTK format. Note that Mesh::PrintVTK must be
       called first. The parameter ref must match the one used in
       Mesh::PrintVTK. */
   void SaveVTK(ostream &out, const string &field_name, int ref);

   void SaveSTL(ostream &out, int TimesToRefine = 1);

   /// Destroys grid function.
   virtual ~GridFunction();
};

void ComputeFlux(BilinearFormIntegrator &blfi,
                 GridFunction &u,
                 GridFunction &flux, int wcoef = 1, int sd = -1);

void ZZErrorEstimator(BilinearFormIntegrator &blfi,
                      GridFunction &u,
                      GridFunction &flux, Vector &ErrorEstimates,
                      int wsd = 1);

#endif
