#include "mfem.hpp"
using namespace mfem;
extern "C" void
dgecon_(char *, int *, double *,  int *, double *,
        double *, double *,  int *,  int *);

extern "C" void
dgelss_(int *, int *, int *, double *, int *, double *, int *, double *,
        double *, int *, double *, int *, int *);
extern "C" void
dgels_(char *, int *, int *, int *, double *, int *, double *, int *, double *,
       int *, int *);

extern "C" void
dgelsy_(int *, int *, int *, double *, int *, double *, int *, int *, double *,
       int *, double *, int *, int *);
void buildLSInterpolation(int dim, int degree, const DenseMatrix &x_center,
                          const DenseMatrix &x_quad, DenseMatrix &interp);
void buildInterpolation(int dim, int degree, int output,
                        const DenseMatrix &x_center, const DenseMatrix &x_quad,
                        DenseMatrix &interp);
/// Abstract class for Galerkin difference method using patch construction
class GalerkinDifference : public FiniteElementSpace
{
protected:
   /// mesh dimension
   int dim;
   /// number of elements in mesh
   int nEle;
   /// degree of lagrange interpolation
   int degree;
   /// number of unknowns per dof
   int vdim;
   /// finite element collection
   const mfem::FiniteElementCollection *fec; // not owned
   /// vector containing info if an element is embedded or not
   std::vector<bool> EmbeddedElements;
   /// mesh 
   Mesh *mesh;
   /// cut cell size
   double scale;
public:
   /// Class constructor.
   GalerkinDifference(mfem::Mesh *pm, int di, int ne, const mfem::FiniteElementCollection *f,
                      std::vector<bool> EmbeddedElems, int vdi = 1, int ordering = mfem::Ordering::byVDIM,
                      int de = 0)
       : FiniteElementSpace(pm, f, vdi, ordering), mesh(pm), dim(di), nEle(ne), fec(f), EmbeddedElements(EmbeddedElems),
         degree(de), vdim(vdi) { BuildGDProlongation(); }
   //    GalerkinDifference::GalerkinDifference(Mesh *pm, const FiniteElementCollection *f,
   //    int vdim, int ordering, int de)
   //    : SpaceType(pm, f, vdim, ordering)
   // {
   //    degree = de;
   //    nEle = mesh->GetNE();
   //    dim = mesh->Dimension();
   //    fec = f;
   //    BuildGDProlongation();
   // }

   /// constructs the neighbour matrices for all mesh elements. 
   /// and second neighbours (shared vertices).
   /// \param[out] nmat1 - matrix of first neighbours
   /// \param[out] nmat1 - matrix of second neighbours
   /// \warning this function is going to be removed soon
   void BuildNeighbourMat(DenseMatrix &nmat1, DenseMatrix &nmat2);

   /// An overload function for build the densmatrix
   void BuildNeighbourMat(const Array<int> &els_id,
                          DenseMatrix &mat_cent,
                          DenseMatrix &mat_quad) const;
   
      /// An overload function for build the densmatrix
   void BuildNeighbourMat(const Array<int> &els_id,
                          DenseMatrix &mat_cent,
                          DenseMatrix &mat_quad, int di) const;
   /// constructs the neighbour set for given mesh element. 
   /// \param[in]  id - the id of the element for which we need neighbour
   /// \param[in]  req_n - the required number of neighbours for patch
   /// \param[out] nels - the set of neighbours (may contain more element than required)
   void GetNeighbourSet(int id, int req_n, Array<int> &nels) const;
  
  /// constructs the neighbour set for given mesh element. 
   /// \param[in]  id - the id of the element for which we need neighbour
   /// \param[in]  req_n - the required number of neighbours for patch
   /// \param[out] nels_x - the set of neighbours in `x` dir (may contain more element than required)
   /// \param[out] nels_y - the set of neighbours in `y` dir (may contain more element than required)
   virtual void GetNeighbourSet(int id, int req_n, Array<int> &nels_x, Array<int> &nels_y) const;
   /// provides the center (barycenter) of an element
   /// \param[in]  id - the id of the element for which we need barycenter
   /// \param[out] cent - the vector of coordinates of center of an element
   void GetElementCenter(int id, mfem::Vector &cent) const;
   SparseMatrix *GetCP() { return cP; }
   /// Get the prolongation matrix in GD method
   virtual const Operator *GetProlongationMatrix() const
   { 
      if (!cP)
      {
         BuildGDProlongation();
         return cP;
      }
      else
      {
         return cP; 
      }
   }

   void checkpcp()
   {
      if (cP) {std::cout << "cP is set.\n";}
      mfem::SparseMatrix *P = dynamic_cast<mfem::SparseMatrix *> (cP);
      if (P) {std::cout << "convert succeeded.\n";}
   }

   /// Build the prolongation matrix in GD method
   void BuildGDProlongation() const;

   /// Assemble the local reconstruction matrix into the prolongation matrix
   /// \param[in] id - vector of element id in patch
   /// \param[in] local_mat - the local reconstruction matrix
   /// problem to be solved: how the ensure the oder of dofs consistent with other forms?
   void AssembleProlongationMatrix(const Array<int> &id,
                           const DenseMatrix &local_mat) const;

   /// check the duplication of quadrature points in the quad matrix
   bool duplicated(const mfem::Vector quad, const std::vector<double> data);

   virtual int GetTrueVSize() const {return nEle * vdim; }

};
