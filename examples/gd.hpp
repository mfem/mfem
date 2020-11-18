#ifndef MFEM_GD
#define MFEM_GD

#include "mfem.hpp"
using namespace mfem;

namespace mfem
{
   /// Abstract class for Galerkin difference method using patch construction
    class GalerkinDifference : public FiniteElementSpace
    {

    public:
        /// Class constructor.
        /// \param[in] opt_file_name - file where options are stored

        GalerkinDifference(mfem::Mesh *pm, const mfem::FiniteElementCollection *f,
                           int vdim = 1, int ordering = mfem::Ordering::byVDIM,
                           int degree = 0);

        /// constructs the neighbour matrices for all mesh elements.
        /// and second neighbours (shared vertices).
        void BuildNeighbourMat(const Array<int> &els_id,
                               DenseMatrix &mat_cent,
                               DenseMatrix &mat_quad) const;

        /// constructs the neighbour set for given mesh element.
        /// \param[in]  id - the id of the element for which we need neighbour
        /// \param[in]  req_n - the required number of neighbours for patch
        /// \param[out] nels - the set of neighbours (may contain more element than required)
        void GetNeighbourSet(int id, int req_n, Array<int> &nels) const;

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
            if (cP)
            {
                std::cout << "cP is set.\n";
            }
            mfem::SparseMatrix *P = dynamic_cast<mfem::SparseMatrix *>(cP);
            if (P)
            {
                std::cout << "convert succeeded.\n";
            }
        }

        /// Build the prolongation matrix in GD method
        void BuildGDProlongation() const;

        /// Assemble the local reconstruction matrix into the prolongation matrix
        /// \param[in] id - vector of element id in patch
        /// \param[in] local_mat - the local reconstruction matrix
        /// problem to be solved: how the ensure the oder of dofs consistent with other forms?
        void AssembleProlongationMatrix(const Array<int> &id,
                                        const DenseMatrix &local_mat) const;

        virtual int GetTrueVSize() const { return nEle * vdim; }

    protected:
        /// mesh dimension
        int dim;
        /// number of elements in mesh
        int nEle;
        /// degree of lagrange interpolation
        int degree;

        const mfem::FiniteElementCollection *fec; // not owned
    };
} // end of namespace mfem
#endif // end of GALERKIN DIFF
