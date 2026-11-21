#include "fsi_utils.hpp"

namespace mfem
{

    namespace common_fsi_utils
    {

        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                            Linalg utils                                              ///
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        void print_matrix(const DenseMatrix &A)
        {
            std::cout << std::scientific;
            std::cout << "{";
            for (int i = 0; i < A.NumRows(); i++)
            {
                std::cout << "{";
                for (int j = 0; j < A.NumCols(); j++)
                {
                    std::cout << A(i, j);
                    if (j < A.NumCols() - 1)
                    {
                        std::cout << ", ";
                    }
                }
                if (i < A.NumRows() - 1)
                {
                    std::cout << "}, ";
                }
                else
                {
                    std::cout << "}";
                }
            }
            std::cout << "}\n";
            std::cout << std::fixed;
            std::cout << std::endl
                      << std::flush; // Debugging print
        }



        // Mult and AddMult for full matrix (using matrices modified with WliminateRowsCols)
        void FullMult(HypreParMatrix *mat, HypreParMatrix *mat_e, Vector &x, Vector &y)
        {
            mat->Mult(x, y);      // y =  mat x
            mat_e->AddMult(x, y); // y += mat_e x
        }

        void FullAddMult(HypreParMatrix *mat, HypreParMatrix *mat_e, Vector &x, Vector &y, double a)
        {
            mat->AddMult(x, y, a);   // y +=  a mat x
            mat_e->AddMult(x, y, a); // y += a mat_e x
        }

        void MeanZero(ParGridFunction &gf, ParLinearForm *mass_lf, real_t volume)
        {
            // Make sure not to recompute the inner product linear form if already provided
            if (mass_lf == nullptr)
            {
                ConstantCoefficient onecoeff(1.0);
                mass_lf = new ParLinearForm(gf.ParFESpace());
                auto *dlfi = new DomainLFIntegrator(onecoeff);
                mass_lf->AddDomainIntegrator(dlfi);
                mass_lf->Assemble();

                ParGridFunction one_gf(gf.ParFESpace());
                one_gf.ProjectCoefficient(onecoeff);

                volume = mass_lf->operator()(one_gf);
            }

            if (volume == 0.0)
            {
                ConstantCoefficient onecoeff(1.0);
                ParGridFunction one_gf(gf.ParFESpace());
                one_gf.ProjectCoefficient(onecoeff);
                volume = mass_lf->operator()(one_gf);
            }

            real_t integ = mass_lf->operator()(gf);
            gf -= integ / volume;
        }

        void Orthogonalize(Vector &v, const MPI_Comm &comm)
        {
            real_t loc_sum = v.Sum();
            real_t global_sum = 0.0;
            int loc_size = v.Size();
            int global_size = 0;

            MPI_Allreduce(&loc_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(&loc_size, &global_size, 1, MPI_INT, MPI_SUM, comm);

            v -= global_sum / static_cast<real_t>(global_size);
        }


        // Implement logic && operator for Array<int>
        Array<int> operator&&(const Array<int> &a, const Array<int> &b)
        {
            MFEM_ASSERT(a.Size() == b.Size(), "Arrays must have the same size.");
            Array<int> result(a.Size());
            for (int i = 0; i < a.Size(); i++)
            {
                result[i] = a[i] && b[i];
            }
            return result;
        }


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///                                              Mesh utils                                              ///
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////

        void ExportMeshwithPartitioning(const std::string &outfolder, Mesh &mesh, const int *partitioning_, const std::string &filename)
        {
            // Extract the partitioning
            Array<int> partitioning;
            partitioning.MakeRef(const_cast<int *>(partitioning_), mesh.GetNE(), false);

            // Assign partitioning to the mesh
            FiniteElementCollection *attr_fec = new L2_FECollection(0, mesh.Dimension());
            FiniteElementSpace *attr_fespace = new FiniteElementSpace(&mesh, attr_fec);
            GridFunction attr(attr_fespace);
            for (int i = 0; i < mesh.GetNE(); i++)
            {
                attr(i) = partitioning[i] + 1;
            }

            // Create paraview datacollection
            ParaViewDataCollection paraview_dc(filename, &mesh);
            paraview_dc.SetPrefixPath(outfolder);
            paraview_dc.SetDataFormat(VTKFormat::BINARY);
            paraview_dc.SetCompressionLevel(9);
            paraview_dc.RegisterField("partitioning", &attr);
            paraview_dc.Save();

            // Clean up
            delete attr_fespace;
            delete attr_fec;
        }




    } // namespace common_fsi_utils

} // namespace mfem
