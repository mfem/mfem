#include "mfem.hpp"

using namespace mfem;
using namespace std;

class AzimuthalECoefficient : public Coefficient
{
private:
   const GridFunction * vgf;
public:
   AzimuthalECoefficient(const GridFunction * vgf_)
      : Coefficient(), vgf(vgf_) {}
   virtual real_t Eval(ElementTransformation &T,
                       const IntegrationPoint &ip);
};

class EpsilonMatrixCoefficient : public MatrixArrayCoefficient
{
private:
   Mesh * mesh = nullptr;
   ParMesh * pmesh = nullptr;
   int num_procs = Mpi::WorldSize();
   int myid = Mpi::WorldRank();
   Array<ParGridFunction * > pgfs;
   Array<GridFunctionCoefficient * > gf_cfs;
   GridFunction * vgf = nullptr;
   int dim;
   int sdim;
public:
   EpsilonMatrixCoefficient(const char * filename, Mesh * mesh_, ParMesh * pmesh_,
                            real_t scale = 1.0);

   // Visualize the components of the matrix coefficient
   // in separate GLVis windows for each component
   void VisualizeMatrixCoefficient();
   // Update the Gridfunctions after mesh refinement
   void Update();

   ~EpsilonMatrixCoefficient();

};