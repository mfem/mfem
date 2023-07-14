#include "mfem.hpp"

using namespace std;
using namespace mfem;


class MPICommunicator
{
private:
    MPI_Comm comm;
    int myid, num_procs;
    int offset, lsize, gsize;
    std::vector<int> offsets; 
    Array<int> send_count;
    Array<int> send_displ;
    Array<int> recv_count;
    Array<int> recv_displ;

public:
    MPICommunicator(MPI_Comm comm_, int offset_, int gsize_);

    int get_rank(int dof);

    void Communicate(const Vector & x_s, Vector & x_r);
    void Communicate(const SparseMatrix & mat_s , SparseMatrix & mat_r);
    void Communicate(const Array<SparseMatrix*> & vmat_s, Array<SparseMatrix*> & vmat_r);
};
