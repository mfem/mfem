
#include <mfem.hpp>
#include <mfem-performance.hpp>

using namespace mfem;
using namespace mfem::future;

// Dimension to instantiate:
constexpr int dimension = 3;

template <typename scalar_t, int... tensor_sizes>
void print_tensor_array(const tensor_array<scalar_t,tensor_sizes...> &ta)
{
   const auto nq = ta.size();
   for (std::size_t q = 0; q < nq; q++)
   {
      std::cout << "=== q = " << q << '\n';
      std::cout << ta.get_tensor(q) << '\n';
      // ta.get_tensor() works with both const and non-const scalar_t
   }
}

template <typename scalar_t, int dim>
MFEM_HOST_DEVICE
inline auto adj(const tensor<scalar_t,dim,dim> &a)
{
   using tensor_layout = StridedLayout2D<dim,dim,dim,1>; // row-major layout

   tensor<scalar_t,dim,dim> adja;
   auto adja_data = &adja(0,0);
   TAdjugateHD<scalar_t>(tensor_layout{}, &a(0,0),
                         tensor_layout{}, adja_data);
   return adja;
}

template <int dim>
void test_q_function(tensor_array<const real_t,dim,dim> &a,
                     tensor_array<real_t,dim,dim> &b)
{
   mfem::forall(a.size(), [=] MFEM_HOST_DEVICE (int q)
   {
      b(q) = adj(a(q));
   });
}

// Clang inlines the above function, so we explicitly instantiate it for
// dim = dimension to be able to look at the generated code on its own.
constexpr int d = dimension;
template void test_q_function<d>(tensor_array<const real_t,d,d> &,
                                 tensor_array<real_t,d,d> &);

template <int dim>
void test(std::size_t nq)
{
   // data for nq dim x dim tensors:
   Vector data(nq*dim*dim);
   for (int i = 0; i < data.Size(); i++)
   {
      data[i] = 1_r + i; // the real_t sequence: { 1, 2, 3, ... }
   }
   // tensor_array<const real_t,dim,dim> ta(data.HostRead(), {nq});
   auto ta = make_tensor_array<dim,dim>(data.HostRead(), nq);

   std::cout << "===== ta:\n";
   print_tensor_array(ta);

   Vector data2(nq*dim*dim);
   data2 = 0_r;
   // tensor_array<real_t,dim,dim> ta2(data2.HostReadWrite(), {nq});
   auto ta2 = make_tensor_array<dim,dim>(data2.HostReadWrite(), nq);

   std::cout << "\n===== test_q_function: begin";
   std::cout << "\n===== test_q_function: ta -> ta2";
   test_q_function<dim>(ta, ta2);
   std::cout << "\n===== test_q_function: end\n";

   std::cout << "\n===== ta:\n";
   print_tensor_array(ta);

   std::cout << "\n===== ta2:\n";
   print_tensor_array(ta2);

   // tensor_array<const real_t> sa(data.HostRead(),
   //                               {(std::size_t)data.Size()});
   auto sa = make_tensor_array(data.HostRead(), data.Size());
   std::cout << "\n===== 0D tensor (scalar) array, using ta's data, sa:\n";
   print_tensor_array(sa);

   // tensor_array<const real_t,dim,dim> tap(data.HostRead(), {nq});
   auto tap = make_tensor_array<dim,dim>(data.HostRead(), nq);
   tap.set_layout({2,1,0});
   std::cout << "\n===== tap: ta permuted with {2,1,0} (row-major layout):\n";
   print_tensor_array(tap);

   std::cout << "\n===== test_q_function: tap -> ta2\n";
   test_q_function<dim>(tap, ta2);

   std::cout << "\n===== ta2:\n";
   print_tensor_array(ta2);
}

int main(int argc, char *argv[])
{
   constexpr int dim = dimension;

   MFEM_VERIFY(argc == 2, "exactly 1 argument is expected!");
   std::size_t nq = atoi(argv[1]);
   MFEM_VERIFY(1 <= nq && nq <= 16,
               "the first argument must be an integer in [1,16]!");
   std::cout << "using nq = " << nq << std::endl;

   test<dim>(nq);

   return 0;
}
