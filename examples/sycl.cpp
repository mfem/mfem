#include <sycl/sycl.hpp>

#define MFEM_DEBUG_COLOR 51
#include "../general/debug.hpp"
#include "../general/forall.hpp"

int main(int argc, char *argv[]) {
  dbg();

  mfem::forall(4, [=] MFEM_HOST_DEVICE(int) {});

  /*sycl::queue q{};

  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(128), sycl::range<1>(32)),
        [=](sycl::nd_item<1> item) {
          sycl::multi_ptr<int[64], sycl::access::address_space::local_space>
              ptr = sycl::ext::oneapi::group_local_memory<int[64]>(
                  item.get_group());
          auto &ref = *ptr;
          ref[2 * item.get_local_linear_id()] = 42;
        });
  });*/
}