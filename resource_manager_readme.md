# Introduction

Currently when working with Memory objects aliases, users must manually ensure that where they are valid is kept up to sync. For example:

```c++
    // full range: [0, 10)
    Memory<int> base(10);
    // alias0 points to [0, 5)
    Memory<int> alias0(base, 0, 5);
    // alias1 points to [5, 10)
    Memory<int> alias1(base, 5, 5);
    {
      auto ptr = alias1.Write(MemoryClass::HOST);
      for (int i = 0; i < alias1.Size(); ++i)
      {
        ptr[i] = i;
      }
    }
    {
      auto ptr = alias0.Write(MemoryClass::DEVICE);
      forall(alias0.Size(), [=] MFEM_HOST_DEVICE(int i) { ptr[i] = i + 10; });
    }
    // actual valid on device: [0, 5)
    // actual valid on host: [5, 10)
    // flags state:
    // - base is valid on host
    // - alias1 is valid on host
    // - alias0 is valid on device

    // in order to use base, alias0 and alias1 must be made valid somewhere
    // then have their flags manually synchronized with base
    // example: we want base to be valid on device
    alias1.Read(MemoryClass::DEVICE);
    // TODO: when should you use Sync vs. SyncAlias?
    base.Sync(alias0);

    // now safe to use base however we want
    {
      auto ptr = base.ReadWrite(MemoryClass::HOST);
      for (int i = 0; i < base.Size(); ++i)
      {
        ptr[i] *= 2;
      }
    }
    // need to re-sync aliases now if we want to use them again...
    // TODO: when should you use Sync vs. SyncAlias?
    alias0.Sync(base);
    alias1.Sync(base);
```

This is error prone and difficult to debug, especially if `base` or one of the aliased memories is passed into another function which will do the `Read`/`ReadWrite`/`Write` operation.

The goal of this PR is to remove the need to ever call `Sync` or `SyncAlias`, and just have everything work seamlessly for you.

```c++
    // full range: [0, 10)
    Resource<int> base(10, ResourceManager::HOST);
    // alias0 points to [0, 5)
    Resource<int> alias0(base, 0, 5);
    // alias1 points to [5, 10)
    Resource<int> alias1(base, 5, 5);
    {
      auto ptr = alias1.Write(ResourceManager::ANY_HOST);
      for (int i = 0; i < alias1.Size(); ++i)
      {
        ptr[i] = i;
      }
    }
    {
      auto ptr = alias0.Write(ResourceManager::ANY_DEVICE);
      forall(alias0.Size(), [=] MFEM_HOST_DEVICE(int i) { ptr[i] = i + 10; });
    }
    // actual valid on device: [0, 5)
    // actual valid on host: [5, 10)

    {
      // [0, 5) is copied from device to host
      auto ptr = base.ReadWrite(ResourceManager::ANY_HOST);
      for (int i = 0; i < base.Size(); ++i)
      {
        ptr[i] *= 2;
      }
    }

    // alias0 and alias1 both are valid on host, no copies are required
    {
      auto ptr0 = alias0.ReadWrite(ResourceManager::ANY_HOST);
      auto ptr1 = alias1.Read(ResourceManager::ANY_HOST);
      for (int i = 0; i < alias0.Size(); ++i)
      {
        ptr0[i] += ptr1[i];
      }
    }
```

Note: right now this PR implements a `Resource` and `ResourceManager` class. The goal is at some point `Resource` will be renamed to `Memory`, and similarly `ResourceManager` will be renamed to `MemoryManager` when the API is sufficient to be a drop-in replacement for the existing `Memory`/`MemoryManager` with no changes required in user code.

Existing supported features:

- External pointers

```c++
  int buffer[3];
  Resource<int> res(buffer, 3, ResourceManager::HOST);
```

- Temporary buffers

```c++
  for (int i = 0; i < 100; ++i)
  {
    // re-uses existing allocation, similar to PR #4065
    Resource<int> res(3, ResourceManager::HOST, true);
  }
```

- Support for host-pinned and managed memory spaces

```c++
  // HOSTPINNED and MANAGED memory spaces can also be used for temporary pools
  Resource<int> res(10, ResourceManager::HOSTPINNED);
  {
    auto ptr = res.Write(ResourceManager::ANY_HOST);
    for (int i = 0; i < res.Size(); ++i)
    {
      ptr[i] = i;
    }
  }
  {
    // no extra allocation/copies performed
    auto ptr = res.ReadWrite(ResourceManager::ANY_DEVICE);
    forall(res.Size(), [=] MFEM_HOST_DEVICE (int i) { ptr[i] += 1; });
  }
```

- Reference counted allocations

```c++
  // HOSTPINNED and MANAGED memory spaces can also be used for temporary pools
  Resource<int> res0(10, ResourceManager::HOST);

  // res1 is an alias of res0
  Resource<int> res1 = res0;
  {
    auto ptr = res0.Write(ResourceManager::ANY_HOST);
    for (int i = 0; i < res.Size(); ++i)
    {
      ptr[i] = i;
    }
  }
  // res0 allocates a new resource, res1 is still vallid to use
  res0 = Resource<int>(5, ResourceManager::MANAGED);
  {
    auto ptr1 = res1.Read(ResourceManager::ANY_DEVICE);
    // res1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    forall(ptr1.Size(), [=] MFEM_HOST_DEVICE(int i)
      {
        printf("res1[%d] = %d\n", i, ptr1[i]);
      });
  }
```

- ResourceManager tracks which Device backends are enabled (or not enabled) so common code can be written which supports CPU-only runs or mixed CPU/Device runs.

See `tests/unit/general/test_resource_manager.cpp` for additional usage features/examples.

# Implementation Details

The `ResourceManager` keeps an internal list of "segments" representing regions of memory. Each segment can contain one reference to another segment which potentially belongs to another memory space. The validity of these two segments will be managed simultaneously by `ResourceManager`.

Validity tracking is performed at the byte level. Each segment uses a red-black tree to mark where where the segment changes from valid to invalid (and vice versa). There is an implicit marker at the start that the segment is valid. Marking a region as valid or invalid is `O(log(M) + K)`, where `M` is the total number of valid/invalid transitions in the segment and `K` is the number of `valid/invalid` transitions within the region being marked.

# TODO

- Umpire support
- (deep) Copy data from one `Resource` (or raw pointer) to another `Resource`.
- General API and code cleanup/matching Memory/MemoryManager API:
  - `ResourceManager::ResourceLocation` -> `MemoryType`
  - What to do with `MemoryClass`? `ResourceManager::ResourceLocation` currently merges the use of `MemoryType` and `MemoryClass`
  - It might be possible to replace `RBase`/`RBTree` with `std::map`? At the very least `RBase` and `RBTree` can be cleaned up.
- Using `Resource` in a multi-thread context is basically impossible. Right now the only supported model is getting a raw pointer from `Read`/`ReadWrite`/`Write` on the main thread and passing the pointer to worker threads. What multi-threading model do we want to support?
- How to handle passing a duplicate pointer when requesting a new Resource/segment? Right now this will create a new unassociated segment.
```c++
int buffer[3];
Resource<int> res0(buffer, 3, ResourceManager::HOST);
Resource<int> res1(buffer, 3, ResourceManager::HOST);
// res0 and res1 will have distinct tracking of validity flags, with no easy way to synchronize the two.
auto ptr0 = res0.Write(ResourceManager::ANY_DEVICE);
auto ptr1 = res1.Write(ResourceManager::ANY_DEVICE);
// ptr0 != ptr1, there will be two device allocations
```
- Documentation
- Test renaming `Resource`->`Memory` and `ResourceManager`->`MemoryManager` and other API updates/cleanups to test compatibility.
- Performance testing
- Anything else?