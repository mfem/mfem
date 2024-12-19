#pragma once

#include <mfem.hpp>

class SharedMemoryManager
{
private:
   struct MemoryBlock
   {
      char* ptr;
      int size;
      bool used;
   };

   MFEM_HOST_DEVICE static const int MAX_BLOCKS = 16;
   MFEM_HOST_DEVICE static MemoryBlock blocks[MAX_BLOCKS];
   MFEM_HOST_DEVICE static int num_blocks;
   MFEM_HOST_DEVICE static char* base_ptr;

public:
   MFEM_HOST_DEVICE static void init(void* shmem, int total_size)
   {
      base_ptr = static_cast<char*>(shmem);
      num_blocks = 1;
      blocks[0] = {base_ptr, total_size, false};
   }

   template<typename T>
   MFEM_HOST_DEVICE static T* reserve(int n)
   {
      int size_bytes = n * sizeof(T);
      for (int i = 0; i < num_blocks; ++i)
      {
         if (!blocks[i].used && blocks[i].size >= size_bytes)
         {
            blocks[i].used = true;
            if (blocks[i].size > size_bytes)
            {
               // Split block
               if (num_blocks < MAX_BLOCKS)
               {
                  blocks[num_blocks] = {blocks[i].ptr + size_bytes, blocks[i].size - size_bytes, false};
                  ++num_blocks;
                  blocks[i].size = size_bytes;
               }
            }
            return reinterpret_cast<T*>(blocks[i].ptr);
         }
      }
      return nullptr; // Allocation failed
   }

   MFEM_HOST_DEVICE static void release(void* ptr)
   {
      for (int i = 0; i < num_blocks; ++i)
      {
         if (blocks[i].ptr == ptr)
         {
            blocks[i].used = false;
            return;
         }
      }
   }

   MFEM_HOST_DEVICE static void release_and_try_merge(void* ptr)
   {
      for (int i = 0; i < num_blocks; ++i)
      {
         if (blocks[i].ptr == ptr)
         {
            blocks[i].used = false;
            merge_adjacent_free_blocks();
            return;
         }
      }
   }

private:
   MFEM_HOST_DEVICE static void merge_adjacent_free_blocks()
   {
      // Simple bubble sort for simplicity (can be optimized)
      for (int i = 0; i < num_blocks - 1; ++i)
      {
         for (int j = 0; j < num_blocks - i - 1; ++j)
         {
            if (blocks[j].ptr > blocks[j + 1].ptr)
            {
               MemoryBlock temp = blocks[j];
               blocks[j] = blocks[j + 1];
               blocks[j + 1] = temp;
            }
         }
      }

      for (int i = 0; i < num_blocks - 1; ++i)
      {
         if (!blocks[i].used && !blocks[i + 1].used)
         {
            blocks[i].size += blocks[i + 1].size;
            for (int j = i + 1; j < num_blocks - 1; ++j)
            {
               blocks[j] = blocks[j + 1];
            }
            --num_blocks;
            --i;
         }
      }
   }
};

MFEM_HOST_DEVICE SharedMemoryManager::MemoryBlock
SharedMemoryManager::blocks[SharedMemoryManager::MAX_BLOCKS];

MFEM_HOST_DEVICE int SharedMemoryManager::num_blocks;

MFEM_HOST_DEVICE char* SharedMemoryManager::base_ptr;
