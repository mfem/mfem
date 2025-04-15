#ifndef MFEM_POINTER_UTILS // saved off as #include "pointer_utils.hpp"
#define MFEM_POINTER_UTILS

#include <memory>
#include <type_traits>
#include <utility>

namespace mfem
{

namespace ptr_utils
{

// --------------------- Smart Pointer Utilities -----------------------

/**
 * @brief Creates a shared_ptr from a raw pointer without taking ownership.
 * 
 * Use this when you have a raw pointer that's managed elsewhere (e.g., by a parent object)
 * and you need to create a temporary shared_ptr that won't delete the object.
 * 
 * @warning Be extremely careful with this function! It should only be used when
 * you're certain the target object's lifetime exceeds the shared_ptr's lifetime.
 */
template<typename T>
std::shared_ptr<T> borrow_ptr(T* raw_ptr) {
    // Create shared_ptr with empty deleter - it won't delete the pointer
    return std::shared_ptr<T>(raw_ptr, [](T*){});
}

/**
 * @brief Creates a shared_ptr that takes ownership of a raw pointer.
 * 
 * Use this when transitioning legacy code that allocates with new
 * to code that uses smart pointers.
 */
template<typename T>
std::shared_ptr<T> take_ownership(T* raw_ptr) {
    return std::shared_ptr<T>(raw_ptr);
}

/**
 * @brief Creates a shared_ptr from a raw pointer with a custom deleter.
 * 
 * Use this when the object requires special cleanup logic.
 */
template<typename T, typename Deleter>
std::shared_ptr<T> custom_deleter(T* raw_ptr, Deleter deleter) {
    return std::shared_ptr<T>(raw_ptr, deleter);
}

/**
 * @brief Gets a raw pointer from a shared_ptr or returns the input if it's already a raw pointer.
 * 
 * This is useful for functions that need to accept either smart or raw pointers.
 */
template<typename T>
T* get_raw(T* ptr) {
    return ptr;
}

template<typename T>
T* get_raw(const std::shared_ptr<T>& ptr) {
    return ptr.get();
}

/**
 * @brief Type trait to check if T is a shared_ptr
 */
template<typename T>
struct is_shared_ptr : std::false_type {};

template<typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};

// --------------------- MFEM-Specific Smart Pointer Utilities -----------------------

// Forward declarations of common MFEM classes that often need this treatment
class Mesh;
class FiniteElementSpace;
class FiniteElementCollection;
class GridFunction;
class QuadratureSpaceBase;
class QuadratureSpace;
class FaceQuadratureSpace;
class QuadratureFunction;

/**
 * @brief Creates a shared_ptr<Mesh> from a raw Mesh pointer without taking ownership.
 * 
 * This is commonly needed when working with existing MFEM code that passes raw Mesh pointers.
 */
inline std::shared_ptr<Mesh> BorrowMesh(Mesh* mesh) {
    return borrow_ptr(mesh);
}

/**
 * @brief Creates a shared_ptr<QuadratureSpaceBase> from a raw QuadratureSpaceBase pointer without taking ownership.
 */
inline std::shared_ptr<QuadratureSpaceBase> BorrowQSpace(QuadratureSpaceBase* qspace) {
    return borrow_ptr(qspace);
}

/**
 * @brief Function to safely determine ownership of a raw pointer when creating a shared_ptr.
 * 
 * This can be used in constructors that currently take raw pointers but are being 
 * transitioned to use shared_ptr internally.
 * 
 * @param ptr The raw pointer
 * @param take_ownership Flag indicating if the created shared_ptr should take ownership
 * @return A shared_ptr that either owns or borrows the raw pointer
 */
template<typename T>
std::shared_ptr<T> MakeShared(T* ptr, bool take_ownership = false) {
    if (take_ownership) {
        return std::shared_ptr<T>(ptr);
    } else {
        return borrow_ptr(ptr);
    }
}

// ------------------ Adapter Pattern for Backward Compatibility ------------------

/**
 * @brief Base class for adapters that can work with both raw and smart pointers.
 * 
 * This class provides a unified interface for objects that need to work with
 * both raw pointers and smart pointers during the transition period.
 */
template<typename T>
class PointerAdapter {
protected:
    std::shared_ptr<T> ptr_;

public:
    // Constructor from raw pointer (borrowing, not taking ownership)
    PointerAdapter(T* raw_ptr) : ptr_(borrow_ptr(raw_ptr)) {}
    
    // Constructor from shared_ptr
    PointerAdapter(std::shared_ptr<T> smart_ptr) : ptr_(std::move(smart_ptr)) {}
    
    // Get the raw pointer (for backward compatibility)
    T* get() const { return ptr_.get(); }
    
    // Get the shared_ptr (for modern code)
    std::shared_ptr<T> get_shared() const { return ptr_; }
    
    // Dereference operators
    T& operator*() const { return *ptr_; }
    T* operator->() const { return ptr_.get(); }
    
    // Bool conversion for null checking
    operator bool() const { return ptr_ != nullptr; }
};

/**
 * @brief Specialization for Mesh pointers which are especially common in MFEM.
 */
using MeshAdapter = PointerAdapter<Mesh>;

/**
 * @brief Specialization for QuadratureSpace pointers.
 */
using QSpaceAdapter = PointerAdapter<QuadratureSpaceBase>;

} // namespace ptr_utils

} // namespace mfem

#endif // MFEM_POINTER_UTILS