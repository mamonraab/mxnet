/*!
 *  Copyright (c) 2015 by Contributors
 * \file ndarray.h
 * \brief NDArray interface that handles array arithematics.
 */
#ifndef MXNET_NDARRAY_H_
#define MXNET_NDARRAY_H_

#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <dmlc/registry.h>
#include <nnvm/node.h>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include "./base.h"
#include "./storage.h"
#include "./engine.h"
#if MKL_EXPERIMENTAL == 1
#include <mkl_memory.h>
#endif
// check c++11
#if DMLC_USE_CXX11 == 0
#error "cxx11 was required for ndarray module"
#endif

namespace mxnet {
// forward declaration
class NDArray;
namespace autograd {
class AGNode;

using AGNodePtr = std::shared_ptr<AGNode>;

class AGNodeEntry {
 public:
  AGNodePtr ag_node;
  uint32_t index;
  uint32_t version;

  void clear() {
    ag_node.reset();
    index = version = 0;
  }

  nnvm::NodeEntry nn_entry() const;
};

class AutogradRuntime;
}  // namespace autograd

// enum for storage types
#define CSR_IND_PTR_TYPE mshadow::kInt32
#define CSR_IDX_DTYPE mshadow::kInt32
#define ROW_SPARSE_IDX_TYPE mshadow::kInt32
// FIXME int64_t is not available mshadow
namespace csr {
enum CSRAuxType {kIndPtr, kIdx};
}

namespace rowsparse {
enum RowSparseAuxType {kIdx};
}

enum NDArrayStorageType {
  kUndefinedStorage = -1,  // undefined storage
  kDefaultStorage,         // dense
  kRowSparseStorage,       // row sparse
  kCSRStorage,             // csr
};

/*!
 * \brief issue an copy operation from one NDArray to another
 *  the two ndarray can sit on different devices
 *  this operation will be scheduled by the engine
 *
 * \param from the ndarray we want to copy data from
 * \param to the target ndarray
 * \param priority Priority of the action.
 * \param alloc_output whether to allocate memory for the output ndarray
 * \note The function name explicitly marks the order of from and to
 *     due to different possible convention carried by copy function.
 */
void CopyFromTo(const NDArray &from, NDArray *to, int priority = 0, bool alloc_output = true);

/*!
 * \brief ndarray interface
 */
class NDArray {
 public:
  /*! \brief default cosntructor */
  NDArray() {
#if MKL_EXPERIMENTAL == 1
      Mkl_mem_ = MKLMemHolder::create();
#endif
  }
  /*!
   * \brief constructs a new dynamic NDArray
   * \param shape the shape of array
   * \param ctx context of NDArray
   * \param delay_alloc whether delay the allocation
   * \param dtype data type of this ndarray
   */
  NDArray(const TShape &shape, Context ctx,
          bool delay_alloc = false, int dtype = mshadow::default_type_flag)
      : ptr_(std::make_shared<Chunk>(shape, ctx, delay_alloc, dtype)),
        shape_(shape), offset_(0), dtype_(dtype), entry_({nullptr, 0, 0}) {
#if MKL_EXPERIMENTAL == 1
      Mkl_mem_ = std::make_shared<MKLMemHolder>();
#endif
  }
  /*! \brief constructor for NDArray with storage type
   */
  NDArray(const NDArrayStorageType storage_type, const TShape &shape, Context ctx,
          bool delay_alloc = true, int dtype = mshadow::default_type_flag,
          std::vector<int> aux_types = {})
      : shape_(shape), offset_(0), dtype_(dtype), entry_({nullptr, 0, 0}) {
      // Assign default aux types if not given
      if (aux_types.size() == 0) {
        if (storage_type == kRowSparseStorage) {
          aux_types = {ROW_SPARSE_IDX_TYPE};
        } else if (storage_type == kCSRStorage) {
          aux_types = {CSR_IND_PTR_TYPE, CSR_IDX_DTYPE};
        } else {
          LOG(FATAL) << "Unknown storage type";
        }
      }
      ptr_ = std::make_shared<Chunk>(ctx, delay_alloc, aux_types, storage_type);
#if MKL_EXPERIMENTAL == 1
      Mkl_mem_ = std::make_shared<MKLMemHolder>();
#endif
      CHECK(storage_type == kRowSparseStorage) << "Only kRowSparseStorage is supported";
  }
  /*!
   * \brief constructing a static NDArray that shares data with TBlob
   *  Use with caution: allocate ONLY ONE NDArray for each TBlob,
   *  make sure the memory region is available through out the life of NDArray
   * \param data the memory content of static data
   * \param dev_id the device id this tensor sits at
   * \param shared_var the same var handle shared with others.
            It will not be deleted during destruction.
   */
  NDArray(const TBlob &data, int dev_id, Engine::VarHandle shared_var = nullptr)
      : ptr_(std::make_shared<Chunk>(data, dev_id, shared_var)), shape_(data.shape_), offset_(0),
        dtype_(data.type_flag_), entry_({nullptr, 0, 0}) {
#if MKL_EXPERIMENTAL == 1
      Mkl_mem_ = std::make_shared<MKLMemHolder>();
#endif
  }
  NDArray(NDArray data, const std::vector<NDArray> aux_data, Context ctx,
          NDArrayStorageType storage_type, const TShape &shape)
      : ptr_(std::make_shared<Chunk>(data, aux_data, ctx, storage_type)), shape_(shape),
        offset_(0), dtype_(data.data().type_flag_), entry_({nullptr, 0, 0}) {
#if MKL_EXPERIMENTAL == 1
      Mkl_mem_ = std::make_shared<MKLMemHolder>();
#endif
      CHECK(aux_data.size() == 1) << "Multiple aux_data not supported yet";
  }

  /*!
   * \return the shape of current NDArray.
   */
  inline const TShape &shape() const {
    return shape_;
  }
  /*!
   * \return the shape of underlying chunk which stores the NDArray values. 
   *  For default storage, it is the same as shape(). For row-sparse storage, it is the shape of
   *  the tensor which stores the non-zero values.
   */
  inline const TShape &storage_shape() const {
    CHECK(ptr_ != nullptr);
    return ptr_->storage_shape;
  }
  /*!
   * \return the shape of aux data at ith index. If it doesn't exist, return an empty one.
   */
  // TODO(haibin) CamelCase
  inline const TShape aux_shape(size_t i) const {
    CHECK(storage_type() != kDefaultStorage);
    if (i >= ptr_->aux_shapes.size()) return TShape();
    return ptr_->aux_shapes[i];
  }
  inline void SetAuxShape(size_t i, const TShape& shape) const {
    ptr_->aux_shapes[i] = shape;
  }
  /*!
   * \return the data TBlob
   */
  inline TBlob data() const {
    CHECK(ptr_ != nullptr);
    TBlob res;
    TShape shape = shape_;
    if (storage_type() != kDefaultStorage) {
      CHECK(offset_ == 0) << "Non-default storage should never set offset_";
      shape = storage_shape();
    }
    MSHADOW_TYPE_SWITCH(dtype(), DType, {
      CHECK(ptr_->shandle.dptr != nullptr);
      res = TBlob(static_cast<DType*>(ptr_->shandle.dptr)
        + offset_, shape, ptr_->shandle.ctx.dev_mask(), dtype());
    });
#if MKL_EXPERIMENTAL == 1
    res.Mkl_mem_ = Mkl_mem_;
#endif
    return res;
  }
  /*!
   * \return the aux TBlob
   */
  inline TBlob aux_data(size_t i) const {
    CHECK(storage_type() != kDefaultStorage);
    TBlob res;
    CHECK(i < ptr_->aux_handles.size());
    MSHADOW_TYPE_SWITCH(aux_type(i), DType, {
      res = TBlob(static_cast<DType*>(ptr_->aux_handles[i].dptr), aux_shape(i),
                  ptr_->aux_handles[i].ctx.dev_mask(), aux_type(i));
    });
#if MKL_EXPERIMENTAL == 1
    res.Mkl_mem_ = Mkl_mem_;
#endif
    return res;
  }
  /*!
   * \return a chunk of raw data in TBlob
   */
  inline TBlob raw_data(index_t offset, index_t length) const {
    CHECK(storage_type() == kDefaultStorage);
    TBlob res;
    TShape raw_shape(1);
    raw_shape[0] = length;
    MSHADOW_TYPE_SWITCH(dtype_, DType, {
      res = TBlob(static_cast<DType*>(ptr_->shandle.dptr)
        + offset_ + offset, raw_shape, ptr_->shandle.ctx.dev_mask());
    });
#if MKL_EXPERIMENTAL == 1
    res.Mkl_mem_ = Mkl_mem_;
#endif
    return res;
  }
  /*!
   * \return the context of NDArray, this function is only valid when the NDArray is not empty
   */
  inline Context ctx() const {
    return ptr_->shandle.ctx;
  }
  /*!
   * \return the data type of NDArray, this function is only valid when the NDArray is not empty
   */
  inline int dtype() const {
    return dtype_;
  }
  inline int aux_type(size_t i) const {
    CHECK(ptr_ != nullptr);
    return ptr_->aux_types[i];
  }
  inline NDArrayStorageType storage_type() const {
    if (is_none()) return kUndefinedStorage;
    return ptr_->storage_type;
  }
  /*! \return whether this ndarray is not initialized */
  inline bool is_none() const {
    return ptr_.get() == nullptr;
  }
  /*!
   * \brief Block until all the pending write operations with respect
   *    to current NDArray are finished, and read can be performed.
   */
  inline void WaitToRead() const {
    if (is_none()) return;
    Engine::Get()->WaitForVar(ptr_->var);
  }
  /*!
   * \brief Block until all the pending read/write operations with respect
   *    to current NDArray are finished, and write can be performed.
   */
  inline void WaitToWrite() const {
    if (is_none()) return;
    /*!
     * Push an empty mutable function to flush all preceding reads to the
     * variable.
     */
    Engine::Get()->PushSync([](RunContext) {}, Context{}, {}, {ptr_->var});
    Engine::Get()->WaitForVar(ptr_->var);
  }
  /*! \return the associated variable of the ndarray.*/
  inline Engine::VarHandle var() const {
    return ptr_->var;
  }
  /*!
   * \brief save the content into binary stream
   * \param strm the output stream
   */
  void Save(dmlc::Stream *strm) const;
  /*!
   * \brief load the content from binary stream
   * \param strm the output stream
   * \return whether the load is successful
   */
  bool Load(dmlc::Stream *strm);
  /*!
   * \brief set all the elements in ndarray to be scalar
   * \param scalar the scalar to set
   * \return reference of self
   */
  NDArray &operator=(real_t scalar);
  /*!
   * \brief elementwise add to current space
   *  this mutate the current NDArray
   * \param src the data to add
   * \return reference of self
   */
  NDArray &operator+=(const NDArray &src);
  /*!
   * \brief elementwise add to current space
   *  this mutate the current NDArray
   * \param src the data to add
   * \return reference of self
   */
  NDArray &operator+=(const real_t &src);
  /*!
   * \brief elementwise subtract from current ndarray
   * this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator-=(const NDArray &src);
  /*!
   * \brief elementwise subtract from current ndarray
   * this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator-=(const real_t &src);
  /*!
   * \brief elementwise multiplication to current ndarray
   *  this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator*=(const NDArray &src);
  /*!
   * \brief elementwise multiplication to current ndarray
   *  this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator*=(const real_t &src);
  /*!
   * \brief elementwise division from current ndarray
   *  this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator/=(const NDArray &src);
  /*!
   * \brief elementwise division from current ndarray
   *  this mutate the current NDArray
   * \param src the data to subtract
   * \return reference of self
   */
  NDArray &operator/=(const real_t &src);
  /*!
   * \brief return transpose of current NDArray
   * \return a new transposed NDArray
   */
  NDArray T() const;
  /*!
   * \brief return a new copy this NDArray
   * \param ctx the new context of this NDArray
   * \return the new copy
   */
  NDArray Copy(Context ctx) const;
  /*!
   * \brief Do a synchronize copy from a continugous CPU memory region.
   *
   *  This function will call WaitToWrite before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copy from.
   * \param size the size of the source array, in sizeof(DType) not raw btyes.
   */
  void SyncCopyFromCPU(const void *data, size_t size) const;
  /*!
   * \brief Do a synchronize copy to a continugous CPU memory region.
   *
   *  This function will call WaitToRead before the copy is performed.
   *  This is useful to copy data from existing memory region that are
   *  not wrapped by NDArray(thus dependency not being tracked).
   *
   * \param data the data source to copyinto.
   * \param size the memory size we want to copy into, in sizeof(DType) not raw btyes.
   */
  void SyncCopyToCPU(void *data, size_t size) const;
  /*!
   * \brief Slice a NDArray
   * \param begin begin index in first dim
   * \param end end index in first dim
   * \return sliced NDArray
   */
  inline NDArray Slice(index_t begin, index_t end) const {
    NDArray ret = *this;
    CHECK(!is_none()) << "NDArray is not initialized";
    CHECK_GE(shape_[0], end) << "Slice end index out of range";
    CHECK(storage_type() == kDefaultStorage) << "Slice not yet implemented for storage "
                                             << storage_type();
    size_t length = shape_.ProdShape(1, shape_.ndim());
    ret.offset_ += begin * length;
    ret.shape_[0] = end - begin;
    return ret;
  }
  /*!
   * \brief Index a NDArray
   * \param idx the index
   * \return idx-th sub array NDArray
   */
  inline NDArray At(index_t idx) const {
    NDArray ret = *this;
    CHECK(!is_none()) << "NDArray is not initialized";
    CHECK_GT(shape_[0], idx) << "index out of range";
    CHECK(storage_type() == kDefaultStorage) << "Storage type "
                                             << storage_type() << " doesn't support At()";
    size_t length = shape_.ProdShape(1, shape_.ndim());
    ret.offset_ += idx * length;
    if (shape_.ndim() > 1) {
      ret.shape_ = TShape(shape_.data()+1, shape_.data()+shape_.ndim());
    } else {
      ret.shape_ = mshadow::Shape1(1);
    }
    return ret;
  }
  /*!
   * \brief Create a NDArray that shares memory with current one
   *  The new array must have smaller memory size than the current array.
   * \param shape new shape
   * \param dtype The data type.
   * \return NDArray in new shape and type.
   */
  inline NDArray AsArray(const TShape &shape, int dtype) const {
    CHECK_EQ(storage_type(), kDefaultStorage) << "Not implemented yet";
    CHECK_GE(shape_.Size() * mshadow::mshadow_sizeof(dtype_),
             shape.Size() * mshadow::mshadow_sizeof(dtype))
        << "NDArray.AsArray: target memory size is bigger";
#if MKL_EXPERIMENTAL == 1
    if (Mkl_mem_ != nullptr) {
      // convert prv to cpu
      Mkl_mem_->check_and_prv_to_cpu(ptr_->shandle.dptr);
    }
#endif
    NDArray ret = *this;
    ret.shape_ = shape;
    ret.dtype_ = dtype;
    return ret;
  }
  /*!
   * \brief Get an reshaped NDArray
   * \param shape new shape
   * \return NDArray in new shape
   */
  inline NDArray Reshape(const TShape &shape) const {
    CHECK(storage_type() == kDefaultStorage) << "Not implemented yet";
    CHECK_GE(shape_.Size(), shape.Size())
        << "NDArray.Reshape: target shape size is different from current shape";
    NDArray ret = *this;
    ret.shape_ = shape;
    return ret;
  }
  /*!
   * \brief Allocate the space if it is delayed allocated.
   * This is an internal function used by system that normal user should not use
   */
  inline void CheckAndAlloc() const {
    CHECK_EQ(storage_type(), kDefaultStorage);
    ptr_->CheckAndAlloc();
  }
  /* !
   * \brief Alloc memory for non-default storage
   * aux_shape is only known at run time
   */
  inline void CheckAndAlloc(const std::vector<TShape> &aux_shapes) const {
    CHECK_NE(storage_type(), kDefaultStorage);
    ptr_->CheckAndAlloc(shape_, aux_shapes, dtype_);
  }
  inline void CheckAndAllocData(const TShape &storage_shape) const {
    CHECK_NE(storage_type(), kDefaultStorage);
    ptr_->CheckAndAllocData(storage_shape, dtype_);
  }
  inline void CheckAndAllocAuxData(size_t i, const TShape &aux_shape) const {
    CHECK_NE(storage_type(), kDefaultStorage);
    ptr_->CheckAndAllocAuxData(i, aux_shape);
  }
  /*!
   * \brief Save list of narray into the Stream.x
   * \param fo The stream of output.
   * \param data the NDArrays to be saved.
   * \param names the name of the NDArray, optional, can be zero length.
   */
  static void Save(dmlc::Stream* fo,
                   const std::vector<NDArray>& data,
                   const std::vector<std::string>& names);
  /*!
   * \brief Load list of narray into from the stream.
   * \param fi The stream of the input file.
   * \param data the NDArrays to be loaded
   * \param keys the name of the NDArray, if saved in the file.
   */
  static void Load(dmlc::Stream* fi,
                   std::vector<NDArray>* data,
                   std::vector<std::string>* keys);

 private:
  friend class autograd::AutogradRuntime;
  /*! \brief the real data chunk that backs NDArray */
  // shandle is used to store the actual values in the NDArray
  // aux_handles store the aux data(such as indices) if it's needed by non-default storage.
  struct Chunk {
    /*! \brief storage handle from storage engine.
               for non-default storage, shandle stores the data(value) array.
     */
    Storage::Handle shandle;
    /*! \brief storage handles for aux data (e.g index)
               for row_sparse, aux_handles[0] = indices
               for csr, aux_handles[0] = indptr, aux_handles[1] = indices
    */
    std::vector<Storage::Handle> aux_handles;
    /*! \brief variable from engine */
    Engine::VarHandle var;
    /*!
     * \brief if this is true, this means the data do not come
     * from Storage, and do not need to be freed
     */
    bool static_data;
    /*! \brief whether allocation is delayed. */
    bool delay_alloc;
    /*! \brief construct from static data */
    NDArrayStorageType storage_type = kDefaultStorage;
    /*! \brief type of aux */
    std::vector<int> aux_types;
    // context of data
    Context ctx;
    // The shape of the chunk data.
    // This might not be the same shape as the NDArray, since the storage may be sparse.
    TShape storage_shape;
    // The shape of aux data. The default value for the shape is 0.
    std::vector<TShape> aux_shapes;
    // \brief skip the deletion of var handle. Usually set when shared_var is present.
    bool skip_delete_var = false;

    /*! \brief construct a new chunk */
    Chunk(TShape shape, Context ctx_, bool delay_alloc_, int dtype)
        : static_data(false), delay_alloc(true), ctx(ctx_) {
      auto size = shape.Size();
      storage_shape = shape;
      var = Engine::Get()->NewVariable();
      shandle.size = size * mshadow::mshadow_sizeof(dtype);
      shandle.ctx = ctx_;
      if (!delay_alloc_) this->CheckAndAlloc();
    }
    // construct a chunk by copying over data
    Chunk(const NDArray &nd, const std::vector<NDArray> &nd_aux, Context ctx_,
          NDArrayStorageType storage_type_)
        : static_data(false), delay_alloc(false), storage_type(storage_type_), ctx(ctx_) {
      // Vars
      var = Engine::Get()->NewVariable();
      // Single threaded copy may not saturate memory bandwidth
      // Aux shapes, types and storage
      for (size_t i = 0; i < nd_aux.size(); i++) {
        const auto &aux_d = nd_aux[i].data();
        aux_types.emplace_back(aux_d.type_flag_);
        // Empty. Skip copy
        if (aux_d.shape_.ndim() == 0) continue;
        CheckAndAllocAuxData(i, aux_d.shape_);
        // Copy aux data
        CHECK_EQ(nd_aux[i].storage_type(), kDefaultStorage);
        TBlob aux_blob(aux_handles[i].dptr, aux_shapes[i], ctx.dev_mask(), aux_types[i]);
        NDArray aux_wrapper(aux_blob, ctx.dev_id, var);
        CopyFromTo(nd_aux[i], &aux_wrapper, 0, false);
      }
      // Data Storage
      const auto &data = nd.data();
      // Empty. Skip Copy
      if (data.shape_.ndim() == 0) return;
      CheckAndAllocData(data.shape_, data.type_flag_);
      CHECK_EQ(nd.storage_type(), kDefaultStorage);
      auto data_blob = TBlob(shandle.dptr, storage_shape, shandle.ctx.dev_mask(), data.type_flag_);
      NDArray data_wrapper(data_blob, ctx.dev_id, var);
      CopyFromTo(nd, &data_wrapper, 0, false);
    }

    Chunk(const TBlob &data, int dev_id, Engine::VarHandle shared_var)
        : static_data(true), delay_alloc(false) {
      CHECK(storage_type == kDefaultStorage);
      // init var
      if (shared_var == nullptr) {
        var = Engine::Get()->NewVariable();
      } else {
        skip_delete_var = true;
        var = shared_var;
      }
      // init ctx
      if (data.dev_mask_ == cpu::kDevMask) {
        ctx = Context::CPU();
      } else {
        CHECK_EQ(data.dev_mask_, gpu::kDevMask);
        ctx = Context::GPU(dev_id);
      }
      // init shandle
      shandle.ctx = ctx;
      shandle.dptr = data.dptr_;
      shandle.size = data.shape_.Size() * mshadow::mshadow_sizeof(data.type_flag_);
      storage_shape = data.shape_;
    }
    Chunk(Context ctx_, bool delay_alloc_, std::vector<int> aux_types_,
          NDArrayStorageType storage_type_)
        : static_data(false), delay_alloc(delay_alloc_), storage_type(storage_type_),
          aux_types(aux_types_), ctx(ctx_) {
      var = Engine::Get()->NewVariable();
      // Assume alloc is always delayed for non-default storage type
      CHECK(delay_alloc_);
      if (!delay_alloc_) {
        this->CheckAndAlloc();
      }
    }
    /*! \brief check if delay alloc is on, do alloc if not yet done */
    inline void CheckAndAlloc(void) {
      if (delay_alloc) {
        shandle = Storage::Get()->Alloc(shandle.size, shandle.ctx);
        delay_alloc = false;
      }
    }
    inline void CheckAndAlloc(const TShape &shape, const std::vector<TShape> &aux_shapes,
                              int dtype) {
      // calculate size, perform allocation
      if (delay_alloc) {
        CHECK_EQ(storage_type, kRowSparseStorage) << "Not yet implemented";
        // For row sparse, aux_shape indicates the number of rows to allocate
        auto aux_shape = aux_shapes[0];
        CHECK_EQ(shape.ndim(), 2) << "High dim RowSparse not yet implemented";
        CheckAndAllocAuxData(rowsparse::kIdx, aux_shape);
        TShape storage_shape(shape);
        storage_shape[0] = aux_shape[0];
        CheckAndAllocData(storage_shape, dtype);
      }
    }
    // create storage handle for data based on shape and dtype, assuming ctx is set
    // shandle and storage shape are updated
    inline void CheckAndAllocData(const TShape &shape, int dtype) {
      CHECK_NE(aux_shapes.size(), 0) << "data is expected to be allocated after aux_data";
      // init shape
      storage_shape = shape;
      // init storage
      auto dbytes = shape.Size() * mshadow::mshadow_sizeof(dtype);
      shandle = Storage::Get()->Alloc(dbytes, ctx);
      // delay_alloc is only set when data storage handle is present
      delay_alloc = false;
    }
    // create storage handle for aux data based on shape, assuming ctx and aux type are set
    // aux_handle and aux shape are updated
    inline void CheckAndAllocAuxData(size_t i, const TShape &shape) {
      CHECK_EQ(aux_shapes.size(), aux_handles.size());
      if (aux_shapes.size() <= i) {
        aux_shapes.resize(i + 1);
        aux_handles.resize(i + 1);
      }
      // init shape
      aux_shapes[i] = shape;
      // init aux storage
      Storage::Handle aux_handle;
      if (storage_type == kRowSparseStorage) {
        auto aux_bytes = shape[0] * mshadow::mshadow_sizeof(aux_types[i]);
        aux_handle = Storage::Get()->Alloc(aux_bytes, ctx);
      } else if (storage_type == kCSRStorage) {
        LOG(FATAL) << "Not implemented";
      }
      aux_handles[i] = aux_handle;
    }
    /*! \brief destructor */
    ~Chunk() {
      if (skip_delete_var) return;
      bool skip_free = static_data || delay_alloc;
      Storage::Handle h = this->shandle;
      std::vector<Storage::Handle> aux_h = this->aux_handles;
      Engine::Get()->DeleteVariable([h, aux_h, skip_free](RunContext s) {
        if (skip_free == false) {
          Storage::Get()->Free(h);
          for (size_t i = 0; i < aux_h.size(); i++) {
            Storage::Get()->Free(aux_h[i]);
          }
        }
      }, shandle.ctx, var);
    }
  };

#if MKL_EXPERIMENTAL == 1
  std::shared_ptr<MKLMemHolder> Mkl_mem_;
#endif
  /*! \brief internal data of NDArray */
  std::shared_ptr<Chunk> ptr_{nullptr};
  /*! \brief shape of current NDArray */
  TShape shape_;
  /*! \brief offset in chunk */
  size_t offset_;
  /*! \brief type of data */
  int dtype_ = -1;
  /*! \brief node entry for autograd */
  autograd::AGNodeEntry entry_;
};


/*!
 * \brief Perform elementwise sum over each data from source, store result into out.
 * \param source the ndarray we want to sum
 * \param out the target ndarray
 * \param priority Priority of the action.
 */
void ElementwiseSum(const std::vector<NDArray> &source, NDArray *out, int priority = 0);

/*!
 * \brief elementwise add
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator+(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise add
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator+(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise subtraction
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator-(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise subtraction
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator-(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise multiplication
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator*(const NDArray &lhs, const NDArray &rhs); \
/*!
 * \brief elementwise multiplication
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator*(const NDArray &lhs, const real_t &rhs);
/*!
 * \brief elementwise division
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator/(const NDArray &lhs, const NDArray &rhs);
/*!
 * \brief elementwise division
 * \param lhs left operand
 * \param rhs right operand
 * \return a new result ndarray
 */
NDArray operator/(const NDArray &lhs, const real_t &rhs);

/*!
 * \brief Seed the random number generator.
 * \param seed the seed to set to global random number generators.
 */
void RandomSeed(uint32_t seed);
/*!
 * \brief Sample uniform distribution for each elements of out.
 * \param begin lower bound of distribution.
 * \param end upper bound of distribution.
 * \param out output NDArray.
 */
void SampleUniform(real_t begin, real_t end, NDArray *out);
/*!
 * \brief Sample gaussian distribution for each elements of out.
 * \param mu mean of gaussian distribution.
 * \param sigma standard deviation of gaussian distribution.
 * \param out output NDArray.
 */
void SampleGaussian(real_t mu, real_t sigma, NDArray *out);
/*!
 * \brief Sample gamma distribution for each elements of out.
 * \param alpha parameter (shape) of the gamma distribution
 * \param beta parameter (scale) of the gamma distribution
 * \param out output NDArray.
 */
void SampleGamma(real_t alpha, real_t beta, NDArray *out);
/*!
 * \brief Sample exponential distribution for each elements of out.
 * \param lambda parameter (rate) of the exponential distribution
 * \param out output NDArray.
 */
void SampleExponential(real_t lambda, NDArray *out);
/*!
 * \brief Sample Poisson distribution for each elements of out.
 * \param lambda parameter (rate) of the Poisson distribution
 * \param out output NDArray.
 */
void SamplePoisson(real_t lambda, NDArray *out);
/*!
 * \brief Sample negative binomial distribution for each elements of out.
 * \param k failure limit
 * \param p success probability 
 * \param out output NDArray.
 */
void SampleNegBinomial(int32_t k, real_t p, NDArray *out);
/*!
 * \brief Sample generalized negative binomial distribution for each elements of out.
 * \param mu parameter (mean) of the distribution
 * \param alpha parameter (over dispersion) of the distribution
 * \param out output NDArray.
 */
void SampleGenNegBinomial(real_t mu, real_t alpha, NDArray *out);


//--------------------------------------------------------------
// The following part are API Registration of NDArray functions.
//--------------------------------------------------------------

/*! \brief definition of NDArray function */
typedef std::function<void (NDArray **used_vars,
                            real_t *scalars,
                            NDArray **mutate_vars,
                            int num_params,
                            char **param_keys,
                            char **param_vals)> NDArrayAPIFunction;
/*! \brief mask information on how functions can be exposed */
enum NDArrayFunctionTypeMask {
  /*! \brief all the use_vars should go before scalar */
  kNDArrayArgBeforeScalar = 1,
  /*! \brief all the scalar should go before use_vars */
  kScalarArgBeforeNDArray = 1 << 1,
  /*!
   * \brief whether this function allows the handles in the target to
   *  be empty NDArray that are not yet initialized, and will initialize
   *  them when the function is invoked.
   *
   *  most function should support this, except copy between different
   *  devices, which requires the NDArray to be pre-initialized with context
   */
  kAcceptEmptyMutateTarget = 1 << 2
};
/*! \brief Registry entry for NDArrayFunction */
struct NDArrayFunctionReg
    : public dmlc::FunctionRegEntryBase<NDArrayFunctionReg,
                                        NDArrayAPIFunction> {
  /*! \brief number of variable used by this function */
  unsigned num_use_vars;
  /*! \brief number of variable mutated by this function */
  unsigned num_mutate_vars;
  /*! \brief number of scalars used by this function */
  unsigned num_scalars;
  /*! \brief information on how function should be called from API */
  int type_mask;
  /*!
   * \brief constructor
   */
  NDArrayFunctionReg()
      : num_use_vars(0),
        num_mutate_vars(0),
        num_scalars(0),
        type_mask(0) {}
  /*!
   * \brief set the function body to a NDArray setvalue function
   *  this will also auto set the parameters correctly
   * \param fsetvalue function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fsetvalue)(const real_t &rhs,
                                                            NDArray *out)) {
    body = [fsetvalue] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                        int num_params, char **param_keys, char **param_vals) {
      (*fsetvalue)(s[0], mutate_vars[0]);
    };
    num_mutate_vars = 1; num_scalars = 1;
    this->add_argument("src", "real_t", "Source input to the function.");
    return *this;
  }
  /*!
  * \brief set the function body to a ternary NDArray function
  *  this will also auto set the parameters correctly
  * \param fternary function body to set
  * \return ref to the registered entry, used to set properties
  */
  inline NDArrayFunctionReg &set_function(void(*fternary)(const NDArray &lhs,
                                                          const NDArray &mhs,
                                                          const NDArray &rhs,
                                                                NDArray *out)) {
    body = [fternary](NDArray **used_vars,
      real_t *s, NDArray **mutate_vars,
      int num_params, char **param_keys, char **param_vals) {
      (*fternary)(*used_vars[0], *used_vars[1], *used_vars[2], mutate_vars[0]);
    };
    num_use_vars = 3; num_mutate_vars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NDArray", "Left operand to the function.");
    this->add_argument("mhs", "NDArray", "Middle operand to the function.");
    this->add_argument("rhs", "NDArray", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a binary NDArray function
   *  this will also auto set the parameters correctly
   * \param fbinary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fbinary)(const NDArray &lhs,
                                                          const NDArray &rhs,
                                                          NDArray *out)) {
    body = [fbinary] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                      int num_params, char **param_keys, char **param_vals) {
      (*fbinary)(*used_vars[0], *used_vars[1], mutate_vars[0]);
    };
    num_use_vars = 2; num_mutate_vars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NDArray", "Left operand to the function.");
    this->add_argument("rhs", "NDArray", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a binary NDArray function
   *  this will also auto set the parameters correctly
   * \param fscalar function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*fscalar)(const NDArray &lhs,
                                                          const real_t &rhs,
                                                          NDArray *out)) {
    body = [fscalar] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                      int num_params, char **param_keys, char **param_vals) {
      (*fscalar)(*used_vars[0], s[0], mutate_vars[0]);
    };
    num_use_vars = 1; num_mutate_vars = 1; num_scalars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("lhs", "NDArray", "Left operand to the function.");
    this->add_argument("rhs", "real_t", "Right operand to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a unary NDArray function
   *  this will also auto set the parameters correctly
   * \param funary function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(void (*funary)(const NDArray &src,
                                                         NDArray *out)) {
    body = [funary] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                     int num_params, char **param_keys, char **param_vals) {
      (*funary)(*used_vars[0], mutate_vars[0]);
    };
    num_use_vars = 1; num_mutate_vars = 1;
    type_mask = kNDArrayArgBeforeScalar | kAcceptEmptyMutateTarget;
    this->add_argument("src", "NDArray", "Source input to the function.");
    return *this;
  }
  /*!
   * \brief set the function body to a unary NDArray function
   *  this will also auto set the parameters correctly
   * \param fgeneric function body to set
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_function(
    void (*fgeneric)(NDArray **used_vars,
                     real_t *s,
                     NDArray **mutate_vars,
                     const std::map<std::string, std::string>& param)) {
    body = [fgeneric] (NDArray **used_vars, real_t *s, NDArray **mutate_vars,
                       int num_params, char **param_keys, char **param_vals) {
      std::map<std::string, std::string> param;
      for (int i = 0; i < num_params; ++i) {
        param[param_keys[i]] = param_vals[i];
      }
      fgeneric(used_vars, s, mutate_vars, param);
    };
    return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_use_vars(unsigned n) {
    num_use_vars = n; return *this;
  }
  /*!
   * \brief set the number of mutate variables
   * \param n number of mutate variablesx
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_mutate_vars(unsigned n) {
    num_mutate_vars = n; return *this;
  }
  /*!
   * \brief set the number of scalar arguments
   * \param n number of scalar arguments
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_num_scalars(unsigned n) {
    num_scalars = n; return *this;
  }
  /*!
   * \brief set type mask
   * \param tmask typemask
   * \return ref to the registered entry, used to set properties
   */
  inline NDArrayFunctionReg &set_type_mask(int tmask) {
    type_mask = tmask; return *this;
  }
};  // NDArrayFunctionReg

/*!
 * \brief Macro to register NDArray function
 *
 * Example: the following code is example to register a plus
 * \code
 *
 * REGISTER_NDARRAY_FUN(Plus)
 * .set_function(Plus);
 *
 * \endcode
 */
#define MXNET_REGISTER_NDARRAY_FUN(name)                                 \
  DMLC_REGISTRY_REGISTER(::mxnet::NDArrayFunctionReg, NDArrayFunctionReg, name)

}  // namespace mxnet

namespace dmlc {
/*!\brief traits */
DMLC_DECLARE_TRAITS(has_saveload, mxnet::NDArray, true);
}  // namespace dmlc
#endif  // MXNET_NDARRAY_H_
