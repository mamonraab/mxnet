/*!
 *  Copyright (c) 2017 by Contributors
 * \file cast_storage-inl.h
 * \brief cast_storage implementation for dense and sparse tensors
 */
#ifndef MXNET_OPERATOR_NN_CAST_STORAGE_INL_H_
#define MXNET_OPERATOR_NN_CAST_STORAGE_INL_H_

#include <dmlc/timer.h>
#include <mxnet/ndarray.h>
#include <vector>
#include "../mxnet_op.h"
#include "../operator_common.h"
#ifdef __CUDACC__
#include "./cast_storage-inl.cuh"
#endif  // __CUDACC__


namespace mxnet {
namespace op {

/*!
 * \brief Kernel for marking row_idx of a RSP matrix per row
 */
struct MarkRspRowIdx {
  // i represents the row index of the matrix data
  template<typename DType, typename RType>
  MSHADOW_XINLINE static void Map(int i, RType* row_idx, const DType* data,
                                  const index_t num_cols) {
    index_t j = 0;
    index_t offset = i * num_cols;
    for (; j < num_cols; ++j) {
      if (data[offset+j] != 0) {
        break;
      }
    }
    if (num_cols == j) {
      row_idx[i] = 0;  // mark as zero for zero row
    } else {
      row_idx[i] = 1;  // mark as one for non-zero row
    }
  }
};

/*!
 * \brief
 * CPU implementation of casting a dns tensor to rsp type.
 */
inline void CastStorageDnsRspImpl(mshadow::Stream<cpu>* s, const TBlob& dns, NDArray* rsp) {
  CHECK(rsp != nullptr);
  CHECK_EQ(rsp->storage_type(), kRowSparseStorage);
  CHECK_EQ(dns.shape_, rsp->shape());
  MSHADOW_TYPE_SWITCH(dns.type_flag_, DType, {  // data type
    MSHADOW_INT_TYPE_SWITCH(rsp->aux_type(rowsparse::kIdx), RType, {  // row idx type
      const index_t num_rows = dns.shape_[0];
      const index_t num_cols = dns.shape_[1];
      rsp->CheckAndAllocAuxData(rowsparse::kIdx, mshadow::Shape1(num_rows));
      TBlob row_idx_blob = rsp->aux_data(rowsparse::kIdx);
      RType* row_idx = row_idx_blob.dptr<RType>();
      mxnet_op::Kernel<MarkRspRowIdx, cpu>::Launch(s, num_rows, row_idx,
          dns.dptr<DType>(), num_cols);
      index_t nnr = 0;
      nnr = mxnet::common::ParallelAccumulate(row_idx, num_rows, nnr);
      rsp->set_aux_shape(rowsparse::kIdx, mshadow::Shape1(nnr));
      if (0 == nnr) return;
      rsp->CheckAndAllocData(mshadow::Shape2(nnr, num_cols));
      mshadow::Tensor<cpu, 2, DType> dns_data = dns.FlatTo2D<cpu, DType>(s);
      mshadow::Tensor<cpu, 2, DType> rsp_data = rsp->data().FlatTo2D<cpu, DType>(s);
      size_t idx = 0;
      for (index_t i = 0; i < num_rows; ++i) {
        if (row_idx[i] > 0) {
          row_idx[idx] = i;
          mshadow::Copy(rsp_data[idx], dns_data[i], s);
          ++idx;
        }
      }
    });
  });
}

// TODO(haibin) Use memcopy instead will be much faster than assigning each individual element
struct CastStorageRspDnsKernel {
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, const index_t width, const IType* idx, const DType *data,
                                  DType* dns) {
    auto rid = idx[i];
    auto dns_offset = rid * width;
    auto rsp_offset = i * width;
    for (size_t col = 0; col < width; col++) {
      dns[dns_offset + col] = data[rsp_offset + col];
    }
  }
};

/*!
 * \brief This function assumes that the meomry for dns has been allocated already
 * since the shape is known at binding stage.
 */
template<typename xpu>
void CastStorageRspDnsImpl(mshadow::Stream<xpu>* s, const NDArray& rsp, TBlob* dns) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(rsp.storage_type(), kRowSparseStorage);
  MSHADOW_TYPE_SWITCH(dns->type_flag_, DType, {
    MSHADOW_INT_TYPE_SWITCH(rsp.aux_type(rowsparse::kIdx), IType, {
      // assign zeros
      mxnet_op::Kernel<mxnet_op::set_zero, xpu>::Launch(s, dns->Size(), dns->dptr<DType>());
      if (rsp.storage_initialized()) {
        // copy over row by row
        auto in_idx = rsp.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s).dptr_;
        auto in_data = rsp.data().FlatTo2D<xpu, DType>(s).dptr_;
        auto out_data = dns->FlatTo2D<xpu, DType>(s).dptr_;
        auto num_rows = rsp.aux_shape(rowsparse::kIdx).Size();
        auto rsp_shape = rsp.shape();
        auto width = rsp_shape.ProdShape(1, rsp_shape.ndim());
        mxnet_op::Kernel<CastStorageRspDnsKernel, xpu>::Launch(s, num_rows, width, in_idx,
                                                               in_data, out_data);
      }
    });
  });
}

/*!
 * \brief This is the kernel for initializing the indptr in a csr tensor.
 */
struct FillCsrIndPtr {
  /*!
   * \brief
   * \param i the i-th row of the dns tensor
   * \param indptr indptr of the csr tensor
   * \param dns the dns tensor
   * \param num_rows
   * \param num_cols
   */
  template<typename DType, typename IType>
  MSHADOW_XINLINE static void Map(int i, IType* indptr, const DType* dns,
                                  const int num_rows, const int num_cols) {
    indptr[i+1] = 0;
    const int offset = i * num_cols;
    for (int j = 0; j < num_cols; ++j) {
      if (dns[offset+j] != 0) {
        ++indptr[i+1];
      }
    }
  }
};

/*!
 * \brief This is the kernel for initializing the col_idx and value array
 * of the csr tensor
 */
struct FillCsrColIdxAndVals {
  /*!
   * \brief
   * \param i the i-th row of the dns tensor
   * \param val value array of the csr
   * \param col_idx column idx array of the csr
   * \param indptr indptr array of the csr
   * \param dns the dns tensor
   * \param num_rows number of rows of the dns
   * \param num_cols number of columns of the dns
   */
  template<typename DType, typename IType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* val, CType* col_idx,
                                  const IType* indptr, const DType* dns,
                                  const int num_rows, const int num_cols) {
    const int offset = i * num_cols;
    int k = indptr[i];
    for (int j = 0; j < num_cols; ++j) {
      if (dns[offset+j] != 0) {
        val[k] = dns[offset+j];
        col_idx[k] = j;
        ++k;
      }
    }
  }
};

/*!
 * \brief
 * CPU implementation of casting a dns tensor to csr type.
 */
inline void CastStorageDnsCsrImpl(mshadow::Stream<cpu>* s, const TBlob& dns, NDArray* csr) {
  CHECK(csr != nullptr);
  CHECK_EQ(csr->storage_type(), kCSRStorage);
  CHECK_EQ(dns.shape_.ndim(), 2);
  CHECK_EQ(dns.shape_, csr->shape());
  MSHADOW_TYPE_SWITCH(dns.type_flag_, DType, {  // data type
    MSHADOW_INT_TYPE_SWITCH(csr->aux_type(csr::kIndPtr), IType, {  // indptr type
      MSHADOW_INT_TYPE_SWITCH(csr->aux_type(csr::kIdx), CType, {  // col idx type
        const index_t num_rows = dns.shape_[0];
        const index_t num_cols = dns.shape_[1];
        csr->CheckAndAllocAuxData(csr::kIndPtr, mshadow::Shape1(num_rows+1));
        IType* indptr = csr->aux_data(csr::kIndPtr).dptr<IType>();
        DType* dns_data = dns.dptr<DType>();
        mxnet_op::Kernel<FillCsrIndPtr, cpu>::Launch(s, num_rows, indptr,
            dns_data, num_rows, num_cols);
        // single thread to accumulate indptr
        // indptr[num_rows] indicates the number of non-zero elements
        indptr[0] = 0;
        for (index_t i = 0; i < num_rows; ++i) {
          indptr[i+1] += indptr[i];
        }
        // allocate column idx array and value array
        csr->CheckAndAllocAuxData(csr::kIdx,
                                  mshadow::Shape1(static_cast<index_t>(indptr[num_rows])));
        csr->CheckAndAllocData(mshadow::Shape1(static_cast<index_t>(indptr[num_rows])));
        // fill col_idx and value arrays of the csr
        mxnet_op::Kernel<FillCsrColIdxAndVals, cpu>::Launch(s, num_rows,
            csr->data().dptr<DType>(), csr->aux_data(csr::kIdx).dptr<CType>(),
            indptr, dns_data, num_rows, num_cols);
      });
    });
  });
}

/*!
 * \brief This is the kernel for copying csr.data to its corresponding dns tensor.
 */
struct CopyCsrDataToDns {
  /*!
   * \brief
   * \param i the i-th row of the dns tensor
   * \param dns_data data blob of the dns tensor
   * \param col_idx column idx array of the csr
   * \param indptr indptr array of the csr
   * \param csr_data data blob of the csr tensor
   * \param num_cols number of columns of the dns
   */
  template<typename DType, typename IType, typename CType>
  MSHADOW_XINLINE static void Map(int i, DType* dns_data, const CType* col_idx,
                                  const IType* indptr, const DType* csr_data,
                                  const int num_cols) {
    const int offset = i * num_cols;
    for (auto j = indptr[i]; j < indptr[i+1]; ++j) {
      dns_data[offset+col_idx[j]] = csr_data[j];
    }
  }
};

/*!
 * \brief Casts a csr tensor to dns format.
 */
template<typename xpu>
void CastStorageCsrDnsImpl(mshadow::Stream<xpu>* s, const NDArray& csr, TBlob* dns) {
  CHECK(dns != nullptr);
  CHECK_EQ(csr.storage_type(), kCSRStorage);
  CHECK_EQ(dns->shape_.ndim(), 2);
  CHECK_EQ(dns->shape_, csr.shape());
  MSHADOW_TYPE_SWITCH(dns->type_flag_, DType, {  // data type
    MSHADOW_INT_TYPE_SWITCH(csr.aux_type(csr::kIndPtr), IType, {  // indptr type
      MSHADOW_INT_TYPE_SWITCH(csr.aux_type(csr::kIdx), CType, {  // col idx type
        const index_t num_rows = dns->shape_[0];
        const index_t num_cols = dns->shape_[1];
        DType* dns_data = dns->dptr<DType>();
        mxnet_op::Kernel<mxnet_op::set_zero, xpu>::Launch(s, dns->shape_.Size(), dns_data);
        if (!csr.storage_initialized()) return;
        const IType* indptr = csr.aux_data(csr::kIndPtr).dptr<IType>();
        const CType* col_idx = csr.aux_data(csr::kIdx).dptr<CType>();
        const DType* csr_data = csr.data().dptr<DType>();
        mxnet_op::Kernel<CopyCsrDataToDns, xpu>::Launch(s, num_rows, dns_data,
            col_idx, indptr, csr_data, num_cols);
      });
    });
  });
}

template<typename xpu>
void CastStorageComputeImpl(mshadow::Stream<xpu>* s,
                            const NDArray& input,
                            const NDArray& output) {
  using namespace mshadow;
  using namespace mshadow::expr;
  const auto src_stype = input.storage_type();
  const auto dst_stype = output.storage_type();
  if (src_stype == kRowSparseStorage && dst_stype == kDefaultStorage) {
    TBlob ret = output.data();
    CastStorageRspDnsImpl<xpu>(s, input, &ret);
  } else if (src_stype == kDefaultStorage && dst_stype == kRowSparseStorage) {
    NDArray ret = output;  // get rid of the const qualifer
    CastStorageDnsRspImpl(s, input.data(), &ret);
  } else if (src_stype == kDefaultStorage && dst_stype == kCSRStorage) {
    NDArray ret = output;  // get rid of the const qualifer
    CastStorageDnsCsrImpl(s, input.data(), &ret);
  } else if (src_stype == kCSRStorage && dst_stype == kDefaultStorage) {
    TBlob ret = output.data();
    CastStorageCsrDnsImpl<xpu>(s, input, &ret);
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

struct CastStorageParam : public dmlc::Parameter<CastStorageParam> {
  int storage_type;
  DMLC_DECLARE_PARAMETER(CastStorageParam) {
    DMLC_DECLARE_FIELD(storage_type)
    .add_enum("default", kDefaultStorage)
    .add_enum("row_sparse", kRowSparseStorage)
    .add_enum("csr", kCSRStorage)
    .describe("Output storage type.");
  }
};

inline bool CastStorageInferStorageType(const nnvm::NodeAttrs& attrs,
                                        std::vector<int> *in_attrs,
                                        std::vector<int> *out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  CHECK_NE(in_attrs->at(0), kUndefinedStorage)
    << "src ndarray's storage type must be specified";
  const CastStorageParam& param = nnvm::get<CastStorageParam>(attrs.parsed);
  CHECK_NE(param.storage_type, kUndefinedStorage)
    << "dst ndarray's storage type must be specified";
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.storage_type);
  return true;
}

template<typename xpu>
void CastStorageComputeEx(const nnvm::NodeAttrs& attrs,
                          const OpContext& ctx,
                          const std::vector<NDArray>& inputs,
                          const std::vector<OpReqType>& req,
                          const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), 1);
  CastStorageComputeImpl<xpu>(s, inputs[0], outputs[0]);
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NN_CAST_STORAGE_INL_H_
