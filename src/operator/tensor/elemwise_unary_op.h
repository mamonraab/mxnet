/*!
 *  Copyright (c) 2015 by Contributors
 * \file elementwise_unary_op-inl.h
 * \brief Function defintion of elementwise unary operators
 */
#ifndef MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_
#define MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_

#include <mxnet/operator_util.h>
#include <vector>
#include <utility>
#include "../mshadow_op.h"
#include "../elemwise_op_common.h"
#include "../special_functions-inl.h"

namespace mxnet {
namespace op {
template<typename GRAD_OP>
struct unary_bwd {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return DType(a*GRAD_OP::Map(b));
  }
};

template<typename xpu, typename OP>
void UnaryCompute(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx,
                  const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], F<OP>(inputs[0].FlatTo1D<xpu, DType>(s)));
  });
}


template<typename xpu>
void IdentityCompute(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (req[0] == kNullOp) return;
  if (req[0] == kWriteInplace) {
    CHECK_EQ(inputs[0].dptr_, outputs[0].dptr_); return;
  }
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DType, {
    Tensor<xpu, 1, DType> out = outputs[0].FlatTo1D<xpu, DType>(s);
    ASSIGN_DISPATCH(out, req[0], F<mshadow_op::identity>(inputs[0].FlatTo1D<xpu, DType>(s)));
  });
}

template<typename xpu>
void IdentityComputeRsp(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<NDArray>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  NDArrayStorageType storage_type = inputs[1].storage_type();
  CHECK_EQ(storage_type, kRowSparseStorage);
  if (req[0] == kNullOp) {
    LOG(FATAL) << "kNullOp in IdentityComputeEx not supported yet";
  }
  if (req[0] == kWriteInplace) {
    LOG(FATAL) << "kWriteInplace for sparse storage not supported yet";
  }
  TShape shape = inputs[1].aux_shape(rowsparse::kIdx);
  if (shape.ndim() == 0) return;
  outputs[0].CheckAndAlloc({shape});
  MSHADOW_TYPE_SWITCH(outputs[0].dtype(), DType, {
    MSHADOW_TYPE_SWITCH(outputs[0].aux_type(rowsparse::kIdx), AuxType, {
      auto out_d = outputs[0].data().FlatTo1D<xpu, DType>(s);
      auto out_aux = outputs[0].aux_data(rowsparse::kIdx).FlatTo1D<xpu, AuxType>(s);
      auto in_aux = inputs[1].aux_data(rowsparse::kIdx).FlatTo1D<xpu, AuxType>(s);
      ASSIGN_DISPATCH(out_d, req[0],
                      F<mshadow_op::identity>(inputs[1].data().FlatTo1D<xpu, DType>(s)));
      ASSIGN_DISPATCH(out_aux, req[0], F<mshadow_op::identity>(in_aux));
    });
  });
}

// FIXME the index is hard coded for _identity_with_attr_like_rhs op
template<typename xpu>
void IdentityComputeEx(const nnvm::NodeAttrs& attrs,
                     const OpContext& ctx,
                     const std::vector<NDArray>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(outputs.size(), 1);
  Stream<xpu> *s = ctx.get_stream<xpu>();
  NDArrayStorageType stype = inputs[1].storage_type();
  CHECK_EQ(stype, kRowSparseStorage) << "Not implemented yet";
  IdentityComputeRsp<xpu>(attrs, ctx, inputs, req, outputs);
}

struct CastParam : public dmlc::Parameter<CastParam> {
  // use int for enumeration
  int dtype;
  DMLC_DECLARE_PARAMETER(CastParam) {
    DMLC_DECLARE_FIELD(dtype)
    .add_enum("float32", mshadow::kFloat32)
    .add_enum("float64", mshadow::kFloat64)
    .add_enum("float16", mshadow::kFloat16)
    .add_enum("uint8", mshadow::kUint8)
    .add_enum("int32", mshadow::kInt32)
    .describe("Output data type.");
  }
};

inline bool CastType(const nnvm::NodeAttrs& attrs,
                     std::vector<int> *in_attrs,
                     std::vector<int> *out_attrs) {
  const CastParam& param = nnvm::get<CastParam>(attrs.parsed);
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  TYPE_ASSIGN_CHECK(*out_attrs, 0, param.dtype);
  return (*in_attrs)[0] != -1;
}

template<typename xpu>
void CastCompute(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<TBlob>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  MSHADOW_TYPE_SWITCH(outputs[0].type_flag_, DstDType, {
    Tensor<xpu, 1, DstDType> out = outputs[0].FlatTo1D<xpu, DstDType>(s);
    MSHADOW_TYPE_SWITCH(inputs[0].type_flag_, SrcDType, {
      Tensor<xpu, 1, SrcDType> data = inputs[0].FlatTo1D<xpu, SrcDType>(s);
      Assign(out, req[0], tcast<DstDType>(data));
    });
  });
}

struct CastStorageParam : public dmlc::Parameter<CastStorageParam> {
  // use int for enumeration
  // TODO(haibin) add enum for storage_type. Probably also aux-types
  int storage_type;
  DMLC_DECLARE_PARAMETER(CastStorageParam) {
    DMLC_DECLARE_FIELD(storage_type)
    .describe("Output storage type.");
  }
};

template<typename xpu>
void CastStorageComputeRspDns(const nnvm::NodeAttrs& attrs,
                 const OpContext& ctx,
                 const std::vector<NDArray>& inputs,
                 const std::vector<OpReqType>& req,
                 const std::vector<NDArray>& outputs) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(inputs.size(), 1);
  CHECK_EQ(outputs.size(), 1);
  auto out = outputs[0];
  auto in = inputs[0];
  auto stype = in.storage_type();
  CHECK_EQ(stype, kRowSparseStorage);
  CHECK_EQ(out.storage_type(), kDefaultStorage);
  MSHADOW_TYPE_SWITCH(in.dtype(), DType, {
    MSHADOW_TYPE_SWITCH(in.aux_type(rowsparse::kIdx), IType, {
      // Fill in zeros. SLOW
      out.data().FlatTo1D<xpu, DType>(s) = 0;
      // data() is not empty
      if (in.storage_shape().ndim() != 0) {
        // Copy over
        auto in_data = in.data().FlatTo2D<xpu, DType>(s);
        auto out_data = out.data().FlatTo2D<xpu, DType>(s);
        CHECK(out.shape().ndim() > 0);
        auto invalid_row_id = out.shape()[0];
        auto num_rows = in.aux_shape(rowsparse::kIdx)[0];
        auto in_idx = in.aux_data(rowsparse::kIdx).FlatTo1D<xpu, IType>(s);
        for (size_t i = 0; i < num_rows; i += 1) {
          size_t row_id = in_idx[i];
          if (row_id == invalid_row_id) continue;
          mshadow::Copy(out_data[in_idx[i]], in_data[i], s);
        }
      }
    });
  });
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
  auto stype = inputs[0].storage_type();
  auto out_stype = outputs[0].storage_type();
  if (stype == kRowSparseStorage && out_stype == kDefaultStorage) {
    CastStorageComputeRspDns<xpu>(attrs, ctx, inputs, req, outputs);
  } else {
    LOG(FATAL) << "Not implemented";
  }
}

#define MXNET_OPERATOR_REGISTER_UNARY(name)                         \
  NNVM_REGISTER_OP(name)                                            \
  .set_num_inputs(1)                                                \
  .set_num_outputs(1)                                               \
  .set_attr<nnvm::FInferShape>("FInferShape", ElemwiseShape<1, 1>)  \
  .set_attr<nnvm::FInferType>("FInferType", ElemwiseType<1, 1>)     \
  .set_attr<nnvm::FInplaceOption>("FInplaceOption",                 \
    [](const NodeAttrs& attrs){                                     \
      return std::vector<std::pair<int, int> >{{0, 0}};             \
    })                                                              \
  .add_argument("data", "NDArray-or-Symbol", "The input array.")

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_TENSOR_ELEMWISE_UNARY_OP_H_
