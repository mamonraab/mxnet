/*!
 * Copyright (c) 2017 by Contributors
 * \file utils.cc
 * \brief cpu implementation of util functions
 */

#include "./utils.h"
#include "../operator/nn/cast_storage-inl.h"

namespace mxnet {
namespace common {


template<>
void CastStorageDispatch<cpu>(mshadow::Stream<cpu>* s,
                              const NDArray& input,
                              const NDArray& output) {
  mxnet::op::CastStorageComputeImpl(s, input, output);
}


}  // namespace common
}  // namespace mxnet
