/*!
 *  Copyright (c) 2015 by Contributors
 * \file iter_libsvm.cc
 * \brief define a LibSVM Reader to read in arrays
 */
#include <mxnet/io.h>
#include <dmlc/base.h>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <dmlc/data.h>
#include "./iter_sparse_prefetcher.h"
#include "./iter_sparse_batchloader.h"

namespace mxnet {
namespace io {
// LibSVM parameters
struct LibSVMIterParam : public dmlc::Parameter<LibSVMIterParam> {
  /*! \brief path to data libsvm file */
  std::string data_libsvm;
  /*! \brief data shape */
  TShape data_shape;
  /*! \brief path to label libsvm file */
  std::string label_libsvm;
  /*! \brief label shape */
  TShape label_shape;
  // declare parameters
  DMLC_DECLARE_PARAMETER(LibSVMIterParam) {
    DMLC_DECLARE_FIELD(data_libsvm)
        .describe("The input LibSVM file or a directory path.");
    DMLC_DECLARE_FIELD(data_shape)
        .describe("The shape of one example.");
    DMLC_DECLARE_FIELD(label_libsvm).set_default("NULL")
        .describe("The input LibSVM file or a directory path. "
                  "If NULL, all labels will be read from ``data_libsvm``.");
    index_t shape1[] = {1};
    DMLC_DECLARE_FIELD(label_shape).set_default(TShape(shape1, shape1 + 1))
        .describe("The shape of one label.");
  }
};

class LibSVMIter: public SparseIIterator<DataInst> {
 public:
  LibSVMIter() {}
  virtual ~LibSVMIter() {}

  // intialize iterator loads data in
  virtual void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) {
    param_.InitAllowUnknown(kwargs);
    data_parser_.reset(dmlc::Parser<uint32_t>::Create(param_.data_libsvm.c_str(),
                                                      0, 1, "libsvm"));
    CHECK_EQ(param_.data_shape.ndim(), 1) << "dimension of data_shape is expected to be 1";
    if (param_.label_libsvm != "NULL") {
      label_parser_.reset(dmlc::Parser<uint32_t>::Create(param_.label_libsvm.c_str(),
                                                         0, 1, "libsvm"));
      CHECK_GT(param_.label_shape.Size(), 1)
        << "label_shape is not expected to be (1,) when param_.label_libsvm is set.";
    } else {
      CHECK_EQ(param_.label_shape.Size(), 1)
        << "label_shape is expected to be (1,) when param_.label_libsvm is NULL";
    }
    // both data and label are of CSRStorage in libsvm format
    if (param_.label_shape.Size() > 1) {
      out_.data.resize(6);
    } else {
      // only data is of CSRStorage in libsvm format.
      out_.data.resize(4);
    }
  }

  virtual void BeforeFirst() {
    data_parser_->BeforeFirst();
    if (label_parser_.get() != nullptr) {
      label_parser_->BeforeFirst();
    }
    data_ptr_ = label_ptr_ = 0;
    data_size_ = label_size_ = 0;
    inst_counter_ = 0;
    end_ = false;
  }

  virtual bool Next() {
    if (end_) return false;
    while (data_ptr_ >= data_size_) {
      if (!data_parser_->Next()) {
        end_ = true; return false;
      }
      data_ptr_ = 0;
      data_size_ = data_parser_->Value().size;
    }
    out_.index = inst_counter_++;
    CHECK_LT(data_ptr_, data_size_);
    const auto data_row = data_parser_->Value()[data_ptr_++];
    // data, indices and indptr
    out_.data[0] = AsDataBlob(data_row);
    out_.data[1] = AsIdxBlob(data_row);
    out_.data[2] = AsIndPtrPlaceholder(data_row);

    if (label_parser_.get() != nullptr) {
      while (label_ptr_ >= label_size_) {
        CHECK(label_parser_->Next())
            << "Data LibSVM's row is smaller than the number of rows in label_libsvm";
        label_ptr_ = 0;
        label_size_ = label_parser_->Value().size;
      }
      CHECK_LT(label_ptr_, label_size_);
      const auto label_row = label_parser_->Value()[label_ptr_++];
      // data, indices and indptr
      out_.data[3] = AsDataBlob(label_row);
      out_.data[4] = AsIdxBlob(label_row);
      out_.data[5] = AsIndPtrPlaceholder(label_row);
    } else {
      out_.data[3] = AsScalarLabelBlob(data_row);
    }
    return true;
  }

  virtual const DataInst &Value(void) const {
    return out_;
  }

  virtual const NDArrayStorageType GetStorageType(bool is_data) const {
    if (is_data) return kCSRStorage;
    return param_.label_shape.Size() > 1 ? kCSRStorage : kDefaultStorage;
  }

  virtual const TShape GetShape(bool is_data) const {
    if (is_data) return param_.data_shape;
    return param_.label_shape;
  }

 private:
  inline TBlob AsDataBlob(const dmlc::Row<uint32_t>& row) {
    const real_t* ptr = row.value;
    TShape shape(mshadow::Shape1(row.length));
    return TBlob((real_t*) ptr, shape, cpu::kDevMask);  // NOLINT(*)
  }

  inline TBlob AsIdxBlob(const dmlc::Row<uint32_t>& row) {
    const uint32_t* ptr = row.index;
    TShape shape(mshadow::Shape1(row.length));
    return TBlob((int32_t*) ptr, shape, cpu::kDevMask, CSR_IDX_DTYPE);  // NOLINT(*)
  }

  inline TBlob AsIndPtrPlaceholder(const dmlc::Row<uint32_t>& row) {
    return TBlob(nullptr, mshadow::Shape1(0), cpu::kDevMask, CSR_IND_PTR_TYPE);
  }

  inline TBlob AsScalarLabelBlob(const dmlc::Row<uint32_t>& row) {
    const real_t* ptr = row.label;
    return TBlob((real_t*) ptr, mshadow::Shape1(1), cpu::kDevMask);  // NOLINT(*)
  }

  LibSVMIterParam param_;
  // output instance
  DataInst out_;
  // internal instance counter
  unsigned inst_counter_{0};
  // at end
  bool end_{false};
  // label parser
  size_t label_ptr_{0}, label_size_{0};
  size_t data_ptr_{0}, data_size_{0};
  std::unique_ptr<dmlc::Parser<uint32_t> > label_parser_;
  std::unique_ptr<dmlc::Parser<uint32_t> > data_parser_;
};


DMLC_REGISTER_PARAMETER(LibSVMIterParam);

MXNET_REGISTER_IO_ITER(LibSVMIter)
.describe(R"code(Returns the LibSVM file iterator. This iterator is experimental and
should be used with care.

The input data is similar to libsvm file format, except that the indices are expected to be
zero-based instead of one-based. Details of the libsvm format are available at
`https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/`

In this function, the `data_shape` parameter is used to set the shape of each line of the data.
The dimension of both `data_shape` and `label_shape` are expected to be 1.

When `label_libsvm` is set to ``NULL``, both data and label are read from the same file specified
by `data_libsvm`. Otherwise, data is read from `data_libsvm` and label from `label_libsvm`,
in this case, if `data_libsvm` contains label, it will ignored.

The `LibSVMIter` only support `round_batch` parameter set to ``True`` for now. So, if `batch_size`
is 3 and there are 4 total rows in libsvm file, 2 more examples
are consumed at the first round. If `reset` function is called after first round,
the call is ignored and remaining examples are returned in the second round.

If ``data_libsvm = 'data/'`` is set, then all the files in this directory will be read.

Examples::

  // Contents of libsvm file ``data.t``.
  1.0 0:0.5 2:1.2
  -2.0
  -3.0 0:0.6 1:2.4 2:1.2
  4 2:-1.2

  // Creates a `LibSVMIter` with `batch_size`=3.
  LibSVMIter = mx.io.LibSVMIter(data_libsvm = 'data.t', data_shape = (3,),
  batch_size = 3)

  // The first batch (data and label)
  [[ 0.5         0.          1.2 ]
   [ 0.          0.          0.  ]
   [ 0.6         2.4         1.2 ]]

  [ 1. -2. -3.]

  // The second batch (data and label)
  [[ 0.          0.         -1.2 ]
   [ 0.5         0.          1.2 ]
   [ 0.          0.          0. ]]

  [ 4.  1. -2.]

  // Contents of libsvm file ``label.t``
  1.0
  -2.0 0:0.125
  -3.0 2:1.2
  4 1:1.0 2:-1.2

  // Creates a `LibSVMIter` with specified label file
  LibSVMIter = mx.io.LibSVMIter(data_libsvm = 'data.t', data_shape = (3,),
  label_libsvm = 'label.t', label_shape = (3,), batch_size = 3)

  // Two batches of data read from the above iterator are as follows(data and label):
  // The first batch
  [[ 0.5         0.          1.2       ]
   [ 0.          0.          0.        ]
   [ 0.6         2.4         1.2      ]]

  [[ 0.          0.          0.        ]
   [ 0.125       0.          0.        ]
   [ 0.          0.          1.2      ]]

  // The second batch
  [[ 0.          0.         -1.2       ]
   [ 0.5         0.          1.2       ]
   [ 0.          0.          0.        ]]

  [[ 0.          1.         -1.2       ]
   [ 0.          0.          0.        ]
   [ 0.125       0.          0.        ]]

)code" ADD_FILELINE)
.add_arguments(LibSVMIterParam::__FIELDS__())
.add_arguments(BatchParam::__FIELDS__())
.add_arguments(PrefetcherParam::__FIELDS__())
.set_body([]() {
    return new SparsePrefetcherIter(
        new SparseBatchLoader(
            new LibSVMIter()));
  });

}  // namespace io
}  // namespace mxnet