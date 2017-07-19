import mxnet as mx
from mxnet.test_utils import *
from get_data import get_libsvm_data
import time
import argparse
import os

parser = argparse.ArgumentParser(description="Run sparse regression with distributed kvstore",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--profiler', type=int, default=0,
                    help='whether to use profiler')
parser.add_argument('--num-epoch', type=int, default=1,
                    help='number of epochs to train')
parser.add_argument('--batch-size', type=int, default=512,
                    help='number of examples per batch')
parser.add_argument('--num-batch', type=int, default=10,
                    help='number of batches per epoch')
parser.add_argument('--dummy-iter', type=int, default=0,
                    help='whether to use dummy iterator to exclude io cost')
parser.add_argument('--kvstore', type=str, default='dist_sync',
                    help='what kvstore to use (local, dist_sync, etc)')
parser.add_argument('--logging', type=int, default=0,
                    help='whether to print the result of metric at the end of each epoch')
parser.add_argument('--dataset', type=str, default='avazu',
                    help='what dataset to use')

class DummyIter(mx.io.DataIter):
    "A dummy iterator that always return the same batch, used for speed testing"
    def __init__(self, real_iter):
        super(DummyIter, self).__init__()
        self.real_iter = real_iter
        self.provide_data = real_iter.provide_data
        self.provide_label = real_iter.provide_label
        self.batch_size = real_iter.batch_size

        for batch in real_iter:
            self.the_batch = batch
            break

    def __iter__(self):
        return self

    def next(self):
        return self.the_batch

avazu = {
    'data_name': 'avazu-app.t',
    'data_origin_name': 'avazu-app.t.bz2',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/avazu-app.t.bz2",
    'feature_dim': 1000000,
}

kdda = {
    'data_name': 'kdda.t',
    'data_origin_name': 'kdda.t.bz2',
    'url': "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/kdda.t.bz2",
    'feature_dim': 20216830,
}
datasets = { 'kdda' : kdda, 'avazu' : avazu }

def dummy_data_iter(num_batch, batch_size, feature_dim):
    data = np.load('x_512_' + str(feature_dim) + '.npz')
    indices = data['indices']
    values = data['values']
    indptr = data['indptr']
    data = mx.sparse_nd.csr(values, indptr, indices,
                            (num_batch * batch_size, feature_dim))
    dns_label = mx.nd.zeros((num_batch * batch_size, 1))

    train_iter = DummyIter(mx.io.NDArrayIter(data=data,
                                             label={'softmax_label':dns_label},
                                             batch_size=batch_size))
    return train_iter, mx.nd.array(indices, dtype='int64')

def regression_model(feature_dim):
     initializer = mx.initializer.Normal()
     x = mx.symbol.Variable("data", stype='csr')
     norm_init = mx.initializer.Normal(sigma=0.01)
     v = mx.symbol.Variable("v", shape=(feature_dim, 1), init=norm_init, stype='row_sparse')
     embed = mx.symbol.dot(x, v)
     y = mx.symbol.Variable("softmax_label")
     model = mx.symbol.LinearRegressionOutput(data=embed, label=y, name="out")
     return model

if __name__ == '__main__':
    # arg parser
    args = parser.parse_args()
    num_epoch = args.num_epoch
    num_batch = args.num_batch
    kvstore = args.kvstore
    profiler = args.profiler > 0
    logging = args.logging > 0
    batch_size = args.batch_size
    dummy_iter = args.dummy_iter
    dataset = args.dataset

    # create kvstore
    kv = mx.kvstore.create(kvstore)
    rank = kv.rank
    num_worker = kv.num_workers
    logging = logging and rank == 0

    # data
    data_dict = avazu
    feature_dim = data_dict['feature_dim']
    if logging:
        print('preparing data ... ')
    data_dir = os.path.join(os.getcwd(), 'data')
    path = os.path.join(data_dir, data_dict['data_name'])
    if not os.path.exists(path):
        get_libsvm_data(data_dir, data_dict['data_name'], data_dict['url'],
                        data_dict['data_origin_name'])
        assert os.path.exists(path)

    if dummy_iter:
        train_data, first_batch_rowids = dummy_data_iter(1, batch_size, feature_dim)
    else:
        train_data = mx.io.LibSVMIter(data_libsvm=path, data_shape=(feature_dim,),
                                      batch_size=batch_size, num_parts=num_worker,
                                      part_index=rank)
        first_batch = next(iter(train_data))
        #TODO(haibin) no need to copy after ndarray refactoring
        first_batch_rowids = first_batch.data[0].indices.copyto(mx.cpu())
    # model
    model = regression_model(feature_dim)

    # module
    mod = mx.mod.Module(symbol=model, data_names=['data'], label_names=['softmax_label'])
    mod.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label)
    mod.init_params(initializer=mx.init.Uniform(scale=.1))
    sgd = mx.optimizer.SGD(momentum=0.1, clip_gradient=5.0,
                           learning_rate=0.1, rescale_grad=1.0/batch_size/num_worker)
    mod.init_optimizer(optimizer=sgd, kvstore=kv,
                       sparse_pull_dict={'v': [first_batch_rowids]})
    # use accuracy as the metric
    metric = mx.metric.create('MSE')

    # start profiler
    if profiler:
        import random
        name = 'profile_output_' + str(num_worker) + '.json'
        mx.profiler.profiler_set_config(mode='all', filename=name)
        mx.profiler.profiler_set_state('run')

    if logging:
        print('start training ...')
    start = time.time()
    for epoch in range(num_epoch):
        nbatch = 0
        end_of_batch = False
        data_iter = iter(train_data)
        data_iter.reset()
        metric.reset()
        next_batch = next(data_iter)
        while not end_of_batch:
            nbatch += 1
            batch = next_batch
            mod.forward_backward(batch)
            try:
                # pre fetch next batch to determine what to pull
                next_batch = next(data_iter)
                # TODO(haibin) remove this copy after ndarray is refactored
                row_ids = next_batch.data[0].indices.copyto(mx.cpu())
                if nbatch == num_batch:
                    raise StopIteration
            except StopIteration:
                row_ids = first_batch_rowids
                end_of_batch = True
            mod.update(sparse_pull_dict={'v': [row_ids]})  # update parameters
            mod.update_metric(metric, batch.label)         # accumulate prediction accuracy
        if logging:
            print('epoch %d, %s' % (epoch, metric.get()))
    end = time.time()
    if profiler:
        mx.profiler.profiler_set_state('stop')
    if rank == 0:
        time_cost = end - start
        print('num_worker = ' + str(num_worker) + ', time cost = ' + str(time_cost))
