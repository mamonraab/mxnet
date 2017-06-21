#!/usr/bin/env python
# pylint: skip-file
import sys
sys.path.insert(0, "../../python/")
import mxnet as mx
import numpy as np
import time

def check_diff_to_scalar(A, x):
    """ assert A == x"""
    assert(np.sum(np.abs((A - x).asnumpy())) == 0), A.asnumpy()

# setup
keys = [3, 5, 7]
str_keys = ['a', 'b', 'c']

rate = 2
shape = (2, 2)
big_shape = (1200, 1200)        # bigger than BIGARRAY_BOUND

def init_kv():
    kv = mx.kv.create('dist_sync')
    # init kv
    kv.init(keys, [mx.nd.ones(shape)] * len(keys))
    kv.init(99, mx.nd.ones(big_shape))
    my_rank = kv.rank
    nworker = kv.num_workers
    # init updater on servers
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate))

    return kv, my_rank, nworker

def init_kv_rsp():
    kv = mx.kv.create('dist_sync')
    # init kv
    kv.init(str_keys, [mx.nd.ones(shape).to_rsp()] * len(str_keys))
    # kv.init('z', mx.nd.ones(big_shape))
    my_rank = kv.rank
    nworker = kv.num_workers
    # init updater on servers
    kv.set_optimizer(mx.optimizer.create('test', rescale_grad=rate))
    return kv, my_rank, nworker

def test_sync_push_pull():
    kv, my_rank, nworker = init_kv()
    nrepeat = 3
    for i in range(nrepeat):
        kv.push(3, mx.nd.ones(shape)*(my_rank+1))
        kv.push(99, mx.nd.ones(big_shape)*(my_rank+1))

    num = (nworker + 1 ) * nworker * rate / 2 * nrepeat + 1
    val = mx.nd.zeros(shape)
    kv.pull(3, out = val)
    check_diff_to_scalar(val, num)
    # print val.asnumpy()

    val2 = mx.nd.zeros(big_shape)
    kv.pull(99, out = val2)
    check_diff_to_scalar(val2, num)

def test_sync_push_pull_row_sparse():
    kv, my_rank, nworker = init_kv_rsp()
    nrepeat = 3
    import time

    for i in range(nrepeat):
        # TODO(haibin) generate random rsp
        v = mx.nd.ones(shape)*(my_rank+1)
        kv.push('a', v.to_rsp())
        # kv.push('z', mx.nd.ones(big_shape)*(my_rank+1))

    time.sleep(1)
    print('done push')
    num = (nworker + 1 ) * nworker * rate / 2 * nrepeat + 1
    print("expect ", num)
    #num = 1
    val = mx.nd.ones(shape).to_rsp()
    time.sleep(1)
    print('start pull')
    kv.pull('a', out = val)
    val.wait_to_read()
    print('done pull')
    print(val)
    print(val.indices)
    print(val.values)
    #print(val.indices.asnumpy())
    #print(val.values.asnumpy())
    
    print('about to_dense')
    time.sleep(1)
    d = val.to_dense()
    #d.wait_to_read()
    print(d)
    '''
    #check_diff_to_scalar(val, num)
    '''
    print('done')
    time.sleep(2)


if __name__ == "__main__":
    # test_sync_push_pull()
    test_sync_push_pull_row_sparse()
