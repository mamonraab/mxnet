import logging
logging.getLogger().setLevel(logging.INFO)
import mxnet as mx
import numpy as np
from movielens_data import get_data_iter, max_id

# data iter
cpu_ctx = mx.cpu(0)
gpu_ctx = mx.gpu(1)
group2ctx={'dev1': cpu_ctx, 'dev2': gpu_ctx}
train_iter, test_iter = get_data_iter(batch_size=50)
max_user, max_item = max_id('./ml-100k/u.data')

# model
def plain_net(k):
    with mx.AttrScope(ctx_group='dev1'):
        # input
        user = mx.symbol.Variable('user')
        item = mx.symbol.Variable('item')
        # user feature lookup
        user = mx.symbol.Embedding(data = user, input_dim = max_user, output_dim = k)
        # item feature lookup
        item = mx.symbol.Embedding(data = item, input_dim = max_item, output_dim = k)
        # predict by the inner product, which is elementwise product and then sum
    with mx.AttrScope(ctx_group='dev2'):
        pred = user * item
        pred = mx.symbol.sum(data = pred, axis = 1)
        pred = mx.symbol.Flatten(data = pred)
        # loss layer
        score = mx.symbol.Variable('score')
        pred = mx.symbol.LinearRegressionOutput(data = pred, label = score)
    return pred

k = 20
net = plain_net(k)

# create module
mod = mx.mod.Module(symbol=net, data_names=['item', 'user'], label_names=['score'])
# allocate memory by given the input data and label shapes
mod.bind(data_shapes=train_iter.provide_data,
         label_shapes=train_iter.provide_label,
         group2ctx=group2ctx)
# initialize parameters by uniform random numbers
mod.init_params(initializer=mx.init.Uniform(scale=.1))
# use SGD with learning rate 0.1 to train
mod.init_optimizer(optimizer='sgd', optimizer_params=(('learning_rate', 0.1), ))
# use accuracy as the metric
metric = mx.metric.create('MSE')
# train 5 epochs, i.e. going over the data iter one pass
for epoch in range(5):
    train_iter.reset()
    metric.reset()
    for batch in train_iter:
        mod.forward(batch, is_train=True)       # compute predictions
        mod.update_metric(metric, batch.label)  # accumulate prediction accuracy
        mod.backward()                          # compute gradients
        mod.update()                            # update parameters
    print('Epoch %d, Training %s' % (epoch, metric.get()))
