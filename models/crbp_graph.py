import six
import numpy as np
import tensorflow.compat.v1 as tf



def lipschitz_penalty(
        last_state,
        prev_state,
        tau=0.95,  # Changed 2/25/20 from 0.9
        compute_hessian=True,
        pen_type='l1'):
    """Compute the cRBP penalty."""
    norm_1_vect = tf.ones_like(last_state)
    norm_1_vect.requires_grad = False
    vj_prod = tf.grad(
        last_state,
        prev_state,
        grad_outputs=[norm_1_vect],
        retain_graph=True,
        create_graph=compute_hessian,
        allow_unused=True)[0]
    vj_penalty = (vj_prod - tau).clamp(0) ** 2  # Squared to emphasize outliers
    return vj_penalty.sum()  # Save memory with the mean


def rbp(x, h1, h2, f, nstep, lbd=0.9, debug=False):
    """Recurrent backprop.
    Implementation taken from:
    https://github.com/renmengye/inc-few-shot-attractor-public/blob/8970557d8d799c6ec85a9ddc4e20a1544e21c35f/fewshot/models/rbp.py

    Args:
        x: Inputs to the dynamical process.
        h1: List of final hidden state.
        h2: List of second last hidden state.
        f: Final cost, function to optimize.
        nstep: Number of recurrent backprop steps.
        lbd: Damping constant, default 0.9.
        debug: Whether to print out intermediate values.
    Returns:
        grad_x: Gradients of f wrt. x.
    Note: You should only unroll the graph once."""
    if type(h1) != list:
        h1 = [h1]
    if type(h2) != list:
        h2 = [h2]
    if type(x) != list:
        x = [x]
    assert lbd >= 0.0

    grad_h = tf.gradients(f, h1, gate_gradients=1)
    nv = [tf.stop_gradient(_) for _ in grad_h]
    ng = [tf.stop_gradient(_) for _ in grad_h]

    for step in six.moves.xrange(nstep):
        j_nv = tf.gradients(h1, h2, grad_ys=nv, gate_gradients=1)
        if lbd > 0.0:
            nv = [j_nv_ - lbd * nv_ for j_nv_, nv_ in zip(j_nv, nv)]
        else:
            nv = j_nv
        if debug:
            # Debug mode, print ng values.
            ng_norm = tf.add_n(
                [tf.sqrt(tf.reduce_sum(tf.square(_))) for _ in ng])
            nv_norm = tf.add_n(
                [tf.sqrt(tf.reduce_sum(tf.square(_))) for _ in nv])
            print_ng = tf.Print(tf.constant(0.0), ['ng norm', step, ng_norm])
            print_nv = tf.Print(tf.constant(0.0), ['nv norm', step, nv_norm])
            with tf.control_dependencies([print_ng, print_nv]):
                nv = [tf.identity(_) for _ in nv]
                ng = [tf.identity(_) for _ in ng]
        ng = [ng_ + nv_ for ng_, nv_ in zip(ng, nv)]
    grad = tf.gradients(h1, x, grad_ys=ng, gate_gradients=1)
    return grad


def _mlp(
        inp,
        hidden_dims=[20],
        activations=None,
        kernel_initializer=tf.variance_scaling_initializer(scale=0.001, seed = 0),  # noqa
        bias_initializer=tf.constant_initializer(0.),
        scope='mlp',
        dropout=None,
        seed=0,
        share_weights=False,
        trainable=True,
        **kwargs):

    # flatten input
    inp_shape = inp.shape.as_list()
    if len(inp_shape) != 2:
        inp = tf.reshape(inp, [-1, inp_shape[-1]])

    # Convert hidden_dims to list
    if not isinstance(hidden_dims, (list, tuple)):
        hidden_dims = [hidden_dims]

    # Make sure there are the same number of activations and feature layers
    if activations is None:
        activations = tf.nn.elu
    if isinstance(activations, type(tf.identity)):
        activations = [activations] * len(hidden_dims)
        activations[-1] = tf.identity

    assert len(hidden_dims) == len(activations), ('One activation per feature layer!')

    for i, (num_feature, activation) in enumerate(zip(hidden_dims, activations)):
        iscope = scope if share_weights else scope + str(i)
        with tf.variable_scope(iscope, reuse=tf.AUTO_REUSE):
            # Infer kernel shape
            kernel_shape = inp.get_shape().as_list()
            kernel_shape = kernel_shape[:-2] + kernel_shape[-1:] + [num_feature]  # noqa
            # Initialize kernel variable
            kernel = tf.get_variable(
                initializer=kernel_initializer,
                shape=kernel_shape,
                dtype=inp.dtype,
                name='weights',
                trainable=trainable)

            # Initialize bias variable
            bias = tf.get_variable(
                initializer = bias_initializer,
                shape = [num_feature],
                dtype = inp.dtype,
                name = 'bias',
                trainable = trainable)
            # Compute fully connected
            inp = activation(tf.matmul(inp, kernel) + bias)
            if dropout is not None:
                inp = tf.nn.dropout(inp, dropout, seed)

    # restore inp shape
    if len(inp_shape) != 2:
        inp = tf.reshape(inp, inp_shape[:-1] + [-1])

    return inp


def _normalize_adj(adj, eps=1e-6):
    deg_inv_sqrt = tf.reduce_sum(adj, axis=-1)
    deg_inv_sqrt = tf.clip_by_value(deg_inv_sqrt, 1.0, tf.float32.max)  # clamp min = 1.0
    deg_inv_sqrt = 1. / tf.sqrt(deg_inv_sqrt + eps)
    adj = tf.expand_dims(deg_inv_sqrt, -1) * adj * tf.expand_dims(deg_inv_sqrt, -2)
    return adj


if __name__ == '__main__':

    ## constants
    BATCH_SIZE = 2
    NUM_NODES = 512
    NODE_DIM_INP = 24
    NODE_DIM_OUT = 36
    HIDDEN_DIMS = []
    ACTIVATION = tf.nn.elu
    NUM_ITERS = 5
    MLP_KWARGS = {}

    ## graph_rnn inputs
    nodes_inp = tf.random.normal([BATCH_SIZE, NUM_NODES, NODE_DIM_INP], dtype=tf.float32)
    affinities = tf.random.normal([BATCH_SIZE, NUM_NODES, NUM_NODES], dtype=tf.float32)

    ## preprocess
    affinities = 5. * (affinities + tf.transpose(affinities, [0,2,1])) # symmetric
    affinities = tf.nn.sigmoid(affinities) # in range (0.,1.)
    affinities = _normalize_adj(affinities) # normalized by node degree; optional

    ## define computation graph
    with tf.variable_scope('embedding'):
        nodes_out = _mlp(nodes_inp, hidden_dims=(HIDDEN_DIMS + [NODE_DIM_OUT]), activations=ACTIVATION)

    for it in range(NUM_ITERS):
        print("iter %s" % it)
        ## channelwise op
        nodes_out = _mlp(nodes_out, hidden_dims=(HIDDEN_DIMS + [NODE_DIM_OUT]), activations=tf.identity, **MLP_KWARGS)

        ## exchange info across nodes
        nodes_out = tf.matmul(affinities, nodes_out)

        ## nonlinearity
        nodes_out = (ACTIVATION or tf.identity)(nodes_out)

    ## compute gradients
    grads = tf.gradients(nodes_out, [nodes_inp, affinities])

    ## run on cpu; note that gpu kernels, including gradients, must be registered separately in tensorflow 1
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    ## run graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    nodes_out, node_node_grads, node_affs_grads = sess.run([nodes_out] + grads)

    print("nodes_out shape", nodes_out.shape)
    print("node_node_gradients abs mean", np.abs(node_node_grads).max())
    print("node_affs_gradients abs mean", np.abs(node_affs_grads).max())
