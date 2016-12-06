
def load_alexnet_to_icnn(sess, caffetf_modelpath):
    """ caffemodel: np.array, """

    caffemodel = np.load(caffetf_modelpath)
    data_dict = caffemodel.item()

    for l in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7']:
        name = 'i_' + l

        # historical grouping by alexnet
        if l == 'conv2' or l == 'conv4' or l == 'conv5':
            _load_param(sess, name, data_dict[l], group=2)
        else:
            _load_param(sess, name, data_dict[l])

def _load_param(sess, name, layer_data, group=1):
    w, b = layer_data

    if group != 1:
        w = np.concatenate((w, w), axis=2)

    with tf.variable_scope(name, reuse=True):
        for subkey, data in zip(('weights', 'biases'), (w, b)):
            print 'loading ', name, subkey

            try:
                var = tf.get_variable(subkey)
                sess.run(var.assign(data))
            except ValueError as e:
                print 'varirable not found in graph:', subkey
