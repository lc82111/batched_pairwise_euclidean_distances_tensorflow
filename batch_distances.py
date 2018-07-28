def batch_pairwise_distances(A, B, squared=False):
    """Compute the euclidean distances between all the embeddings in batch-wise .
    ┊   (ref https://omoindrot.github.io/triplet-loss)
    Args:
    ┊   A, B: tensor of shape (batch_size, nb_samples, embed_dim)
    ┊   squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
    ┊   ┊   ┊   ┊If false, output is the pairwise euclidean distance matrix.
    Returns:
    ┊   batch pairwise distances: tensor  with shape (batch_size, nb_sample, nb_sample)
    """

    # convert inputs to tensor float32
    A = tf.cast(tf.convert_to_tensor(A), tf.float32)
    B = tf.cast(tf.convert_to_tensor(B), tf.float32)

    # Using trick to computer distance
    squared_norm_A = tf.reduce_sum(tf.square(A), -1)  # (B, nb_sample)
    squared_norm_B = tf.reduce_sum(tf.square(B), -1)  # (B, nb_sample)
    dot_product_AB = tf.keras.backend.batch_dot(A, B, axes=[2, 2])  # (B, nb_sample, nb_sample), dot product of all channel pair combinations
    distances = tf.expand_dims(squared_norm_A, -1) - 2.0*dot_product_AB + tf.expand_dims(squared_norm_B, 1)  # (B, nb_sample, nb_sample)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = tf.maximum(distances, 0.0)

    if not squared:
        # Because the gradient of sqrt is infinite when distances == 0.0
        # we need to add a small epsilon where distances == 0.0
        mask = tf.to_float(tf.equal(distances, 0.0))
        distances = distances + mask * 1e-16

        distances = tf.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances

def test_batched_pairwise_distances(B=2, nb_sample=3, ch=3):
    # fake data
    Ls_np = np.random.randint(low=0, high=3, dtype=np.int, size=(B, nb_sample, ch)) * 1.
    Rs_np = np.random.randint(low=0, high=3, dtype=np.int, size=(B, nb_sample, ch)) * 1.
    Ls = tf.constant(Ls_np, dtype=tf.float32)
    Rs = tf.constant(Rs_np, dtype=tf.float32)  
    print 'Ls:' + str(Ls)
    print 'Rs:' + str(Rs)
    
    #pdb.set_trace()

    # numpy
    dis_np = distance.cdist(Ls[0], Rs[0], 'euclidean')  # (nb_sample, nb_sample)
    print 'distance np batch_num 1 \n:' + str(dis_np)
    dis_np = distance.cdist(Ls[1], Rs[1], 'euclidean')  # (nb_sample, nb_sample)
    print 'distance np batch_num 2 \n:' + str(dis_np)
    
    # tf
    dist_tf = batch_pairwise_distances(Ls, Rs, False)
    print 'distance tf \n:' + str(dist_tf)
  
