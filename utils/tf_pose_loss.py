# this contains the poss loss if you want to polish 3D-2D matches by using a classification network
# for tensorflow, not pytorch as I found SVD is not accurate for pytorch in my computer
# Using the following scripts, you can easily modified the codes in https://github.com/vcg-uvic/learned-correspondence-release


#----------------------------------------------------------------------------------
# replace all the codes in _build_model(self) from line 122 in https://github.com/vcg-uvic/learned-correspondence-release/blob/master/network.py

# use DLT and SVD to estimate the rotation and translation
# the shape is [batch x 1 x number x (3+2)]
# input data (num_img_pair x num_corr x 5)
# after reshape, xx is (num_img_pair x 5 x num_corr), the first 3 is 3D points, and the next 2 is 2D points
xx = tf.transpose(tf.reshape(self.x_in, (x_shp[0], x_shp[2], 5)), (0, 2, 1))
# Create the matrix to be used for the 6-point algorithm
# first row, see E.q. (7.2) in Page 179
X = tf.transpose(tf.stack([
    tf.zeros_like(xx[:, 0]), tf.zeros_like(xx[:, 0]), tf.zeros_like(xx[:, 0]), tf.zeros_like(xx[:, 0]),
    -xx[:, 0], -xx[:, 1], -xx[:, 2], -tf.ones_like(xx[:, 0]),
    xx[:, 4] * xx[:, 0], xx[:, 4] * xx[:, 1], xx[:, 4] * xx[:, 2], xx[:, 4]
], axis=1), (0, 2, 1))

# second row, see E.q. (7.2) in Page 179
Y = tf.transpose(tf.stack([
    xx[:, 0], xx[:, 1], xx[:, 2], tf.ones_like(xx[:, 0]),
    tf.zeros_like(xx[:, 0]), tf.zeros_like(xx[:, 0]), tf.zeros_like(xx[:, 0]), tf.zeros_like(xx[:, 0]),
    -xx[:, 3] * xx[:, 0], -xx[:, 3] * xx[:, 1], -xx[:, 3] * xx[:, 2], -xx[:, 3]
], axis=1), (0, 2, 1))

X = tf.concat([X, Y], axis=1)

print("X shape = {}".format(X.shape))
wX = tf.tile(tf.reshape(weights, (x_shp[0], x_shp[2], 1)), [1, 2, 1]) * X
print("wX shape = {}".format(wX.shape))
XwX = tf.matmul(tf.transpose(X, (0, 2, 1)), wX)
print("XwX shape = {}".format(XwX.shape))

# Recover rotation matrix and tranlsation vector from self-adjoing eigen
e, v = tf.self_adjoint_eig(XwX)

# normalize R and t independantly
# the solution v stores [P1,P2,P3], where P_k is the k-th row of the projection matrix P with dim at 3x4
self.RT = tf.reshape(v[:, :, 0], (x_shp[0], 3, 4))
# normalize the rotation, Frobenius norm, for numerical stablity
self.RR = tf.reshape(self.RT[:,:,:3], (x_shp[0], 9))
self.RR /= tf.norm(self.RR, axis=1, keep_dims=True)
# normalize the translation, for numerical stablity
self.tt = tf.reshape(self.RT[:, :, 3], (x_shp[0], 3))
self.tt /= tf.norm(self.tt, axis=1, keep_dims=True)



#----------------------------------------------------------------------------------
# replace all the codes among line 159-176 in _build_loss(self) in https://github.com/vcg-uvic/learned-correspondence-release/blob/master/network.py


# the pose loss
# normalize the ground-truth rotation, Frobenius norm
RR_gt = tf.reshape(self.R_in, (x_shp[0], 9))
RR_gt /= tf.norm(RR_gt, axis=1, keep_dims=True)
# normalize the ground-truth tranlsation, L_2 norm
tt_gt = self.t_in / tf.norm(self.t_in, axis=1, keep_dims=True)

R_loss = tf.reduce_mean(tf.minimum(tf.reduce_sum(tf.square(self.RR - RR_gt), axis=1),
                                   tf.reduce_sum(tf.square(self.RR + RR_gt), axis=1)))
t_loss = tf.reduce_mean(tf.minimum(tf.reduce_sum(tf.square(self.tt - tt_gt), axis=1),
                                   tf.reduce_sum(tf.square(self.tt + tt_gt), axis=1)))
# pose loss is the summation of rotation and translation loss
pose_loss = R_loss + 1.0 * t_loss

tf.summary.scalar("pose_loss", pose_loss)
tf.summary.scalar("rotation_loss", R_loss)
tf.summary.scalar("translation_loss", t_loss)