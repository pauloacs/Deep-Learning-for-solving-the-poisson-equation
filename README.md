# Deep-Learning-for-solving-the-poisson-equation

The two approaches can be mainly classified by the way the simulation data is treated. 
In: 
the data (in this case 2D unsteady simulations): each frame of the simulation is used as an image in order to use typical convolutional models.

To overcome the drawbacks of the above method, and be able to take data from any mesh without interpolation to an uniform grid (ressembling pixels of an image) losing information in zones of particular interest in the further methods the data is used as points on a domain (representing the cells of a mesh - including those at the boundaries).

# I - Image data

# II - Point data

In:
No info is given about the geometry, either way the model is able to learn. 
IMPORTANT NOTE: This way the features (physical quantities) are mapped directly and it may be influenced by the order in wich those are given making the model not able to generalize to a totally different set of points of the same flow. To archieve invariance in this aspect, proceed to section II. 

## CONVLSTMMODEL : (sims, times, points, feature) --> (sims, times, points, feature)
Trainined with all the timesteps but the goal is to use it with n previous know times, for example:

(sim, 1, ...) ---predict---> (sim, 2, ...) 

(sim, 1&2, ...) ---predict---> (sim, 2&3, ...)

(sim, 1&2&3, ...) ---predict---> (sim, 2&3&4, ...) ...and so on...

**Problem:**  Loss becomes nan - **solved: Do not use "relu" activation in LSTM - it leads to exploding gradients** (can also be solved with clipvalue in the adam optimizer but it harms the training(a lot))

## CONVMODEL : (sims * times , points, feature) --> (sims * times, points, feature)
test loss ~ 2e-3.   

---loss = mean(square(p-p_true)+square(ux-ux_true)+square(uy-uy_true)) ---

# III -  POint data - with POintNet

In:

PointNet (https://github.com/charlesq34/pointnet) concept is used joined with the last models giving information about the geometry. Getting the spatial coordinates of each cell center the OpenFOAM's post-processing utility - writeCellCentres - is used. 
![alt text](http://stanford.edu/~rqi/pointnet/images/pointnet.jpg)

PointNet is successfully used to predict flow quantities in https://arxiv.org/abs/2010.09469 but for stationary flow, ence only giving coordinate information. Here the ultimate goal is to tackle the instationary evolution of a flow hence the network is not used directly as presented in the above image.  




# First results: 

## Every cell
Since the ultimate goal is to, from the previous velocity field, predict the pressure and it need to be done for every cell of the mesh the training of model was done with all cells. 

CONV+PointNet : 50 epoch trainig with 100 simulations in the training set. (with padding and not ingnoring the padded values)

![alt text](https://github.com/pauloacs/Deep-Learning-for-solving-the-poisson-equation/blob/main/ux_movie.gif)

## 1/4 
It revealed itself too expensive so now I'm using only 1/4 of the data. 
Maybe it can be trained with N cells:  input:(...,N) --> output(...,N) but be able to do:  input:(...,other_N) --> output(...,other_N). **Try this**




## TEST OTHER 3D POINTCLOUD FRAMEWORKS:

-PointNET++
-
-
-
