# Deep-Learning-for-solving-the-poisson-equation

The two approaches can be mainly classified by the way the simulation data is treated. 
In: 
the data (in this case 2D unsteady simulations): each frame of the simulation is used as an image in order to use typical convolutional models.

To overcome the drawbacks of the below method, and be able to take data from any mesh without interpolation to an uniform grid (ressembling pixels of an image) losing information in zones of particular interest in the further methods the data is used as points on a domain (representing the cells of a mesh - including those at the boundaries).
In:
No info is given about the geometry, either way the model is able to learn. 

CONVLSTMMODEL : (sims, times, points, feature) --> (sims, times, points, feature)
Trainined with all the timesteps but the goal is to use it with n previous know times, for example:
(sim, 1, ...) ---predict---> (sim, 2, ...) 

(sim, 1&2, ...) ---predict---> (sim, 2&3, ...)

(sim, 1&2&3, ...) ---predict---> (sim, 2&3&4, ...) ...and so on...

CONVMODEL : (sims * times , points, feature) --> (sims * times, points, feature)

In:

PointNet (https://github.com/charlesq34/pointnet) concept is used joined with the last models giving information about the geometry. Getting the spatial coordinates of each cell center the OpenFOAM's post-processing utility - writeCellCentres - is used. 


First results: 

CONV+PointNet : 50 epoch trainig with 100 simulations in the training set. (with padding and not ingnoring the padded values)

![alt text](https://github.com/pauloacs/Deep-Learning-for-solving-the-poisson-equation/blob/main/ux_movie.gif)

CONV+PointNet : 50 epoch trainig with 100 simulations in the training set. (with padding and not ingnoring the padded values)


