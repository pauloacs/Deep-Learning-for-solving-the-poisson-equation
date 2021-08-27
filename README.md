# Deep-Learning-for-solving-the-poisson-equation

The approaches can be mainly classified by the way the simulation data is treated. 
In **I** the data (in this case 2D unsteady simulations): each frame of the simulation is used as an image in order to use typical convolutional models.

To overcome the drawbacks of the above method, and be able to take data from any mesh without interpolation to an uniform grid (resembling pixels of an image) and without losing information in zones of particular interest in the further methods, in **II, III, IV** the data extracted as points off a domain (representing the cells of a mesh - including those at the boundaries).

# I - Image data

# II - Point data

No info is given about the geometry, either way the model is able to learn. 
IMPORTANT NOTE: In this method the features (physical quantities) are mapped directly and that process may be influenced by the order in wich the points are given therefore making the model not able to generalize to a differently organized set of points of the same flow. To archieve **invariance** in this aspect, proceed to section **II**. 

## CONVLSTMMODEL 

** Esquema da rede **

The model takes all the previous known values and based on those, predicts the value for the next time. 

(sims, times, points, feature) --> (sims, times, points, feature)

Trainined with all the timesteps but the goal is to use it with n previous know times, for example:

(sim, 1, ...) ---predict---> (sim, 2, ...) 

(sim, 1&2, ...) ---predict---> (sim, 2&3, ...)

(sim, 1&2&3, ...) ---predict---> (sim, 2&3&4, ...) ...and so on...

**Problem:**  Loss becomes nan - **solved: Do not use "relu" activation in LSTM - it leads to exploding gradients** (can also be solved with clipvalue in the adam optimizer but it harms the training(a lot))

- 1 780 000 parameters

## CONVMODEL 

** Esquema da rede **

This model predicts only the next time value of a field, having as input the field at the present time.

(sims * times , points, feature) --> (sims * times, points, feature)

best test loss archived so far ~ 2e-3.   

Defining the loss as:

<img src="https://latex.codecogs.com/svg.image?loss&space;=&space;mean(&space;(u_{x}-u_{x,true})^{2}&plus;(u_{y}-u_{y,true})^{2}&plus;(p-p_{true})^{2}&space;)" title="loss = mean( (u_{x}-u_{x,true})^{2}+(u_{y}-u_{y,true})^{2}+(p-p_{true})^{2} )" />

# III -  POint data - with POintNet

PointNet (https://github.com/charlesq34/pointnet) concept is joined with the networks presented in the last section giving information about the geometry (x and y values) . For getting the spatial coordinates of each cell center the OpenFOAM's post-processing utility - writeCellCentres - is used. The original PointNet architecture is presented in the following image, the segmentation network is the one of interest in this case.

![alt text](http://stanford.edu/~rqi/pointnet/images/pointnet.jpg)

PointNet is successfully used to predict flow quantities in https://arxiv.org/abs/2010.09469 for stationary flows, hence only giving coordinate information. Here the ultimate goal is to tackle the instationary evolution of a flow.

The networks here developed will be introduced now:

## CONVLSTMMODEL + PointNet 

** Esquema da rede **

- 2 560 000 parameters

## CONVMODEL + PointNet 

** Esquema da rede **

- 2 670 000 parameters

# First results: 

## Every cell

Since the ultimate goal is to, from the previous velocity field, predict the pressure and it need to be done for every cell of the mesh the training of model was done with all cells. 

CONV+PointNet : 50 epoch trainig with 100 simulations in the training set. (with padding and not ingnoring the padded values)

![alt text](https://github.com/pauloacs/Deep-Learning-for-solving-the-poisson-equation/blob/main/ux_movie.gif)

## 1/4 

Using every cell for training revealed itself too expensive so now I'm using only 1/4 of the data. 
Maybe it can be trained with N cells:  input:(...,N) --> output(...,N) but be able to do:  input:(...,other_N) --> output(...,other_N) with other_N > N. **Try this**

![alt text](https://github.com/pauloacs/Deep-Learning-for-solving-the-poisson-equation/blob/main/images/ux_movie%20(1).gif)

Stop trying to predict from the 0-th time. The evolution from the initial conditions to the first time is very different from the evolution between times. As can be seen in the shown predictions, the model can not, with so little data, learn to predict that.

## IV - Physics informed neural network

Since the models have too much parameters, differentiate multiple times , for each batch in each epoch becomes prohibitively expensive (and very RAM demanding, crashing the google Colab when using 12,7 GB of RAM). To overcome this, multiple methods are being studied:

1 - using a different neural network to refine the result of one of the previous models minimizing the residuals of the governing equations and matching the boundary conditions - not knowing the true values for the values in the interior of the domain. Using adam optimization but also L-BFGS as in https://github.com/maziarraissi/PINNs .

The loss is defined as:

<img src="https://latex.codecogs.com/svg.image?&space;&space;&space;&space;&space;&space;&space;Loss=&space;L_{B}&space;&plus;&space;L_{GE}" title=" Loss= L_{B} + L_{GE}" />

*B* - Boundaries and *GE* - Governing equations


2 - retrain the big model (update its parameters) for only one prediction with the loss as defined above.


## TEST OTHER 3D POINTCLOUD FRAMEWORKS:

- PointNET++
- PointConv
- SO-NET
- PSTNET


## Data details

1st simulation: squared cylinder 

2nd : squared cylinder  + cylinder + elipses


To increase training performance, the data was further normalized to be in [0-1] range making the pressure (typical bigger value) having the same relevance as the velocity. 

<img src="https://latex.codecogs.com/svg.image?\varphi^{*}&space;=&space;\frac{\varphi&space;-&space;min(\varphi)}{max(\varphi)-&space;min(\varphi)}" title="\varphi^{*} = \frac{\varphi - min(\varphi)}{max(\varphi)- min(\varphi)}" />

where <img src="https://latex.codecogs.com/svg.image?\varphi" title="\varphi" /> is some physical quantity. The max and min values are those of the whole training set. 

As all the simulations do not have the same number of cells, when exporting the data, the data was padded and since convolutional models do not allow masking, in the loss function the padded values are not accounted. 

# Create and export simulation data -> details in data folder
