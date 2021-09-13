# Deep-Learning-for-solving-the-poisson-equation

The approaches can be mainly classified by the way the simulation data is treated.
In **I** the data (in this case 2D unsteady simulations): each frame of the simulation is used as an image to use typical convolutional models.

To overcome the drawbacks of the above method, and be able to take data from any mesh without interpolation to a uniform grid (resembling pixels of an image) and without losing information in zones of particular interest in the further methods, in **II, III, IV** the data extracted as points of a domain (representing the cells of a mesh - including those at the boundaries).

# I - Image data

# II - Point data

No info is given about the geometry, either way, the model can learn. 
IMPORTANT NOTE: In this method, the features (physical quantities) are mapped directly and that process may be influenced by the order in which the points are given therefore making the model not able to generalize to a differently organized set of points of the same flow. To achieve **invariance** in this aspect, proceed to section **II**. 

## CONVLSTMMODEL 

** Esquema da rede **

The model takes all the previous known values and based on those, predicts the value for the next time. 

(sims, times, points, feature) --> (sims, times, points, feature)

Training with all the timesteps, but the goal is to use it with n previous know times, for example:

(sim, 1, ...) ---predict---> (sim, 2, ...) 

(sim, 1&2, ...) ---predict---> (sim, 2&3, ...)

(sim, 1&2&3, ...) ---predict---> (sim, 2&3&4, ...) ...and so on...

**Problem:**  Loss becomes nan - **solved: Do not use "relu" activation in LSTM - it leads to exploding gradients** (can also be solved with clipvalue in the adam optimizer but it harms the training(a lot))

- 1 780 000 parameters

## CONVMODEL 

** Esquema da rede **

This model predicts only the next time value of a field, having as input the field at the present time.

(sims * times , points, feature) --> (sims * times, points, feature)

The best test loss archived so far ~ 2e-3.   

Defining the loss as:

<img src="https://latex.codecogs.com/svg.image?loss&space;=&space;mean(&space;(u_{x}-u_{x,true})^{2}&plus;(u_{y}-u_{y,true})^{2}&plus;(p-p_{true})^{2}&space;)" title="loss = mean( (u_{x}-u_{x,true})^{2}+(u_{y}-u_{y,true})^{2}+(p-p_{true})^{2} )" />

# III -  POint data - with POintNet

PointNet (https://github.com/charlesq34/pointnet) concept is joined with the networks presented in the last section giving information about the geometry (x and y values). For getting the spatial coordinates of each cell center the OpenFOAM's post-processing utility - writeCellCentres - is used. The original PointNet architecture is presented in the following image, the segmentation network is the one of interest in this case.

![alt text](http://stanford.edu/~rqi/pointnet/images/pointnet.jpg)

PointNet is successfully used to predict flow quantities in https://arxiv.org/abs/2010.09469 for stationary flows, hence only giving coordinate information. Here the ultimate goal is to tackle the unsteady evolution of a flow.

The networks here developed will be introduced now:

## CONVLSTMMODEL + PointNet 

** Esquema da rede **

- 2 560 000 parameters

## CONVMODEL + PointNet 

** Esquema da rede **

- 2 670 000 parameters

# First results: 

## Every cell

Since the ultimate goal is to, from the previous velocity field, predict the pressure and it needs to be done for every cell of the mesh, the training of the model was done with all cells. 

CONV+PointNet : 50 epoch training with 100 simulations in the training set. (with padding and not ignoring the padded values)

![alt text](https://github.com/pauloacs/Deep-Learning-for-solving-the-poisson-equation/blob/main/ux_movie.gif)

## 1/4 

Using every cell for training revealed itself too expensive so now I'm using only 1/4 of the data. 
Maybe it can be trained with N cells:  input:(...,N) --> output(...,N) but be able to do:  input:(...,other_N) --> output(...,other_N) with other_N > N. **Try this**

![alt text](https://github.com/pauloacs/Deep-Learning-for-solving-the-poisson-equation/blob/main/images/ux_movie%20(1).gif)
**  Representar os resultados reais e não os normalizados na próxima atualização, faltou "desnormalizar" ** 

Stop trying to predict from the 0-th time. The evolution from the initial conditions to the first time is very different from the evolution between times. As can be seen in the shown predictions, the model can not, with so little data, learn to predict that.




TRAIN TO PREDICT THE STREAM FUNCTION IN THE PREVIOUS MODELS - WILL ENFORCE CONTINUITY -
THE DRAWBACK - ONLY WORKS FOR 2D


Using OpenFOAM's postprocessing utility by typing : 

```
postprocess -func streamfunction 
```

Now modifying the above models to predict <img src="https://latex.codecogs.com/svg.image?\psi&space;" title="\psi " /> instead of the velocity vector, it will ensure continuity.


Yet to do: Interpolate point field to vol field: 

```
pointVolInterpolation::interpolate(phi);
```
(??)



## IV - Physics informed neural network

## a) No data PINN

To refine the understanding and implement one of these networks, firstly (as it is common in literature), the implementation of a model which can predict the flow given the boundary, initial conditions and governing equations is implemented. It needs no data from the CFD solver since it only needs to be given the coordinates of a sample of points and the parameters at the initial time. 

The governing equations are not evaluated in every point, instead, a random sample of the points' coordinates is taken and the residuals are calculated for those. 
Random sample first - cheaper. Use **Latin hypercube sampling** (lhs) sampling later. 

### i - input [x, y, t] -> output: [ux, uy, p]

Loss = Loss_boundary + Loss_Initial + Loss_equations

Where loss_equations = NS_x + NS_y + Continuity

- Does not converge. Needs 2nd order derivations

### ii - input [x, y, t] -> output: [<img src="https://latex.codecogs.com/svg.image?\psi&space;" title="\psi " />, p]

Ux and Uy are derived from <img src="https://latex.codecogs.com/svg.image?\psi&space;" title="\psi " />:

<img src="https://latex.codecogs.com/svg.image?u_{x}=\frac{\partial&space;\psi&space;}{\partial&space;y}&space;\quad&space;\quad&space;&space;u_{y}=-\frac{\partial&space;\psi&space;}{\partial&space;x}" title="u_{x}=\frac{\partial \psi }{\partial y} \quad \quad u_{y}=-\frac{\partial \psi }{\partial x}" />

The stream function enforces continuity since:

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;u_{x}}{\partial&space;x}&space;&plus;&space;\frac{\partial&space;u_{y}}{\partial&space;y}=\frac{\partial&space;}{\partial&space;x}\left&space;(&space;\frac{\partial&space;\psi}{\partial&space;x}&space;\right&space;)&space;&plus;&space;\frac{\partial&space;}{\partial&space;y}\left&space;(-&space;\frac{\partial&space;\psi}{\partial&space;x}&space;\right&space;)&space;=0" title="\frac{\partial u_{x}}{\partial x} + \frac{\partial u_{y}}{\partial y}=\frac{\partial }{\partial x}\left ( \frac{\partial \psi}{\partial x} \right ) + \frac{\partial }{\partial y}\left (- \frac{\partial \psi}{\partial x} \right ) =0" />

- Leads to convergence.  Needs 3rd order derivations!


### iii - input [x, y, t] -> output: [<img src="https://latex.codecogs.com/svg.image?\psi&space;" title="\psi " />, p, <img src="https://latex.codecogs.com/svg.image?\sigma" title="\sigma" />]

The Cauchy momentum equations are used here:

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;\textbf{u}}{\partial&space;t}&space;&plus;&space;\left&space;(&space;\textbf{u}&space;\cdot&space;\nabla&space;\right&space;)\cdot&space;\textbf{u}&space;=&space;\nabla&space;\mathbf{\sigma^{*}}&space;" title="\frac{\partial \textbf{u}}{\partial t} + \left ( \textbf{u} \cdot \nabla \right )\cdot \textbf{u} = \nabla \mathbf{\sigma^{*}} " />


with the constitutive equation for incompressible newtonian fluid:

<img src="https://latex.codecogs.com/svg.image?\sigma^{*}&space;=&space;\frac{\sigma}{\rho}&space;=&space;-\frac{p}{\rho}&space;I&space;&plus;\nu&space;(\nabla&space;u&space;&plus;&space;\nabla&space;u^{T})" title="\sigma^{*} = \frac{\sigma}{\rho} = -\frac{p}{\rho} I +\nu (\nabla u + \nabla u^{T})" />

- Leads to convergence.  Needs only 2nd order derivations while ensuring continuity which makes the optimization problem easier. **Higher-order derivations lead to much higher computational and storage costs. ** 

Concept from: https://arxiv.org/abs/2002.10558

## b) PINN with incorrect values

Since the models have too many parameters, differentiate multiple times, each batch in each epoch becomes prohibitively expensive (and very RAM demanding, crashing the google Colab when using 12,7 GB of RAM). To overcome this, multiple methods are being studied:

## 1 - using a different neural network to refine the result of one of the previous models minimizing the residuals of the governing equations and matching the boundary conditions 

- not knowing the true values for the values in the interior of the domain. Using adam optimization but also L-BFGS as in https://github.com/maziarraissi/PINNs .

The loss is defined as:

<img src="https://latex.codecogs.com/svg.image?&space;&space;&space;&space;&space;&space;&space;Loss=&space;L_{B}&space;&plus;&space;L_{GE}" title=" Loss= L_{B} + L_{GE}" />

*B* - Boundaries and *GE* - Governing equations

The input coordinates are normalized as :

<img src="https://latex.codecogs.com/svg.image?\phi^{*}&space;=&space;\frac{2&space;\left&space;(&space;&space;\phi&space;-&space;\phi_{min}&space;\right&space;)}{\phi_{max}&space;-&space;\phi_{min}}&space;-&space;1" title="\phi^{*} = \frac{2 \left ( \phi - \phi_{min} \right )}{\phi_{max} - \phi_{min}} - 1" />

Ideas: 

### i- Coordinates as inputs (3 input features) and parameters as outputs having the parameters predicted by the "Convmodel" helping in the training. 

### ii - Coordinates and parameters predicted by the "ConvModel" as inputs (6 input features) - fast to approach the loss of the ConvModel but can overfit in a way hard to overcome. 

### iii - Correction network 

![alt text](https://github.com/pauloacs/Deep-Learning-for-solving-the-poisson-equation/blob/main/images/123.jpg)

The layers connecting to the left side of the "add layer" can be thought of as providing an offset to the results provided on the right-hand side.  

version 2:

![alt text](https://github.com/pauloacs/Deep-Learning-for-solving-the-poisson-equation/blob/main/images/123%20(1).jpg)


## 2 - retrain the big model (update its parameters) for only one prediction with the loss as defined above.


To do: Test "adaptive activation functions" to speed up training. (https://www.researchgate.net/publication/337511438_Adaptive_activation_functions_accelerate_convergence_in_deep_and_physics-informed_neural_networks)

## TEST OTHER 3D POINTCLOUD FRAMEWORKS:

- PointNET++
- PointConv
- SO-NET
- PSTNET


## Data details

The Navier-Stokes equations for incompressible flow are (neglecting the gravity term):



and can be non-dimensionalized to:

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;\textbf{u}^{*}}{\partial&space;x}&plus;\left&space;(&space;\textbf{u}^{*}&space;\cdot&space;{\nabla}^{*}&space;\right&space;)&space;\textbf{u}^{*}&space;=&space;-&space;{\nabla}^{*}&space;{p}^{*}&space;&plus;&space;\frac{1}{Re}&space;{\nabla}^{*^2}&space;\textbf{u}^{*}" title="\frac{\partial \textbf{u}^{*}}{\partial x}+\left ( \textbf{u}^{*} \cdot {\nabla}^{*} \right ) \textbf{u}^{*} = - {\nabla}^{*} {p}^{*} + \frac{1}{Re} {\nabla}^{*^2} \textbf{u}^{*}" />

with  

<img src="https://latex.codecogs.com/svg.image?\mathbf{u}^{*}=&space;\frac{u}{U}&space;&space;\quad&space;\quad&space;t^{*}=\frac{t}{H/U}&space;\quad&space;\quad&space;\mathbf{\nabla}^{*}=H\nabla&space;\quad&space;\quad&space;p^{*}=\frac{p}{\rho&space;U^{2}}&space;\quad&space;\quad&space;Re&space;=&space;\frac{U}{\nu}&space;&space;" title="\mathbf{u}^{*}= \frac{u}{U} \quad \quad t^{*}=\frac{t}{H/U} \quad \quad \mathbf{\nabla}^{*}=H\nabla \quad \quad p^{*}=\frac{p}{\rho U^{2}} \quad \quad Re = \frac{U}{\nu} " />

U is the mean velocity (a parabolic profile is implemented at the inlet) fixed at 1. 
H = 1 , nu = 1.51e-4 

1st simulation: squared cylinder 

2nd : squared cylinder  + cylinder + elipses


To increase training performance, the data was further normalized to be in [0-1] range making the pressure (typical bigger value) having the same relevance as the velocity. 

<img src="https://latex.codecogs.com/svg.image?\varphi^{*}&space;=&space;\frac{\varphi&space;-&space;min(\varphi)}{max(\varphi)-&space;min(\varphi)}" title="\varphi^{*} = \frac{\varphi - min(\varphi)}{max(\varphi)- min(\varphi)}" />

where <img src="https://latex.codecogs.com/svg.image?\varphi" title="\varphi" /> is some physical quantity. The max and min values are those of the whole training set. 

As all the simulations do not have the same number of cells, when exporting the data, the data was padded, and since convolutional models do not allow masking, in the loss function the padded values are not accounted for. 


# Create and export simulation data -> details in the data folder
