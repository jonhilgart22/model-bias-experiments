#  Inductive Bias in Computer Vision Models

This repo contains a number experiments for determining how architecture of a model will impact the performance on given tasks.

## Overview

The notebook `notebooks/mnist_model_experiments.ipynb` contain an overview of a genetic algorithm that combines a number of 'primitive' functions in an attempt to classify MNIST digits. This algortihm was inspried by the paper [Weight Agnostic Neural Networks](https://weightagnostic.github.io/) which leveraged a genetic algortihm called NEAT to find networks with inductive bias to accomplish tasks such as MNIST or the cartpole balancing task. 

### Primitive Functions

Here are the functions I explored combining in different orders to classify MNIST images. 

1. cos
2. sin
3. running sum
4. running average
5. discrete average
6. absolute value
7. inverse
8. negative step function
9.  positive step function
10. max pool
11. sigmoid
12. linear transform
13. power
14. convolution

Here are what some of these functions look like on a range of -100 to 100.

<image src="media/sum.png" alt="drawing" width = "300">
<image src="media/power.png" alt="drawing" width = "300">
<image src="media/positive_step.png" alt="drawing" width = "300">
<image src="media/log.png" alt="drawing" width = "300">


### Process

The algorithm works as follows. 
1. Have an input MNIST image (128x128 pixels)
2. While the size of the MNIST new image is > the `N_PIXELS_TO_PROCESS` parameter, pick a new function at random from the primitive functions list and apply it to the inputs. This will generate a list of functions such as ['cos', 'abs', 'inverse', 'max_pool', 'sigmoid', .....]
3. Once the size of the altered image after applying this list of functions, is less than the `N_PIXELS_TO_PROCESS` parameter, take the average of the remaining values and use that as our 'prediction' for what the input image was.
4. Repeat this process for we have reached the `NUM_ARCHITECTURES` parameter. If `NUM_ARCHITECTURES` is 25, we will have 25 models for the first generation. 
5. Find the model with the lowest loss in this generation. This becomes our `BEST_RMA_CLASS` (best random model architecture class) which we will update if we find a lower loss in future generations.
6. Find the top half models, defined as lowest loss, in this generation. Carry these models into the next generation.
7. For the top class, and the top half models, update their architecture for the next generation using the `PERCENT_BEST_ARCHITECTURE_TO_KEEP_NEXT_GEN` to indicate what percent of the functions to update. For example, if `PERCENT_BEST_ARCHITECTURE_TO_KEEP_NEXT_GEN` is 20%, and the model architecture is ['cos', 'sign', 'power', 'linear_transform', 'abs'], we would keep the 'cos' function, (1/5 or 20% of the functions), and randomlly generate the rest.
8. For any models that we haven't carried over from the previous generations, randomlly generate them until we have `NUM_ARCHITECTURES` in this generation. 
9. Repeat until you've reached the `N_GENERATIONS` defined.

### Computational Complexity

My search space for this problem was quite large. With 15 primitive functions, 2-9 pixels to process, and total architecture size ranging from 4 to a possible infinite upper bound (which we'll use as 30), we are looking at `1.9e35` combinations of functions, times the number of possible pixels to process, for a complete search space over `1.52e36` possibilities. If we can search over 4 architectures per second,  it would take us `1.2e28` years to look through every combination. 

 Therefore I employed a number of heuristics in order to simplify the search problem and move away from an exhaustive search. For example, I replicate the best model from a given generation as a starting point for the next generation. I also keep the top half of models from a given generation to seed the next. 

### Results

If we were randomlly guessing on the MNIST dataset, we'd expect a 10% accuracy(10 numbers). For the final model I found after ~800 generations, the overall accruacy was ~ **9%**.  However, there are some classes that we perform better than random on.


<image src="media/performance_per_class.png" alt="drawing" width = "300">

Here is the final pipeline of functions that performed the best.  The `N_PIXELS_TO_PROCESS` parameter was 5.

```
['log', 'mf_inverse', 'tan', 'mf_inverse', 'mf_inverse', 'log',
'sin', 'abs', 'cos', 'tan', 'mf_convolve', 'mf_inverse', 'abs',
'mf_max_pool', 'mf_linear_transform', 'mf_discrete_avg', 'tan',
'sin', 'mf_convolve', 'mf_pos_step_function',
'mf_linear_transform', 'mf_sum', 'mf_pos_step_function', 'mf_pow',
'mf_linear_transform', 'cos', 'mf_convolve', 'mf_running_avg',
'mf_neg_step_function', 'mf_inverse', 'mf_discrete_avg']
```


Another way to verify the genetic algorithm is getting better, minimizing the loss, is to look at the best loss per generation over time. Ideally, we'd want to see a downward trend which we find below. 

<image src="media/loss_per_generation.png" alt="drawing" width = "300">

#### Functions used

Given the list of primitive functions used, I would expect that some would be better at helping to classify these MNIST digits. Therefore, we can compare the distribution of functions found in the first generation compared to the last generation.

<image src="media/first_gen_functions.png" alt="drawing" width = "500">

<image src="media/last_gen_functions.png" alt="drawing" width = "500">

### TODOs and Next Steps

1. Genetic algortihms are used generally because they are easy to parallelize. I didn't explore this strategy in my code, but this could help cut the search time down by orders of magnitude.
2. I only explored one type of `mutation` for each generation, random. Instead, it might be interesting to look at swapping the order of functions used, or combining functions from different models.
3. My final models make the naive assumption that the functions should be applied to all input pixels. In a future iteration, I'd like to test adding/removing connections from this processing pipeline.