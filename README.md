# cs5478 2D Path Planning 


## Usage

#### Plan with Classic Planner
`python3 eval_planner.py [RRT, BiRRT, RRTStar, APF, CHOMP]`

![](vis/classic_planners_results.png)

#### Plan with Diffusion Planner
`python3 diffusion_planner.py`
![](vis/diffusion_planner_result.png)

![](vis/diffusion_posterior.png)
(general posterior)
![](vis/diffusion_task_cond_posterior.png)
(task conditioned posterior)
![](vis/diffusion_new_obstacles.png)
(task conditioned posterior with new obstacles added in the environment)


