# PID Project Reflection
## Describe the effect each of the P, I, D components had in your implementation.

Term **P** is proportional to the current value of the SP-PV error,which is *Cross Track Error* or *Speed Error*in our project. **P** is to reduce *SP-PV error* from current state.

Term **I** accounts for past values of the SP-PV error and integrates them over time to produce the **I** term. It aim to reduce the system error by regularizing sum error counting from start.

Term **D** is a best estimate of the future trend of the SP-PV error, based on its current rate of change. It aim to avoid the system from change too fast, which means oscllating in steer or throllte, as it regularized the derivate of SP-PV error. 



## Describe how the final hyperparameters were chosen.
I initialized the hyperparameters by choose a mature PID sample in the course, then I tuning the hyperparameters manualy according to strategy as follow:

* If the car always steering not enough, increase the P;

* If the car keep driving with a bias from center line, increase I;

* If the car always turn too heavy, increase D.