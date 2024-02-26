# Multi-stage training scheme for neural networks

Codes associated with the manuscript titled "Multi-stage neural networks: Function approximator of machine precision" authored by Yongji wang and Ching-Yao Lai. We provide two examples, one for multi-stage training for regression problem and the other for the physics-informed neural networks.

# Abstract

Deep learning techniques are increasingly applied to scientific problems, where the precision of networks is crucial. Despite being deemed as universal function approximators, neural networks, in practice, struggle to reduce the prediction errors below $`O(10^{-5})`$ even with large network size and extended training iterations. To address this issue, we developed the multi-stage neural networks that divides the training process into different stages, with each stage using a new network that is optimized to fit the residue from the previous stage. Across successive stages, the residue magnitudes decreases substantially and follows an inverse power-law relationship with the residue frequencies. The multi-stage neural networks effectively mitigate the spectral biases associated with regular neural networks, enabling them to capture the high frequency feature of target functions. We demonstrate that the prediction error from the multi-stage training for both regression problems and physics-informed neural networks can nearly reach the machine-precision $`O(10^{-16})`$ of double-floating point within a finite number of iterations. Such levels of accuracy are rarely attainable using single neural networks alone.

# Citation
Yongji Wang and Ching-Yao Lai.
*Multi-stage neural networks: Function approximator of machine precision.* Journal of Computational Physics, Volume 504, 2024, 112865, ISSN 0021-9991, https://doi.org/10.1016/j.jcp.2024.112865.

**BibTex:**
```
@article{WANG2024112865,
          title = {Multi-stage Neural Networks: Function Approximator of Machine Precision},
          journal = {Journal of Computational Physics},
          pages = {112865},
          volume = {504},
          year = {2024},
          issn = {0021-9991},
          doi = {https://doi.org/10.1016/j.jcp.2024.112865},
          url = {https://www.sciencedirect.com/science/article/pii/S0021999124001141},
          author = {Yongji Wang and Ching-Yao Lai},
          keywords = {Scientific machine learning, Neural networks, Physics-informed neural networks, Multi-stage training}
}
```
