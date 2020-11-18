# Paper-on-Large-scale-Machine-Learning
This page is organized based on our paper：
- A Survey on Large-Scale Machine Learning, IEEE TKDE, 2020．

Please refer the paper for more details. If you need further discussion, please feel free to contact us with fwj.edu@gmail.com

#### We review LML according to three computational perspectives:
Model Simplification, which reduces Computational Complexities by simplifying predictive models; 

Optimization Approximation, which enhances Computational Efficiency by designing better optimization algorithms; 

Computation Parallelism, which improves Computational Capabilities by scheduling multiple computing devices.

#### Related Surveys
- Efficient machine learning for big data: A review, 
  - O. Y. Al-Jarrah, P. D. Yoo, S. Muhaidat, G. K. Karagiannidis, and K. Taha, Big Data Research, vol. 2, no. 3, pp. 87–93, 2015.
- Optimization methods for large-scale machine learning
  - L. Bottou, F. E. Curtis, and J. Nocedal, SIAM Review, vol. 60, no. 2, pp. 223–311, 2018.
- Big data analytics: a survey
  - C.-W. Tsai, C.-F. Lai, H.-C. Chao, and A. V. Vasilakos, Journal of Big data, vol. 2, no. 1, p. 21, 2015.
- A survey of open source tools for machine learning with big data in the hadoop ecosystem
  - S. Landset, T. M. Khoshgoftaar, A. N. Richter, and T. Hasanin, Journal of Big Data, vol. 2, no. 1, p. 24, 2015.
- A survey of optimization methods from a machine learning perspective
  - S. Sun, Z. Cao, H. Zhu, and J. Zhao, IEEE Trans on Cybernetics, 2019.

### Model Simplification.
#### Kernel-based Models
- Sampling methods for the nystrom method
  - S. Kumar, M. Mohri, and A. Talwalkar, JMLR, vol. 13, no. Apr, pp. 981–1006, 2012.
- Nystrom method vs random fourier features: A theoretical and empirical comparison
  - T. Yang, Y.-F. Li, M. Mahdavi, R. Jin, and Z.-H. Zhou, NeurIPS, 2012, pp. 476–484.
- Scaling up graph-based semisupervised learning via prototype vector machine
  - K. Zhang, L. Lan, J. T. Kwok, S. Vucetic, and B. Parvin, IEEE TNNLS, vol. 26, no. 3, pp. 444–457, 2015.
- Sampling with minimum sum of squared similarities for nystrom-based large scale spectral clustering
  - D. Bouneffouf and I. Birol, IJCAI, 2015, pp. 2313–2319.
- A novel greedy algorithm for nystrom approximation
  - A. Farahat, A. Ghodsi, and M. Kamel, AISTATS, 2011, pp. 269–277.
- Improved nystrom low-rank approximation and error analysis
  - K. Zhang, I. W. Tsang, and J. T. Kwok, ICML, 2008, pp. 1232–1239.
- A randomized algorithm for the decomposition of matrices
  - P.-G. Martinsson, V. Rokhlin, and M. Tygert, Applied and Computational Harmonic Analysis, vol. 30, no. 1, pp. 47–68, 2011.
- Randomized sketches for kernels: Fast and optimal nonparametric regression
  - Y. Yang, M. Pilanci, M. J. Wainwright et al., The Annals of Statistics, vol. 45, no. 3, pp. 991–1023, 2017.
#### Graph-based Models
- Fast knn graph construction with locality sensitive hashing
  - Y.-M. Zhang, K. Huang, G. Geng, and C.-L. Liu, ECML PKDD, 2013, pp. 660–674.
- Scalable k-nn graph construction for visual descriptors
  - J. Wang, J. Wang, G. Zeng, Z. Tu, R. Gan, and S. Li, CVPR, 2012, pp. 1106–1113.
- Fast approximate k nn graph construction for high dimensional data via recursive lanczos bisection
  - J. Chen, H. R. Fang, and Y. Saad, JMLR, vol. 10, no. 5, pp. 1989–2012, 2009.
- Locally optimized product quantization for approximate nearest neighbor search
  - Y. Kalantidis and Y. Avrithis, CVPR, 2014, pp. 2321–2328.
- Large graph construction for scalable semi-supervised learning
  - W. Liu, J. He, and S.-F. Chang, ICML, 2010, pp. 679–686.
- FLAG: Faster learning on anchor graph with label predictor optimization
  - W. Fu, M. Wang, S. Hao, and T. Mu, IEEE TBD, no. 1, pp. 1–1, 2017.
- Learning on big graph: Label inference and regularization with anchor hierarch
  - M. Wang, W. Fu, S. Hao, H. Liu, and X. Wu, IEEE TKDE, vol. 29, no. 5, pp. 1101–1114, 2017.
- Scalable semisupervised learning by efficient anchor graph regularization
  - M. Wang, W. Fu, S. Hao, D. Tao, and X. Wu, IEEE TKDE, vol. 28, no. 7, pp. 1864–1877, 2016.
#### Deep Models
- Mobilenets: Efficient convolutional neural networks for mobile vision applications
  - A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, arXiv preprint arXiv:1704.04861, 2017.
- Rigid-motion scattering for image classification
  - L. Sifre and S. Mallat, Ph. D. thesis, 2014. 
- Going deeper with convolutions
  - C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, CVPR, 2015, pp. 1–9.
- Shufflenet: An extremely efficient convolutional neural network for mobile devices
  - X. Zhang, X. Zhou, M. Lin, and J. Sun, CVPR, 2018, pp. 6848–6856.
- Rethinking the inception architecture for computer vision
  - C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, CVPR, 2016, pp. 2818–2826.
- Multi-scale context aggregation by dilated convolutions
  - F. Yu and V. Koltun, ICLR, 2016.
- Bounded activation functions for enhanced training stability of deep neural networks on visual pattern recognition problems
  - S. S. Liew, M. Khalil-Hani, and R. Bakhteri, Neurocomputing, vol. 216, pp. 718–734, 2016.
- Addernet: Do we really need multiplications in deep learning?
  - H. Chen, Y. Wang, C. Xu, B. Shi, C. Xu, Q. Tian, and C. Xu, arXiv preprint arXiv:1912.13200, 2019.
#### Tree-based Models
- Fast and balanced: Efficient label tree learning for large scale object recognition
  - J. Deng, S. Satheesh, A. C. Berg, and F. Li, NeurIPS, 2011, pp. 567–575.
- Xgboost: A scalable tree boosting system
  - T. Chen and C. Guestrin, SIGKDD, 2016, pp. 785–794.
- Lightgbm: A highly efficient gradient boosting decision tree
  - G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.- Y. Liu, NeurIPS, 2017, pp. 3146–3154.
- Random forests
  - L. Breiman, Machine learning, vol. 45, no. 1, pp.5–32, 2001.
- Xgboost: A scalable tree boosting system
  - T. Chen and C. Guestrin, SIGKDD, 2016, pp. 785–794.
- A streaming parallel decision tree algorithm
  - Y. Ben-Haim and E. Tom-Tov, JMLR, vol. 11, no. 2, 2010.

### Optimization Approximation.
#### For Mini-batch Gradient Descent
- Variance reduction in sgd by distributed importance samplin
  - G. Alain, A. Lamb, C. Sankar, A. Courville, and Y. Bengio, arXiv preprint arXiv:1511.06481, 2015.
- Accurate, large minibatch sgd: Training imagenet in 1 hour
  - P. Goyal, P. Dollar, R. Girshick, P. Noordhuis, L. Wesolowski, A. Kyrola, A. Tulloch, Y. Jia, and K. He, arXiv preprint arXiv:1706.02677, 2017.
- Accelerating stochastic gradient descent using predictive variance reduction
  - R. Johnson and T. Zhang, NeurIPS, 2013, pp. 315–323.
- Gradient methods for minimizing composite functions
  - Y. Nesterov, Mathematical Programming, vol. 140, no. 1, pp. 125–161, 2013.
- On the momentum term in gradient descent learning algorithms
  N. Qian, Neural networks, vol. 12, no. 1, pp. 145–151, 1999.
- Minimizing finite sums with the stochastic average gradient
  - M. Schmidt, N. Le Roux, and F. Bach, Mathematical Programming, vol. 162, no. 1-2, pp. 83–112, 2017.
- A stochastic quasi-newton method for large-scale optimization
  - R. H. Byrd, S. L. Hansen, J. Nocedal, and Y. Singer, SIAM Journal on Optimization, vol. 26, no. 2, pp. 1008–1031, 2016.
- Lsd-slam: Large-scale direct monocular slam
  - J. Engel, T. Schops, and D. Cremers, ECCV, 2014, pp. 834–849.
- On optimization methods for deep learning
  - Q. V. Le, J. Ngiam, A. Coates, A. Lahiri, B. Prochnow, and A. Y. Ng, ICML, 2011, pp. 265–272.
- Towards optimal one pass large scale learning with averaged stochastic gradient descent
  - W. Xu, arXiv preprint arXiv:1107.2490, 2011.
- Adaptive subgradient methods for online learning and stochastic optimization
  - J. Duchi, E. Hazan, and Y. Singer, JMLR, vol. 12, no. Jul, pp. 2121–2159, 2011.
- Adam: A method for stochastic optimization
  - D. P. Kingma and J. Ba, arXiv preprint arXiv:1412.6980, 2014.
- Adadelta: an adaptive learning rate method
  - M. D. Zeiler, arXiv preprint arXiv:1212.5701, 2012.

#### For Coordinate Gradient Descent
- Coordinate descent method for large-scale l2-loss linear support vector machines
  - K.-W. Chang, C.-J. Hsieh, and C.-J. Lin, JMLR, vol. 9, no. Jul, pp. 1369–1398, 2008.
- A dual coordinate descent method for large-scale linear svm
  - C.-J. Hsieh, K.-W. Chang, C.-J. Lin, S. S. Keerthi, and S. Sundararajan, ICML, 2008, pp. 408–415.
- Nearest neighbor based greedy coordinate descent
  - I. S. Dhillon, P. K. Ravikumar, and A. Tewari, NeurIPS, 2011, pp. 2160–2168.
- Coordinate descent converges faster with the gauss-southwell rule than random selection
  - J. Nutini, M. Schmidt, I. Laradji, M. Friedlander, and H. Koepke, ICML, 2015, pp. 1632–1641.
- Efficient accelerated coordinate descent methods and faster algorithms for solving linear systems
  - Y. T. Lee and A. Sidford, in IEEE FOCS, 2013, pp. 147–156.
- Efficiency of coordinate descent methods on hugescale optimization problems
  - Y. Nesterov, SIAM Journal on Optimization, vol. 22, no. 2, pp. 341–362, 2012.
- A fast iterative shrinkage-thresholding algorithm for linear inverse problem
  - A. Beck and M. Teboulle, SIAM journal on imaging sciences, vol. 2, no. 1, pp. 183–202, 2009.
- An accelerated proximal coordinate gradient method
  - Q. Lin, Z. Lu, and L. Xiao, NeurIPS, 2014, pp. 3059–3067.
- Accelerated proximal gradient methods for nonconvex programming
  - H. Li and Z. Lin, NeurIPS, 2015, pp. 379–387.
- Distributed optimization and statistical learning via the alternating direction method of multipliers
  - S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein et al., Foundations and Trends R in Machine learning, vol. 3, no. 1, pp. 1–122, 2011.

#### For Numerical Integration with MCMC
- Bayesian learning via stochastic gradient langevin dynamics
  - M. Welling and Y. W. Teh, ICML, 2011, pp. 681–688.
- Bayesian posterior sampling via stochastic gradient fisher scoring
  - S. Ahn, A. Korattikara, and M. Welling, arXiv preprint arXiv:1206.6380, 2012.
- Stochastic gradient hamiltonian monte carlo
  - T. Chen, E. Fox, and C. Guestrin, ICML, 2014, pp. 1683–1691.
- Stochastic gradient riemannian langevin dynamics on the probability simplex
  - S. Patterson and Y. W. Teh, NeurIPS, 2013, pp. 3102–3110.
- A complete recipe for stochastic gradient mcmc
  - Y.-A. Ma, T. Chen, and E. Fox, NeurIPS, 2015, pp. 2917–2925.

### Computation Parallelism.
#### For Multi-core Machines
- Parallel dual coordinate descent method for large-scale linear classification in multi-core environments
  - W.-L. Chiang, M.-C. Lee, and C.-J. Lin, SIGKDD, 2016, pp. 1485–1494.
- A fast parallel stochastic gradient method for matrix factorization in shared memory systems
  - W.-S. Chin, Y. Zhuang, Y.-C. Juan, and C.-J. Lin, ACM TIST, vol. 6, no. 1, pp. 1–24, 2015.
- The shogun machine learning toolbox
  - S. Sonnenburg, S. Henschel, C. Widmer, J. Behr, A. Zien, F. d. Bona, A. Binder, C. Gehl, V. Franc et al., JMLR, vol. 11, no. 6, pp. 1799–1802, 2010.
- Hogwild: A lock-free approach to parallelizing stochastic gradient descent
  - B. Recht, C. Re, S. Wright, and F. Niu, NeurIPS, 2011, pp. 693–701.
- Theano: Deep learning on gpus with python
  - J. Bergstra, F. Bastien, O. Breuleux, et al., NeurIPS, vol. 3. Citeseer, 2011, pp. 1–48.
- Caffe: Convolutional architecture for fast feature embedding
  - Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. B. Girshick, S. Guadarrama, and T. Darrell, ACM MM, 2014, pp. 675–678.
- Mxnet: A flexible and efficient machine learning library for heterogeneous distributed systems
  - T. Chen, M. Li, Y. Li, M. Lin, N. Wang, M. Wang, T. Xiao, B. Xu, C. Zhang, and Z. Zhang, arXiv preprint arXiv:1512.01274, 2015.
- Graphchi: Large-scale graph computation on just a PC
  - A. Kyrola, G. Blelloch, and C. Guestrin, OSDI, 2012, pp. 31–46.
- X-stream: Edgecentric graph processing using streaming partitions
  - A. Roy, I. Mihailovic, and W. Zwaenepoel, SOSP, 2013, pp. 472–488.
- Gridgraph: Large-scale graph processing on a single machine using 2-level hierarchical partitioning
  - X. Zhu, W. Han, and W. Chen, ,” in USENIX ATC, 2015, pp. 375–386.
- vdnn: Virtualized deep neural networks for scalable, memory-efficient neural network design
  - M. Rhu, N. Gimelshein, J. Clemons, A. Zulfiqar, and S. W. Keckler, MICRO. IEEE, 2016, pp. 1–13.

#### For Multi-machine Clusters
- Mapreduce: simplified data processing on large clusters
  - J. Dean and S. Ghemawat, Communications of the ACM, vol. 51, no. 1, pp. 107–113, 2008.

J. Rosen, N. Polyzotis, V. Borkar, Y. Bu, M. J. Carey, M. Weimer, T. Condie, and R. Ramakrishnan, “Iterative mapreduce for large scale machine learning,” arXiv preprint arXiv:1303.3517, 2013.

M. Boehm, S. Tatikonda, B. Reinwald, P. Sen, Y. Tian, D. R. Burdick, and S. Vaithyanathan, “Hybrid parallelization strategies for large-scale machine learning in systemml,” Proceedings of VLDB, vol. 7, no. 7, pp. 553–564, 2014.

M. Zaharia, M. Chowdhury, M. J. Franklin, S. Shenker, and I. Stoica, “Spark: cluster computing with working sets,” in Proceedings of Hot Topics in Cloud Computing, 2010, pp. 10–10.

X. Meng, J. K. Bradley, B. Yavuz, E. R. Sparks, S. Venkataraman, D. Liu, J. Freeman, D. B. Tsai, M. Amde, S. Owen et al., “Mllib: machine learning in apache spark,” JMLR, vol. 17, no. 34, pp. 1–7, 2016.

G. Malewicz, M. H. Austern, A. J. Bik, J. C. Dehnert, I. Horn, N. Leiser, and G. Czajkowski, “Pregel: a system for large-scale graph processing,” in Proceedings of SIGMOD, 2010, pp. 135–146.

Y. Low, D. Bickson, J. E. Gonzalez, C. Guestrin, A. Kyrola, and J. M. Hellerstein, “Distributed graphlab: a framework for machine learning and data mining in the cloud,” in Proceedings of VLDB, vol. 5, no. 8, 2012, pp. 716–727.

J. E. Gonzalez, Y. Low, H. Gu, D. Bickson, and C. Guestrin, “Powergraph: distributed graph-parallel computation on natural graphs.” in Proceedings of OSDI, vol. 12, no. 1, 2012, p. 2.

R. Chen, J. Shi, Y. Chen, B. Zang, H. Guan, and H. Chen, “Powerlyra: Differentiated graph computation and partitioning on skewed graphs,” ACM Transactions on Parallel Computing (TOPC), vol. 5, no. 3, pp. 1–39, 2019.

J. Dean, G. Corrado, R. Monga, K. Chen, M. Devin, M. Mao, A. Senior, P. Tucker, K. Yang, Q. V. Le et al., “Large scale distributed deep networks,” in Proceedings of NeurIPS, 2012, pp. 1223–1231.

M. Li, D. G. Andersen, J. W. Park, A. J. Smola, A. Ahmed, V. Josifovski, J. Long, E. J. Shekita, and B.-Y. Su, “Scaling distributed machine learning with the parameter server,” in Proceedings of OSDI, vol. 14, 2014, pp. 583–598.

T. Chen, M. Li, Y. Li, M. Lin, N. Wang, M. Wang, T. Xiao, B. Xu, C. Zhang, and Z. Zhang, “Mxnet: A flexible and efficient machine learning library for heterogeneous distributed systems,” arXiv preprint arXiv:1512.01274, 2015.

J. Jiang, B. Cui, C. Zhang, and F. Fu, “Dimboost: Boosting gradient boosting decision tree to higher dimensions,” in Proceedings of ICDM, 2018, pp. 1363–1376.

E. P. Xing, Q. Ho, W. Dai, J. K. Kim, J. Wei, S. Lee, X. Zheng, P. Xie, A. Kumar, and Y. Yu, “Petuum: A new platform for distributed machine learning on big data,” IEEE TBD, vol. 1, no. 2, pp. 49–67, 2015.

T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in Proceedings of SIGKDD, 2016, pp. 785–794.

G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.Y. Liu, “Lightgbm: A highly efficient gradient boosting decision tree,” in Proceedings of NeurIPS, 2017, pp. 3146–3154.

P. Patarasuk and X. Yuan, “Bandwidth optimal all-reduce algorithms for clusters of workstations,” Elsevier JPDC, vol. 69, no. 2, pp. 117–124, 2009.

A. Sergeev and M. Del Balso, “Horovod: fast and easy distributed deep learning in tensorflow,” arXiv preprint arXiv:1802.05799, 2018.

P. Goyal, P. Dollar, R. Girshick, P. Noordhuis, L. Wesolowski, A. Kyrola, A. Tulloch, Y. Jia, and K. He, “Accurate, large minibatch sgd: Training imagenet in 1 hour,” arXiv preprint arXiv:1706.02677, 2017.

### Hybrid Collaboration.
F. Seide, H. Fu, J. Droppo, G. Li, and D. Yu, “1-bit stochastic gradient descent and its application to data-parallel distributed training of speech dnns,” in Proceedings of INTERSPEECH, 2014.

D. Alistarh, D. Grubic, J. Li, R. Tomioka, and M. Vojnovic, “Qsgd: Communication-efficient sgd via gradient quantization and encoding,” in Proceedings of NeurIPS, 2017, pp. 1709–1720.

W. Wen, C. Xu, F. Yan, C. Wu, Y. Wang, Y. Chen, and H. Li, “Terngrad: Ternary gradients to reduce communication in distributed deep learning,” in Proceedings of NeurIPS, 2017, pp. 1509–1519.

Y. Lin, S. Han, H. Mao, Y. Wang, and W. J. Dally, “Deep gradient compression: Reducing the communication bandwidth for distributed training,” arXiv preprint arXiv:1712.01887, 2017.

J. Jiang, F. Fu, T. Yang, and B. Cui, “Sketchml: Accelerating distributed machine learning with data sketches,” in Proceedings of SIGMOD, 2018, pp. 1269–1284.

H. Zhang, J. Li, K. Kara, D. Alistarh, J. Liu, and C. Zhang, “Zipml: Training linear models with end-to-end low precision, and a little bit of deep learning,” in Proceedings of ICML, 2017, pp. 4035–4043.

J. Langford, A. Smola, and M. Zinkevich, “Slow learners are fast,” arXiv preprint arXiv:0911.0491, 2009.

A. Agarwal and J. C. Duchi, “Distributed delayed stochastic optimization,” in Proceedings of NeurIPS, 2011, pp. 873–881.

M. Jaggi, V. Smith, M. Takac, J. Terhorst, S. Krishnan, T. Hofmann, and M. I. Jordan, “Communication-efficient distributed dual coordinate ascent,” in Proceedings of NeurIPS, 2014, pp. 3068–3076.

P. Richtarik and M. Tak ´ a´c, “Distributed coordinate descent method for learning with big data,” JMLR, vol. 17, no. 1, pp. 2657–2681, 2016.

L. Y. Zhang S, Choromanska AE, “Deep learning with elastic averaging sgd,” in Proceedings of NeurIPS, 2015, pp. 685–693.

M. Zinkevich, M. Weimer, L. Li, and A. J. Smola, “Parallelized stochastic gradient descent,” in Proceedings of NeurIPS, 2010, pp. 2595–2603.

S. Zheng, Q. Meng, T. Wang, W. Chen, N. Yu, Z.-M. Ma, and T.-Y. Liu, “Asynchronous stochastic gradient descent with delay compensation,” in Proceedings of ICML, 2017, pp. 4120–4129.

H. Tang, X. Lian, T. Zhang, and J. Liu, “Doublesqueeze: Parallel stochastic gradient descent with double-pass error-compensated compression,” arXiv preprint arXiv:1905.05957, 2019.

J. Wu, W. Huang, J. Huang, and T. Zhang, “Error compensated quantized sgd and its applications to large-scale distributed optimization,” arXiv preprint arXiv:1806.08054, 2018.

