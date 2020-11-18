# Paper-on-Large-scale-Machine-Learning
#### Surveys
O. Y. Al-Jarrah, P. D. Yoo, S. Muhaidat, G. K. Karagiannidis, and K. Taha, “Efficient machine learning for big data: A review,” Big Data Research, vol. 2, no. 3, pp. 87–93, 2015.

L. Bottou, F. E. Curtis, and J. Nocedal, “Optimization methods for large-scale machine learning,” SIAM Review, vol. 60, no. 2, pp. 223–311, 2018.

C.-W. Tsai, C.-F. Lai, H.-C. Chao, and A. V. Vasilakos, “Big data analytics: a survey,” Journal of Big data, vol. 2, no. 1, p. 21, 2015.

S. Landset, T. M. Khoshgoftaar, A. N. Richter, and T. Hasanin, “A survey of open source tools for machine learning with big data in the hadoop ecosystem,” Journal of Big Data, vol. 2, no. 1, p. 24, 2015.

S. Sun, Z. Cao, H. Zhu, and J. Zhao, “A survey of optimization methods from a machine learning perspective,” IEEE Trans on Cybernetics, 2019.


#### We review LML according to three computational perspectives:
#### model simplification, which reduces computational complexities by simplifying predictive models; 
#### optimization approximation, which enhances computational efficiency by designing better optimization algorithms; 
#### computation parallelism, which improves computational capabilities by scheduling multiple computing devices.
### Model Simplification.
#### Kernel-based Models
S. Kumar, M. Mohri, and A. Talwalkar, “Sampling methods for the nystrom method,” JMLR, vol. 13, no. Apr, pp. 981–1006, 2012.

T. Yang, Y.-F. Li, M. Mahdavi, R. Jin, and Z.-H. Zhou, “Nystrom method vs random fourier features: A theoretical and empirical comparison,” in Advances in neural information processing systems, 2012, pp. 476–484.

K. Zhang, L. Lan, J. T. Kwok, S. Vucetic, and B. Parvin, “Scaling up graph-based semisupervised learning via prototype vector machines,” IEEE TNNLS, vol. 26, no. 3, pp. 444–457, 2015.

D. Bouneffouf and I. Birol, “Sampling with minimum sum of squared similarities for nystrom-based large scale spectral clustering.” in Proceedings of IJCAI, 2015, pp. 2313–2319.

A. Farahat, A. Ghodsi, and M. Kamel, “A novel greedy algorithm for nystrom approximation,” in Proceedings of AISTATS, 2011, pp. 269–277.

K. Zhang, I. W. Tsang, and J. T. Kwok, “Improved nystrom low-rank approximation and error analysis,” in Proceedings of ICML, 2008, pp. 1232–1239.

P.-G. Martinsson, V. Rokhlin, and M. Tygert, “A randomized algorithm for the decomposition of matrices,” Applied and Computational Harmonic Analysis, vol. 30, no. 1, pp. 47–68, 2011.

Y. Yang, M. Pilanci, M. J. Wainwright et al., “Randomized sketches for kernels: Fast and optimal nonparametric regression,” The Annals of Statistics, vol. 45, no. 3, pp. 991–1023, 2017.

#### Graph-based Models
Y.-M. Zhang, K. Huang, G. Geng, and C.-L. Liu, “Fast knn graph construction with locality sensitive hashing,” in ECML PKDD, 2013, pp. 660–674.

J. Wang, J. Wang, G. Zeng, Z. Tu, R. Gan, and S. Li, “Scalable k-nn graph construction for visual descriptors,” in Proceedings of CVPR, 2012, pp. 1106–1113.

J. Chen, H. R. Fang, and Y. Saad, “Fast approximate k nn graph construction for high dimensional data via recursive lanczos bisection,” JMLR, vol. 10, no. 5, pp. 1989–2012, 2009.

Y. Kalantidis and Y. Avrithis, “Locally optimized product quantization for approximate nearest neighbor search,” in Proceedings of CVPR, 2014, pp. 2321–2328.

W. Liu, J. He, and S.-F. Chang, “Large graph construction for scalable semi-supervised learning,” in Proceedings of ICML, 2010, pp. 679–686.

W. Fu, M. Wang, S. Hao, and T. Mu, “Flag: Faster learning on anchor graph with label predictor optimization,” IEEE TBD, no. 1, pp. 1–1, 2017.

M. Wang, W. Fu, S. Hao, H. Liu, and X. Wu, “Learning on big graph: Label inference and regularization with anchor hierarchy,” IEEE TKDE, vol. 29, no. 5, pp. 1101–1114, 2017.

M. Wang, W. Fu, S. Hao, D. Tao, and X. Wu, “Scalable semisupervised learning by efficient anchor graph regularization,” IEEE TKDE, vol. 28, no. 7, pp. 1864–1877, 2016.

#### Deep Models
A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto, and H. Adam, “Mobilenets: Efficient convolutional neural networks for mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017.

L. Sifre and S. Mallat, “Rigid-motion scattering for image classification,” Ph. D. thesis, 2014. 

C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions,” in Proceedings of CVPR, 2015, pp. 1–9.

X. Zhang, X. Zhou, M. Lin, and J. Sun, “Shufflenet: An extremely efficient convolutional neural network for mobile devices,” in Proceedings of CVPR, 2018, pp. 6848–6856.

C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the inception architecture for computer vision,” in Proceedings of CVPR, 2016, pp. 2818–2826.

F. Yu and V. Koltun, “Multi-scale context aggregation by dilated convolutions,” in Proceedings of ICLR, 2016.

S. S. Liew, M. Khalil-Hani, and R. Bakhteri, “Bounded activation functions for enhanced training stability of deep neural networks on visual pattern recognition problems,” Neurocomputing, vol. 216, pp. 718–734, 2016.

H. Chen, Y. Wang, C. Xu, B. Shi, C. Xu, Q. Tian, and C. Xu, “Addernet: Do we really need multiplications in deep learning?”arXiv preprint arXiv:1912.13200, 2019.

#### Tree-based Models
J. Deng, S. Satheesh, A. C. Berg, and F. Li, “Fast and balanced: Efficient label tree learning for large scale object recognition,” in Proceedings of NeurIPS, 2011, pp. 567–575.

T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in Proceedings of SIGKDD, 2016, pp. 785–794.

G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.- Y. Liu, “Lightgbm: A highly efficient gradient boosting decision tree,” in Proceedings of NeurIPS, 2017, pp. 3146–3154.

L. Breiman, “Random forests,” Machine learning, vol. 45, no. 1, pp.5–32, 2001.

T. Chen and C. Guestrin, “Xgboost: A scalable tree boosting system,” in Proceedings of SIGKDD, 2016, pp. 785–794.

Y. Ben-Haim and E. Tom-Tov, “A streaming parallel decision tree algorithm.” JMLR, vol. 11, no. 2, 2010.

### Optimization Approximation.
#### For Mini-batch Gradient Descent
G. Alain, A. Lamb, C. Sankar, A. Courville, and Y. Bengio, “Variance reduction in sgd by distributed importance sampling,” arXiv preprint arXiv:1511.06481, 2015.

P. Goyal, P. Dollar, R. Girshick, P. Noordhuis, L. Wesolowski, A. Kyrola, A. Tulloch, Y. Jia, and K. He, “Accurate, large minibatch sgd: Training imagenet in 1 hour,” arXiv preprint arXiv:1706.02677, 2017.

R. Johnson and T. Zhang, “Accelerating stochastic gradient descent using predictive variance reduction,” in Proceedings of NeurIPS, 2013, pp. 315–323.

Y. Nesterov, “Gradient methods for minimizing composite functions,” Mathematical Programming, vol. 140, no. 1, pp. 125–161, 2013.

N. Qian, “On the momentum term in gradient descent learning algorithms,” Neural networks, vol. 12, no. 1, pp. 145–151, 1999.

M. Schmidt, N. Le Roux, and F. Bach, “Minimizing finite sums with the stochastic average gradient,” Mathematical Programming, vol. 162, no. 1-2, pp. 83–112, 2017.

R. H. Byrd, S. L. Hansen, J. Nocedal, and Y. Singer, “A stochastic quasi-newton method for large-scale optimization,” SIAM Journal on Optimization, vol. 26, no. 2, pp. 1008–1031, 2016.

J. Engel, T. Schops, and D. Cremers, “Lsd-slam: Large-scale direct monocular slam,” in Proceedings of ECCV, 2014, pp. 834–849.

Q. V. Le, J. Ngiam, A. Coates, A. Lahiri, B. Prochnow, and A. Y. Ng, “On optimization methods for deep learning,” in Proceedings of ICML, 2011, pp. 265–272.

W. Xu, “Towards optimal one pass large scale learning with averaged stochastic gradient descent,” arXiv preprint arXiv:1107.2490, 2011.

J. Duchi, E. Hazan, and Y. Singer, “Adaptive subgradient methods for online learning and stochastic optimization,” JMLR, vol. 12, no. Jul, pp. 2121–2159, 2011.

D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.
 
M. D. Zeiler, “Adadelta: an adaptive learning rate method,” arXiv preprint arXiv:1212.5701, 2012.


#### For Coordinate Gradient Descent
K.-W. Chang, C.-J. Hsieh, and C.-J. Lin, “Coordinate descent method for large-scale l2-loss linear support vector machines,” Journal of Machine Learning Research, vol. 9, no. Jul, pp. 1369–1398,
2008.

C.-J. Hsieh, K.-W. Chang, C.-J. Lin, S. S. Keerthi, and S. Sundararajan, “A dual coordinate descent method for large-scale linear svm,” in Proceedings of ICML, 2008, pp. 408–415.

I. S. Dhillon, P. K. Ravikumar, and A. Tewari, “Nearest neighbor based greedy coordinate descent,” in Proceedings of NeurIPS, 2011, pp. 2160–2168.

J. Nutini, M. Schmidt, I. Laradji, M. Friedlander, and H. Koepke, “Coordinate descent converges faster with the gauss-southwell rule than random selection,” in Proceedings of ICML, 2015, pp. 1632–1641.

Y. T. Lee and A. Sidford, “Efficient accelerated coordinate descent methods and faster algorithms for solving linear systems,” in IEEE FOCS, 2013, pp. 147–156.

Y. Nesterov, “Efficiency of coordinate descent methods on hugescale optimization problems,” SIAM Journal on Optimization, vol. 22, no. 2, pp. 341–362, 2012.

A. Beck and M. Teboulle, “A fast iterative shrinkage-thresholding algorithm for linear inverse problems,” SIAM journal on imaging sciences, vol. 2, no. 1, pp. 183–202, 2009.
 
Q. Lin, Z. Lu, and L. Xiao, “An accelerated proximal coordinate gradient method,” in Proceedings of NeurIPS, 2014, pp. 3059–3067.

H. Li and Z. Lin, “Accelerated proximal gradient methods for nonconvex programming,” in Proceedings of NeurIPS, 2015, pp. 379–387.

S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein et al., “Distributed optimization and statistical learning via the alternating direction method of multipliers,” Foundations and Trends R in Machine learning, vol. 3, no. 1, pp. 1–122, 2011.

#### For Numerical Integration with MCMC
M. Welling and Y. W. Teh, “Bayesian learning via stochastic gradient langevin dynamics,” in Proceedings of ICML, 2011, pp. 681–688.

S. Ahn, A. Korattikara, and M. Welling, “Bayesian posterior sampling via stochastic gradient fisher scoring,” arXiv preprint arXiv:1206.6380, 2012.

T. Chen, E. Fox, and C. Guestrin, “Stochastic gradient hamiltonian monte carlo,” in Proceedings of ICML, 2014, pp. 1683–1691.

S. Patterson and Y. W. Teh, “Stochastic gradient riemannian langevin dynamics on the probability simplex,” in Proceedings of NeurIPS, 2013, pp. 3102–3110.

Y.-A. Ma, T. Chen, and E. Fox, “A complete recipe for stochastic gradient mcmc,” in Proceedings of NeurIPS, 2015, pp. 2917–2925.

### Computation Parallelism.
#### For Multi-core Machines
W.-L. Chiang, M.-C. Lee, and C.-J. Lin, “Parallel dual coordinate descent method for large-scale linear classification in multi-core environments,” in Proceedings of SIGKDD, 2016, pp. 1485–1494.

W.-S. Chin, Y. Zhuang, Y.-C. Juan, and C.-J. Lin, “A fast parallel stochastic gradient method for matrix factorization in shared memory systems,” ACM TIST, vol. 6, no. 1, pp. 1–24, 2015.

S. Sonnenburg, S. Henschel, C. Widmer, J. Behr, A. Zien, F. d. Bona, A. Binder, C. Gehl, V. Franc et al., “The shogun machine learning toolbox,” JMLR, vol. 11, no. 6, pp. 1799–1802, 2010.

B. Recht, C. Re, S. Wright, and F. Niu, “Hogwild: A lock-free approach to parallelizing stochastic gradient descent,” in Proceedings of NeurIPS, 2011, pp. 693–701.

J. Bergstra, F. Bastien, O. Breuleux, et al., “Theano: Deep learning on gpus with python,” in Proceedings of NeurIPS, vol. 3. Citeseer, 2011, pp. 1–48.

Y. Jia, E. Shelhamer, J. Donahue, S. Karayev, J. Long, R. B. Girshick, S. Guadarrama, and T. Darrell, “Caffe: Convolutional architecture for fast feature embedding,” in Proceedings of ACM MM, 2014, pp. 675–678.

T. Chen, M. Li, Y. Li, M. Lin, N. Wang, M. Wang, T. Xiao, B. Xu, C. Zhang, and Z. Zhang, “Mxnet: A flexible and efficient machine learning library for heterogeneous distributed systems,” arXiv preprint arXiv:1512.01274, 2015.

A. Kyrola, G. Blelloch, and C. Guestrin, “Graphchi: Large-scale graph computation on just a {PC},” in OSDI, 2012, pp. 31–46.

A. Roy, I. Mihailovic, and W. Zwaenepoel, “X-stream: Edgecentric graph processing using streaming partitions,” in Proceedings of SOSP, 2013, pp. 472–488.

X. Zhu, W. Han, and W. Chen, “Gridgraph: Large-scale graph processing on a single machine using 2-level hierarchical partitioning,” in USENIX ATC, 2015, pp. 375–386.

M. Rhu, N. Gimelshein, J. Clemons, A. Zulfiqar, and S. W. Keckler, “vdnn: Virtualized deep neural networks for scalable, memory-efficient neural network design,” in MICRO. IEEE,
2016, pp. 1–13.

#### For Multi-machine Clusters
J. Dean and S. Ghemawat, “Mapreduce: simplified data processing on large clusters,” Communications of the ACM, vol. 51, no. 1, pp. 107–113, 2008.

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

