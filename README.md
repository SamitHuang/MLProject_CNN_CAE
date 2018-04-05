
Task 1: CNN
1) To start training and tuning with cross-validation, please run (e.g.): 
    $ python main.py --task=train_cnn --datapath=../datasets --lr=0.01 --batch_size=64 --momentum=0.9
    lr is short for learning rate. 
    It will report the performance (in terms of ccuaracy and loss) on the training set in the end of training.
    
2) To start training without cross-validation to save time, please specify --cross_validate=0, e.g:
    $ python main.py --task=train_cnn --datapath=../datasets --cross_validate=0 --lr=0.01 --batch_size=64 --momentum=0.9

3) To test the well-trained model and report the performance on test set, please run:
    $ python main.py --task=test_cnn 
    
    It will load the model which is trained under the best parameter settings that I tuned: lr=0.01, batch_size=64, momentum=0.9
    Plase make sure  the "checkpoints" directory is in the right position as I have already uploaded.


My tuning result using 5-fold cross validation:

LR      Acc     Loss    Time per fold(s) 
0.01    0.8419  0.5098  81
0.001   0.8324  0.5321  81
0.0001  0.5966  1.4059  81

BS  Acc     Loss    Time per fold(s)
64  0.8476  0.475   101
256 0.8419  0.5098  71
512 0.8394  0.519   66
(BS - Batch Size)

Momentum    Acc     Loss    Time per fold(s) 
0.1         0.8393  0.541   101
0.5         0.84    0.5207  101
0.9         0.8476  0.475   101

Best params: lr=0.01, batch_size=64, momentum=0.9
Performance:
Acc on training set     0.956
Loss on training set    0.1297
Acc on test set         0.8375
Loss on test set        0.591



Task 2: CAE
1) To tune the parameters for CAE training, please run (e.g.):
    $ python main.py --task=train_ae --datapath=../datasets --lr=0.01 --batch_size=64
    It will compute the loss of the CAE model on the training set while training.
    
2) To evaluate the CAE model and compute the feature maps and reconstructed images for CAE training, please run: 
    $ python main,py --task=evaluate_ae --datapath=../datasets
    
    It will report the loss of the trained CAE on evaluation set and show visualization results.
    Plase make sure  the "checkpoints" directory is in the right position as I have already uploaded.
    

My tunning result:
LR      loss        time
0.01    0.085291    4m5s
0.001   0.003436    4m5s
0.0001  0.004201    4m5s

BS  loss        time
32  0.003168    10m42s
64  0.003196    6m43
256 0.003436    4m5s
512 0.003602    3m37s

Best params: learning rate = 0.001, batch size=32
Best performance: 
loss on training set    0.003165 
loss on evulation set   0.003168
