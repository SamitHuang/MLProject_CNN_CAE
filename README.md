
Task 1: CNN
To start training and tuning with cross-validation using, please run: 
    $ python main.py --task=train_cnn --datapath=../datasets --lr=0.01 --batch_size=64 --momentum=0.9
    lr is short for learning rate. 
    It will report the performance (in terms of ccuaracy and loss) on the training set in the end of training.
    
To start training without cross-validation to save time, please specify --cross_validate=0, e.g:
    $ python main.py --task=train_cnn --datapath=../datasets --cross_validate=0 --lr=0.01 --batch_size=64 --momentum=0.9

To test the well-trained model and report the performance on test set, please run:
    $ python main.py --task=test_cnn 
    It will load the model which is trained under the best parameter settings that I tuned: lr=0.01, batch_size=64, momentum=0.9
    Plase make sure  the "checkpoints" directory is in the right position as I have already uploaded.



 
 
