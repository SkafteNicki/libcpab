#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:54:24 2019

@author: nsde
"""

class args:
    experiment = 'regression_benchmark'

if __name__ == '__main__':
    if args.experiment == 'regression_benchmark':
        from sklearn.model_selection import train_test_split
        from sklearn import preprocessing
    
        # Load data
        dataset = np.load('data/regression_datasets/' + args.dataset + '.npz')
        X, y = dataset['data'], dataset['target'].reshape(-1,1)
    
        log_score, rmse_score = [ ], [ ]
        # Train multiple models
        for i in range(args.repeats):
            print("==================== Model {0}/{1} ====================".format(i+1, args.repeats))
            # Make train/test split
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, 
                                                            test_size=args.test_size,
                                                            random_state=(i+1)*args.seed)
            # Normalize data
            scaler = preprocessing.StandardScaler()
            scaler.fit(Xtrain)
            Xtrain = scaler.transform(Xtrain)
            Xtest = scaler.transform(Xtest)

            # Fit
            model = NeuralNet(X.shape[1], 50, 'cuda')
            model.fit(Xtrain, ytrain, n_epochs = 1000, batch_size = 256, lr = 1e-2)
        
            # Predict
            log, rmse = model.evaluate(Xtest, ytest)
        
            log_score.append(log)
            rmse_score.append(rmse)
        
        log_score = np.array(log_score)
        rmse_score = np.array(rmse_score)

        # Print the results
        print('log(px): {0:.3f} +- {1:.3f}'.format(log_score.mean(), log_score.std()))
        print('rmse:    {0:.3f} +- {1:.3f}'.format(rmse_score.mean(), rmse_score.std()))