"""Wrapper file for testing the Multiple Time Scale RNN given in Yamashita, Tani 2008.
This script contains:
    
    -MTSRNN: The chainer chain of links that defines the network structure and 
        some properties. Call and Reset functions sepcified here.
    -RNNPredictor: wraps a MTSRNN object. Train contains the training algorithm.
        Predict is a function for testing the model given some data.
        Save and Load are functions to save/load a pre-trained MTSRNN predictor.
    

"""

import chainer
from chainer import Variable
import tqdm
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer.functions.activation import softmax
from chainer import cuda
from chainer.functions import concat
import numpy as np
import My_Links as ML
#import cupy as cp
from chainer.functions.loss.mean_squared_error import mean_squared_error

class MTSRNN(Chain):
    """
    Multiple Time Scales Recurrent Neural Network
    Inout: input-output group of neurons, alpha = 1 (Elman units)
    Fast : fast reacting group of context neurons, alpha = ?
    Slow : slow reacting group of context neurons, alpha = ?
    """
    
    def __init__(self, mode, n_out, n_hidd, n_input, R, S):

        #super(EncoderDecoderRNN, self).__init__(encoder=L.GRU(enc_input, n_hidden), decoder=L.GRU(dec_input, n_hidden), output=L.Linear(n_hidden, dec_output))
        
        if mode == 'Static':
            super(MTSRNN, self).__init__(hidden=ML.DRU_static(R, S, n_hidd, n_input), readout=L.Linear(n_hidd, n_out))
        elif mode == 'LearnAlpha_local':
            super(MTSRNN, self).__init__(hidden=ML.DRU(n_hidd, n_input), readout=L.Linear(n_hidd, n_out))
        elif mode == 'LearnAlpha_global':
            super(MTSRNN, self).__init__(hidden=ML.DRU(n_hidd, n_input, dim=1), readout=L.Linear(n_hidd, n_out))
        else:
            print('--Error, specify mode--')
            from sys import exit
            exit()
            
        self.mode = mode
        self.n_out = n_out
        self.n_hidd = n_hidd
        self.n_input = n_input

    def __call__(self, x):
        """

        :param x:
        :param encoding: run network in encoding mode if true and decoding mode if false
        :return:
        """
        
        self.hidden(x)
        self.out = self.readout(self.hidden.y)
        self.hidden.firing_rate()        
        # Output signlas is firing rates of readout units
        
        return self.out

    def reset_state(self):
        self.hidden.reset_state()
        #self.readout.reset_state()

class RNNPredictor(object):

    "n_clamp is the number of time steps during which input is fed"
    
    def __init__(self, model, LR, device_id):

        self.model = model
        if device_id == -1:
            self.xp = np
        else:
            chainer.cuda.get_device_from_id(device_id).use()
            self.xp = cp
            self.model.to_gpu(device=device_id)
        self.device_id = device_id

        # Setup an optimizer
        self.optimizer = chainer.optimizers.Adam(alpha=LR) #SGD(lr = lr)
        self.optimizer.setup(self.model)
        self.optimizer.add_hook(chainer.optimizer.GradientClipping(2))
        self.optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))
        

    def train(self, train_data, train_input, test_data, test_input, n_epochs, filename = None, KL_loss = False, Add_training=False):
        """
        
        :param train_data: data in the form n_batches x batch_size x n_steps x n_outputs
        :param test_data: data in the form n_batches x batch_size x n_steps x n_outputs
        :param n_epochs: nr of training epochs
        :param dec_input: this is the input to the decoder, which can modulate input dynamics; size: step_size x n_inputs
        :return:
        """
        
        # keep track of loss
        train_loss = np.zeros(n_epochs)
        test_loss = np.zeros(n_epochs)
        batches_loss = np.zeros(train_data.shape[0]*n_epochs)
        
        # keep track of learned alphas and weights
        if self.model.mode is not 'Static':
            learning_alphasS = np.empty((n_epochs+1, self.model.hidden.alphaS.alpha.size))
            learning_alphasR = np.empty((n_epochs+1, self.model.hidden.alphaR.alpha.size))
            learning_alphasS[0,:] = self.model.hidden.alphaS.alpha
            learning_alphasR[0,:] = self.model.hidden.alphaR.alpha

#        saved_U_fast = np.zeros((n_epochs,self.model.n_fast, self.model.n_slow+self.model.n_inout))
#        saved_U_inout= np.zeros((n_epochs, self.model.n_inout, self.model.n_fast))
#        saved_W_fast = np.zeros((n_epochs, self.model.n_fast, self.model.n_fast))
#        saved_W_inout= np.zeros((n_epochs, self.model.n_inout, self.model.n_inout))
        
        index = 0 #for batches_wise loss
        best_loss = 4000
        
        if Add_training:
            self.optimizer.setup(self.model.slow)
	
    	#self.model.inout.W.W.data = np.zeros((25,25))  #NO RECURRENT CONNECTION IN OUTPUT LAYER!        
        
        
        for epoch in tqdm.tqdm(xrange(n_epochs)):
        #for epoch in xrange(n_epochs):
            with chainer.using_config('train', True):

                n_batches = train_data.shape[0]
                batch_size = train_data.shape[1]
                n_steps = train_data.shape[2]

                for i in range(n_batches):
                    #print('Sample number %i' %i)
                    loss = Variable(self.xp.array(0, 'float32'))
                    self.model.reset_state()
                    
                    #initialization for this batch
                    data0 = Variable(train_data[i, :, 0, :])
                    
                    self.model.hidden.initialize_state(batch_size)
                    #self.model.readout.initialize_state(batch_size)
                    
                    for t in xrange(0,n_steps,1):
                        x = Variable(train_input[i,:,t,:])
                        data = self.xp.array(train_data[i, :, t, :])                       
                        _loss = mean_squared_error(self.model(x), data)  # prediction mode
                        if KL_loss:
                            _loss = self.KL_divergence(self.model(), data)
                        #print _loss
                        train_loss[epoch] += cuda.to_cpu(_loss.data)
                        loss += _loss
                                                           
                    
                    batches_loss[index] = loss.data
                    index = index+1
                    
                    self.model.cleargrads()  #look into this function to clear grad of a whole link
                    loss.backward()
                    loss.unchain_backward()
		    
                    #self.model.inout.W.disable_update() #NO RECURRENT CONNECTIONS IN OUTPUT LAYER!
                    #if self.model.mode == 'Static':
                    #    self.model.hidden.alphaS.disable_update()
                    #    self.model.hidden.alphaR.disable_update()
                        
                    if Add_training:     #delete grads to be deleted! or use enable_update()
                        self.model.fast.U1.disable_update()
                        self.model.fast.W.disable_update()
                        self.model.inout.disable_update()
                        self.model.slow.W.disable_update()
                    self.optimizer.update()
                    #print 'UPDATE'                   
                
#            saved_U_fast[epoch,:,:] = self.model.fast.U.W.data
#            saved_U_inout[epoch,:,:]= self.model.inout.U.W.data
#            saved_W_fast[epoch,:,:] = self.model.fast.W.W.data
#            saved_W_inout[epoch, :,:]= self.model.inout.W.W.data
#            
#            #save learning of time constants
            if self.model.mode is not 'Static':
                learning_alphasS[epoch+1, :] = self.model.hidden.alphaS.alpha.data
                learning_alphasR[epoch+1, :] = self.model.hidden.alphaR.alpha.data

            # compute loss per epoch
            train_loss[epoch] /= (n_batches * batch_size * self.model.n_out)
            
            # save model at some epoch
            #epochs_save = np.linspace(0, n_epochs-n_epochs/10, num=10, dtype=int)
            #if epoch in epochs_save:
            #    thisname = 'model_at_epoch_%i' %epoch
            #    self.save('saved_models/'+filename+'/'+thisname) 

            # validation
            with chainer.using_config('train', False):

                n_batches = test_data.shape[0]
                batch_size = test_data.shape[1]
                n_steps = test_data.shape[2]
#                assert(n_steps == n_clamp+n_pred)
                
                for i in range(n_batches):
                    
                    self.model.reset_state()
                    data0 = Variable(test_data[i, :, t, :])
                    
                    self.model.hidden.initialize_state(batch_size)
#                    self.model.readout.initialize_state(batch_size)
                            
                    for t in xrange(0,n_steps,1):
                        x = Variable(test_input[i,:,t,:])
                        data = self.xp.array(test_data[i, :, t, :])
                        _loss = mean_squared_error(self.model(x), data)  # prediction mode
                        if KL_loss:
                            _loss = self.KL_divergence(self.model(), data)
                        test_loss[epoch] += cuda.to_cpu(_loss.data)
		    

            # compute loss per epoch
            test_loss[epoch] /= (n_batches * batch_size * self.model.n_out)
            
            #method do avoid overfitting
            if test_loss[epoch] < best_loss:
                best_loss = test_loss[epoch]
                self.save('saved_models/'+filename+'/best')
                np.save('saved_models/'+filename+'/conv_epoch', epoch)
            # end of training cycle

 	    np.save('saved_models/'+filename+'/best_loss', best_loss)
        if self.model.mode is not 'Static':
            np.save('saved_models/'+filename+'/learning_alphaS', learning_alphasS)
            np.save('saved_models/'+filename+'/learning_alphaR', learning_alphasR)
#        np.save('saved_U_fast', saved_U_fast)
#        np.save('saved_W_fast', saved_W_fast)
#        np.save('saved_U_inout', saved_U_inout)
#        np.save('saved_W_inout', saved_W_inout)     
#        np.save('saved_models/'+filename+'/saved_alphas_fast', learning_alphas_fast)
#        np.save('saved_models/'+filename+'/saved_alphas_slow', learning_alphas_slow)
#        np.save('saved_models/'+filename+'/saved_alphas_inout', learning_alphas_inout)
#        

        return train_loss, test_loss, batches_loss

    def predict(self, sample, sample_input):        
        n_steps = sample.shape[1]
        predicted = np.zeros((n_steps,self.model.n_out))
        u = np.zeros((n_steps,self.model.n_hidd))
        y = np.zeros((n_steps,self.model.n_hidd))

        # validation
        with chainer.using_config('train', False):
            self.model.reset_state()                
            self.model.reset_state()
            data0 = Variable(sample[0,0, :])
                    
            self.model.hidden.initialize_state()
#            self.model.readout.initialize_state()
                            
            for t in xrange(0,n_steps,1):
                x = Variable(sample_input[:,t,:])
                data = self.xp.array(sample[:,t, :])    
                _predicted = self.model(x)
                predicted[t,:] = _predicted.data
                u[t,:] = self.model.hidden.y.data
                y[t,:] = self.model.hidden.u.data
                               

        return predicted, u, y
    
    def save(self, fname):
        chainer.serializers.save_npz('{}_model'.format(fname), self.model)
    
    def load(self, fname):
        chainer.serializers.load_npz('{}_model'.format(fname), self.model)
        
    def KL_divergence(self, Ypred, Ydata):
        E = F.sum(Ydata*F.log(Ydata/Ypred))
        return E
        
