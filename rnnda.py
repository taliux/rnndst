import numpy
from dA import dA
import theano
import theano.tensor as T

"""
	Simple RNN for belief tracking:
	
	bt = T.nnet.softmax([lambda xt_i,mt_i,i 
			              : T.dot( w,[  dA(xt_i),
						   			 	mt_i,
						   			 	btm1(i), 
						   			 	T.sum(btm1)-btm1(i) ] )
						, bias])
	where:
		bt -- belief state at turn t
		i  -- hypothesis index
		xt_i -- observation, i.e. word ngrams, with respect to hypothesis i at turn t
		mt_i -- system action features with respect to hypothesis i at turn t
		btm1 -- belief state at turn t-1
		
	model parameter:
		w	 -- (tied) weights to learn
		bias -- for the "none" hypothesis 
		dA(W,b,b_prime) -- dA parameters, see dA.py
"""

class rnnda(object):

	def __init__(self,n_visible, n_hidden, n_mact):
	'''Construct a symbolic RNN-dA and initialize parameters.

    n_visible : integer
        Number of visible units.
    n_hidden : integer
        Number of hidden units of the denoising autoencoder.
    n_mact :
    	Number of system action features
    
	Return a (belief, params, updates_train) tuple:

    belief : Theano vector
        Symbolic variable holding belief state
    params : tuple of Theano shared variables
        The parameters of the model to be optimized during training.
    updates_train : dictionary of Theano variable -> Theano variable
        Update object that should be passed to theano.function when compiling
        the training function.
	'''

		rng = numpy.random.RandomState(123);
		
		
		self.wx = theano.shared(numpy.asarray
				                  (rng.normal(size=(1,n_hidden), scale= .01, loc = .0)
				                  , dtype = theano.config.floatX)
				               );

		self.wm = theano.shared(numpy.asarray
				                  (rng.normal(size=(1,n_mact), scale= .01, loc = .0)
				                  , dtype = theano.config.floatX)
				               );

		self.wb1 = theano.shared(numpy.asscalar
				                  (rng.normal(size=(1,1), scale= .01, loc = .0)
				                  , dtype = theano.config.floatX)
				               );
		self.wb2 = theano.shared(numpy.asscalar
				                  (rng.normal(size=(1,1), scale= .01, loc = .0)
				                  , dtype = theano.config.floatX)
				               );
		self.wb3 = theano.shared(numpy.asscalar
				                  (rng.normal(size=(1,1), scale= .01, loc = .0)
				                  , dtype = theano.config.floatX)
				               );

    	self.bias = theano.shared(numpy.asscalar
				                  (rng.normal(size=(1,1), scale= .01, loc = .0)
				                  , dtype = theano.config.floatX)
				               );
		
    	self.da = dA(n_visible=n_visible,n_hidden=n_hidden);
    	self.params = [self.wx, self.wm, self.wb1, self.wb2, self.wb3, self.bias, self.da.params];
    	self.b0 = T.zeros((1,1));
    	
    def get_beliefs(self, input_x, input_m):
    	
    	
    	def recurrence(xt, mt, btm1):
    		
    		bt_, updates = theano.scan(
    			lambda x,m,i : T.dot(self.da.get_hidden_values(x)-0.5,self.wx)
    								+T.dot(m-0.5,self.wm)
    								+self.wb1*(btm1[i]-0.5)
    								+self.wb2*(T.sum(btm1)-btm1[i]-btm1[-1]-0.5)
    								+self.wb3*(btm1[-1]-0.5),
    			sequences=[xt,mt,T.arange(xt.shape[0])],
    			non_sequences=btm1
    		);
    		bt_.append(self.bias);
    		return T.nnet.softmax(bt_), updates;
    
    	bt, rnn_updates = theano.scan(
    		lambda xt, mt, btm1, *_ : recurrence(xt,mt,btm1),
    		sequences=[input_x,input_m],
    		outputs_info=[self.b0,None]
    	);
    	
    	return bt, rnn_updates;
    