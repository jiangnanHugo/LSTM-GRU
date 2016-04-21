import numpy as np
import theano
import  theano.tensor as T
import operator

class GRU:

    def __init__(self,word_dim,hidden_dim=128,bptt_truncate=-1):
        self.word_dim=word_dim
        self.hidden_dim=hidden_dim
        self.bptt_truncate=bptt_truncate

        initial_E=np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim,word_dim))
        initial_U=np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(3,hidden_dim,hidden_dim))
        initial_W=np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(3,hidden_dim,hidden_dim))
        initial_V=np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(word_dim,hidden_dim))
        initial_b=np.zeros((3,hidden_dim))
        initial_c=np.zeros(word_dim)

        self.E=theano.shared(name='E',value=initial_E.astype(theano.config.floatX))
        self.U=theano.shared(name='U',value=initial_U.astype(theano.config.floatX))
        self.W=theano.shared(name='W',value=initial_W.astype(theano.config.floatX))
        self.V=theano.shared(name='V',value=initial_V.astype(theano.config.floatX))
        self.b=theano.shared(name='b',value=initial_b.astype(theano.config.floatX))
        self.c=theano.shared(name='c',value=initial_c.astype(theano.config.floatX))
        self.params=[self.E, self.V, self.U, self.W, self.b, self.c]
        self._build()

    def _build(self):
        E, V, U, W, b, c = self.E, self.V, self.U, self.W, self.b, self.c

        x=T.ivector('x')
        y=T.ivector('y')

        def _recurrence(x_t,s_tm1):
            x_e=E[:,x_t]

            # GRU Layer 1
            z_t=T.nnet.hard_sigmoid( T.dot( U[0], x_e ) + T.dot( W[0], s_tm1 ) + b[0])
            r_t=T.nnet.hard_sigmoid( T.dot( U[1], x_e ) + T.dot( W[1], s_tm1 ) + b[1])
            c_t=T.tanh ( T.dot( U[2], x_e ) + T.dot( W[2] , s_tm1 * r_t) + b[2])
            s_t=( T.ones_like(z_t) - z_t) * c_t + z_t * s_tm1

            o_t=T.nnet.softmax( T.dot( V, s_t ) + c )[0]
            return [o_t, s_tm1]

        [o,s],_=theano.scan(
            fn=_recurrence,
            sequences=x,
            truncate_gradient=self.bptt_truncate,
            outputs_info=[dict(initial=T.zeros(self.hidden_dim))]
        )

        prediction=T.argmax(o,axis=1)
        o_error=T.sum(T.nnet.categorical_crossentropy(o,y))

        gparams=T.grad(o_error,self.params)

        learning_rate=T.iscalar('Learning_rate')
        updates=[(param, param-learning_rate*gparam) for param,gparam in zip(self.params,gparams)]
        self.predict=theano.function([x],o)
        self.predict_class=theano.function([x],prediction)
        self.ce_error=theano.function([x,y],o_error)
        self.sgd_step=theano.function([x,y,learning_rate],[],updates=updates)


    def calculate_toal_loss(self,X,Y):
        num_words=np.sum([len(y) for y in Y])
        return np.sum([self.ce_error(x,y) for x,y in zip(X,Y)])/float(num_words)

  
