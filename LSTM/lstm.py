import numpy as np
import theano
import theano.tensor as T


class LSTM:

    def __init__(self,n_input,n_hidden):
        self.n_input=n_input
        self.n_hidden=n_hidden
        self.f=T.nnet.hard_sigmoid

        # forget gate parameters.
        initial_Wf = np.asarray(
            np.random.uniform(
                low=-np.sqrt(1./n_input),
                high=np.sqrt(1./n_input),
                size=(n_hidden,n_input+n_hidden)),
            dtype=theano.config.floatX)

        initial_bf = np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wf=theano.shared(value=initial_Wf,name='Wf')
        self.bf=theano.shared(value=initial_bf,name='bf')

        # Input gate parameters.
        initial_Wi = np.asarray(
            np.random.uniform(
                low=-np.sqrt(1./n_input),
                high=np.sqrt(1./n_input),
                size=(n_hidden,n_input+n_hidden)),
            dtype=theano.config.floatX)

        initial_bi=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wi=theano.shared(value=initial_Wi,name='Wi')
        self.bi=theano.shared(value=initial_bi,name='bi')

        # Cell gate parameters
        initial_Wc=np.asarray(np.random.uniform(
            low=-np.sqrt(1./n_input),
            high=np.sqrt(1./n_input),
            size=(n_hidden,n_input+n_hidden)),
        dtype=theano.config.floatX)

        initial_bc=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wc=theano.shared(value=initial_Wc,name='Wc')
        self.bc=theano.shared(value=initial_bc,name='bc')

        # Output gate parameters.
        initial_Wo=np.asarray(
            np.random.uniform(
                low=-np.sqrt(1./n_input),
                high=np.sqrt(1./n_hidden),
                size=(n_hidden,n_input+n_hidden)),
            dtype=theano.config.floatX)

        initial_bo=np.zeros((n_hidden),dtype=theano.config.floatX)

        self.Wo=theano.shared(value=initial_Wo,name='Wi')
        self.bo=theano.shared(value=initial_bo,name='bo')

        self.params=[self.Wi,self.Wf,self.Wc,self.Wo,self.bi,self.bf,self.bc,self.bo]
        self._build()

    def _build(self):
        x=T.fmatrix('x')
        y=T.fmatrix('y')

        '''
            Compute hidden state in an LSTM
            :param x_t: Input vector
            :param h_prev: Hidden variable from previous time step.
            :param c_prev: Cell state from previous time step.
            :return: [new hidden variable, updated cell state]
        '''
        def _recurrence(x_t,h_tm1,c_tm1):
            concated=T.concatenate([x_t,h_tm1])
            # Forget gate
            f_t=self.f(T.dot(self.Wf,concated)+self.bf)
            # Input gate
            i_t=self.f(T.dot(self.Wi,concated)+self.bi)

            # Cell Update
            c_tilde_t=T.tanh(T.dot(self.Wc,concated)+self.bc)
            c_t=f_t * c_tm1 + i_t * c_tilde_t

            # output gate
            o_t=self.f(T.dot(self.Wo,concated)+self.bo)

            # Hidden state
            h_t= o_t * T.tanh(c_t)

            return [h_t,c_t]

        [h_t,c_t],_=theano.scan(
            fn=_recurrence,
            sequences=x,
            truncate_gradient=-1,
            outputs_info=[dict(initial=T.zeros(self.n_hidden)),
                          dict(initial=T.zeros(self.n_hidden))])

        o_error=((y-h_t)**2).sum()

        gparams=T.grad(o_error,self.params)
        learning_rate=T.scalar('learning_rate')
        #decay=T.scalar('decay')
        updates=[(param,param-learning_rate*gparam)
                 for param, gparam in zip(self.params,gparams)]

        self.predict=theano.function(
            inputs=[x],
            outputs=[h_t,c_t]
        )

        self.error=theano.function(
            inputs=[x,y],
            outputs=o_error
        )

        self.sgd_step=theano.function(
            inputs=[x,y,learning_rate],
            outputs=o_error,
            updates=updates
        )

    def compute_cost(self,x,y):
        return self.error(x,y)


