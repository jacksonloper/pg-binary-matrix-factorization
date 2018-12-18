from . import pgfacts
import numpy as np
import numpy.random as npr

def gaukl1Z(muhat,sighatsq):
    '''
    KL(N(muhat,sighatsq) || N(0,1))

    muhat**2 / 2 + .5 ( sighatsq - 1 - log(sighatsq))

    '''

    return .5*(muhat**2+sighatsq-1-np.log(sighatsq))

def Y_from_ABZ(muhat,sighatsq,alpha,beta):
    '''
    muhat: batch
    sighatsq: batch
    alpha: Ng
    beta: Ng

    returns batch x Ng

    '''

    return np.sqrt((np.outer(muhat,alpha)+beta[None])**2+np.outer(sighatsq,alpha**2))

def Z_from_ABXNU(nu,x,alpha,beta):
    '''
    nu: batch x Ng
    x: batch x Ng
    alpha: Ng
    beta: Ng

    returns batch (mu),batch (sighatsq)

    '''

    sighatsqi = 1+nu@alpha**2 # batch
    sighatsq = 1.0/(sighatsqi)

    muhat = np.sum((x-.5)*alpha[None]- beta[None]*alpha[None]*nu,axis=1) * sighatsq

    return muhat,sighatsq

def AB_from_ABXZ(x,nu,muhat,sighatsq,pseudocount=.01):
    '''
    x: batch x Ng
    nu: batch x Ng
    muhat: batch
    sighatsq: batch

    '''

    k1=(x.T-.5)@muhat # Ng
    k2=np.sum(x-.5,axis=0) #Ng

    m = nu.T@muhat # Ng
    s = nu.T@(muhat**2+sighatsq)
    w = np.sum(nu,axis=0)

    w=w+pseudocount
    s=s+pseudocount

    beta= (k2*s - k1*m) / (w*s-m**2)
    alpha = (k1 - beta*m) / s

    return alpha,beta 


class BinaryMatrixFactorization1D:
    def __init__(self,Ng,pseudocount=1.0):
        self.Ng=Ng
        self.alpha=np.zeros(Ng)
        self.beta=np.zeros(Ng)
        self.pseudocount=1.0

    def initialize(self,logits=None,alphainit=.0001,counts=None):
        if logits is not None:
            assert counts is None,"Can't supply both logits and counts"
            logits=np.require(logits,dtype=np.float64).ravel()
            assert logits.shape==(self.Ng,)
        elif counts is not None:
            counts=np.require(counts,dtype=np.float64)
            assert counts.shape==(2,self.Ng)
            logits=np.log((counts[1] + self.pseudocount)/(counts[0] + 2*self.pseudocount))
        else:
            logits=np.zeros(ng)

        self.beta=logits.copy()
        self.alpha=(npr.rand(self.Ng)-.5)*alphainit

    def calc_amortized_posterior(self,data,niter=4,chat=None,ms=None):
        '''
        Calcs posterior variational parameters for Z,Y,
        using niter iterations of CAVI.  
        '''

        data=np.require(data,dtype=np.float64)
        Nbatch,Ng = data.shape
        assert Ng==self.Ng

        if (chat is None) and (ms is None): # make up initial conditiosn
            chat=np.zeros((Nbatch,Ng))
            nu = pgfacts.pge_safe(chat)
        elif (chat is not None): # we have Y initial conditions
            nu = pgfacts.pge_safe(chat)
        elif (ms is not None): # we have Z initial conditions
            muhat,sighatsq=ms
            chat = Y_from_ABZ(muhat,sighatsq,self.alpha,self.beta)
            nu=pgfacts.pge_safe(chat)

        for i in range(niter):
            muhat,sighatsq=Z_from_ABXNU(nu,data,self.alpha,self.beta)
            chat = Y_from_ABZ(muhat,sighatsq,self.alpha,self.beta)
            nu=pgfacts.pge_safe(chat)

        return (muhat,sighatsq),chat


    def calc_ELBO(self,x,muhat,sighatsq,chat):
        x=np.require(x,dtype=np.float64)
        nu=pgfacts.pge_safe(chat)

        muhatalphapbeta=np.outer(muhat,self.alpha)+self.beta[None] # batch x Ng

        ELBO=-2*np.log(self.Ng)
        ELBO += -np.sum(gaukl1Z(muhat,sighatsq))
        ELBO += -np.sum(pgfacts.pkl(chat))
        ELBO += np.sum((x-.5)*muhatalphapbeta)
        ELBO += -.5*np.sum(nu*(muhatalphapbeta**2 + np.outer(sighatsq,self.alpha**2)))

        return ELBO/np.prod(x.shape)

    def estimate_alphabeta(self,x,muhat,sighatsq,chat,LR=.1,pseudocount=None):
        if pseudocount is None:
            pseudocount=self.pseudocount

        nu=pgfacts.pge_safe(chat)
        return AB_from_ABXZ(x,nu,muhat,sighatsq,pseudocount=pseudocount)

    def update_alphabeta(self,x,muhat,sighatsq,chat,LR=.1,pseudocount=None):
        alpha,beta = self.estimate_alphabeta(x,muhat,sighatsq,chat,LR=LR,pseudocount=pseudocount)
        self.alpha=alpha*LR+(1-LR)*self.alpha
        self.beta=beta*LR+(1-LR)*self.beta