import numpy as np

debug_flag = 0
def debug(*args):
    if(debug_flag):
        print(*args)

##########################################
# Bayesian Belief Network
class BBN():
    def __init__(self,):
        self.table = CPT()

    ####################
    # Handle BNN call
    def __call__(self,query):
        # Separate terms in query
        print('Query:',query)
        num = []
        den = []
        den_flag = False
        for q in query:
            if(q=='|'):
                den = []
                den_flag = True
            else:
                num.append(q)
                den.append(q)

        if(den_flag):
            return round( self.P(num) / self.P(den) , 3 )
        else:
            return round( self.P(num) , 3 )

    #####################
    # Calculate prob
    def P(self,terms):
        debug('P_terms:',terms)

        new_terms  = []
        new_index = []
        for t in terms:
            t = self.table.lookup(t)
            new_terms.append(t['label'])
            new_index.append(t['index'])

        ## a
        a_lim_0 = 0
        a_lim_1 = 4
        if('a' in new_terms):
            a_lim_0 = new_index[new_terms.index('a')] - 1
            a_lim_1 = a_lim_0 + 1

        ## b
        b_lim_0 = 0
        b_lim_1 = 2
        if('b' in new_terms):
            b_lim_0 = new_index[new_terms.index('b')] - 1
            b_lim_1 = b_lim_0 + 1

        ## x
        x_lim_0 = 0
        x_lim_1 = 2
        if('x' in new_terms):
            x_lim_0 = new_index[new_terms.index('x')] - 1
            x_lim_1 = x_lim_0 + 1
        
        ## c
        c_lim_0 = 0
        c_lim_1 = 3
        if('c' in new_terms):
            c_lim_0 = new_index[new_terms.index('c')] - 1
            c_lim_1 = c_lim_0 + 1
        
        ## d
        d_lim_0 = 0
        d_lim_1 = 2
        if('d' in new_terms):
            d_lim_0 = new_index[new_terms.index('d')] - 1
            c_lim_1 = c_lim_0 + 1

        ## Do Calculation
        a_len = a_lim_1-a_lim_0
        b_len = b_lim_1-b_lim_0

        debug('a_limits:',a_lim_0,a_lim_1)
        debug('b_limits:',b_lim_0,b_lim_1)
        debug('x_limits:',x_lim_0,x_lim_1)
        debug('c_limits:',c_lim_0,c_lim_1)
        debug('d_limits:',d_lim_0,d_lim_1)

        x = self.table.Px[:,a_lim_0:a_lim_1,b_lim_0:b_lim_1]
        a = self.table.Pa[a_lim_0:a_lim_1]
        b = self.table.Pb[b_lim_0:b_lim_1]

        debug('x shape:',x.shape)
        debug(x)
        debug('a shape:',a.shape)
        debug(a)
        debug('b shape:',b.shape)
        debug(b)


        c = x * b.T
        c = np.reshape(np.sum(c,2),(2,a_len,1))
        d = c * a
        d = np.sum(d,1)
        debug('absum:',d.shape)
        debug(d)
        e = d * self.table.Pc[c_lim_0:c_lim_1].T
        e = np.sum(e,1)
        debug('csum:',e.shape)
        debug(e)
        f = e * self.table.Pd[d_lim_0:d_lim_1]
        g = np.sum(f[:,x_lim_0:x_lim_1])
        debug('dsum:',g.shape)

        return g
   
        

##########################################
# Class for conditional probability table
class CPT():
    def __init__(self,):
        self.Pa = [.25,.25,.25,.25]
        self.Pb = [.6,.4]

        self.Px = [
            [
                [.5,.7],
                [.6,.8],
                [.4,.1],
                [.2,.3]
            ],
            [
                [.5,.3],
                [.4,.2],
                [.6,.9],
                [.8,.7]
            ]
        ]

        self.Pc = [
            [.6,.2],
            [.2,.3],
            [.2,.5]
        ]

        self.Pd = [
            [.3,.6],
            [.7,.4]
        ]

        self.Pa = np.reshape(np.array(self.Pa),(4,1))
        self.Pb = np.reshape(np.array(self.Pb),(2,1))
        self.Px = np.reshape(np.array(self.Px),(2,4,2))
        self.Pc = np.reshape(np.array(self.Pc),(3,2))
        self.Pd = np.reshape(np.array(self.Pd),(2,2))

        # Build label -> var label,num
        self.fish_dict = dict()
        labels = ['a','b','x','c','d']
        keys = [
            ['winter','spring','summer','autumn'],
            ['north','south'],
            ['salmon','seabass'],
            ['light','medium','dark'],
            ['wide','thin']
        ]
        for akeys,lab in zip(keys,labels):
            for i,a in enumerate(akeys):
                self.fish_dict[a]          = dict()
                self.fish_dict[a]['label'] = lab
                self.fish_dict[a]['index'] = i+1


    def lookup(self,key):
        return self.fish_dict[key]