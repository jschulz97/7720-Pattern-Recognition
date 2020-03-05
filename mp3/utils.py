##########################################
# Class for conditional probability table
class BBN():
    def __init__(self,):
        self.table = CPT()

    def P(self,query):
        #Separate terms in query
        terms = []
        for q in query:
            if(q=='|'):
                prob = terms
                terms = []
            else:
                terms.append(q)
        given = terms

        



##########################################
# Class for conditional probability table
class CPT():
    def __init__(self,):
        self.Pa = dict()
        Pakeys = ['winter','spring','summer','autumn']
        vals = [.25,.25,.25,.25]
        for k,v in zip(Pakeys,vals):
            self.Pa[k] = v

        self.Pb = dict()
        Pbkeys = ['north','south']
        vals = [.6,.4]
        for k,v in zip(Pbkeys,vals):
            self.Pb[k] = v

        self.Px = dict()
        Pxkeys = ['salmon','seabass']
        vals = [.5,.7,.6,.8,.4,.1,.2,.3,.5,.3,.4,.2,.6,.9,.8,.7]
        for x in Pxkeys:
            self.Px[x] = dict()
            for a in Pakeys:
                self.Px[x][a] = dict()
                for b in Pbkeys:
                    self.Px[x][a][b] = vals.pop(0)

        self.Pc = dict()
        Pckeys = ['light','medium','dark']
        vals = [.6,.2,.2,.3,.2,.5]
        for c in Pckeys:
            self.Pc[c] = dict()
            for x in Pxkeys:
                self.Pc[c][x] = vals.pop(0)


        self.Pd = dict()
        Pdkeys = ['wide','thin']
        vals = [.3,.6,.7,.4]
        for d in Pdkeys:
            self.Pd[d] = dict()
            for x in Pxkeys:
                self.Pd[d][x] = vals.pop(0)
