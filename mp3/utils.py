##########################################
# Class for conditional probability table
class BBN():
    def __init__(self,):
        self.table = CPT()

    def P(self,query):
        # Separate terms in query
        #terms = []
        terms = dict()
        keys = ['a','b','x','c','d']
        for k in keys:
            terms[k] = False
        for q in query:
            if(q=='|'):
                _=0
                #prob = terms
                #terms = []
            else:
                #terms.append(q)
                if(q in self.table.Pakeys):
                    terms['a'] = q
                elif(q in self.table.Pbkeys):
                    terms['b'] = q
                elif(q in self.table.Pxkeys):
                    terms['x'] = q
                elif(q in self.table.Pckeys):
                    terms['c'] = q
                elif(q in self.table.Pdkeys):
                    terms['d'] = q
        # given = terms
        # terms.extend(prob)

        # If term specified
        if(terms['a']):
            # multiply to product

        # If not specified
        else:
            # 



        product = 1 
        if(not terms['a'] and not terms['b'] and terms['x'] and terms['c'] and terms['d']):
            sum = 0
            for a in self.table.Pakeys:
                for b in self.table.Pbkeys:
                    sum += self.table.Px[terms['x']][a][b]
            product *= sum  
            print('HERE')

            product *= self.table.Pc[terms['c']][terms['x']] * self.table.Pd[terms['d']][terms['x']]

        elif(not terms['a'] and terms['b'] and not terms['x'] and terms['c'] and terms['d']):
            sum = 0
            for a in self.table.Pakeys:
                for x in self.table.Pxkeys:
                    sum += self.table.Px[x][a][terms['b']]
            product *= sum
            sum = 0
            for x in self.table.Pxkeys:
                sum += self.table.Pc[terms['c']][x]
            product *= sum
            sum = 0
            for x in self.table.Pxkeys:
                sum += self.table.Pd[terms['d']][x]
            product *= sum

        elif(terms['a'] and not terms['b'] and not terms['x'] and terms['c'] and terms['d']):
            sum = 0
            for b in self.table.Pbkeys:
                for x in self.table.Pxkeys:
                    sum += self.table.Px[x][terms['a']][b]
            product *= sum
            sum = 0
            for x in self.table.Pxkeys:
                sum += self.table.Pc[terms['c']][x]
            product *= sum
            sum = 0
            for x in self.table.Pxkeys:
                sum += self.table.Pd[terms['d']][x]
            product *= sum

        elif(terms['a'] and terms['b'] and not terms['x'] and not terms['c'] and terms['d']):
            sum = 0
            for c in self.table.Pckeys:
                for x in self.table.Pxkeys:
                    sum += self.table.Pc[c][x]
            product *= sum
            sum = 0
            for x in self.table.Pxkeys:
                sum += self.table.Pd[terms['d']][x]
            product *= sum

        elif(terms['a'] and terms['b'] and not terms['x'] and terms['c'] and not terms['d']):
            sum = 0
            for d in self.table.Pdkeys:
                for x in self.table.Pxkeys:
                    sum += self.table.Pd[d][x]
            product *= sum
            sum = 0
            for x in self.table.Pxkeys:
                sum += self.table.Pd[terms['c']][x]
            product *= sum

        elif(not terms['a'] and not terms['b'] and not terms['x'] and terms['c'] and terms['d']):
            sum = 0
            for a in self.table.Pakeys:
                for b in self.table.Pbkeys:
                    for x in self.table.Pxkeys:
                        sum += self.table.Px[x][a][b]
            product *= sum
            sum = 0
            for x in self.table.Pxkeys:
                sum += self.table.Pc[terms['c']][x]
            product *= sum
            sum = 0
            for x in self.table.Pxkeys:
                sum += self.table.Pd[terms['d']][x]
            product *= sum

        return product
        # for t in prob:
        #     if(t in self.table.Pakeys):
        #         product *= self.table.Pa[t]

        # flag = False
        # for t in self.table.Pakeys:
        #     if(t in terms):
        #         flag = True
        #         product *= self.table.Pa[t]
        # if(!flag):
        #     sum 



        # if(any(s in self.table.Pakeys for s in prob)):
        #     _=0
        # elif(any(s in self.table.Pxkeys for s in prob)):
        #     _=0


##########################################
# Class for conditional probability table
class CPT():
    def __init__(self,):
        self.Pa = dict()
        self.Pakeys = ['winter','spring','summer','autumn']
        vals = [.25,.25,.25,.25]
        for k,v in zip(self.Pakeys,vals):
            self.Pa[k] = v

        self.Pb = dict()
        self.Pbkeys = ['north','south']
        vals = [.6,.4]
        for k,v in zip(self.Pbkeys,vals):
            self.Pb[k] = v

        self.Px = dict()
        self.Pxkeys = ['salmon','seabass']
        vals = [.5,.7,.6,.8,.4,.1,.2,.3,.5,.3,.4,.2,.6,.9,.8,.7]
        for x in self.Pxkeys:
            self.Px[x] = dict()
            for a in self.Pakeys:
                self.Px[x][a] = dict()
                for b in self.Pbkeys:
                    self.Px[x][a][b] = vals.pop(0)

        self.Pc = dict()
        self.Pckeys = ['light','medium','dark']
        vals = [.6,.2,.2,.3,.2,.5]
        for c in self.Pckeys:
            self.Pc[c] = dict()
            for x in self.Pxkeys:
                self.Pc[c][x] = vals.pop(0)


        self.Pd = dict()
        self.Pdkeys = ['wide','thin']
        vals = [.3,.6,.7,.4]
        for d in self.Pdkeys:
            self.Pd[d] = dict()
            for x in self.Pxkeys:
                self.Pd[d][x] = vals.pop(0)
