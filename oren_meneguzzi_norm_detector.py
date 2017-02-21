from norm_detector import norm_detector
from planlib import goal_nodes,start_nodes,Goal
import logging as log

class common_norm_detector(norm_detector):
    
    translate_norms = True
    
    def convert_from_bayesian_norms(self,norms):
        """Converts norms in the bayesian form to our form"""
        nm_norms = set([])
        for n in norms:
            if len(n) == 2:
                (modality,node) = n
                if(modality == 'never'):
                    nm_norms.add( ("forbidden",node) )
                elif (modality == 'eventually'):
                    nm_norms.add( ("obliged",node) )
                else:
                    #log.warning("Unsupported norm ("+str(modality)+","+str(node)+")")
                    pass
            else:
                (pre, modality,node) = n
                #log.warning("Unsupported norm ("+str(pre)+","+str(modality)+","+str(node)+")")
        return nm_norms
    
    def convert_to_bayesian_norms(self, norms):
        bayesian_norms = set([])
        for (modality,node) in norms:
            if(modality == "obliged"):
                bayesian_norms.add( ("eventually",node))
            elif(modality == "forbidden"):
                bayesian_norms.add( ("never",node))
        
        return bayesian_norms
    
    def get_norm_hypotheses(self):
        """Returns a set of the possible norms considered by the detector (mostly for debugging)"""
        hypotheses = set([("obliged",s) for s in self.all_possible_states(self.planlib)])
        hypotheses |= set([("forbidden",s) for s in self.all_possible_states(self.planlib)])
        return hypotheses
    
    def separate_norms(self,norms):
        """ Separate /norms/ into each individual type of norm"""
        if(self.translate_norms): norms = self.convert_from_bayesian_norms(norms)
        o = set([])
        f = set([])
        
        for (modality,state) in norms:
            if modality == "obliged":
                o.add((modality,state))
            elif modality == "forbidden":
                f.add((modality,state))
            else:
                print "!! Invalid norm %s !!" % (modality)
                
        return (o,f)
    
    def count_violations(self, plan, norms):
        (o,f) = self.separate_norms(norms)
        v = 0; # number of violations
        indexes = []
        idx = 0
        for a in plan:
            for (modality,node) in f:
                if (a == node):
                    v += 1
                    indexes.append(idx)
            for (modality,node) in o: # If we see an obligation, it is fulfilled
                if (a == node):
                    o -= set([(modality,node)])
            
            idx += 1
        
        for remaining in o:
            v +=1
            indexes.append(idx)
        
        return (v,indexes)
    
    def set_goal(self, goal):
        """Updates the goal assumed by the norm detector"""
        self.goal=goal
    
    def get_goal(self):
        return self.goal

class basic_norm_detector(common_norm_detector):
    """A Python implementation of Oren and Meneguzzi's (COIN 2013) http://goo.gl/ZDZu1K basic norm detector"""

    def __init__(self, planlib, goal=None):
        """ Assigns variables for this class, I'm assuming here that planlib is a set of Actions (from the planlib module)"""
        super(basic_norm_detector,self).__init__(planlib)
        self.reinitialise()
        self.past_observations = [] # Make sure this is not in reinitialise, since this will mess up with the #learn_norms algorithm below
        if(goal == None):
            self.goal=Goal(start_nodes(planlib).pop(), goal_nodes(planlib).pop())
        else:
            self.goal = goal
        
    def reinitialise(self):
        """Reinitialises the norm detector to a state in which no observations 
           have been made. Implementations may have parameters"""
        self.potO = set([("obliged", node) for node in self.all_possible_states(self.planlib)])
        self.potF = set([])
        self.notF = set([])
        
    
    def update_with_observations(self,observation):
        """Updates the norm detector with the sequence of actions in /observations/, this is a single iteration of self.learn_norms"""
        self.past_observations+=observation
        self.pO = set([]) #Actual obligations
        self.pF = set([]) #Atual prohibitions
        for s in observation: # For all state s transitioned through as part of (in this case, the states transitioned in the plan)
            self.pO.add( ("obliged",s) )
            self.notF.add( ("forbidden",s) )
        for pi in self.alternative_plans(observation,observation[0],observation[-1],self.planlib): 
            for s in set(pi).difference(observation): # all states s visited as part of pi and not for in observation
                self.pF.add( ("forbidden",s) )
        
        self.potF = (self.potF | self.pF) - self.notF
        self.potO = self.potO & self.pO
        
    
    def learn_norms(self, runs):
        """Algorithm 1 from http://goo.gl/ZDZu1K - Recomputes norms from scratch"""
        self.reinitialise()
        for observation in runs:
            self.pO = set([]) #Actual obligations
            self.pF = set([]) #Atual prohibitions
            for s in observation: # For all state s transitioned through as part of (in this case, the states transitioned in the plan)
                self.pO.add( ("obliged",s) )
                self.notF.add( ("forbidden",s) )
            for pi in self.alternative_plans(observation,observation[0],observation[-1],self.planlib): 
                for s in set(pi).difference(observation): # all states s visited as part of pi and not for in observation
                    self.pF.add( ("forbidden",s) )
            
            self.potF = (self.potF | self.pF) - self.notF
            self.potO = self.potO & self.pO
            
        return self.potO,self.potF
    
    def get_inferred_norms(self,topNorms=1):
        if(self.translate_norms):
            return self.convert_to_bayesian_norms(self.potO | self.potF)
        else:
            return self.potO | self.potF
    
    

class threshold_norm_detector(common_norm_detector):
    """A Python implementation of Oren and Meneguzzi's (COIN 2013) http://goo.gl/ZDZu1K threshold-based filtering heuristic norm detector"""
    def __init__(self,planlib, goal=None):
        super(threshold_norm_detector,self).__init__(planlib)
        self.planlib = planlib
        self.ot = 0.5 # Threshold for obligations
        self.ft = 0.5 # Threshold for prohibitions
        self.reinitialise()
        self.past_observations = []
        self.goal = goal
        if(goal == None):
            self.goal=Goal(start_nodes(planlib).pop(), goal_nodes(planlib).pop())
        else:
            self.goal = goal
    
    def reinitialise(self):
        self.oc = {("obliged",s):(0,0) for s in self.all_possible_states(self.planlib)}
        self.fc = {("forbidden",s):(0,0) for s in self.all_possible_states(self.planlib)}
        self.potO = set([])
        self.potF = set([])
    
    def update_thresholds(self, ot, ft):
        """Updates thresholds for obligations and prohibitions"""
        self.ot = ot
        self.ft = ft
    
    def update_with_observations(self,observation):
        """An implementation of a single iteration of Algorithm 2 from COIN 2013 paper"""
        self.past_observations+=observation
        self.oc,self.fc = self.update_counter(observation, self.oc, self.fc)
        for (obligation,(oy,on)) in self.oc.iteritems():
            if (on == 0 and oy > 0) or (on != 0 and oy/on > self.ot):
                self.potO = self.potO | set([obligation])
        
        for (prohibition,(fy,fn)) in self.fc.iteritems():
            if(fn == 0 or fy/fn > self.ft):
                self.potF = self.potF | set([prohibition])
                
        for s in self.all_possible_states(self.planlib):
            if( ( ("obliged",s) in self.potO) and ( ("forbidden",s) in self.potF)):
                self.potO.remove(("obliged",s))
                self.potF.remove(("forbidden",s))
        
    
    def t_learn_norms(self, runs):
        """An implementation of Algorithm 2 from COIN 2013 paper"""
        self.reinitialise()
        for observation in runs:
            self.oc,self.fc = self.update_counter(observation, self.oc, self.fc)
            
        for (obligation,(oy,on)) in self.oc.iteritems:
            if (on == 0 and oy > 0) or (on != 0 and oy/on > self.ot):
                self.potO = self.potO | set([obligation])
        
        for (prohibition,(fy,fn)) in self.fc.iteritems:
            if(fn == 0 or fy/fn > self.ft):
                self.potF = self.potF | set([prohibition])
                
        for s in self.all_possible_states(self.planlib):
            if( ( ("obliged",s) in self.potO) and ( ("forbidden",s) in self.potF)):
                self.potO.remove(("obliged",s))
                self.potF.remove(("forbidden",s))
        
        return self.potO,self.potF
    
    def update_counter(self,observation,oc,fc):
        """An implementation of Algorithm 3 from COIN 2013 paper"""
        for s in observation: # For all state s transitioned through as part of (in this case, the states transitioned in the plan)
            (oy,on) = oc[("obliged",s)]
            oc[("obliged",s)] = (oy+1,on)
            (fy,fn) = fc[("forbidden",s)]
            fc[("forbidden",s)] = (fy,fn+1)
        for pi in self.alternative_plans(observation,observation[0],observation[-1],self.planlib):
            for s in set(pi).difference(observation): # all states s visited as part of pi and not for in observation
                (fy,fn) = fc[("forbidden",s)]
                fc[("forbidden",s)] = (fy+1,fn)
        return oc,fc
    
    def get_inferred_norms(self,topNorms=1):
        if(self.translate_norms):
            return self.convert_to_bayesian_norms(self.potO | self.potF)
        else:
            return self.potO | self.potF
    