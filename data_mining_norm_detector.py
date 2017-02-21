from norm_detector import norm_detector
from collections import defaultdict
from subprocess import call
from Queue import PriorityQueue

class data_mining_norm_detector(norm_detector):
    """Implementation of the norm_detector interface to wrap around Tony's Java code"""
    def __init__(self, planlib, goal):
        """ Assigns variables for this class, I'm assuming here that planlib is a set of Actions (from the planlib module)"""
        super(data_mining_norm_detector,self).__init__(planlib)
        self.planlib = planlib
        self.goal = goal
        self.inputFileName = "observations.txt"
        self.oniOutputFileName = "oni_out.txt"
        self.pniOutputFileName = "pni_out.txt"
        self.javaAppClass = "AppToCallFromPython"
     
    def reinitialise(self):
        """Reinitialises the norm detector to a state in which no observations 
           have been made. Implementations may have parameters"""

        with open(self.inputFileName, 'w') as fIn:
            fIn.truncate(0)
        with open(self.oniOutputFileName, 'w') as fOutOni:
            fOutOni.truncate(0)
        with open(self.pniOutputFileName, 'w') as fOutPni:
            fOutPni.truncate(0)        
    
    def update_with_observations(self,observation):
        """Updates the norm detector with the sequence of actions in /observations/"""
        with open(self.inputFileName, 'a') as fIn:
            for action in observation:
                fIn.write(action)
            fIn.write('\n')                    
    
    def set_goal(self, goal):
        """Updates the goal assumed by the norm detector"""
        self.goal = goal
    
    def get_goal(self):
        return self.goal
    
    def get_inferred_norms(self, topNorms=1):
        call(["java", self.javaAppClass])
        pq = PriorityQueue()
        normProbabilities = {}
        with open(self.oniOutputFileName, 'r') as fOutOni:
            for line in fOutOni:
                parts = line.split() # norm as a string of action chars, and then probability
                if len(parts) > 0:
                    norm = ('eventually', parts[0][0]) if len(parts[0]) == 1 else (parts[0][0], 'next', parts[0][1])
                    pq.put((float(parts[1]), norm))
                    normProbabilities[norm] = float(parts[1])
        with open(self.pniOutputFileName, 'r') as fOutPni:
            for line in fOutPni:
                parts = line.split() # norm as a string of action chars, and then probability
                if len(parts) > 0:
                    norm = ('never', parts[0][0]) if len(parts[0]) == 1 else (parts[0][0], 'not next', parts[0][1])
                    pq.put((float(parts[1]), norm))
                    normProbabilities[norm] = float(parts[1])
        sorted_norms = []
        while not pq.empty():
            sorted_norms += [pq.get()[1]]
        norms = [x for x in reversed(sorted_norms)]
        # Check that we select either the topNorms, or the first ones with the same odds
        for (i,n) in enumerate(norms):
            if normProbabilities[n] == 0:
                topNorms = i
                break
            if(i+1 == topNorms):
                tied = len(norms) > i+1 and (normProbabilities[n] == normProbabilities[norms[i+1]])
                if (tied): 
                    topNorms += 1
                else: 
                    break
        #endfor
        return norms[0:topNorms]

    def get_norm_hypotheses(self):
        """Returns a set of the possible norms considered by the detector (mostly for debugging)"""
        nodes = { node for action in self.planlib for node in action.path }
        successors = defaultdict(set)
        for action in self.planlib:
            for i, node in enumerate(action.path[0:-1]):
                successors[node].add(action.path[i+1])
        conditional_norms = [ (context, modality, node) for context in nodes for node in nodes \
                                                       for modality in 'next', 'not next' \
                                                       if node in successors[context] ]
        # Note: the data mining approach does NOT handle "a eventually b" and "a never b"
        
        unconditional_norms = [ (modality, node) for node in nodes for modality in 'eventually', 'never' ]
        hypotheses = unconditional_norms + conditional_norms
        return hypotheses
    
    def separate_norms(self,norms):
        """ Separate /norms/ into each individual type of norm"""
        nxt = set([])
        not_next = set([])
        eventually = set([])
        never = set([])
        ## Get rid of the None norm in the set if there is one
        nnorms = set(norms)
        nnorms.discard(None)
        for (context,modality,node) in nnorms:
            if(modality == 'next'):
                nxt.add((context,modality,node))
            elif(modality == 'not next'):
                not_next.add((context,modality,node))
            elif(modality == 'eventually'):
                eventually.add((context,modality,node))
            elif(modality == 'never'):
                never.add((context,modality,node))
            else:
                print "!! Invalid norm %s !!" % (modality)
        return (nxt,not_next,eventually,never)

    # Copied from norm_identification_logodds.py, with "hypothesis" added as an argument
    # This is here only to check that it and count_violations(...) below implement the same logic
    def violation_indices(self, hypothesis):
        if len(hypothesis) == 2:
            (modality, node) = hypothesis
            if modality == "eventually": # Interpreted as meaning eventually <node> within the plan"
                violation_indices = lambda path: {len(path)-1} if all(item!=node for item in path) else set([])
            elif modality == "never":
                # Assumption: A "never" norm may be sanctioned multiple times on separate breaches (an agent may miss earlier breaches)
                violation_indices = lambda path: {index for index,item in enumerate(path) if item==node}
            else:
                raise ValueError("Invalid modality in hypothesis %s" % hypothesis)
        elif len(hypothesis) == 3:
            (context_node, modality, node) = hypothesis
            if modality == "next":  # Interpreted as only applying if there *is* a next node after the context node
                violation_indices = \
                lambda path: {i+1 for i in range(len(path)-1) if path[i]==context_node and path[i+1]!=node}
            elif modality == "not next":  # Interpreted as only applying if there *is* a next node after the context node
                violation_indices = \
                lambda path: {i+1 for i in range(len(path)-1) if path[i]==context_node and path[i+1]==node}
            elif modality == "eventually":
                # Interpreted as meaning "after the context state (if there is a next state), eventually <node> within the plan".
                # The "after the current state" is for the case when context_node == node
                def pred(path):
                    last_context_index = len(path) # off end of list
                    # Loop code adapted from http://stackoverflow.com/a/9836681
                    for index, item in enumerate(reversed(path[:-1])): # ignore last element of path: context not relevant there
                        if item == context_node:
                            last_context_index = len(path)-index-2
                            break
                    if last_context_index < len(path)-1 and all(item!=node for item in path[last_context_index+1:]):
                        #print "violation_indices = %s" % {len(path)-1}
                        return {len(path)-1}
                    else:
                        #print "violation_indices = %s" % set([])
                        return set([])
                violation_indices = pred
            elif modality == "never": # If we allow context_node == node then this means "next never"
                def pred(path):
                    first_context_index = len(path) # off end of list
                    for index, item in enumerate(path[:-1]): # ignore last element of path: context not relevant there
                        if item == context_node:
                            first_context_index = index
                            break
                        #print "First context index is %s" % first_context_index
                        #print "violation_indices = %s" % {index for index,item in enumerate(path[first_context_index+1:]) if item==node}
                    return {index+first_context_index+1 for index,item in enumerate(path[first_context_index+1:]) if item==node}
                violation_indices = pred
            else:
                raise ValueError("Invalid modality in hypothesis %s" % hypothesis)
        return violation_indices
    
    
    def count_violations(self, plan,norms):
        """Counts the number of violations of /norms/ in /plan/ and returns a list with the indexes of the violations"""
        indices = []
        for norm in norms:
            if(norm is not None):
                violation_indices_fn = self.violation_indices(norm)
                indices+= violation_indices_fn(plan)
        
        indices.sort()

        return (len(indices),indices) # And we return the number of violations
