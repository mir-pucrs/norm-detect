from priorityqueue import PriorityQueue
# from norm_identification2 import *
from norm_identification_logodds import *
import random
from collections import defaultdict
from norm_detector import norm_detector
from planlib import start_nodes,goal_nodes, plan_library_to_graphviz, generate_all_plans
import re

import logging as log
from numpy import sqrt
from __builtin__ import True

class NormBehaviour:
    """An abstract class representing behaviour generation for a particular norm representation"""
    def generate_random_observations(self, norm_detector, scenario, runs, shift_goals=False, violation_signal=True):
        """Generates random observations for a particular norm representation"""
        s_nodes = start_nodes(norm_detector.planlib)
        goals = goal_nodes(norm_detector.planlib)
        goal = norm_detector.get_goal().end
        observations = []
        for i in range(runs):
            node = random.sample(s_nodes,1)[0]
            if(shift_goals):
                goal = None # Need to try goals until we find a compliant one
                while goal is None:
                    if(len(goals)==0): 
                        #log.warning("No goals possible under current norms, skipping initial node")
                        print "No goals possible under current norms, skipping initial node"
                        node = random.sample(s_nodes,1)[0]
                        continue
                    goal = random.sample(goals,1)[0]
                    if not self.goal_is_possibly_compliant(norm_detector, goal, node, scenario.norms):
                        goals.remove(goal) #remove this goal from possible ones to make sure we don't get stuck here
                        goal = None
                      
            
            plan = self.choose_norm_compliant_plan_with_prob(norm_detector, goal, node, scenario.norms, scenario.prob_non_compliance) # TODO Here I should add the probability of detection/sanctioning
                
            if(violation_signal):
                plan = self.annotate_violation_signals(plan,norm_detector, scenario.norms) #TODO Here I should add the probability of random punishment
            observations.append(plan)
        return observations

    def choose_norm_compliant_plan_with_prob(self, norm_detector, goal, node, norms, prob):
        """Chooses a norm compliant plan with probability /prob/ or otherwise choose any plan"""
        (compliant_plans,non_compliant_plans) = self.generate_norm_compliant_plans(norm_detector, goal, node, norms)
        if(random.random() < prob and len(non_compliant_plans) > 0): # Choosing compliant plan
    #         print "Popping in a non-compliant plan" 
            return random.choice(non_compliant_plans)
        else:
            return random.choice(compliant_plans)
    
    def choose_norm_compliant_plan(self, norm_detector, goal, node, norms):
        """Returns a random norm compliant plan out of all possible plans"""
        (compliant_plans,non_compliant_plans) = self.generate_norm_compliant_plans(norm_detector, goal, node, norms)
        if(len(compliant_plans) == 0):
            return None
        else:
            return random.choice(compliant_plans)
    
    def generate_norm_compliant_plans(self, norm_detector, goal, node, norms):
        """Generate all possible compliant (and non-compliant) plans in a plan library"""
        compliant_plans = []
        non_compliant_plans = []
        # print "Generating all plans for goal "+str(node)+" "+str(goal)
        for plan in norm_detector.generate_all_plans(norm_detector.planlib, goal, node, []):
            if(self.is_norm_compliant(plan, norm_detector, norms)):
                compliant_plans.append(plan)
            else:
    #             v = count_violations(plan, norms)
    #             non_compliant_plans.append( (v, plan ) )
                non_compliant_plans.append(plan)
        if(len(compliant_plans) == 0): # No compliant plans
            print "No compliant plans found for goal "+str(node)+" "+str(goal)+" under "+str(norms)+" selecting minimally violating?"
        return (compliant_plans,non_compliant_plans)
        
    def goal_is_possibly_compliant(self, norm_detector, goal, node, norms):
        """Returns whether the supplied goal is possibly compliant in the available plans within the norm_detector"""
        for plan in norm_detector.generate_all_plans(norm_detector.planlib, goal, node, []):
            if(self.is_norm_compliant(plan, norm_detector, norms)): return True
        
        return False
    
    def is_norm_compliant(self, plan, norm_detector, norms):
        """Returns whether or not /plan/ complies with /norms/"""
        (v,indices) = self.count_violations(plan, norms)
        return (v == 0)

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
    
    def annotate_violation_signals(self, plan, norm_detector, norms):
        "Annotates violation signals into a plan"
        (v,indices) = norm_detector.count_violations(plan, norms)
        annotated_plan = []
        idx = 0
        for a in plan:
            annotated_plan.append(a)
            if(idx in indices):
                annotated_plan.append('!')
            idx+=1
        return annotated_plan
    
    def generate_plan(self, norm_detector, node):
        """Randomly generates a plan, completely ignoring norms. This is mainly for testing the norm driven algorithm"""
        plan = [node]
        next_actions = norm_detector.next_actiofns(norm_detector.planlib, node)
        # print "Next actions ", next_actions
        while (next_actions != []):
            a = random.sample(next_actions,1)[0]
            node = a.path[1:]
            plan[len(plan):] = node
            node = node[-1] # if we have a sequence of actions
            next_actions = norm_detector.next_actions(norm_detector.planlib, node)
        return plan
    
    def start_nodes(self, norm_detector):
        """Just forward the call to the function in planlib"""
        return start_nodes(norm_detector.planlib)
    
    def goal_nodes(self, norm_detector):
        """Just forward the call to the function in planlib"""
        return goal_nodes(norm_detector.planlib)
    
    def goal_from_plan(self, plan):
        start = plan[0]
        
        i = 1
        while plan[-i] == '!':
            i+=1
        end = plan[-i]
        return Goal(start,end)
    
    def norms_to_text(self,norms,filename=None):
        "Generates a textual representation of a set of norms, writes the norms into a file if a name is provided"
        ret = ""
        for norm in norms:
            ret +=  str(norm)+"\n"
            
        if(filename is not None):
            f = open(str(filename),'w')
            f.write(ret)
            f.close()
            
        return ret

    def parse_norms(self,filename):
        f = open(filename,'r')
        text = f.read()
        triple_norms = re.findall("\(\W*(\w+)\W*,\W*((?:next|not next|eventually|never))\W*,\W*(\w+)\W*\)",text)
#         print str(triple_norms)
        #for n in triple_norms:
        double_norms = re.findall("\(\W*((?:eventually|never))\W*,\W*(\w+)\W*\)",text)
#         print str(double_norms)
        return set(double_norms + triple_norms)
    
    
    def is_possibly_compliant(self,planlib,norms):
        """Returns whether or not a plan library is possible compliant with a set of norms"""
        s_nodes = start_nodes(planlib)
        nd = norm_detector(planlib)
        for start in s_nodes:
            for plan in generate_all_plans(planlib, start, goal=None):
                if(self.is_norm_compliant(plan, nd, norms)):
                    return True
        
        return False
    
    def gen_random_norms(self,planlib,num_norms=None,norm_file=None):
        """Generates random norms for a given plan library"""
        if(num_norms is None):
            num_norms = sqrt(len(planlib)) # A magic number of of norms in case no number is provided
        
        print "Generating "+str(num_norms)+ " random norms"
        
        modalities = ['never','eventually','next','not next']
        norms = set([])
        
        gvg = plan_library_to_graphviz(planlib)
        nodes = gvg.nodes()
        
        discarded_norms = set([])
        
        norms_to_go = num_norms
        
        while norms_to_go>0:
            log.info(str(num_norms)+" to go")
            modality = random.choice(modalities)
            node1 = None
            if modality in set(['never','eventually']):
                node1 = random.choice(nodes+[None])
            else:
                node1 = random.choice(nodes)
            
            if(node1 is None):
                node2 = random.choice(nodes)
                norm = (modality,str(node2))
            else:
                if(len(gvg.successors(node1)) > 0):
                    node2 = random.choice(gvg.successors(node1))
                    norm = (str(node1),modality,str(node2))
                else: continue
            
            print "Checking norm "+str(norm)+" for viability"
            if(norm not in norms and norm not in discarded_norms): # Check that we are not repeating norms
                if(self.is_possibly_compliant(planlib, norms)):
                    norms.add(norm)
                    norms_to_go-=1
                else:
                    #print "Adding norm "+str(norm)+" to "+str(norms)+ " makes plan library impossible to comply"
                    discarded_norms.add(norm)
                    print "Discarding norm "+str(norm)
            else:
                print "Discarding norm "+str(norm)
            
            if(len(discarded_norms) > num_norms):
                print "Discarded norms more than total number of norms, breaking"
                break
        
        if(norm_file is not None):
            self.norms_to_text(norms, norm_file)
        
        return norms