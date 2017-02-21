# This file is kept to make sure that aamas_experiments.py remain working

from priorityqueue import PriorityQueue
# from norm_identification2 import *
from norm_identification_logodds import *
import random
from collections import defaultdict


# This is DEPRECATED
def build_norm_suite(goal, actions, prob_non_compliance=0.01, prob_viol_detection=0.99, \
                    prob_sanctioning=0.99, prob_random_punishment=0.01, prior=log(0.5), prior_none = log(1)):
    nodes = { node for action in actions for node in action.path }
#     conditional_norms = [ (context, modality, node) for context in nodes for node in nodes \
#                                                     for modality in 'next', 'not next', 'eventually', 'never' ]
    successors = defaultdict(set)
    for action in actions:
        for i, node in enumerate(action.path[0:-1]):
            successors[node].add(action.path[i+1])
    conditional_norms = [ (context, modality, node) for context in nodes for node in nodes \
                                                  for modality in 'eventually', 'never' ]
    conditional_norms += [ (context, modality, node) for context in nodes for node in nodes \
                                                   for modality in 'next', 'not next' \
                                                   if node in successors[context] ]
    
    unconditional_norms = [ (modality, node) for node in nodes for modality in 'eventually', 'never' ]
    poss_norms = unconditional_norms + conditional_norms
    
    # hypotheses = dict.fromkeys(poss_norms, 0.05)
    # hypotheses[None] = 1 # Set prior odds ration for hypothesis None
    # hypotheses = dict.fromkeys(poss_norms, log(0.05))
    # hypotheses[None] = log(1) # Set prior odds ration for hypothesis None
    # hypotheses = dict.fromkeys(poss_norms, log(1))
    # hypotheses[None] = log(1) # Set prior odds ration for hypothesis None
    hypotheses = dict.fromkeys(poss_norms, prior)
    hypotheses[None] = prior_none # Set prior odds ration for hypothesis None
    
    suite = NormSuite(goal, hypotheses, actions, prob_non_compliance, prob_viol_detection, prob_sanctioning, prob_random_punishment)
    return suite

# ============================================
# = Suite Reinitialization for repeated runs =
# ============================================
# This is DEPRECATED
def reinitialise_suite(suite):
    prior_none = suite.initial_hypotheses[None]
    prior = None
    for hyp_key in suite.initial_hypotheses.iterkeys():
        if (hyp_key != None):
            prior = suite.initial_hypotheses[hyp_key]
            break
     
    assert(prior != None)
     
    new_suite = build_norm_suite(suite.inferred_goal, suite.actions, suite.prob_non_compliance, suite.prob_viol_detection, suite.prob_sanctioning, suite.prob_random_punishment, prior, prior_none)
    return new_suite

# ====================================
# = Observation Generation Functions =
# ====================================
def generate_random_observations(suite,norms,runs,shift_goals=False,violation_signal=False):
    "Generates random observations for the scenarios"
    s_nodes = start_nodes(suite)
    goals = goal_nodes(suite)
    goal = suite.inferred_goal.end
    generative_norms = convert_norms_to_generative(norms)
    observations = []
    for i in range(runs):
        node = random.sample(s_nodes,1)[0]
        if(shift_goals):
            goal = None # Need to try goals until we find a compliant one
            while goal is None:
                goal = random.sample(goals,1)[0]
                if not goal_is_possibly_compliant(suite, goal, node, generative_norms):
                    goals.remove(goal) #remove this goal from possible ones to make sure we don't get stuck here
                    goal = None
                  
        
        plan = choose_norm_compliant_plan_with_prob(suite, goal, node, generative_norms) # TODO Here I should add the probability of detection/sanctioning
            
        if(violation_signal):
            plan = annotate_violation_signals(plan,generative_norms) #TODO Here I should add the probability of random punishment
        observations.append(plan)
    return observations

def choose_norm_compliant_plan_with_prob(suite, goal, node, norms):
    "Chooses a norm compliant plan with probability /prob/ or otherwise choose any plan"
    (compliant_plans,non_compliant_plans) = generate_norm_compliant_plans(suite, goal, node, norms)
    if(random.random() < suite.prob_non_compliance and len(non_compliant_plans) > 0): # Choosing compliant plan
#         print "Popping in a non-compliant plan" 
        return random.choice(non_compliant_plans)
    else:
        return random.choice(compliant_plans)

def choose_norm_compliant_plan(suite, goal, node, norms):
    (compliant_plans,non_compliant_plans) = generate_norm_compliant_plans(suite, goal, node, norms)
    if(len(compliant_plans) == 0):
        return None
    else:
        return random.choice(compliant_plans)

def generate_norm_compliant_plans(suite, goal, node, norms):
    """Randomly generate a norm compliant plan for /goal/"""
    compliant_plans = []
    non_compliant_plans = []
    # print "Generating all plans for goal "+str(node)+" "+str(goal)
    for plan in generate_all_plans(suite, goal, node, []):
        if(is_norm_compliant(plan, norms)):
            compliant_plans.append(plan)
        else:
#             v = count_violations(plan, norms)
#             non_compliant_plans.append( (v, plan ) )
            non_compliant_plans.append(plan)
    if(len(compliant_plans) == 0): # No compliant plans
        print "No compliant plans found for goal "+str(node)+" "+str(goal)+" under "+str(norms)+" selecting minimally violating?"
    return (compliant_plans,non_compliant_plans)

def goal_is_possibly_compliant(suite, goal, node, norms):
    "Returns whether or not the state goal is possibly compliant"
    for plan in generate_all_plans(suite, goal, node, []):
        if(is_norm_compliant(plan, norms)): return True
    
    return False

def separate_norms(norms):
    """ Separate /norms/ into each individual type of norm"""
    nxt = set([])
    not_next = set([])
    eventually = set([])
    never = set([])
    for (context,modality,node) in norms:
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

def is_norm_compliant(plan, norms):
    """Returns whether or not /plan/ complies with /norms/"""
    (v,indexes) = count_violations(plan, norms)
    return (v == 0)

def annotate_violation_signals(plan, norms):
    "Annotates violation signals into a plan"
    (v,indexes) = count_violations(plan, norms)
    annotated_plan = []
    idx = 0
    for a in plan:
        annotated_plan.append(a)
        if(idx in indexes):
            annotated_plan.append('!')
        idx+=1
    return annotated_plan

def count_violations(plan,norms):
    """Counts the number of violations of /norms/ in /plan/ and returns a list with the indexes of the violations"""
    (nxt,not_next,eventually,never) = separate_norms(norms)
    v = 0; # number of violations
    indexes = []
    idx = 0
    active_norms = compute_active_norms(None, norms, set([]))
    to_remove = set([])
    for a in plan:
        for(context,modality,node) in nxt & active_norms: # things here should be nxt
            assert(modality == 'next')
            if(a != node): 
                v = v+1
                indexes.append(idx)
            to_remove.add((context,modality,node)) # nxt should be removed immediately afterwards
        for(context,modality,node) in not_next & active_norms: # things here should not be nxt
            assert(modality == 'not next')
            if(a == node): 
                v = v+1
                indexes.append(idx)
            to_remove.add((context,modality,node))
        for(context,modality,node) in eventually & active_norms: # things here should eventually happen
            assert(modality == 'eventually')
            if(a == node): 
                if(context is True): #If this is an unconditional norm, then we don't need to care about it
                    eventually.remove((context,modality,node))
                to_remove.add((context,modality,node)) 
        for(context,modality,node) in never & active_norms: # things here should never happen
            assert(modality == 'never')
            if(a == node): 
                v = v+1
                indexes.append(idx)
                if(context is not True): # If this is a conditional norm, when it is violated it is removed
                    to_remove.add((context,modality,node))
        active_norms = compute_active_norms(a, nxt | not_next | eventually | never, active_norms)
        active_norms -= to_remove # So we remove the inactive norms from the list of active_norms
        idx+=1
    #Once we are through iterating the plan, any remaining eventually norms are violated
    for (context,modality,node) in eventually:
        if(context == True): 
            v+=1
            indexes.append(idx-1)
        elif((context,modality,node) in active_norms):
            v+=1
            indexes.append(idx-1)
            
    return (v,indexes) # And we return the number of violations

def compute_active_norms(action, norms, active_norms):
    new_active_norms = set([])
    for (context,mod,node) in norms - active_norms:
        if(context is True or context == action):
            new_active_norms.add((context,mod,node))
    return new_active_norms | active_norms
        
    
#     def generate_norm_compliant_plan(self, suite, node, norms):
#         """Randomly generates a plan, given a set of norms """
#         plan = [node]
#         next_actions = self.next_compliant_actions(suite,node,norms)
#         # print "Next actions ", next_actions
#         while (next_actions != []):
#             a = random.sample(next_actions,1)[0]
#             node = a.path[1:]
#             plan[len(plan):] = node
#             node = node[-1] # if we have a sequence of actions
#             next_actions = self.next_actions(suite,node)
#         return plan

#     def next_compliant_actions(self,suite,node,norms):
#         next_actions =[]
#         for a in self.next_actions(suite,node):
#             compliant = True # We start assuming the action is compliant
#             value = 1 # (one compliance)
#             for (modality,node) in norms:
#                 if(modality == 'obliged'):
#                     if(a.path.count(node) > 0):
#                         value = value + 1
#                         next_actions[len(next_actions):] = [a]
#                     else:
#                         compliant = False
#                         value = value - 1
#                 elif(modality == 'forbidden'):
#                     if(a.path.count(node) > 0):
#                         compliant = False
#                         value = value - 1
#                     else:
#                         compliant = False
#                         value = value + 1
#             if(compliant):
#                 next_actions[len(next_actions):] = [a]
#         if(len(next_actions)==0):
#             next_actions[len(next_actions):] = [random.choice(self.next_actions(suite,node))]
#         return next_actions

def generate_plan(suite, node):
    """Randomly generates a plan, completely ignoring norms. This is mainly for testing the norm driven algorithm"""
    plan = [node]
    next_actions = next_actions(suite,node)
    # print "Next actions ", next_actions
    while (next_actions != []):
        a = random.sample(next_actions,1)[0]
        node = a.path[1:]
        plan[len(plan):] = node
        node = node[-1] # if we have a sequence of actions
        next_actions = next_actions(suite,node)
    return plan

#     def generate_all_plans(suite, goal, node, prefix, plans):
#         if(prefix == []):
#             prefix = [node]
#         if(node == goal):
#             plans.append(prefix)
#         else:
#             n_actions = next_actions(suite,node)
#             for a in n_actions:
#                 plan = prefix[:]
#                 plan = plan + a.path[1:]
#                 subnode = a.path[-1]
#                 generate_all_plans(suite, goal, subnode, plan, plans)

# Deprecated into individual classes
def generate_all_plans(suite, goal, node, prefix):
    if(prefix == []):
        prefix = [node]
    if(node == goal):
        yield prefix
    else:
        for a in next_actions(suite,node):
            plan = prefix[:]
            plan = plan + a.path[1:]
            subnode = a.path[-1]
            for p in generate_all_plans(suite, goal, subnode, plan):
                yield p
    
def convert_norms_to_generative(norms):
    """Converts /norms/ into the internal format used by the behaviour generation algorithms """
    conv_norms = []
    for n in norms:
        if n is None: continue
        if(len(n) == 2):
            (modality,node) = n
            conv_norms.append((True,modality,node))
        else:
            conv_norms.append(n)
    return set(conv_norms)

# Deprecated into individual classes
def next_actions(suite, node):
    """Returns a list of next actions given a current node"""
    next_actions =[]
    for a in suite.actions:
        if(a.path[0] == node):
            next_actions[len(next_actions):] = [a]
            
    return next_actions

def start_nodes(suite):
    start_nodes = set([])
    for al in suite.actions:
        start = True
        for ar in suite.actions:
            if(al.path[0] == ar.path[-1]):
                start = False
                break
        if(start):
            start_nodes.add(al.path[0])
    return start_nodes

def goal_nodes(suite):
    end_nodes = set([])
    for al in suite.actions:
        end = True
        for ar in suite.actions:
            if(al.path[-1] == ar.path[0]):
                end = False
                break
        if(end):
            end_nodes.add(al.path[-1])
    return end_nodes

def goal_from_plan(plan):
    start = plan[0]
    
    i = 1
    while plan[-i] == '!':
        i+=1
    end = plan[-i]
    return Goal(start,end)


def print_plan_library(suite,filename="graph.dot"):
    f = open(filename,'w')
    f.write('digraph {\n')
    for a in suite.actions:
        for i in range(len(a.path)-1):
            f.write(a.path[i]+' -> '+a.path[i+1]+'\n')
    f.write("}\n")
    f.close()