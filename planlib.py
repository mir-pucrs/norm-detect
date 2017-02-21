# Data structures to represent a plan library
from operator import concat

from pygraphviz import AGraph

import logging as log

class Action():
    def __init__(self, path):
        self.path = path
        self.start = path[0]
        self.end = path[-1]
    
    def __eq__(self, other):
        return isinstance(other, Action) and self.path == other.path
    
    def __hash__(self):
        return hash(tuple(self.path))
    
    def __repr__(self):
        return "Action(%r)" % self.path

class Goal():
    def __init__(self, start, end, plans=None):
        self.start = start
        self.end = end
        self.plans = plans
    
    def setPlans(self, plans):
        self.plans = plans
  
    def __repr__(self):
        return "Goal(%r,%r,%r)" % (self.start, self.end, self.plans)
  
    def __str__(self):
        if not self.plans:
            return "Goal(%s,%s)" % (self.start, self.end)
        else:
            return "Goal(%s,%s,plans=or(%s))" % (self.start, self.end, self.plans)
  
def planned(goal, actions, visited_nodes=set([])):
    "sets 'plans' property and returns true (if successful) or false. Doesn't do loop checking"
    act_list = filter(lambda p: p.start == goal.start and p.end == goal.end, \
                      actions)
    compound_plans = filter(lambda goal_list: \
                            planned(goal_list[1], \
                                    actions - set([goal_list[0].plans[0]]), \
                                    visited_nodes.union({a.end})), \
                          [ [Goal(a.start, a.end, [a]), Goal(a.end, goal.end)] \
                            for a in actions \
                            if a.start == goal.start and a.end != goal.end and a.end not in visited_nodes])
    plans = act_list + compound_plans
    if len(plans) > 0:
        goal.setPlans(plans)
        return True
    else:
        return False

def start_nodes(planlib):
    start_nodes = set([])
    for al in planlib:
        start = True
        for ar in planlib:
            if(al.path[0] == ar.path[-1]):
                start = False
                break
        if(start):
            start_nodes.add(al.path[0])
    return start_nodes
    
def goal_nodes(planlib):
    end_nodes = set([])
    for al in planlib:
        end = True
        for ar in planlib:
            if(al.path[-1] == ar.path[0]):
                end = False
                break
        if(end):
            end_nodes.add(al.path[-1])
    return end_nodes

def flattened_plan_tree(goal_or_plan, prefix=[]):
    if isinstance(goal_or_plan, Goal):
        plan_list = goal_or_plan.plans
        return reduce(concat, (flattened_plan_tree(plan, prefix) for plan in plan_list), [])
    elif isinstance(goal_or_plan, Action):
        return [prefix + [goal_or_plan]]
    elif isinstance(goal_or_plan, list): # A plan (list of goals)
        if len(goal_or_plan) == 1:
            return flattened_plan_tree(goal_or_plan[0], prefix)
        elif len(goal_or_plan) == 2:
            # First goal must be satisfied by an atomic plan, and so results in a singleton flattened plan list
            [ new_prefix ] = flattened_plan_tree(goal_or_plan[0], prefix)
            return flattened_plan_tree(goal_or_plan[1], new_prefix)
        else:
            raise ValueError("flattened_plan_tree called with incorrect plan length (not 1 or 2): %s" % goal_or_plan)
    else:
        raise TypeError("flattened_plan_tree called with argument of wrong type: %s" % goal_or_plan)

def generate_all_plans(planlib, node, goal = None, prefix=[]):
    """Generates all plans for a plan library - changed from norm_detector, with reordered parameters, TODO double check!!"""
    if(prefix == []):
        prefix = [node]
    if(node == goal):
        yield prefix
    else:
        end_node = True
        for a in next_actions(planlib,node):
            end_node = False
            plan = prefix[:]
            plan = plan + a.path[1:]
            subnode = a.path[-1]
            for p in generate_all_plans(planlib, subnode, goal, plan):
                yield p
        if(end_node and goal is None): # If this is an end node
            yield prefix

def next_actions(planlib,node):
    """Returns a list of next actions given a current node"""
    next_actions =[]
    for a in planlib:
        if(a.path[0] == node):
            next_actions[len(next_actions):] = [a]
        
    return next_actions

def dot_to_plan_library(filename):
    G=AGraph()
    G.read(filename)
    planlib = set([])
    for edge in G.edges():
        a = Action(list(edge))
        planlib.add(a)
    
    log.info("Loaded graph: "+str(planlib))
    return planlib

def plan_library_to_graphviz(planlibrary):
    """Converts a plan library into a graphviz data structure"""
    G = AGraph()
    
    for a in planlibrary:
        G.add_edge(a.path[0], a.path[1])
    
    return G

def plan_library_to_dot(planlibrary,filename="graph.dot"):
    f = open(filename,'w')
    f.write('digraph {\n')
    for a in planlibrary:
        for i in range(len(a.path)-1):
            f.write(a.path[i]+' -> '+a.path[i+1]+'\n')
    f.write("}\n")
    f.close()
