import random
import unittest
from norm_identification import *
from compiler.ast import Node
from rospkg.environment import on_ros_path
from mercurial.templater import if_
from __builtin__ import str

class TestNormSuite(unittest.TestCase):
    
    def setUp(self):
        goal = Goal('a','d')
        actions = set([Action(['a','b']), Action(['b','e']), Action(['b','c']), Action(['b','d']), Action(['a','f']), Action(['a','c','e']), Action(['e','d'])])
        observation = ['a','c','e','d']

        nodes = { node for action in actions for node in action.path }
        poss_norms = [norm for node in nodes for norm in ("forbidden", node), ("obliged", node)]
        
        hypotheses = dict.fromkeys(poss_norms, 0.1)
        hypotheses[None] = 1 # Set prior odds ration for hypothesis None
        
#         hypotheses = dict.fromkeys(poss_norms, 1.0/(len(nodes)+1))
#         hypotheses[None] = 1.0/(len(nodes)+1) # Set prior odds ration for hypothesis None
        
        suite = NormSuite(goal, hypotheses, actions)
        
        norms = {}
        self.executions = 10 # Number of behaviour executions
        
    def test_basic(self):
        print "Goal: ", goal
        print "Actions: ", actions
        print "Norm hypotheses (with prior odds ratios): ", hypotheses

        print "Updating odds rations after observing ", observation
        suite.UpdateOddsRatioVsNoNorm(observation)

        print "The posterior odds ratios are:"
        suite.Print()
        
        all_plans = []
        for plan in self.generate_all_plans(suite, goal.end, 'a', []):
            print plan
            all_plans.append(plan)
        
        assert(len(all_plans)> 0)
#         print(all_plans)
        
        print("Basic test success")

    
    def test_generate_norm_compliant_plan(self):
        plan = self.generate_all_plans(suite, 'd', 'a', []).next()
        planC = self.generate_norm_compliant_plans(suite,'d','a',[])[0]
        self.assertEqual(plan,planC)
         
        for i in range(10):
            planC = self.choose_norm_compliant_plan(suite,'d','a',[('forbidden','b')])
            self.assertEquals(planC.count('b'), 0)
            
        for i in range(10):
            planC = self.choose_norm_compliant_plan(suite,'d','a',[('obliged','b')])
            self.assertEquals(planC.count('b'), 1)
        
        self.assertEqual(self.generate_norm_compliant_plans(suite,'d','a',[('forbidden','a')]),[])
        # print "XXXX: ",planC
        
        print "Test test_generate_norm_compliant_plan Passed"
        
    def test_norm_driven_behaviour_one_obligation(self):
        print "******************************************************"
        start_nodes = self.start_nodes(suite)
        norms = [('obliged','e')]
        
        print "Norms are: "+str(norms)
        print "Prob "+str(norms[0])+ str(suite.d.get(norms[0]))
        print "Generated plans:"
        for i in range(10):
            node = random.sample(start_nodes,1)[0]
            plan = self.choose_norm_compliant_plan(suite, 'd', node, norms)
            print "\t "+str(plan)
#             goal = Goal(node,plan[-1])
            suite.UpdateOddsRatioVsNoNorm(plan)
            print "\t\t Updated prob: "+str(norms[0])+ str(suite.d.get(norms[0]))
        
        suite.Print()
        
        print "Test test_norm_driven_behaviour_one_obligation Passed"
        print "******************************************************"
        
    def test_norm_driven_behaviour_one_prohibition(self):
        print "******************************************************"
        start_nodes = self.start_nodes(suite)
        norms = [('forbidden','e')]
        print "Norms are: "+str(norms)
        print "Prob "+str(norms[0])+ str(suite.d.get(norms[0]))
        print "Generated plans:"
        for i in range(10):
            node = random.sample(start_nodes,1)[0]
            plan = self.choose_norm_compliant_plan(suite, 'd', node, norms)
            print "\t "+str(plan)
#             goal = Goal(node,plan[-1])
            suite.UpdateOddsRatioVsNoNorm(plan)
            print "\t\t Updated prob: "+str(norms[0])+ str(suite.d.get(norms[0]))
        
        suite.Print()
            # print "Plan:", plan
        print "Test test_norm_driven_behaviour_one_prohibition Passed"
        print "******************************************************"
    
    def choose_norm_compliant_plan(self, suite, goal, node, norms):
        return random.choice(self.generate_norm_compliant_plans(suite, goal, node, norms))
    
    def generate_norm_compliant_plans(self, suite, goal, node, norms):
        """Randomly generate a norm compliant plan for /goal/"""
        compliant_plans = []
        non_compliant_plans = []
        for plan in self.generate_all_plans(suite, goal, node, []):
            if(self.is_norm_compliant(plan, norms)):
                compliant_plans.append(plan)
            else:
                (o,f) = self.count_violations(plan, norms)
                non_compliant_plans.append( (o,f, plan ) )
        if(len(compliant_plans) == 0): # No compliant plans
            print "No compliant plans found for "+str(norms)+" selecting minimally violating?"
            return []
        else:
            return compliant_plans
    
    def separate_norms(self, norms):
        """ Separate /norms/ into /obligations/ and /prohibitions/"""
        obligations = set([])
        prohibitions = set([])
        for (modality,node) in norms:
            if(modality == 'obliged'):
                obligations.add((modality,node))
            elif(modality == 'forbidden'):
                prohibitions.add((modality,node))
        return (obligations,prohibitions)
    
    def is_norm_compliant(self, plan, norms):
        """Returns whether or not /plan/ complies with /norms/"""
        (o,f) = self.count_violations(plan, norms)
        return (o == 0 and f == 0)
            
    def count_violations(self,plan,norms):
        """Counts the number of violations of /norms/ in /plan/"""
        (obl,pro) = self.separate_norms(norms)
        o = f = 0; # number of violations
        for a in plan:
            for(modality,node) in pro:
                if(modality == 'forbidden'):
                    if(a == node): 
                        #if we find any instance of a prohibited norm, 
                        #there is a violation
                        f = f + 1
            to_remove = set([])
            for(modality,node) in obl:
                if(modality == 'obliged'):
                    if(a == node): 
                        #if we find an instance of an obliged norm, 
                        #then this plan no longer needs to comply with it
                        to_remove.add((modality,node))
            
            obl -= to_remove # So we remove the obligations from the list of obligations
        #Once we are through iterating the plan, any remaining obligations are violated
        o = len(obl)
        return (o,f) # And we return the number of violations
    
            
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
    
    def generate_plan(self, suite, node):
        """Randomly generates a plan, completely ignoring norms. This is mainly for testing the norm driven algorithm"""
        plan = [node]
        next_actions = self.next_actions(suite,node)
        # print "Next actions ", next_actions
        while (next_actions != []):
            a = random.sample(next_actions,1)[0]
            node = a.path[1:]
            plan[len(plan):] = node
            node = node[-1] # if we have a sequence of actions
            next_actions = self.next_actions(suite,node)
        return plan

#     def generate_all_plans(self, suite, goal, node, prefix, plans):
#         if(prefix == []):
#             prefix = [node]
#         if(node == goal):
#             plans.append(prefix)
#         else:
#             next_actions = self.next_actions(suite,node)
#             for a in next_actions:
#                 plan = prefix[:]
#                 plan = plan + a.path[1:]
#                 subnode = a.path[-1]
#                 self.generate_all_plans(suite, goal, subnode, plan, plans)
                    
    def generate_all_plans(self, suite, goal, node, prefix):
        if(prefix == []):
            prefix = [node]
        if(node == goal):
            yield prefix
        else:
            next_actions = self.next_actions(suite,node)
            for a in next_actions:
                plan = prefix[:]
                plan = plan + a.path[1:]
                subnode = a.path[-1]
                for p in self.generate_all_plans(suite, goal, subnode, plan):
                    yield p
        
    
    def next_actions(self, suite, node):
        """Returns a list of next actions given a current node"""
        next_actions =[]
        for a in suite.actions:
            if(a.path[0] == node):
                next_actions[len(next_actions):] = [a]
                
        return next_actions

    def start_nodes(self, suite):
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

    def goal_nodes(self, suite):
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

    def test_start_nodes(self):
        self.assertEqual(self.start_nodes(suite),set(['a']))

    def print_plan_library(self, suite):
        f = open('graph.dot','w')
        f.write('digraph {\n')
        for a in suite.actions:
            for i in range(len(a.path)-1):
                f.write(a.path[i]+' -> '+a.path[i+1]+'\n')
        f.write("}\n")
        f.close()

if __name__ == '__main__':
    unittest.main()