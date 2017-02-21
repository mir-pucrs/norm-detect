import random
import unittest
import copy
from norm_behaviour_old import *
from aamas_experiments import sum_entry,average_entries
from priorityqueue import PriorityQueue
from norm_identification2 import *
from __builtin__ import str

class TestNormBehaviour(unittest.TestCase):
    
    def setUp(self):
        self.goal = Goal('a','d')
        self.actions = set([Action(['a','b']), Action(['b','e']), Action(['b','c']), Action(['b','d']), Action(['a','f']), Action(['a','c','e']), Action(['e','d'])])
        self.observation1 = ['a','c','e','d']
        self.observation2 = ['a','b','d','!']
        
        self.suite = build_norm_suite(self.goal,self.actions)
        
        self.executions = 10 # Number of behaviour executions
        
    def test_generate_norm_compliant_plan(self):
        plan = generate_all_plans(self.suite, 'd', 'a', []).next()
        planC = generate_norm_compliant_plans(self.suite,'d','a',set([]) )[0][0]
        self.assertEqual(plan,planC)
         
        for i in range(self.executions):
            planC = choose_norm_compliant_plan(self.suite,'d','a',set([(True,'never','b')]) )
            self.assertEqual(planC.count('b'), 0)
            
        for i in range(self.executions):
            planC = choose_norm_compliant_plan(self.suite,'d','a',set([(True,'eventually','b')]) )
            self.assertEqual(planC.count('b'), 1)
        
        self.assertNotEqual(generate_norm_compliant_plans(self.suite,'d','a',set([(True,'never','a')]) ),[])
        # print "XXXX: ",planC
        
        print "Test test_generate_norm_compliant_plan Passed"
    
    def test_start_nodes(self):
        self.assertEqual(start_nodes(self.suite),set(['a']))

    def test_most_probable_norms(self):
        topN = 10
        print "******************************************************"
        print "Testing most_prob_norms - top "+str(topN)+" norms"
        s_nodes = start_nodes(self.suite)
        norms = set([ ('a','never','e') ])
        print "Norms are: "+str(norms)
        print "Prob "+str(norms.__iter__().next())+":"+str(self.suite.d.get(norms.__iter__().next()))
        print "Generated plans:"
        for i in range(self.executions):
            node = random.sample(s_nodes,1)[0]
            plan = choose_norm_compliant_plan(self.suite,'d', node, norms)
            print "\t "+str(plan)
            self.suite.UpdateOddsRatioVsNoNorm(plan)
        prob_norms,topN = self.suite.most_probable_norms(topN)
        self.assertEqual(len(prob_norms),topN)
        self.suite.Print()
        print "Most probable norms: "+str([(n,self.suite.d[n]) for n in prob_norms])
        self.assertIn(('a','never','e'), prob_norms)

    def test_annotate_violation_signals(self):
        print "******************************************************"
        print "Testing annotate_violation_signals"
        norms = set([ ('a','never','e') ])
        plan = ['a','b','c','d','e']
        annotated_plan = annotate_violation_signals(plan,norms)
        self.assertIn('!',annotated_plan)
        self.assertEqual('!',annotated_plan[-1])
        
        norms = set([ ('a','never','e') ])
        plan = ['a','b','c','d']
        annotated_plan = annotate_violation_signals(plan,norms)
        self.assertNotIn('!',annotated_plan)
        
        norms = set([ ('a','not next','b') ])
        plan = ['a','b','c','d','e']
        annotated_plan = annotate_violation_signals(plan,norms)
        self.assertIn('!',annotated_plan)
        self.assertEqual('!',annotated_plan[2])
        
        norms = set([ ('a','next','c') ])
        plan = ['a','b','c','d','e']
        annotated_plan = annotate_violation_signals(plan,norms)
        self.assertIn('!',annotated_plan)
        self.assertEqual('!',annotated_plan[2])

        norms = set([ ('a','eventually','f') ])
        plan = ['a','b','c','d','e']
        annotated_plan = annotate_violation_signals(plan,norms)
        self.assertIn('!',annotated_plan)
        self.assertEqual('!',annotated_plan[-1])
        
        norms = set([ ('a','never','l'), ('0','next','j') ])
        plan = ['a', '0', 'y', '!']
        annotated_plan = annotate_violation_signals(plan,norms)
        self.assertIn('!',annotated_plan)
        self.assertEqual('!',annotated_plan[-1])

    def test_sum_entry(self):
        entries = [(0,1,1,2), (1,1,1,2), (2,1,1,2), (3,1,1,2)]
        entries2 = copy.deepcopy(entries)
        entries3 = [(0,3,2,4), (1,1,1,2), (2,1,1,2), (3,1,1,2)]
        entries0 = [(0,0,0,0), (1,0,0,0), (2,0,0,0), (3,0,0,0)]
        
        for i in range(len(entries)):
            sum_entry(entries,entries[i])
        average_entries(entries,2)

        for i in range(len(entries)):
            self.assertEqual(entries[i],entries2[i])

        entries = entries2

        for i in range(len(entries)):
            sum_entry(entries,entries3[i])
        average_entries(entries,2)

        self.assertEqual(entries[0][1],2)
        self.assertEqual(entries[0][2],1.5)
        self.assertEqual(entries[0][3],3)
        
        entries = entries3
        for r in range(10):
            for i in range(len(entries)):
                sum_entry(entries,entries0[i])
        average_entries(entries,10)
        
        self.assertEqual(entries[0][1],0.3)
        self.assertEqual(entries[0][2],0.2)
        self.assertEqual(entries[0][3],0.4)

    def test_goal_from_plan(self):
        plan1 = ['a','b','c','d']
        plan2 = ['a','b','c','d','!']
        self.assertEqual(goal_from_plan(plan1).start,'a')
        self.assertEqual(goal_from_plan(plan1).end,'d')
        self.assertEqual(goal_from_plan(plan2).start,'a')
        self.assertEqual(goal_from_plan(plan2).end,'d')

if __name__ == '__main__':
    unittest.main()