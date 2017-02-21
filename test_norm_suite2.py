import random
import unittest
from norm_behaviour_old import *
from priorityqueue import PriorityQueue
from norm_identification2 import *
from __builtin__ import str

class TestNormSuite2(unittest.TestCase):
    
    def setUp(self):
        self.goal = Goal('a','d')
        self.actions = set([Action(['a','b']), Action(['b','e']), Action(['b','c']), Action(['b','d']), Action(['a','f']), Action(['a','c','e']), Action(['e','d'])])
        self.observation1 = ['a','c','e','d']
        self.observation2 = ['a','b','d','!']
        nodes = { node for action in self.actions for node in action.path }
        self.conditional_norms = [ (context, modality, node) for context in nodes for node in nodes \
                                                        for modality in 'next', 'not next', 'eventually', 'never' ]
        self.unconditional_norms = [ (modality, node) for node in nodes for modality in 'eventually', 'never' ]
        self.poss_norms = self.unconditional_norms + self.conditional_norms
        
        self.hypotheses = dict.fromkeys(self.poss_norms, 0.05)
        self.hypotheses[None] = 1 # Set prior odds ration for hypothesis None
        
        self.suite = NormSuite(self.goal, self.hypotheses, self.actions)
        
        self.norms = {}
        self.executions = 10 # Number of behaviour executions
        
    def test_basic(self):
        print "Goal: ", self.goal
        print "Actions: ", self.actions
        print "Norm hypotheses (with prior odds ratios): ", self.hypotheses

        print "Updating odds ratios after observing ", self.observation1
        self.suite.UpdateOddsRatioVsNoNorm(self.observation1)
        
#         print "The posterior odds ratios are:"
#         self.suite.Print()
        
        print "Updating odds ratios after observing ", self.observation2
        self.suite.UpdateOddsRatioVsNoNorm(self.observation2)
        
#         print "The posterior odds ratios are:"
#         self.suite.Print()
        
        all_plans = []
        for plan in generate_all_plans(self.suite, self.goal.end, 'a', []):
            print plan
            all_plans.append(plan)
        
        assert(len(all_plans)> 0)
#         print(all_plans)
        
        print("Basic test success")

    
    def test_generate_norm_compliant_plan(self):
        plan = generate_all_plans(self.suite, 'd', 'a', []).next()
        planC = generate_norm_compliant_plans(self.suite,'d','a',set([]) )[0]
        self.assertEqual(plan,planC)
         
        for i in range(self.executions):
            planC = choose_norm_compliant_plan(self.suite,'d','a',set([(True,'never','b')]) )
            self.assertEquals(planC.count('b'), 0)
            
        for i in range(self.executions):
            planC = choose_norm_compliant_plan(self.suite,'d','a',set([(True,'eventually','b')]) )
            self.assertEquals(planC.count('b'), 1)
        
        self.assertEqual(generate_norm_compliant_plans(self.suite,'d','a',set([(True,'never','a')]) ),[])
        # print "XXXX: ",planC
        
        print "Test test_generate_norm_compliant_plan Passed"
        
    def test_norm_driven_behaviour_next(self):
        print "******************************************************"
        print "Testing next "
        s_nodes = start_nodes(self.suite)
        norms = set( [('a','next','c')] )
        # First, the norm compliant cases
        plan = ['a','c','e','d']
        assert(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],0)
        
        #Then non-compliant ones
        plan = ['a','b','e','d']
        self.assertFalse(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],1)
        
        plan = ['a','b','c','d']
        self.assertFalse(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],1)
        
        
        print "Norms are: "+str(norms)
        print "Prob "+str(norms.__iter__().next())+ str(self.suite.d.get(norms.__iter__().next()))
        print "Generated plans:"
        for i in range(self.executions):
            node = random.sample(s_nodes,1)[0]
            plan = choose_norm_compliant_plan(self.suite, 'd', node, norms)
            print "\t "+str(plan)
#             goal = Goal(node,plan[-1])
            self.suite.UpdateOddsRatioVsNoNorm(plan)
            print "\t\t Updated prob: "+str(norms.__iter__().next())+ str(self.suite.d.get(norms.__iter__().next()))
        
        self.suite.Print()
        
        print "Test test_norm_driven_behaviour_next Passed"
        print "******************************************************"
    
    def test_norm_driven_behaviour_not_next(self):
        print "******************************************************"
        print "Testing not next "
        s_nodes = start_nodes(self.suite)
        norms = set( [('a','not next','c')] )
        # First, the norm compliant cases
        plan = ['a','b','e','d']
        assert(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],0)
        
        plan = ['a','b','c','d']
        self.assertTrue(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],0)
        
        #Then non-compliant cases
        plan = ['a','c','e','d']
        self.assertFalse(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],1)
        
        
        print "Norms are: "+str(norms)
        print "Prob "+str(norms.__iter__().next())+ str(self.suite.d.get(norms.__iter__().next()))
        print "Generated plans:"
        for i in range(self.executions):
            node = random.sample(s_nodes,1)[0]
            plan = choose_norm_compliant_plan(self.suite, 'd', node, norms)
            print "\t "+str(plan)
#             goal = Goal(node,plan[-1])
            self.suite.UpdateOddsRatioVsNoNorm(plan)
            print "\t\t Updated prob: "+str(norms.__iter__().next())+ str(self.suite.d.get(norms.__iter__().next()))
        
        self.suite.Print()
        
        print "Test test_norm_driven_behaviour_not_next Passed"
        print "******************************************************"
    
    def test_norm_driven_behaviour_one_never(self):
        print "******************************************************"
        print "Testing never "
        s_nodes = start_nodes(self.suite)
        
        norms = set([('a','never','e')])
        
        # First, the norm compliant cases
        plan = ['e','a','b','d']
        assert(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],0)
        
        #Then non-compliant ones
        plan = ['a','b','e','d']
        self.assertFalse(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],1)
        
        plan = ['a','b','e','e']
        self.assertFalse(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],1)
        
        
        norms = set([(True,'never','e')])
        
        # First, the norm compliant cases
        plan = ['a','b','d']
        assert(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],0)
        
        #Then non-compliant ones
        plan = ['a','b','e','d']
        self.assertFalse(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],1)
        
        plan = ['a','b','c','e']
        self.assertFalse(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],1)
        
        plan = ['a','b','e','e']
        self.assertFalse(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],2)
        
        norms = set([('a','never','e')])
        print "Norms are: "+str(norms)
        print "Prob "+str(norms.__iter__().next())+ str(self.suite.d.get(norms.__iter__().next()))
        print "Generated plans:"
        for i in range(self.executions):
            node = random.sample(s_nodes,1)[0]
            plan = choose_norm_compliant_plan(self.suite, 'd', node, norms)
            print "\t "+str(plan)
#             goal = Goal(node,plan[-1])
            self.suite.UpdateOddsRatioVsNoNorm(plan)
            print "\t\t Updated prob: "+str(norms.__iter__().next())+ str(self.suite.d.get(norms.__iter__().next()))
        
        self.suite.Print()
            # print "Plan:", plan
        print "Test test_norm_driven_behaviour_one_never Passed"
        print "******************************************************"
    
    def test_norm_driven_behaviour_one_eventually(self):
        print "******************************************************"
        print "Testing eventually "
        s_nodes = start_nodes(self.suite)
        
        norms = set([('a','eventually','e')])
        
        # First, the norm compliant cases
        plan = ['a','b','e','d']
        assert(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],0)
        
        #Then non-compliant ones
        plan = ['a','b','c','d']
        self.assertFalse(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],1)
        
        plan = ['e','a','b','d']
        self.assertFalse(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],1)
        
        
        norms = set([(True,'eventually','e')])
        
        # First, the norm compliant cases
        plan = ['a','b','e','d']
        assert(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],0)
        
        plan = ['a','b','e','e']
        self.assertTrue(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],0)
        
        #Then non-compliant ones
        plan = ['a','b','c','d']
        self.assertFalse(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],1)
        
        plan = ['a','b','c']
        self.assertFalse(is_norm_compliant(plan, norms))
        self.assertEqual(count_violations(plan, norms)[0],1)
    
        norms = set([('a','eventually','e')])
        print "Norms are: "+str(norms)
        print "Prob "+str(norms.__iter__().next())+ str(self.suite.d.get(norms.__iter__().next()))
        print "Generated plans:"
        for i in range(self.executions):
            node = random.sample(s_nodes,1)[0]
            plan = choose_norm_compliant_plan(self.suite, 'd', node, norms)
            print "\t "+str(plan)
#             goal = Goal(node,plan[-1])
            self.suite.UpdateOddsRatioVsNoNorm(plan)
            print "\t\t Updated prob: "+str(norms.__iter__().next())+ str(self.suite.d.get(norms.__iter__().next()))
    
        self.suite.Print()
    
        print "Test test_norm_driven_behaviour_one_obligation Passed"
        print "******************************************************"

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
#             goal = Goal(node,plan[-1])
            self.suite.UpdateOddsRatioVsNoNorm(plan)
        prob_norms,topN = self.suite.most_probable_norms(topN)
        self.assertEquals(len(prob_norms),topN)
        self.suite.Print()
        print "Most probable norms: "+str([(n,self.suite.d[n]) for n in prob_norms])
        self.assertIn(('a','never','e'), prob_norms)



if __name__ == '__main__':
    unittest.main()