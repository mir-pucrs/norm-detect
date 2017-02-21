import unittest
from bayesian_norm_detector import bayesian_norm_detector
from norm_detector_test import NormDetectorTest
from planlib import Goal


class BayesianNormDetectorTest(NormDetectorTest):
    @classmethod
    def setUpClass(cls):
        print "******* Testing basic_norm_detector *******"
    
    def setUp(self):
        super(BayesianNormDetectorTest,self).setUp()
        self.normdetector = bayesian_norm_detector(self.planlib,goal = Goal('a','d'))
        
    def test_update_with_observations(self):
        self.normdetector.update_with_observations(self.observation1)
        self.assertIsNotNone(self.normdetector.get_inferred_norms(1), "No potential norms")
    
    def test_get_inferred_norms(self):
        print "Norm hypotheses: "+str(self.normdetector.get_norm_hypotheses())
        
        self.normdetector.update_with_observations(self.observation1)
        self.assertIsNotNone(self.normdetector.get_inferred_norms(), "Inferred norms should not be none")
        self.assertIn((('eventually','a')), self.normdetector.get_inferred_norms(10))
        self.assertIn((('never','b')), self.normdetector.get_inferred_norms())
        print "Past observations "+str(self.normdetector.past_observations)
        print "Inferred norms: " + str(self.normdetector.get_inferred_norms())
        
        self.normdetector.update_with_observations(self.observation2)
        self.assertNotIn((("never",'b')), self.normdetector.get_inferred_norms())
        self.assertNotIn((("eventually",'e')), self.normdetector.get_inferred_norms())
        print "Past observations "+str(self.normdetector.past_observations)
        print "Inferred norms: " + str(self.normdetector.get_inferred_norms())
        
        self.normdetector.update_with_observations(self.observation3)
        self.assertIn((("eventually",'a')), self.normdetector.get_inferred_norms())
        self.assertNotIn((("never",'b')), self.normdetector.get_inferred_norms())
        self.assertNotIn((("eventually",'e')), self.normdetector.get_inferred_norms())
        print "Past observations "+str(self.normdetector.past_observations)
        print "Inferred norms: " + str(self.normdetector.get_inferred_norms())

if __name__ == '__main__':
    unittest.main()