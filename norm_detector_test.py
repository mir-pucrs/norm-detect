import unittest
from norm_detector import norm_detector
from planlib import Action, Goal

class NormDetectorTest(unittest.TestCase):
    
    def setUp(self):
        self.goal = Goal('a','d')
        self.planlib = set([Action(['a','b']), Action(['b','e']), Action(['b','c']), Action(['b','d']), Action(['a','f']), Action(['a','c','e']), Action(['e','d'])])
        self.observation1 = ['a','c','e','d']
        self.observation2 = ['a','b','d']
        self.observation3 = ['a','b','e','d']
        self.norm_detector = norm_detector(self.planlib)
        
    def test_get_inferred_norms(self):
        pass
    
    def test_alternative_plans(self):
        Pi = self.norm_detector.alternative_plans(self.observation1,'a','d',self.planlib)
        print Pi
    
    def test_generate_all_plans(self):
        pass
    
    def test_next_actions(self):
        pass
    
if __name__ == '__main__':
    unittest.main()