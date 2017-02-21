import unittest
import os
from norm_behaviour import NormBehaviour

class NormBehaviourTest(unittest.TestCase):
    def setUp(self):
        self.nb = NormBehaviour()
        norms = set([])
        for norm in ['eventually','never']:
            norms.add( ('a',norm,'b'))
            norms.add( (norm,'c'))
        for norm in ['next','not next']:
            norms.add( ('a',norm,'b'))
            
        self.nb.norms_to_text(norms,"__norms__.txt")
    
    def tearDown(self):
        os.remove("__norms__.txt")
        pass
    
    def test_parse_norms(self):
        norms = self.nb.parse_norms("__norms__.txt")
        self.assertIsNotNone(norms, "Failed to parse norms")

if __name__ == '__main__':
    unittest.main()