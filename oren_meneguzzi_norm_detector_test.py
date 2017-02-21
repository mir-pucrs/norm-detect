import unittest
from oren_meneguzzi_norm_detector import basic_norm_detector,threshold_norm_detector
from norm_detector_test import NormDetectorTest


class BasicNormDetectorTest(NormDetectorTest):
    @classmethod
    def setUpClass(cls):
        print "******* Testing basic_norm_detector *******"
    
    def setUp(self):
        super(BasicNormDetectorTest,self).setUp()
        self.norm_detector = basic_norm_detector(self.planlib)
        self.norm_detector.translate_norms = False
        
    def test_update_with_observations(self):
        self.norm_detector.update_with_observations(self.observation1)
        self.assertIsNotNone(self.norm_detector.potF, "No potential prohibitions")
        self.assertIsNotNone(self.norm_detector.potO, "No potential obligations")
    
    def test_get_inferred_norms(self):
        print "Norm hypotheses: "+str(self.norm_detector.get_norm_hypotheses())
        self.norm_detector.update_with_observations(self.observation1)
        self.assertIsNotNone(self.norm_detector.get_inferred_norms(), "Inferred norms should not be none")
        self.assertIn((("obliged",'a')), self.norm_detector.get_inferred_norms())
        self.assertIn((("forbidden",'b')), self.norm_detector.get_inferred_norms())
        print "Past observations "+str(self.norm_detector.past_observations)
        print "Inferred norms: " + str(self.norm_detector.get_inferred_norms())
        
        self.norm_detector.update_with_observations(self.observation2)
        self.assertNotIn((("forbidden",'b')), self.norm_detector.get_inferred_norms())
        self.assertNotIn((("obliged",'e')), self.norm_detector.get_inferred_norms())
        print "Past observations "+str(self.norm_detector.past_observations)
        print "Inferred norms: " + str(self.norm_detector.get_inferred_norms())
        
        self.norm_detector.update_with_observations(self.observation3)
        self.assertIn((("obliged",'a')), self.norm_detector.get_inferred_norms())
        self.assertNotIn((("forbidden",'b')), self.norm_detector.get_inferred_norms())
        self.assertNotIn((("obliged",'e')), self.norm_detector.get_inferred_norms())
        print "Past observations "+str(self.norm_detector.past_observations)
        print "Inferred norms: " + str(self.norm_detector.get_inferred_norms())
    
class ThresholdNormDetectorTest(BasicNormDetectorTest):
    @classmethod
    def setUpClass(cls):
        print "******* Testing threshold_norm_detector *******"
    
    def setUp(self):
        super(ThresholdNormDetectorTest,self).setUp()
        self.norm_detector = threshold_norm_detector(self.planlib)
        self.norm_detector.translate_norms = False

if __name__ == '__main__':
    unittest.main()