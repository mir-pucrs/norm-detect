import unittest

from norm_detector import norm_detector
from norm_detector_benchmark import NormDetectorBenchmark, Scenario

class PerfectNormDetector(norm_detector):
    def __init__(self, scenario):
        super(PerfectNormDetector,self).__init__(scenario.planlibrary)
        assert(isinstance(scenario, Scenario))
        self.goal=scenario.goal
        self.scenario = scenario
    
    def reinitialise(self):
        pass
    
    def update_with_observations(self, observation):
        pass
    
    def set_goal(self, goal):
        self.goal =  goal
    def get_goal(self):
        return self.goal
    
    def get_inferred_norms(self, topNorms=1):
        if(topNorms == len(self.scenario.norms)): return self.scenario.norms
        else:
            norms = set([])
            true_norms = list(self.scenario.norms)
            for i in range(min(topNorms,len(true_norms))):
                norms.add(true_norms[i])
            return norms
    
    def get_norm_hypotheses(self):
        return self.scenario.norms
    
    def count_violations(self, plan, norms):
        return (0,0)
    
class HalfPerfectNormDetector(PerfectNormDetector):
    
    def get_inferred_norms(self, topNorms=1):
        norms = set([])
        true_norms = list(self.scenario.norms)
        num_norms =  len(self.scenario.norms)
        for i in range(num_norms):
            if(i < num_norms/2.0):
                norms.add(true_norms[i])
            else:
                norms.add( (None,None,None) )
        return norms

class NormDetectorBenchmarkTest(unittest.TestCase):
    
    def setUp(self):
        self.benchmark = NormDetectorBenchmark()
        self.benchmark.runs = 10
        self.benchmark.repeats = 10
        self.benchmark.writeTables = False
    
    def test_experiment_inferred_norms_runs(self):
        scenario = self.benchmark.gen_scenario_1()
        norm_detector = PerfectNormDetector(scenario)
        table = self.benchmark.experiment_inferred_norms_runs(scenario, norm_detector)
        self.assertEqual(table.shape[0],self.benchmark.runs)
        self.assertEqual(table.shape[1],7)
        for i in range(self.benchmark.runs):
            self.assertEqual(table[i][0],i+1)
            self.assertEqual(table[i][1],len(scenario.norms)) # Mean inferred norms (this should be perfect)
            self.assertEqual(table[i][2],0) # Std Dev. inferred norms (this should be 0)
            self.assertEqual(table[i][3],100) # Precision (this should be 100)
            self.assertEqual(table[i][4],0) # Std. Dev of Precision (this should be 0)
            self.assertEqual(table[i][5],100) # Recall (this should be 100)
            self.assertEqual(table[i][6],0) # Std. Dev of Recall (this should be 0)
        
        scenario = self.benchmark.gen_scenario_1_more_norms()
        norm_detector = HalfPerfectNormDetector(scenario)
        table = self.benchmark.experiment_inferred_norms_runs(scenario, norm_detector)
        self.assertEqual(table.shape[0],self.benchmark.runs)
        self.assertEqual(table.shape[1],7)
        for i in range(self.benchmark.runs):
            self.assertEqual(table[i][0],i+1)
            self.assertEqual(table[i][1],len(scenario.norms)) # Mean inferred norms (this should be perfect)
            self.assertEqual(table[i][2],0) # Std Dev. inferred norms (this should be 0)
            self.assertEqual(table[i][3],50) # Precision (this should be 50)
            self.assertEqual(table[i][4],0) # Std. Dev of Precision (this should be 0)
            self.assertEqual(table[i][5],50) # Recall (this should be 50)
            self.assertEqual(table[i][6],0) # Std. Dev of Recall (this should be 0)
        
    
    def test_experiment_precision_recall_over_norms(self):
        scenario = self.benchmark.gen_scenario_1()
        norm_detector = PerfectNormDetector(scenario)
        table = self.benchmark.experiment_precision_recall_over_norms(scenario, norm_detector)
        self.assertEqual(table.shape[0],len(scenario.norms))
        self.assertEqual(table.shape[1],7)
        self.assertEqual(table[0][0],1)
        self.assertEqual(table[0][1],len(scenario.norms)) # Mean inferred norms (this should be perfect)
        self.assertEqual(table[0][2],0) # Std Dev. inferred norms (this should be 0)
        self.assertEqual(table[0][3],100) # Precision (this should be 100)
        self.assertEqual(table[0][4],0) # Std. Dev of Precision (this should be 0)
        self.assertEqual(table[0][5],100) # Recall (this should be 100)
        self.assertEqual(table[0][6],0) # Std. Dev of Recall (this should be 0)
        
        scenario = self.benchmark.gen_scenario_1_more_norms()
        norm_detector = HalfPerfectNormDetector(scenario)
        table = self.benchmark.experiment_precision_recall_over_norms(scenario, norm_detector)
        self.assertEqual(table.shape[0],len(scenario.norms))
        self.assertEqual(table.shape[1],7)
        
        self.assertEqual(table.shape[0],len(scenario.norms))
        self.assertEqual(table.shape[1],7)
        self.assertEqual(table[0][0],1)
        self.assertEqual(table[0][1],len(scenario.norms)) # Mean inferred norms (this should be perfect)
        self.assertEqual(table[0][2],0) # Std Dev. inferred norms (this should be 0)
        self.assertEqual(table[0][3],50) # Precision (this should be 100)
        self.assertEqual(table[0][4],0) # Std. Dev of Precision (this should be 0)
        self.assertEqual(table[0][5],50) # Recall (this should be 100)
        self.assertEqual(table[0][6],0) # Std. Dev of Recall (this should be 0)
    
    def test_reading_large_scenario(self):
        scenario = self.benchmark.gen_scenario_large()
        assert(list(scenario.planlibrary)[0].path[0] is not 'u')
        
if __name__ == '__main__':
    unittest.main()