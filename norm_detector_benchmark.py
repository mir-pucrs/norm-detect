import random
import copy
import os
import subprocess

import logging as log

from norm_detector import norm_detector
from bayesian_norm_detector import bayesian_norm_detector
from hierarchical_bayesian_norm_detector import hierarchical_bayesian_norm_detector
from oren_meneguzzi_norm_detector import threshold_norm_detector
from data_mining_norm_detector import data_mining_norm_detector
from norm_detector_test import NormDetectorTest
from planlib import Goal, Action, dot_to_plan_library, start_nodes, goal_nodes

# from scipy import stats
import numpy as np

# My own timer functions
from stats import start_timer,end_timer
from optparse import OptionParser
# import threading
from multiprocessing import Process
from norm_behaviour import NormBehaviour
# from planlib import *
# import math

class Scenario():
    """A class representing a scenario, scenarios consist of norms, plan libraries and parameters for the behaviour of agents"""
    def __init__(self,norms,planlibrary, goal, prob_non_compliance=0.01, prob_viol_detection=0.99, prob_sanctioning=0.99, prob_random_punishment=0.01):
        self.norms = norms
        self.planlibrary = planlibrary
        self.goal = goal
        self.prob_non_compliance = prob_non_compliance
        self.prob_viol_detection = prob_viol_detection
        self.prob_sanctioning = prob_sanctioning
        self.prob_random_punishment = prob_random_punishment
    

class NormDetectorBenchmark():
    
    genPlot = False # whether or not to write the gnuplot script to plot graphs
    writeTrace = True # whether or not we should write the generated observations to disk
    writeTables= True # whether or not we should write the resulting tables to disk
    repeats = 1
    runs = 50
    shift_goals = True
    violation_signal = True
    output_folder = "./ijcai/"
    nb = NormBehaviour()
    
    def __init__(self, runs = None, repeats = None):
        if runs != None: self.runs = runs
        if repeats != None: self.repeats = repeats

    # First add the scenarios
    def gen_scenario_1(self, prob_non_compliance=0.01, prob_viol_detection=0.99, \
                    prob_sanctioning=0.99, prob_random_punishment=0.01):
        """Scenarios return plan libraries and sets of norms"""
        goal = Goal('a','d')
        planlib = set([Action(['a','b']), Action(['b','e']), Action(['b','c']), Action(['b','d']), Action(['a','f']), Action(['a','c','e']), Action(['e','d'])])
        norms = set( [ ('a','never','e') ] )
        scenario = Scenario(norms, planlib, goal, prob_non_compliance, prob_viol_detection, \
                    prob_sanctioning, prob_random_punishment)
        
        return scenario
    
    def gen_scenario_1_more_norms(self, prob_non_compliance=0.01, prob_viol_detection=0.99, \
                    prob_sanctioning=0.99, prob_random_punishment=0.01):
        """The same as scenario 1, but with more norms"""
        scenario = self.gen_scenario_1(prob_non_compliance, prob_viol_detection, prob_sanctioning, prob_random_punishment)
        scenario.norms.add(('a','not next','c'))
        return scenario
        
    # Add more scenarios
    
    def gen_scenario_2(self, prob_non_compliance=0.01, prob_viol_detection=0.99, \
                    prob_sanctioning=0.99, prob_random_punishment=0.01):
        """Larger but acyclic graph, allows multiple goals"""
        goal = Goal("a","y")
        planlib = set([Action(["a","0"]), Action(["0","y"]), Action(["0","j"]), Action(["a","w"]), 
                       Action(["a","l"]), Action(["a","e"]), Action(["a","s"]), Action(["a","d"]), 
                       Action(["a","o"]), Action(["b","q"]), Action(["c","z"]), Action(["c","f"]), 
                       Action(["c","n"]), Action(["c","g"]), Action(["d","h"]), Action(["d","r"]), 
                       Action(["d","t"]), Action(["d","z"]), Action(["e","s"]), Action(["e","b"]), 
                       Action(["f","i"]), Action(["f","2"]), Action(["f","u"]), Action(["g","k"]), 
                       Action(["h","1"]), Action(["h","v"]), Action(["j","t"]), Action(["j","3"]), 
                       Action(["k","x"]), Action(["l","s"]), Action(["m","y"]), Action(["n","p"]), 
                       Action(["r","1"]), Action(["s","m"]), Action(["o","c"]), Action(["q","y"])])
        norms = set( [ ('a','never','l') ] )
        scenario = Scenario(norms, planlib, goal, prob_non_compliance, prob_viol_detection, \
                    prob_sanctioning, prob_random_punishment)
        return scenario
    
    def gen_scenario_2_many_norms(self, prob_non_compliance=0.01, prob_viol_detection=0.99, \
                    prob_sanctioning=0.99, prob_random_punishment=0.01):
        scenario = self.gen_scenario_2(prob_non_compliance, prob_viol_detection, prob_sanctioning, prob_random_punishment)
        scenario.norms.add(('e','next','b'))
        scenario.norms.add(('f','next','u'))
        scenario.norms.add(('f','not next','2'))
        scenario.norms.add(('o','never','p'))
        scenario.norms.add(('d','eventually','1'))
        scenario.norms.add(('0','not next','y'))
        return scenario
    
    def gen_scenario_large(self, prob_non_compliance=0.01, prob_viol_detection=0.99, \
                    prob_sanctioning=0.99, prob_random_punishment=0.01):
        planlib = dot_to_plan_library('large-planlib.dot')
        goal = Goal(list(start_nodes(planlib))[0],list(goal_nodes(planlib))[0])
        norms = self.nb.parse_norms('large-norms.txt')
        scenario = Scenario(norms, planlib, goal, prob_non_compliance, prob_viol_detection, \
                    prob_sanctioning, prob_random_punishment)
        return scenario
    
    suffix = None
    def gen_save_filename(self,prefix,scenario,suffix=None):
        filename = self.output_folder+"/"+prefix+"-"\
            +str(len(scenario.planlibrary))\
            +"a"+str(len(scenario.norms))\
            +"n"+("-vs" if self.violation_signal else "")\
            +("-sg" if self.shift_goals else "")\
            +(str(scenario.prob_non_compliance)+"-nc" if scenario.prob_non_compliance!=0.01 else "")\
            +(str(scenario.prob_viol_detection)+"-vd" if scenario.prob_viol_detection!=0.99 else "")\
            +(str(scenario.prob_sanctioning)+"-san" if scenario.prob_sanctioning!=0.99 else "")\
            +(str(scenario.prob_random_punishment)+"-rp" if scenario.prob_random_punishment!=0.01 else "")\
            +("-"+suffix if suffix is not None else "")\
            +".txt"
#         print "Writing table to: "+file
        return filename
    
    def write_observations_to_file(self, norms, observations, filename, norm_detector = None):
        """Writes the generated observations as well as the norms used to generate them to file"""
        log.info("Writing observations to "+filename)
        f = open(str(filename),'w')
        for norm in norms:
            f.write(str(norm)+"\n")
        f.write("\n")
        for plan in observations:
            f.write(str(plan))
            
            if(norm_detector is not None):
                f.write(" "+type(norm_detector).__name__)
                inferred = self.nb.is_norm_compliant(plan, norm_detector, norm_detector.get_inferred_norms(len(norms)+self.top_norms))
                reality = self.nb.is_norm_compliant(plan, norm_detector, norms)
                if(inferred):
                    f.write(" inferred: compliant")
                else:
                    f.write(" inferred: violating")
                
                if(reality):
                    f.write(" truth: compliant")
                else:
                    f.write(" truth: violating")
                
                if(inferred and reality): f.write(" TP")
                if(not inferred and not reality): f.write(" TN")
                if(inferred and not reality): f.write(" FP")
                if(not inferred and reality): f.write(" FN")
            f.write("\n")
        f.close()
        
    def gen_gnuplot(self, filename, xlabel=None, ylabel=None,title=None, curves=None):
        """Generates a Gnuplot file to plot a datafile, 
            curves is a list of tuples (i,n) indicating the column index i and curve name n for each curve
            If curves is empty, we assume there is only one column in the data file and print it""" 
        f = open("%s%s.plot" % (self.output_folder,str(filename)),'w')
        f.write("#!/usr/local/bin/gnuplot\n")
        f.write("set term pdf enhanced\n")
        f.write("set output \"%s.pdf\"\n" % (self.output_folder,str(filename)))
        f.write("set key under\n")
        if(title != None):
            f.write("set title \"\"\n" % str(title))
        if(xlabel != None):
            f.write("set xlabel \"\"\n" % str(xlabel))
        if(ylabel != None):
            f.write("set ylabel \"\"\n" % str(ylabel))
        if(curves == None):
            f.write("plot %s with linesp \n" % filename)
        else:
            f.write("plot")
            for (i,n) in curves:
                f.write(" \"%s\" using 1:%s title \"%s\" with linesp,\\\n" % (filename,str(i),str(n)))
                
            f.write("\n")
        f.close()
    
    def replot_all(self):
        for fn in os.listdir(self.output_folder):
            if(fn.endswith(".plot")):
                fn_graph = self.output_folder+fn.replace(".plot",".pdf")
                if(not os.path.exists(fn_graph) or (os.path.getctime(fn_graph) < os.path.getctime(self.output_folder+fn)) ):
                    print "Plotting "+self.output_folder+fn
                    if (subprocess.call(["/usr/local/bin/gnuplot",self.output_folder+fn])==0):
                        print "Plot complete"
                else:
                    print "Skipping "+fn+", graph not updated"
    
    #Variables for selecting the most likely norms
    top_norms = 10
    def compute_detected_norms(self,norm_detector,scenario):
        """Computes the detected norms (checks if they match the supplied norms) and generate accuracy measures (precision and recall)"""
        prob_norms = norm_detector.get_inferred_norms(len(scenario.norms)+self.top_norms)
        totalDectected = len(prob_norms)
        detected = len(set(scenario.norms) & set(prob_norms))
        recall = (detected*100.0)/len(scenario.norms)
        precision = (detected*100.0)/totalDectected
        return (totalDectected,precision,recall)
    
    # Variables for sampling compliant behaviour
    accuracy_samples = 10 # Number of norms from detected ones that will be sampled -- high numbers may result in no possible plans
    plan_samples = 20 # Number of plans that will be sampled for compliance
    
    def sample_compliant_behaviour(self,norm_detector,scenario):
        """Computes precision and recall of compliant behaviour, this can replace compute_detected_norms to create a different measure of accuracy"""
        prob_norms = norm_detector.get_inferred_norms(len(scenario.norms)+self.top_norms)
        norm_samples = min(self.accuracy_samples,len(prob_norms))
        
        totalDectected = len(prob_norms)
        detected = len(scenario.norms & set(prob_norms))
        recall = (detected*100.0)/len(scenario.norms)
        
        #sample_norms = random.sample(prob_norms,norm_samples)
        real_norms = scenario.norms
        
        sample_scenario = copy.deepcopy(scenario)
        # sample_scenario.norms = sample_norms
        sample_scenario.norms = set([])
        
        try:
            observations = self.nb.generate_random_observations(norm_detector, sample_scenario, self.plan_samples, shift_goals=True, violation_signal=False)
            # if observations == []: print "No compliant plans possible"
            if observations == []: print "No plans possible"
            assert(observations != [])
            
        except ValueError:
            print "No compliant plans possible"
            observations = []
            
        tp = 0 # True positives
        tn = 0 # True negatives
        fp = 0 # False positives
        fn = 0 # False negatives
        for plan in observations:
            inferred = self.nb.is_norm_compliant(plan,norm_detector,prob_norms)
            reality = self.nb.is_norm_compliant(plan,norm_detector,real_norms)
            if(inferred == True and  reality == True):
                tp += 1
            elif(inferred == False and  reality == False):
                tn += 1
            elif(inferred == True and  reality == False):
                fp += 1
            elif(inferred == False and  reality == True):
                fn += 1
        
        precision = (tp*100.0)/(tp + fp) if (tp+fp) != 0 else 0
        recall = (tp*100.0)/(tp+fn) if (tp+fn) != 0 else 0
        
        return (totalDectected,precision,recall)
    
    def experiment_inferred_norms_runs(self,scenario,norm_detector):
        """An experiment that measures the number of norms inferred as the number of runs increases"""
        print "Running Inferred Norms over Runs (different precision "\
            +str(self.plan_samples)+" samples), "\
            +str(len(scenario.norms))+" norms, "\
            +str(len(scenario.planlibrary))+" actions"\
            +(", shifting goals" if self.shift_goals else "")\
            +(", violation signal" if self.violation_signal else "")\
            +(", "+str(scenario.prob_non_compliance*100)+"% non-compliance")\
            +(", "+str(scenario.prob_viol_detection*100)+"% violation detection")\
            +(", "+str(scenario.prob_sanctioning*100)+"% sanctioning")\
            +(", "+str(scenario.prob_random_punishment*100)+"% random punishment")
        print str(self.runs)+" runs, "+str(self.repeats)+" repetitions"
        assert(isinstance(scenario, Scenario))
        
        t_detected_norms = np.zeros((self.runs,self.repeats))
        t_precision = np.zeros((self.runs,self.repeats))
        t_recall = np.zeros((self.runs,self.repeats))
        
        if(self.writeTrace) : observation_cache = []
        
        for r in range(self.repeats):
            norm_detector.reinitialise()
            observations = self.nb.generate_random_observations(norm_detector, scenario, self.runs, self.shift_goals, self.violation_signal) #TODO : I need to make a class out of norm_behaviour to conduct this test
            if(self.writeTrace) : observation_cache += observations
            
            for i in range(self.runs):
                plan = observations[i]
                if self.shift_goals: norm_detector.set_goal(self.nb.goal_from_plan(plan)) #Another method in norm_behaviour
                norm_detector.update_with_observations(plan)
                #(detected,precision,recall) = self.compute_detected_norms(norm_detector, scenario)
                (detected,precision,recall) = self.sample_compliant_behaviour(norm_detector, scenario)
                t_detected_norms[i][r] = detected
                t_precision[i][r] = precision
                t_recall[i][r] = recall
                
            
        
        # Generate numbers for table
        mean_detected_norms = t_detected_norms.mean(axis=1)
        std_detected_norms = t_detected_norms.std(axis=1)
        
        mean_precision = t_precision.mean(axis=1)
        std_precision = t_precision.std(axis=1)
        
        mean_recall = t_recall.mean(axis=1)
        std_recall = t_recall.std(axis=1)
        
        table = np.array([[x + 1 for x in range(self.runs)],
                            mean_detected_norms,
                            std_detected_norms,
                            mean_precision,
                            std_precision,
                            mean_recall,
                            std_recall]).T
        
#         print table
        if(self.writeTrace): self.write_observations_to_file(scenario.norms, observation_cache, self.gen_save_filename("norms-runs", scenario,"obs"), norm_detector)
        
        if(self.writeTables):
            np.savetxt(self.gen_save_filename("norms-runs", scenario, self.suffix), table, fmt='%d %.4f %.4f %.4f %.4f %.4f %.4f', delimiter=" ", newline="\n", header="% Run, Mean Detected Norms, Std Dev Mean Detected Norms, Mean Precision, Std Dev Precision, Mean Recall, Std Dev Recall", footer="", comments="")
        return table
    
    # TODO Debug this
    def experiment_precision_recall_over_norms(self, scenario, norm_detector):
        """An experiment that measures the number of norms inferred as the number of *norms* being monitored increases"""
        print "Running Precision and Recall over #Norms (precision "+str(self.accuracy_samples)+" samples), "+str(len(scenario.norms))+" norms, "+str(len(scenario.planlibrary))+" actions"+(", shifting goals" if self.shift_goals else "")+(", violation signal" if self.violation_signal else "")
        print str(self.runs)+" runs, "+str(self.repeats)+" repetitions"
        
        assert(isinstance(scenario, Scenario))
        t_detected_norms = np.zeros((len(scenario.norms),self.repeats))
        t_precision = np.zeros((len(scenario.norms),self.repeats))
        t_recall = np.zeros((len(scenario.norms),self.repeats))
        
        for r in range(self.repeats):
            for ni in range(len(scenario.norms)):
                norm_detector.reinitialise()
                sample_norms = set(random.sample(scenario.norms,ni+1))
                sample_scenario = copy.deepcopy(scenario)
                sample_scenario.norms = sample_norms
                observations = self.nb.generate_random_observations(norm_detector, sample_scenario, self.runs, shift_goals=self.shift_goals, violation_signal=self.violation_signal)
                if(self.writeTrace): self.write_observations_to_file(sample_norms, observations, self.gen_save_filename("prec-norms", scenario,"obs"))
                for i in range(self.runs): # Here we use the maximum number of runs to be able to determine the best norm inference
                    plan = observations[i]
                    if self.shift_goals: norm_detector.set_goal(self.nb.goal_from_plan(plan))
                    norm_detector.update_with_observations(plan)
                #(detected,precision,recall) = self.compute_detected_norms(norm_detector, scenario)
                (detected, precision, recall) = self.sample_compliant_behaviour(norm_detector, scenario)
                t_detected_norms[ni][r] = detected
                t_precision[ni][r] = precision
                t_recall[ni][r] = recall
        
        # Generate numbers for table
        mean_detected_norms = t_detected_norms.mean(axis=1)
        std_detected_norms = t_detected_norms.std(axis=1)
        
        mean_precision = t_precision.mean(axis=1)
        std_precision = t_precision.std(axis=1)
        
        mean_recall = t_recall.mean(axis=1)
        std_recall = t_recall.std(axis=1)
        
        table = np.array([[x + 1 for x in range(len(scenario.norms))],
                            mean_detected_norms,
                            std_detected_norms,
                            mean_precision,
                            std_precision,
                            mean_recall,
                            std_recall]).T
        
        if(self.writeTables):
            np.savetxt(self.gen_save_filename("prec-norms", scenario, self.suffix), table, fmt='%d %.4f %.4f %.4f %.4f %.4f %.4f', delimiter=" ", newline="\n", header="% #Norms, Mean Detected Norms, Std Dev Mean Detected Norms, Mean Precision, Std Dev Precision, Mean Recall, Std Dev Recall", footer="", comments="")

        return table
    

def numpy_stats_demo():
    # http://docs.scipy.org/doc/scipy-0.16.1/reference/tutorial/stats.html
    # http://docs.scipy.org/doc/scipy-0.16.1/reference/tutorial/basic.html
    # http://docs.scipy.org/doc/numpy/reference/arrays.nditer.html
    a = np.ones((5,3))
    j = 1
    for x in np.nditer(a,op_flags=['readwrite']):
        x[...] = j
        j+=1
    print "Array: \n" + str(a)
    m = a.mean(axis=0) # this should get the mean along the columns of the matrix
    print "Mean along columns: "+str(m)
    m = a.mean(axis=1) # this should get the mean along the rows of the matrix
    print "Mean along rows: "+str(m)
    v = a.var(axis=0)
    print "Variance along columns: "+str(v)
    v = a.var(axis=1)
    print "Variance along rows: "+str(v)
    s = a.std(axis=0)
    print "Standard deviation along columns: "+str(s)
    s = a.std(axis=1)
    print "Standard deviation rows: "+str(s)
    
    t = np.array([[x+1 for x in range (5)], m, v, s]).T
    print t
    np.savetxt("test.txt", t, fmt="%.2f")

# Uncomment this if you want to run regular tests
# if __name__ == '__main__':
#     unittest.main()

def do_nothing(runs,repeats):
    pass 

def trivial_experiments(runs,repeats):
    benchmark = NormDetectorBenchmark(50,100)
    scenario = benchmark.gen_scenario_1()
    norm_detector = bayesian_norm_detector(scenario.planlibrary,scenario.goal, scenario.prob_non_compliance, scenario.prob_viol_detection, scenario.prob_sanctioning, scenario.prob_random_punishment)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector)
    
    scenario = benchmark.gen_scenario_1_more_norms()
    norm_detector = bayesian_norm_detector(scenario.planlibrary,scenario.goal, scenario.prob_non_compliance, scenario.prob_viol_detection, scenario.prob_sanctioning, scenario.prob_random_punishment)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector)

def trivial_comparative_experiments(runs,repeats):
    benchmark = NormDetectorBenchmark(50, 100)
    scenario = benchmark.gen_scenario_1()
    
    norm_detector2 = threshold_norm_detector(scenario.planlibrary)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector2)

def medium_experiments_norms_runs(runs,repeats):
    benchmark = NormDetectorBenchmark(runs,repeats)
    
    scenario = benchmark.gen_scenario_2_many_norms()
    norm_detector = bayesian_norm_detector(scenario.planlibrary,scenario.goal, scenario.prob_non_compliance, scenario.prob_viol_detection, scenario.prob_sanctioning, scenario.prob_random_punishment)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector)
    
    benchmark = NormDetectorBenchmark(runs,repeats)
    benchmark.violation_signal = False
    scenario = benchmark.gen_scenario_2_many_norms()
    norm_detector = bayesian_norm_detector(scenario.planlibrary,scenario.goal, scenario.prob_non_compliance, scenario.prob_viol_detection, scenario.prob_sanctioning, scenario.prob_random_punishment)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector)
    
def medium_experiments_norms_runs_violations(runs,repeats):
    benchmark = NormDetectorBenchmark(runs,repeats)
    
    # scenario = benchmark.gen_scenario_2_many_norms(prob_non_compliance=.30)
#     norm_detector = bayesian_norm_detector(scenario.planlibrary,scenario.goal, scenario.prob_non_compliance, scenario.prob_viol_detection, scenario.prob_sanctioning, scenario.prob_random_punishment)
#     benchmark.experiment_inferred_norms_runs(scenario, norm_detector)
    
    # scenario = benchmark.gen_scenario_2_many_norms(prob_non_compliance=.60)
#     norm_detector = bayesian_norm_detector(scenario.planlibrary,scenario.goal, scenario.prob_non_compliance, scenario.prob_viol_detection, scenario.prob_sanctioning, scenario.prob_random_punishment)
#     benchmark.experiment_inferred_norms_runs(scenario, norm_detector)
#
#     scenario = benchmark.gen_scenario_2_many_norms(prob_non_compliance=1)
#     norm_detector = bayesian_norm_detector(scenario.planlibrary,scenario.goal, scenario.prob_non_compliance, scenario.prob_viol_detection, scenario.prob_sanctioning, scenario.prob_random_punishment)
#     benchmark.experiment_inferred_norms_runs(scenario, norm_detector)
    
    benchmark.violation_signal = True

    scenario = benchmark.gen_scenario_2_many_norms(prob_non_compliance=.30)
    initial_guess_prob_non_compliance = 0.5
    norm_detector = hierarchical_bayesian_norm_detector(scenario.planlibrary, scenario.goal, initial_guess_prob_non_compliance, scenario.prob_viol_detection, scenario.prob_sanctioning, scenario.prob_random_punishment)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector)
    
#    scenario = benchmark.gen_scenario_2_many_norms(prob_non_compliance=.30)
#    norm_detector = bayesian_norm_detector(scenario.planlibrary,scenario.goal, scenario.prob_non_compliance, scenario.prob_viol_detection, scenario.prob_sanctioning, scenario.prob_random_punishment)
#    benchmark.experiment_inferred_norms_runs(scenario, norm_detector)
    
    # scenario = benchmark.gen_scenario_2_many_norms(prob_non_compliance=.60)
#     norm_detector = bayesian_norm_detector(scenario.planlibrary,scenario.goal, scenario.prob_non_compliance, scenario.prob_viol_detection, scenario.prob_sanctioning, scenario.prob_random_punishment)
#     benchmark.experiment_inferred_norms_runs(scenario, norm_detector)
#
#     scenario = benchmark.gen_scenario_2_many_norms(prob_non_compliance=1)
#     norm_detector = bayesian_norm_detector(scenario.planlibrary,scenario.goal, scenario.prob_non_compliance, scenario.prob_viol_detection, scenario.prob_sanctioning, scenario.prob_random_punishment)
#     benchmark.experiment_inferred_norms_runs(scenario, norm_detector)

def medium_experiments_precision_recall_over_norms(runs, repeats):
    benchmark = NormDetectorBenchmark(runs,repeats)
    scenario = benchmark.gen_scenario_2_many_norms()
    norm_detector = bayesian_norm_detector(scenario.planlibrary,scenario.goal, scenario.prob_non_compliance, scenario.prob_viol_detection, scenario.prob_sanctioning, scenario.prob_random_punishment)
    benchmark.experiment_precision_recall_over_norms(scenario, norm_detector)
    
    benchmark = NormDetectorBenchmark(runs,repeats)
    benchmark.violation_signal = False
    scenario = benchmark.gen_scenario_2_many_norms()
    norm_detector = bayesian_norm_detector(scenario.planlibrary,scenario.goal, scenario.prob_non_compliance, scenario.prob_viol_detection, scenario.prob_sanctioning, scenario.prob_random_punishment)
    benchmark.experiment_precision_recall_over_norms(scenario, norm_detector)

def large_experiments_norms_runs_violations(runs,repeats):
    benchmark = NormDetectorBenchmark(runs,repeats)
    scenario = benchmark.gen_scenario_large()
    norm_detector = bayesian_norm_detector(scenario.planlibrary,scenario.goal, scenario.prob_non_compliance, scenario.prob_viol_detection, scenario.prob_sanctioning, scenario.prob_random_punishment)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector)
    
def comparative_experiments(runs,repeats):
    benchmark = NormDetectorBenchmark(runs,repeats)
    scenario = benchmark.gen_scenario_2_many_norms()
    
    benchmark.suffix = "threshold_detector"
    norm_detector2 = threshold_norm_detector(scenario.planlibrary)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector2)
    
    benchmark.suffix = "datamining_detector"
    norm_detector3 = data_mining_norm_detector(scenario.planlibrary,scenario.goal)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector3)

def large_comparative_experiments(runs,repeats):
    benchmark = NormDetectorBenchmark(runs,repeats)
    scenario = benchmark.gen_scenario_large()
    
    benchmark.suffix = "threshold_detector"
    norm_detector2 = threshold_norm_detector(scenario.planlibrary)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector2)
    
    benchmark.suffix = "datamining_detector"
    norm_detector3 = data_mining_norm_detector(scenario.planlibrary,scenario.goal)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector3)
    
def comparative_experiments_violations(runs,repeats):
    benchmark = NormDetectorBenchmark(runs,repeats)
    
    benchmark.suffix = "threshold_detector"
    
    scenario = benchmark.gen_scenario_2_many_norms(prob_non_compliance=.30)
    norm_detector2 = threshold_norm_detector(scenario.planlibrary)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector2)
     
    scenario = benchmark.gen_scenario_2_many_norms(prob_non_compliance=.60)
    norm_detector2 = threshold_norm_detector(scenario.planlibrary)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector2)
    
    scenario = benchmark.gen_scenario_2_many_norms(prob_non_compliance=1)
    norm_detector2 = threshold_norm_detector(scenario.planlibrary)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector2)
    
    benchmark.suffix = "datamining_detector"
    scenario = benchmark.gen_scenario_2_many_norms(prob_non_compliance=.30)
    norm_detector3 = data_mining_norm_detector(scenario.planlibrary,scenario.goal)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector3)
    
    scenario = benchmark.gen_scenario_2_many_norms(prob_non_compliance=.60)
    norm_detector3 = data_mining_norm_detector(scenario.planlibrary,scenario.goal)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector3)
    
    scenario = benchmark.gen_scenario_2_many_norms(prob_non_compliance=1)
    norm_detector3 = data_mining_norm_detector(scenario.planlibrary,scenario.goal)
    benchmark.experiment_inferred_norms_runs(scenario, norm_detector3)

if __name__ == '__main__':
#     numpy_stats_demo()
#     exit()
    parser = OptionParser()
    parser.add_option("-d", "--dir-output", dest="output", action="store", type="string",
                  help="write reports to DIR", metavar="DIR")
    parser.add_option("-q", "--quiet",
                  action="store_false", dest="quiet", default=False,
                  help="don't print status messages to stdout")
    parser.add_option("-v", "--verbose",
                  action="store_false", dest="verbose", default=False,
                  help="print extra status messages to stdout, overrides quiet")
    parser.add_option("-c","--cores", dest="cores", action="store", type="int",
                help="Create CORES separate number of processes", metavar="CORES")
    parser.add_option("-r","--repeats", dest="repeats", action="store", type="int",
                help="Repeat experiments REPEATS number of times", metavar="REPEATS")
    parser.add_option("-o","--observations", dest="observations", action="store", type="int",
                help="For each experiment, generate OBS number of observations", metavar="OBS")
    parser.add_option("-p","--plot", dest="replot", action="store_true", default=True,
                help="Replot all graphs once experiments are over")

    (options, args) = parser.parse_args()
    
    runs = 100
    repeats = 20
    
    if(options.quiet):
        log.info("Suppressing most output")
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.CRITICAL)
    
    if(options.verbose):
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.INFO)
        log.info("Verbose output.")
    elif(not options.quiet):
        log.basicConfig(format="%(levelname)s: %(message)s")
    
    if(options.repeats != None):
        repeats = options.repeats
    if(options.observations != None):
        runs = options.observations
    
    timer = start_timer()
    
    experiment_calls = [
                        # trivial_experiments,
                        #medium_experiments_norms_runs,
                        medium_experiments_norms_runs_violations,
                        #medium_experiments_precision_recall_over_norms,
                        #large_experiments_norms_runs_violations,
                        #comparative_experiments,
                        #comparative_experiments_violations,
                        #large_comparative_experiments,
                        # experiment_multiple_priors_regular,
                        # experiment_multiple_priors_equal_prior,
                        # experiment_multiple_priors_over_prior,
                        do_nothing
                        ]
    
    if(options.cores == None): 
        print "Running experiments in a single core"
    
        for e in experiment_calls:
            e(runs,repeats)
    else:
        print "Running experiments in "+str(options.cores)+" cores"
        processes = []
        cores_left = options.cores
        for experiment in experiment_calls:
            if(cores_left > 0):
                p = Process(target=experiment,args=(runs,repeats))
                p.start()
                processes.append(p)
                cores_left -= 1
            else:
                "No more cores left, linearising the remaining experiments"
                experiment(runs,repeats)
        
        print "Waiting for processes to finish"
        for p in processes:
            p.join()
        
        #
    timer = end_timer(timer)

    
    print(str(timer)+"s testing")
    
    #if(options.replot):
        #replot_all() # TODO redo this

