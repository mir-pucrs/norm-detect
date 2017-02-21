#
#  journal_experiments.py
#  NormDetection
#
#  Created by Felipe Meneguzzi on 2015-07-20.
#  Copyright 2015 Felipe Meneguzzi. All rights reserved.
#

import random
from norm_behaviour_old import *
# My own timer functions
from stats import *

from __builtin__ import str

import os
import os.path
import subprocess

import sys
from optparse import OptionParser

# import threading
from multiprocessing import Process

import math

from scipy import stats

outputfolder="plot_journal/"

# ============================================
# = Scenario Definitions for the Experiments =
# ============================================

def scenarioX(
              prob_non_compliance=0.01, 
              prob_viol_detection=0.99, 
              prob_sanctioning=0.99, 
              prob_random_punishment=0.01,
              prior=log(0.5), 
              prior_none = log(1)):
    pass
    
def scenario1(prob_non_compliance=0.01, prob_viol_detection=0.99, \
                    prob_sanctioning=0.99, prob_random_punishment=0.01,
                    prior=log(0.5), prior_none = log(1)):
    goal = Goal('a','d')
    actions = set([Action(['a','b']), Action(['b','e']), Action(['b','c']), Action(['b','d']), Action(['a','f']), Action(['a','c','e']), Action(['e','d'])])
    
    suite = build_norm_suite(goal, actions,prob_non_compliance, prob_viol_detection, prob_sanctioning, prob_random_punishment, prior, prior_none)
    
    norms = set( [ ('a','never','e') ] )
    return (suite,norms)

def scenario1_more_norms(prob_non_compliance=0.01, prob_viol_detection=0.99, \
                    prob_sanctioning=0.99, prob_random_punishment=0.01,
                    prior=log(0.5), prior_none = log(1)):
    """The same as scenario 1, but with more norms"""
    suite,norms = scenario1(prob_non_compliance, prob_viol_detection, prob_sanctioning, prob_random_punishment, prior, prior_none)
    norms.add(('a','not next','c'))
    # norms.add(('b','next','d'))
    return (suite,norms)

def writeObservationsToFile(norms,observations,filename):
    f = open(outputfolder+str(filename)+".obs.txt",'w')
    for norm in norms:
        f.write(str(norm)+"\n")
    f.write("\n")
    for plan in observations:
        f.write(str(plan)+"\n")
    f.close()

# ========================
# = Statistics Gathering =
# ========================
# TODO Redo code to compute statistics using scypi.stats.describe
def compute_stats(entries):
    """Computes mean and standard deviation statistics for the empirical data 
       in entries where:
       entries[i] contains a matrix of repetitions for the ith run
       entries[i][j] contains repetition j of run i
    """
    averages = [None] * len(entries)
    sigma = [None] * len(entries)
    # print(str(entries))
    for i in range(len(entries)):
        total = len(entries[i]) # Total number of repetitions
        averages[i] = [0]*len(entries[i][0]) # Average stat
        sigma[i] = [0]*len(entries[i][0]) # Sigma
        averages[i][0]=i
        sigma[i][0]=i
        # First compute the average (mean)
        for entry in entries[i]:
            for j in range(1,len(entry)):
                averages[i][j] = averages[i][j]+entry[j] 
        for j in range(1,len(averages[i])):
            averages[i][j] = averages[i][j]/total
        # Then compute the standard deviation
        # By first summing the distance from the mean
        for entry in entries[i]:
            for j in range(1,len(entry)):
                sigma[i][j] = sigma[i][j]+math.pow(entry[j] - averages[i][j],2)
        for j in range(1,len(sigma[i])):
            sigma[i][j] = math.sqrt(sigma[i][j]/total)
        averages[i] = tuple(averages[i])
        sigma[i] = tuple(sigma[i])
    return (averages,sigma)

# =======================================================================================
# = Experiment Type 1 - Evolution of the odds of a norm, given a series of observations =
# =======================================================================================
def experiment_odds_over_runs(suite,norms,runs, repeats=1, shift_goals=False, violation_signal=False, genPlot=False,graphname="odds-runs",writeTrace=True):
    graphname+="-"+str(len(suite.actions))+"a"+str(len(norms))+"n"+("-vsignal" if violation_signal else "")+("-shiftgoals" if shift_goals else "") 
    print "Running Odds over Runs, "+str(len(norms))+" norms, "+str(len(suite.actions))+" actions"+(", shifting goals" if shift_goals else "")+(", violation signal" if violation_signal else "")
    print str(runs)+" runs, "+str(repeats)+" repetitions"
    
    plot_entries = [[0 for x in range(repeats)] for x in range(runs+1)];
    
    for r in range(repeats):
        # reinitialise the norm suite
        # suite = build_norm_suite(suite.inferred_goal, suite.actions, suite.prob_non_compliance, suite.prob_viol_detection, suite.prob_sanctioning, suite.prob_random_punishment)
        suite = reinitialise_suite(suite)
        observations = generate_random_observations(suite, norms, runs, shift_goals, violation_signal)
        if(writeTrace): writeObservationsToFile(norms,observations,graphname)
    
        (n,topN) = suite.most_probable_norms(1)
        plot_entries[0][r] = create_entry_odds(0,suite,norms,suite.d[n[0]])
    
        for i in range(runs):
            plan = observations[i]
            # print "Run "+str(i)+": Observed plan: "+str(plan)
            if shift_goals:
                suite.SetGoal(goal_from_plan(plan))
            suite.UpdateOddsRatioVsNoNorm(plan)
#             suite.print_ordered()
            (n,topN) = suite.most_probable_norms(1)
            plot_entries[i+1][r]=create_entry_odds(i+1,suite,norms,suite.d[n[0]])
    
    (averages,sigma) = compute_stats(plot_entries)
    
    
    labels = list(norms)+["Max Odds"]
    plotTitle = "Odds x Runs"+(" (Violation Signal)" if violation_signal else "")
    print_graph(graphname+"-avgs",averages,True,"Runs","Odds",plotTitle,labels)
    print_graph(graphname+"-sigmas",sigma,True,"Runs","Odds",plotTitle,labels)

def create_entry_odds(t,suite,norms,maxOdds):
    entry = [t]
    for n in norms:
        entry.append(suite.d[n])
    entry.append(maxOdds)
    return tuple(entry)

# ================================================
# = Experiment Type 2 - Inferred Norms over Runs =
# ================================================    

def experiment_inferred_norms_over_runs(suite,norms,runs, repeats=1, shift_goals=False, violation_signal=False, genPlot=False,graphname="norms-runs",writeTrace=True):
    graphname+="-"+str(len(suite.actions))+"a"+str(len(norms))+"n"+("-vsignal" if violation_signal else "")+("-shiftgoals" if shift_goals else "") 
    print "Running Inferred norms over Runs, "+str(len(norms))+" norms, "+str(len(suite.actions))+" actions"+(", shifting goals" if shift_goals else "")+(", violation signal" if violation_signal else "")
    print str(runs)+" runs, "+str(repeats)+" repetitions"
    
    plot_entries = [[0 for x in range(repeats)] for x in range(runs+1)];
    
    for r in range(repeats):
        # reinitialise the norm suite
        suite = reinitialise_suite(suite)
        observations = generate_random_observations(suite, norms, runs, shift_goals, violation_signal)
        if(writeTrace): writeObservationsToFile(norms,observations,graphname)

        plot_entries[0][r] = (0,0,0,0)
        for i in range(runs):
            plan = observations[i]
            # print "Run "+str(i)+": Observed plan: "+str(plan)
            if shift_goals:
                suite.SetGoal(goal_from_plan(plan))
            suite.UpdateOddsRatioVsNoNorm(plan)
            plot_entries[i+1][r] = create_entry_inferred_norms(i,suite,norms)
    
    (averages,sigma) = compute_stats(plot_entries)
    
    plotTitle = "Inferred Norms x Runs"+(" (Violation Signal)" if violation_signal else "")
    print_graph(graphname+"-avgs",averages,True,"Runs","Norms",plotTitle, ["Precision%","Recall%","#Top Norms"])
    print_graph(graphname+"-sigmas",sigma,True,"Runs","Norms",plotTitle, ["Precision%","Recall%","#Top Norms"])

def create_entry_inferred_norms(t,suite,norms):
    (prob_norms,topN) = suite.most_probable_norms(len(norms))
    totalDectected = len(prob_norms)
    detected = len(norms & set(prob_norms))
    recall = (detected*100.0)/len(norms)
    precision = (detected*100.0)/totalDectected
    
    return (t+1,precision,recall,totalDectected)

# =========================================================================
# = Experiment Type 3 - Inferred Norms over Runs using Modified Criterion =
# =========================================================================
def experiment_inferred_norms_over_runs_plan_precision(suite, norms, runs, repeats=1, samples=5, shift_goals=False, violation_signal=False, genPlot=False,graphname="norms-runs-plan-prec",writeTrace=True):
    graphname+="-"+str(len(suite.actions))+"a"+str(len(norms))+"n"+("-vsignal" if violation_signal else "")+("-shiftgoals" if shift_goals else "") 
    print "Running Inferred Norms over Runs (different precision "+str(samples)+" samples), "+str(len(norms))+" norms, "+str(len(suite.actions))+" actions"+(", shifting goals" if shift_goals else "")+(", violation signal" if violation_signal else "")
    print str(runs)+" runs, "+str(repeats)+" repetitions"
    
    plot_entries = [[0 for x in range(repeats)] for x in range(runs+1)];
    
    for r in range(repeats):
        # reinitialise the norm suite
        suite = reinitialise_suite(suite)
        observations = generate_random_observations(suite, norms, runs, shift_goals, violation_signal)
        if(writeTrace): writeObservationsToFile(norms,observations,graphname)
    
        plot_entries[0][r] = (0,0,0,0)
        for i in range(runs):
            plan = observations[i]
            # print "Run "+str(i)+": Observed plan: "+str(plan)
            if shift_goals:
                suite.SetGoal(goal_from_plan(plan))
            suite.UpdateOddsRatioVsNoNorm(plan)
    #         plot_entries.append(create_entry_inferred_norms(i,suite,norms))
            plot_entries[i+1][r] = create_entry_inferred_norms_diff_precision(i, suite, norms, runs, len(norms)*samples)
    
    (averages,sigma) = compute_stats(plot_entries)
    
    plotTitle = "Inferred Norms x Runs (Plan Precision "+str(samples)+" per norm)"+(" (Violation Signal)" if violation_signal else "")
    print_graph(graphname+"-avgs",averages,True,"Runs","Norms",plotTitle, ["Precision%","Recall%","#Top Norms"])
    print_graph(graphname+"-sigmas",sigma,True,"Runs","Norms",plotTitle, ["Precision%","Recall%","#Top Norms"])

def create_entry_inferred_norms_diff_precision(t,suite, norms, plan_samples, norm_samples=10):
    (prob_norms,topN) = suite.most_probable_norms(len(norms)+10)
    norm_samples = min(topN,norm_samples)
    totalDectected = len(prob_norms)
    detected = len(norms & set(prob_norms))
    recall = (detected*100.0)/len(norms)
    
    sample_norms = random.sample(prob_norms,norm_samples)
    real_norms = convert_norms_to_generative(norms)
    try:
        observations = generate_random_observations(suite, sample_norms, plan_samples, shift_goals=True, violation_signal=False)
    except ValueError:
        # print "No compliant plans possible"
        observations = []
    correct_plans = 0
    for plan in observations:
        if(is_norm_compliant(plan,real_norms)):
            correct_plans +=1
    precision = (correct_plans*100.0)/plan_samples
    
    return (t+1,precision,recall,totalDectected)

# ====================================================================================
# = Experiment 4 - Compute Precision and Recall as a Function of the Number of Norms =
# ====================================================================================

# TODO double and triple check this
def experiment_precision_recall_over_norms(suite, norms, runs, repeats=1, samples=5, shift_goals=False, violation_signal=False, genPlot=False,graphname="prec-norms",writeTrace=True):
    
    graphname+="-"+str(len(suite.actions))+"a"+str(len(norms))+"n"+("-vsignal" if violation_signal else "")+("-shiftgoals" if shift_goals else "") 
    print "Running Precision and Recall over #Norms (precision "+str(samples)+" samples), "+str(len(norms))+" norms, "+str(len(suite.actions))+" actions"+(", shifting goals" if shift_goals else "")+(", violation signal" if violation_signal else "")
    print str(runs)+" runs, "+str(repeats)+" repetitions"
    
    plot_entries = [[0 for x in range(repeats)] for x in range(runs+1)];
    
    for r in range(repeats):
        plot_entries[0][r] = (0,0,0,0)
        #In each repetition I want to get precision and recall for every size of the norms set
        for ni in range(len(norms)):
            # print "Computing precision and recall for %d norms out of %d norms" % (ni+1, len(norms))
            # reinitialise the norm suite
            # suiteuite(suite.inferred_goal, suite.actions, suite.prob_non_compliance, suite.prob_viol_detection, suite.prob_sanctioning, suite.prob_random_punishment)
            suite = reinitialise_suite(suite)
            norm_sample = set(random.sample(norms,ni+1))
            observations = generate_random_observations(suite, norm_sample, runs, shift_goals, violation_signal)
            if(writeTrace): writeObservationsToFile(norm_sample,observations,graphname)
            for i in range(runs):
                plan = observations[i]
                # print "Run "+str(i)+": Observed plan: "+str(plan)
                if shift_goals:
                    suite.SetGoal(goal_from_plan(plan))
                suite.UpdateOddsRatioVsNoNorm(plan)
                # TODO Check that the indentation here is correct (in the AAMAS version it was outside the runs)
                plot_entries[i+1][r] = create_entry_inferred_norms_diff_precision(ni, suite, norm_sample, runs, len(norm_sample)*samples)
            
    (averages,sigma) = compute_stats(plot_entries)
    plotTitle = "Precision/Recall x #Norms ("+str(samples)+" per norm)"+(" (Violation Signal)" if violation_signal else "")
    print_graph(graphname+"-avgs",averages,True,"#Norms",None,plotTitle, ["Precision%","Recall%","#Top Norms"])
    print_graph(graphname+"-sigmas",sigma,True,"#Norms",None,plotTitle, ["Precision%","Recall%","#Top Norms"])

# ====================================
# = Data writing and Plotting Graphs =
# ====================================
def print_graph(filename,entries,genPlot=False,xlabel=None,ylabel=None,title=None,curves=None):
    datafile = outputfolder+str(filename)+".txt"
    f = open(datafile,'w')
    if(curves != None):
        f.write("# ")
        for c in curves:
            f.write(str(c)+" ")
        f.write("\n")
    
    for e in entries:
        for i in e:
            f.write(str(i)+" ")
        f.write("\n")
    f.write("\n")
    f.close()
    if(genPlot):
        f = open(outputfolder+str(filename)+".plot",'w')
        f.write("#!/usr/local/bin/gnuplot\n")
        f.write("set term pdf enhanced\n")
        f.write("set output \""+outputfolder+str(filename)+".pdf\"\n")
        f.write("set key under\n")
        if(title != None):
            f.write("set title \""+str(title)+"\"\n")
        if(xlabel != None):
            f.write("set xlabel \""+str(xlabel)+"\"\n")
        if(ylabel != None):
            f.write("set ylabel \""+str(ylabel)+"\"\n")
        if(curves == None):
            f.write("plot %s with linesp \n",datafile)
        else:
            ci = 2
            f.write("plot")
            for curve in curves:
                if(ci != 2):
                    f.write(",\\\n")
                f.write(" \""+datafile+"\" using 1:"+str(ci)+" title \""+str(curve)+"\" with linesp")
                ci+=1
            f.write("\n")
        f.close()

def replot_all():
    for fn in os.listdir(outputfolder):
        if(fn.endswith(".plot")):
            fn_graph = outputfolder+fn.replace(".plot",".pdf")
            if(not os.path.exists(fn_graph) or (os.path.getctime(fn_graph) < os.path.getctime(outputfolder+fn)) ):
                print "Plotting "+outputfolder+fn
                if (subprocess.call(["/usr/local/bin/gnuplot",outputfolder+fn])==0):
                    print "Plot complete"
            else:
                print "Skipping "+fn+", graph not updated"
            # subprocess.call("/usr/local/bin/gnuplot")
            
# ====================================================
# = Calls to allow parallel execution of experiments =
# ====================================================

def all_experiments_odds_runs(runs,repeats):
    pass
    # (suite,norms) = scenario1()
    # experiment_odds_over_runs(suite,norms,runs,repeats,True,False,True)
    (suite,norms) = scenario1_more_norms()
    experiment_odds_over_runs(suite,norms,runs,repeats,True,False,True)
    # (suite,norms) = scenario2()
    # experiment_odds_over_runs(suite,norms,runs*2,repeats,True,False,True)
    # (suite,norms) = scenario2_more_norms()
#     experiment_odds_over_runs(suite,norms,runs,repeats,True,False,True)
#
#     (suite,norms) = scenario2_many_norms()
#     experiment_odds_over_runs(suite,norms,runs,repeats,True,False,True)

def do_nothing(runs,repeats):
    pass
# ====================================
# = Main Function to Run Experiments =
# ====================================

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", "--dir-output", dest="output", action="store", type="string",
                  help="write reports to DIR", metavar="DIR")
    parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbose", default=True,
                  help="don't print status messages to stdout")
    parser.add_option("-c","--cores", dest="cores", action="store", type="int",
                help="Create CORES separate number of processes", metavar="CORES")
    parser.add_option("-r","--repeats", dest="repeats", action="store", type="int",
                help="Repeat experiments REPEATS number of times", metavar="REPEATS")
    parser.add_option("-o","--observations", dest="observations", action="store", type="int",
                help="For each experiment, generate OBS number of observations", metavar="OBS")
    parser.add_option("-p","--plot", dest="replot", action="store_true", default=True,
                help="Replot all graphs once experiments are over")

    (options, args) = parser.parse_args()
    
    runs = 10
    repeats = 5
    
    if(options.repeats != None):
        repeats = options.repeats
    if(options.observations != None):
        runs = options.observations
    
    timer = start_timer()
    
    experiment_calls = [
                        all_experiments_odds_runs,
                        # all_experiments_odds_runs_violations,
                        # all_experiments_norms_runs,
                        # all_experiments_norms_runs_violations,
                        # all_experiments_precision_recall_over_norms,
                        # all_experiments_precision_recall_over_norms_violations,
                        # experiment_multiple_priors_regular,
                        # experiment_multiple_priors_equal_prior,
                        # experiment_multiple_priors_over_prior,
                        do_nothing
                        ]
    
    if(options.cores == None): 
        print "Running experiments in a single core"
    
        all_experiments_odds_runs(runs,repeats)
        #all_experiments_odds_runs_violations(runs,repeats)

        # runs=100
        #all_experiments_norms_runs(runs,repeats)
        #all_experiments_norms_runs_violations(runs,repeats)
        #
        #all_experiments_precision_recall_over_norms(runs, repeats)
        #all_experiments_precision_recall_over_norms_violations(runs, repeats)
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
    
    if(options.replot):
        replot_all()

