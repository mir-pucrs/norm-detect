from planlib import plan_library_to_dot, Action
from networkx import fast_gnp_random_graph,minimum_spanning_tree,write_dot
from graph_tool import generation
from norm_behaviour import NormBehaviour

import sys
from optparse import OptionParser

def gen_tree_networkx(filname="graph.dot"):
    g = fast_gnp_random_graph(60,0.1,None,False)
    # g = random_powerlaw_tree(20)
    g = minimum_spanning_tree(g)
    # print(str(g))
    write_dot(g,filname)

# http://graph-tool.skewed.de/static/doc/generation.html?highlight=random#graph_tool.generation.price_network
# http://graph-tool.skewed.de/static/doc/quickstart.html
def gen_price(filename="graph.dot",nodes=100,norms_filename=None):
    """Randomly generates a plan library and, if the optional norms_filename is provided, generate tnorms"""
    # g= graph_tool.generation.price_network(30, m=5, c=None, gamma=1, directed=True, seed_graph=None)
    print "Generating graph with "+str(nodes)+" nodes"
    g= generation.price_network(nodes, m=10, c=1, gamma=2, directed=True, seed_graph=None)
    g.set_reversed(True)
    planlib = graph_tool_to_planlibrary(g)
    print "Resulting plan library has "+str(len(planlib))+" actions"
    plan_library_to_dot(planlib, filename)
    if(norms_filename is not None):
        nb = NormBehaviour()
        nb.gen_random_norms(planlib, norm_file=norms_filename)
    
    # graph_draw(g, vertex_text=g.vertex_index, vertex_font_size=10,output_size=(500, 500), output="price.svg")
    # Drawing to graphviz
    # deg = g.degree_property_map("in")
    # import numpy
    # deg.a = 2 * (numpy.sqrt(deg.a) * 0.5 + 0.4)
    # ebet = betweenness(g)[1]
    # graph_tool.draw.graphviz_draw(g, vcolor=deg, vorder=deg, elen=10, ecolor=ebet, eorder=ebet, output="price.svg")
    # graph_tool.draw.graphviz_draw(g,output="price.svg")

def graph_tool_to_planlibrary(g):
    planlib = set([])
    for e in g.edges():
        vert1 = g.vertex_index[e.source()]
        vert2 = g.vertex_index[e.target()]
        a = Action([str(vert1),str(vert2)])
        planlib.add(a)
    return planlib

if __name__ == '__main__':

    gen_price('large-planlib.dot',50,'large-norms.txt')    
    exit()
    parser = OptionParser()
    parser.add_option("-a", "--algorithm", dest="algorithm", action="store", type="string",
                  help="generates scenarios using algorithm ALGO", metavar="ALGO")
    
    
    
    gen_price()
    exit()
    print sys.argv
    if len(sys.argv) > 1:
        print("Generating plan library into "+str(sys.argv[1]))
        gen_tree_networkx(str(sys.argv[1])+".dot")
    else:
        print "Please provide the name of the target filename"