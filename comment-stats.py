#!/usr/bin/env python2.7
""" 
comment-stats.py
Get data stats for comment list files.

Program scaffold based on
<http://users.soe.ucsc.edu/~karplus/bme205/f12/Scaffold.html>
"""

import argparse, sys, io, pickle, itertools, collections, random, re, tempfile
import os, subprocess, random, numpy

from matplotlib import pyplot

def generate_parser():
    """
    Generate the options parser for this program.
    Borrows heavily from the argparse documentation examples:
    <http://docs.python.org/library/argparse.html>
    """
    
    # Construct the parser (which is stored in parser)
    # Module docstring lives in __doc__
    # See http://python-forum.com/pythonforum/viewtopic.php?f=3&t=36847
    # And a formatter class so our examples in the docstring look good. Isn't it
    # convenient how we already wrapped it to 80 characters?
    # See http://docs.python.org/library/argparse.html#formatter-class
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    # Now add all the options to it.
    parser.add_argument("--in", dest="inFile", type=argparse.FileType('r'), 
        default=sys.stdin, 
        help="serialized comment input file (default: stdin)")
    parser.add_argument("--min_user_comments", type=int, default=100,
        help="miniumum comments a user has to have to be used") 
    
        
    return parser

def parse_args(args):
    """
    Takes in the command-line arguments list (args), and returns a nice argparse
    result with  fields for all the options.
    
    """
    
    # The command line arguments start with the program name, which we don't
    # want to treat as an argument for argparse. So we remove it.
    args = args[1:]
    
    # Get the parser
    # parser holds the program's argparse parser.
    parser = generate_parser()
    
    # Invoke the parser
    return parser.parse_args(args)  
    
def read_comments(stream):
    """
    Yields (username, markdown, creation time) tuples read from the given 
    stream.
    """
    
    try:
        while True:
            yield pickle.load(stream)
    except EOFError:
        # Got everything. End the loop.
        pass
        
def create_user_index(comments):
    """
    Create a dict of lists of comments sorted by user. Comments is an iterator 
    of comment tuples.
    """
    
    # Our dict is really a defaultdict, so it can have empty lists for anyone we
    # don't know about.
    user_index = collections.defaultdict(list)
    
    for comment in comments:
        user_index[comment[0]].append(comment)
        
    return user_index
            
    
def main(args):
    """
    Parses command line arguments and download comments.
    "args" specifies the program arguments, with args[0] being the executable
    name. The return value should be used as the program's exit code.
    """
    
    
    options = parse_args(args) # This holds the nicely-parsed options object
    
    # This holds a set of all comments. We use set in case we downloaded the 
    # same comment twice.
    comment_list = set(read_comments(options.inFile))
    print "{} comments loaded".format(len(comment_list))
            
    # And this an index of comments by user
    user_index = create_user_index(comment_list)
    print "{} total users".format(len(user_index.keys()))
    
    # Throw out the users that don't have enough comments
    user_index = {user: comments for (user, comments) in user_index.iteritems() 
        if len(comments) >= options.min_user_comments}
    print "{} users available for analysis".format(len(user_index.keys()))
    
    # How many comments are left?
    total_comments = sum(map(len, user_index.itervalues()))
    print "{} comments available for analysis.".format(total_comments)
    
    print "Uniform guess correctness rate: {}".format(
        1/float(len(user_index.keys())))
    
    # What's the average comment length?
    total_characters = 0
    for comment_list in user_index.itervalues():
        for comment in comment_list:
            total_characters += len(comment[1])
    print "Mean comment length: {} characters.".format(total_characters / float(total_comments))
    
    # What users are there?
    print "User list: {}".format(", ".join(user_index.iterkeys())) 
    
    return 0
    
if __name__ == "__main__" :
    # No error catching because stack traces are important
    # And I don't want to mess around with a module to get them
    sys.exit(main(sys.argv)) 
