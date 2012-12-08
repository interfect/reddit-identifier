#!/usr/bin/env python2.7
""" 
download.py:
Download Reddit comments and save them to a file.

Program scaffold based on
<http://users.soe.ucsc.edu/~karplus/bme205/f12/Scaffold.html>
"""

import praw, argparse, sys, io, pickle, itertools, pickle

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
    parser.add_argument("--out", dest="outFile", type=argparse.FileType('w'), 
        default=sys.stdout, 
        help="serialized comment output file (default: stdout)") 
    parser.add_argument("--min_user_comments", type=int, default=100,
        help="miniumum comments a user has to have to be counted")
    parser.add_argument("--max_user_comments", type=int, default=1000,
        help="maximum comments to download for any given user (newest first)")
    parser.add_argument("--num_users", type=int, default=100,
        help="number of users to get comment histories for")
    
        
    return parser

def parse_args(args):
    """
    Takes in the command-line arguments list (args), and returns a nice argparse
    result with fields for all the options.
    
    """
    
    # The command line arguments start with the program name, which we don't
    # want to treat as an argument for argparse. So we remove it.
    args = args[1:]
    
    # Get the parser
    # parser holds the program's argparse parser.
    parser = generate_parser()
    
    # Invoke the parser
    return parser.parse_args(args)

def user_generator(reddit):
    """
    Yield Redditors, with no repeats. May eventually run out of users.
    """
    
    seen_users = set()
    
    front_page = reddit.get_front_page(limit=1000) # This ought to be enough
    
    for post in front_page:
        for comment in post.comments_flat:
            if not hasattr(comment, "author") or comment.author == None:
                # Probably a [deleted]
                continue
                
            if comment.author.name not in seen_users:
                seen_users.add(comment.author.name)
                yield comment.author
    
def main(args):
    """
    Parses command line arguments and download comments.
    "args" specifies the program arguments, with args[0] being the executable
    name. The return value should be used as the program's exit code.
    """
    
    options = parse_args(args) # This holds the nicely-parsed options object
    
    reddit = praw.Reddit(user_agent="comment-learner")
    
    accepted_user_count = 0
    for user in user_generator(reddit):
        if accepted_user_count >= options.num_users:
            # Got enough users
            break
            
        # This line here downloads the whole comment history. It can't be that
        # big, right?
        comments = user.get_comments(limit=options.max_user_comments)
        comment_count = 0
        
        print "Saving comments for user \"{}\" ({}/{})".format(user.name, 
            accepted_user_count, options.num_users)
            
        for comment in comments:
            comment_count += 1
            
            # Save the comment user, Markdown, and creation time.
            # Markdown is in Unicode btw.
            comment_tuple = (user.name, comment.body, comment.created)
            pickle.dump(comment_tuple, options.outFile)
            
            print "Saved comment {}: #{}".format(user.name, comment_count)
        
        if comment_count >= options.min_user_comments:
            # Count the user towards our total only if they have enough comments
            accepted_user_count += 1
        
    if accepted_user_count < options.num_users:
        print "WARNING! Could only get data for {}/{} users!".format(
            accepted_user_count, options.num_users)
        return 1
            
    return 0
    
if __name__ == "__main__" :
    # No error catching because stack traces are important
    # And I don't want to mess around with a module to get them
    sys.exit(main(sys.argv)) 
