#!/usr/bin/env python2.7
""" 
learn.py
Learn Reddit comment authors from text.

Program scaffold based on
<http://users.soe.ucsc.edu/~karplus/bme205/f12/Scaffold.html>
"""

import praw, argparse, sys, io, pickle, itertools, collections, random

import nltk
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.classify import DecisionTreeClassifier

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
    
def split_user_index(user_index):
    """
    Splits a dict of comment lists by user name randomly into training and test 
    sets with the same structure.
    
    Returns a tuple of user-indexed comment dicts: (training, test)
    """
    
    # This holds the training set indexed by user
    training = {}
    
    # This holds the test set indexed by user
    test = {}
    
    for (user, comments) in user_index.iteritems():
        # We need two complementary random samples, so we shuffle and split in 
        # half
        
        # Make our own copy to shuffle so we don't mess up the original comments
        to_shuffle = list(comments)
        random.shuffle(to_shuffle)
        
        # Training set gets the first half
        training[user] = to_shuffle[0:len(to_shuffle)/2]
        
        # Test set gets the second half
        test[user] = to_shuffle[len(to_shuffle)/2:]
        
    return (training, test)
        
def make_nltk_labeled_list(user_index, feature_function):
    """
    Given a user-indexed comment dictionary and a function from comments to
    feature dicts, makes a list of (features, label) tuples such as an NLTK
    classifier expects to be trained or tested on.
    """
    
    example_list = []
    for (user, comments) in user_index.iteritems():
        # Run the feature function on each comment's text, and put that as an 
        # example labeled with this user
        example_list += [(feature_function(comment[1]), user) for comment in 
            comments]
            
    return example_list

def bag_of_words_features(comment):
    """
    Makes a feature/value dictionary from a Unicode comment, under a 
    bag-of-words model.
    """    
    return {word: True for word in nltk.word_tokenize(comment)}
    
def content_free_features(comment):
    """
    Makes a feature/value dictionary from a Unicode comment, with content-free
    features.
    """
    # What feature dict are we making
    feature_dict = {}
    
    # What caharacters does every comment need a frequency feature for?
    legal_chars = (u"ABCDEFGHIJKLMNOPQRSTUVWZYZabcdefghijklmnopqrstuvwxyz"
     "!@#$%^&*()_+{}|:\"<>?1234567890-=[]\\;',./~`\n ")
    
    # Character frequency
    for character in legal_chars:
        feature_dict[u"char({})".format(character)] = (
            float(comment.count(character)) / len(comment))
    
    return feature_dict
    
def main(args):
    """
    Parses command line arguments and download comments.
    "args" specifies the program arguments, with args[0] being the executable
    name. The return value should be used as the program's exit code.
    
    Based on http://streamhacker.com/2010/05/10/text-classification-sentiment-analysis-naive-bayes-classifier/
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
    
    # Get the test and training sets
    training_index, test_index = split_user_index(user_index)
    
    # Feature models to try
    feature_models = {
        "Bag-of-Words": bag_of_words_features,
        "Content-Free": content_free_features
    }
    
    # Classifiers to try
    classifiers = {
        "Naive Bayes": NaiveBayesClassifier,
        "Decision Tree": DecisionTreeClassifier
    }
    
    # Which pairings do we use.
    # Not all are valid
    pipelines = [("Bag-of-Words", "Naive Bayes"), 
        ("Content-Free", "Decision Tree")]
    
    for (model_name, classifier_name) in pipelines:
        
        # Get the feature model function
        model_function = feature_models[model_name]
        
        # Get the class of classifier to use
        classifier_class = classifiers[classifier_name]
        
        # Convert to labeled examples       
        print "Producing {} features...".format(model_name) 
        training_set = make_nltk_labeled_list(training_index, model_function)
        test_set = make_nltk_labeled_list(test_index, model_function)

        # Train up a classifier
        print "Training {} Classifier...".format(classifier_name)
        classifier = classifier_class.train(training_set)
        
        # Calculate the accuracy on the test set
        print "Computing accuracy..."
        accuracy = nltk.classify.util.accuracy(classifier, test_set)
        
        # Report the accuracy and most informative features
        print "{}/{} Accuracy: {}".format(classifier_name, model_name, accuracy)
        try:
            # If it supports it, dump the features
            classifier.show_most_informative_features()
        except:
            pass
            
    return 0
    
if __name__ == "__main__" :
    # No error catching because stack traces are important
    # And I don't want to mess around with a module to get them
    sys.exit(main(sys.argv)) 
