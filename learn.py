#!/usr/bin/env python2.7
""" 
learn.py
Learn Reddit comment authors from text.

Program scaffold based on
<http://users.soe.ucsc.edu/~karplus/bme205/f12/Scaffold.html>
"""

import argparse, sys, io, pickle, itertools, collections, random, re, tempfile
import os, subprocess

import nltk
import nltk.grammar
from nltk.tree import Tree

from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# We keep our list of function words in a global
# Function words from Appendix 3 of Narayanan et al. 2012
function_words=set([
"aboard", "about", "above", "absent", "according", "accordingly",  "across",
"after", "against", "ahead", "albeit", "all", "along", "alongside", "although",
"am", "amid", "amidst", "among", "amongst", "amount", "an", "and", "another",
"anti",  "any", "anybody", "anyone", "anything", "are", "around", "as",
"aside", "astraddle",  "astride", "at", "away", "bar", "barring", "be",
"because", "been", "before", "behind",  "being", "below", "beneath", "beside",
"besides", "better", "between", "beyond", "bit",  "both", "but", "by", "can",
"certain", "circa", "close", "concerning", "consequently",  "considering",
"could", "couple", "dare", "deal", "despite", "down", "due", "during",  "each",
"eight", "eighth", "either", "enough", "every", "everybody", "everyone",
"everything", "except", "excepting", "excluding", "failing", "few", "fewer",
" fifth",  " first", " five", "following", "for", "four", "fourth", "from",
"front", "given", "good",  "great", "had", "half", "have", "he", "heaps",
"hence", "her", "hers", "herself", "him",  "himself", "his", "however", "i",
"if", "in", "including", "inside", "instead", "into", "is", "it",  "its",
"itself", "keeping", "lack", "less", "like", "little", "loads", "lots",
"majority", "many",  "masses", "may", "me", "might", "mine", "minority",
"minus", "more", "most", "much",  "must", "my", "myself", "near", "need",
"neither", "nevertheless", "next", "nine",  "ninth", "no", "nobody", "none",
"nor", "nothing", "notwithstanding", "number",  "numbers", "of", "off", "on",
"once", "one", "onto", "opposite", "or", "other", "ought",  "our", "ours",
"ourselves", "out", "outside", "over", "part", "past", "pending", "per",
"pertaining", "place", "plenty", "plethora", "plus", "quantities", "quantity",
"quarter",  "regarding", "remainder", "respecting", "rest", "round", "save",
"saving", "second",  "seven", "seventh", "several", "shall", "she", "should",
"similar", "since", "six", "sixth",  "so", "some", "somebody", "someone",
"something", "spite", "such", "ten", "tenth",  "than", "thanks", "that", "the",
"their", "theirs", "them", "themselves", "then", "thence",  "therefore",
"these", "they", "third", "this", "those", "though", "three", "through",
"throughout", "thru", "thus", "till", "time", "to", "tons", "top", "toward",
"towards",  "two", "under", "underneath", "unless", "unlike", "until", "unto",
"up", "upon", "us",  "used", "various", "versus", "via", "view", "wanting",
"was", "we", "were", "what",  "whatever", "when", "whenever", "where",
"whereas", "wherever", "whether",  "which", "whichever", "while", "whilst",
"who", "whoever", "whole", "whom",  "whomever", "whose", "will", "with",
"within", "without", "would", "yet", "you",  "your", "yours", "yourself",
"yourselves"
])


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
    parser.add_argument("--in", dest="in file", type=argparse. fileType('r'), 
        default=sys.stdin, 
        help="serialized comment input  file (default: stdin)")
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

def write_temp_text(text):
    """
    Write the given string to a temporary file.
    Returns the filename of the temporary file.
    """
    
    (handle, filename) = tempfile.mkstemp(suffix=".txt")
    
    # Convert handle (an int) to a proper stream
    stream = io.open(handle, "w") 
    stream.write(text)
    stream.close()
    
    return filename

def stanford_parse(comment):
    """
    Given a string of multiple sentences, yiled NLTK trees for their
    parsings. Creates these trees using the Stanford parser in ./stanford.
    
    See http://www.cs.ucf.edu/courses/cap5636/fall2011/nltk.pdf
    
    """
    
    # Write the comment and get the filename
    comment_file = write_temp_text(comment)
    
    # Call the parser
    
    command_line = ["java", "-mx150m", "-cp", "\"stanford/*:\"", 
        "edu.stanford.nlp.parser.lexparser.LexicalizedParser", "-outputFormat", 
        "penn", "edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz", 
        comment_file]
    stanford = subprocess.Popen(" ".join(command_line), stdout=subprocess.PIPE, 
        shell=True) 
    
    # Read each tree and yield it
    # This holds all the lines part of this parsing
    current_parsing = []
    
    for line in stanford.stdout:
        if line.rstrip() == "":
            # A blank separator line
            # The s-expression is done, so parse it (ugh) and yield it
            yield Tree.parse(" ".join(current_parsing))
            
            # New parsing
            current_parsing = []
        else:
            # This line is more of this parsing
            current_parsing.append(line.rstrip())
    # The final parsing has a blank line after it
    
    # That's all the data (hopefully)
    if stanford.wait() != 0:
        raise Exception("Stanford broke!")
        
    os.remove(comment_file)
    
    
    
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
        
        # Training set gets the  first half
        training[user] = to_shuffle[0:len(to_shuffle)/2]
        
        # Test set gets the second half
        test[user] = to_shuffle[len(to_shuffle)/2:]
        
    return (training, test)

def bag_of_words_features(comment):
    """
    Makes a feature/value dictionary from a Unicode comment, under a 
    bag-of-words model.
    """    
    return {word: True for word in nltk.word_tokenize(comment)}
    
def raw_content_free_features(comment):
    """
    Just like content_free_features but not normalized.
    """
    
    return content_free_features(comment, normalize=False)
    
def content_free_features(comment, normalize=True):
    """
    Makes a feature/value dictionary from a Unicode comment, with content-free
    features.
    
    Features to compute:
        * Words and characters in post (2 features)
        * Fraction of words of 1 to 20 characters (20 features)
        * Fraction of words in:
            * UPPERCASE
            * lowercase
            * Capitalized
            * CamelCase
            * aNyThInG ElSe
        * Fraction of letters (ignoring case)
        * Fraction of all other characters by type
        * Fraction of words that are each function word
        * Parse tree edges by from, to as a fraction of total
        
    If normalize is false, all frequencies are raw counts instead.
    
    """
    # What feature dict are we making? (a dict of floats)
    features = collections.defaultdict(float)
    
    # Get all the words
    words = nltk.word_tokenize(comment)
    
    # Characters in post
    features[u"characters"] = len(comment)
    
    # Words in post
    features[u"words"] = len(words)
    
    # Word lengths from 1 to 20 characters
    for word in words:
        if len(word) == 0 or len(word) > 20:
            continue
            
        features[u"word-length:" + str(len(word))] += 1
        
    if normalize:
        # Normalize
        for length in xrange(1, 21):
            features[u"word-length:" + str(length)] /= float(len(words))
        
    # Word capitalization counts. This regex matches camelcase (or capitalized,
    # but we check that separately):  first letter is capital, then some
    # lowercase, and then some capital letters each followed by some lowercase
    # ones
    camel_regex = re.compile(r"([A-Z][a-z]+)+")
    for word in words:
        if word.isupper():
            features[u"case:upper"] += 1
        elif word.islower():
            features[u"case:lower"] += 1
        elif word.istitle():
            features[u"case:capitalized"] += 1
        elif camel_regex.match(word):
            features[u"case:camel"] += 1
        else:
            features[u"case:other"] += 1
            
    if normalize:
        # Normalize
        features[u"case:upper"] /= float(len(words))
        features[u"case:lower"] /= float(len(words))
        features[u"case:capitalized"] /= float(len(words))
        features[u"case:camel"] /= float(len(words))
        features[u"case:other"] /= float(len(words))
        
    
    # Character frequencies
    # Keep in a separate dict for easy norming
    character_counts = collections.Counter()
    for character in comment:
        character_counts[u"char:" + character.lower()] += 1
        
    # Normalize and add in
    for key, count in character_counts.iteritems():
        features[key] = count
        if normalize:
            features[key] /= float(len(comment))
            
    
    # Function words
    function_counts = collections.Counter()
    for word in words:
        word = word.lower()
        if word in function_words:
            function_counts["function:" + word] += 1
    
    for key, count in function_counts.iteritems():
        features[key] = count
        if normalize:
            features[key] /= float(len(words))
            
    # Parsing edge features
    # Parse all the sentences
    trees = stanford_parse(comment)
    
    # This holds counts all the (parent, child) tree edges
    edges = collections.Counter()
    
    for tree in trees:
        for production in tree.productions():
            for child in production.rhs():
                if isinstance(child, nltk.grammar.Nonterminal):
                    # Don't let words in
                    edges[u"parse:" + str((production.lhs(), child))] += 1
                    
    # Copy over, normalizing if needed
    for key, count in edges.iteritems():
        features[key] = count
        if normalize:
            features[key] /= float(len(edges))
    
    
    
    return features

def make_sklearn_dataset(user_index, model_function, vectorizer=None):
    """
    Given a dict of comment tuple lists by user name, and a function mapping
    comment strings to dicts of features, produces a feature matrix X and a
    label vector t suitable for use with sklearn classifiers. Converts user
    names to numbers.
    
    if vectorizer is passed, it is used to map feature dicts to feature vectors.
    You would get a vectorizer from calling this function on the training set,
    and use it when calling this function on the test set, so that features that
    we would only know to have by loking at the test set don't get used.
    
    Returns feature matrix, label vector, vectorizer
    
    """
    
    
    
    # Compose a flat list of feature dicts and a list of labels
    feature_dicts = []
    labels = []
    
    # Use the passed feature extraction function to get dicts from comments
    for (user_number, (user_name, comments)) in enumerate(
        user_index.iteritems()):
        
        for comment in comments:
            # Store the extracted features for the comment
            feature_dicts.append(model_function(comment[1]))
            
            # Store the label in the corresponding position in the labels list
            labels.append(user_number)
            
    if vectorizer is None:
        # This is the DictVectorizer that we will use to vectorize the feature dicts
        # for each comment
        vectorizer = DictVectorizer()
        
        # Train on this data
        vectorizer. fit(feature_dicts)
            
    # Transform dicts into vectors
    feature_matrix = vectorizer.transform(feature_dicts)
    
    return feature_matrix, labels, vectorizer
            
    
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
    
    print "Uniform guess correctness rate: {}".format(
        1/float(len(user_index.keys())))
    
    sys.stdout.flush()
    
    # Get the test and training sets
    training_index, test_index = split_user_index(user_index)
    
    # Feature models to try
    feature_models = {
        "Content-Free": content_free_features,
        "Content-Free (Raw)": raw_content_free_features,
        "Bag-of-Words": bag_of_words_features
    }
    
    # Classifiers to try
    classifiers = {
        "Naive Bayes": MultinomialNB,
        "Nearest Neighbor": KNeighborsClassifier,
        "Support Vector Classifier": SVC
    }
    
    for model_name in feature_models.iterkeys():
        
        # Get the feature model function
        model_function = feature_models[model_name]
        
        # Convert to labeled examples       
        print "Producing {} features...".format(model_name) 
        sys.stdout.flush()
        
        # Get a vectorizer from making the training set feature vectors
        training_features, training_labels, vectorizer = make_sklearn_dataset(
            training_index, model_function)
            
        # Use it when making the test set feature vectors
        test_features, test_labels, vectorizer = make_sklearn_dataset(
            test_index, model_function, vectorizer=vectorizer)
        
        for classifier_name in classifiers.iterkeys():

            # Get the class of classifier to use
            classifier_class = classifiers[classifier_name]

            # Train up a classifier
            print "Training {} Classifier...".format(classifier_name)
            sys.stdout.flush()
            
            classifier = classifier_class()
            classifier. fit(training_features, training_labels)
            
            # Calculate the accuracy on the test set
            print "Computing accuracy..."
            sys.stdout.flush()
            
            accuracy = classifier.score(test_features, test_labels)
            
            # Report the accuracy and most informative features
            print "{}/{} Accuracy: {}".format(classifier_name, model_name, 
                accuracy)
            sys.stdout.flush()
            
    return 0
    
if __name__ == "__main__" :
    # No error catching because stack traces are important
    # And I don't want to mess around with a module to get them
    sys.exit(main(sys.argv)) 
