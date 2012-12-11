#!/usr/bin/env python2.7
""" 
learn.py
Learn Reddit comment authors from text.

Program scaffold based on
<http://users.soe.ucsc.edu/~karplus/bme205/f12/Scaffold.html>
"""

import argparse, sys, io, pickle, itertools, collections, random, re, tempfile
import os, subprocess, random, numpy

import nltk

import nltk.tree

from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

import function_words

import pp

# We have a global PP job server
job_server = None

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
    parser.add_argument("--top_predictions", type=int, default=5,
        help="number of best predictions to score")
    parser.add_argument("--pp", action="store_true",
        help="use Parallel Python for feature extraction")
    
        
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
    
    command_line = ["java", "-mx1G", "-cp", "\"stanford/*:\"", 
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
            yield nltk.tree.Tree.parse(" ".join(current_parsing))
            
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
    
    # BAG of words, not a SET of words
    word_counts = collections.Counter()
    
    for word in nltk.word_tokenize(comment):
        word_counts[word] += 1
        
    return word_counts
    
def raw_content_free_features(comment):
    """
    Just like content_free_features with no parsing or normalization.
    """
    
    return content_free_features(comment, normalize=False, parse=False)
    
def noparse_content_free_features(comment):
    """
    Content-free features with no parsing and with normalization.
    """
    
    return content_free_features(comment, normalize=True, parse=False)
    
def content_free_features(comment, normalize=True, parse=True):
    """
    Makes a feature/value dictionary from a Unicode comment, with content-free
    features.
    
    Features to compute:
        * Words and characters in post (2 features)
        * Vocabulary metrics
            * Yule's K (P(two randomly chosen nouns are the same)) (1 feature)
            * Frequency of words appearing exactly 1 or 2 or 3 ... or 10 times
              in the text (10 features)
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
            * This is replaced by PoS counts if you don't do the parse.
        
    If normalize is false, all frequencies are raw counts instead.
    
    If parse is false, don't use any features requiring parsing.
    
    """
    # What feature dict are we making? (a dict of floats)
    features = collections.defaultdict(float)
    
    # Get all the words
    words = nltk.word_tokenize(comment)
    
    # Characters in post
    features[u"characters"] = len(comment)
    
    # Words in post
    features[u"words"] = len(words)
    
    if len(words) == 0:
        # None of the other features make sense, they can all be 0
        return features
    
    # Yule's K
    
    # PoS tag everything
    pos_tagged = nltk.pos_tag(words)
    
    # Get all the nouns
    noun_counts = collections.Counter()
    for (word, pos) in pos_tagged:
        if pos[0] == "N":
            # Assume any tag starting with N is a noun (NN, NNP, etc.)
            noun_counts[word.lower()] += 1
    
    yules_k = 0
    total_nouns = sum(noun_counts.itervalues())
    # OR together all the events (we pick this noun twice)
    for noun, count in noun_counts.iteritems():
        yules_k += (count/float(total_nouns)) ** 2
        
    # Store Yule's K feature
    features["yules-k"] = yules_k
    
    # Words that appear 1 to 10 times each: frequencies thereof
    # First, count all the words
    word_counts = collections.Counter()
    for word in words:
        word_counts[word.lower()] += 1
    
    # Then, count number of unique words with any given count
    words_with_count = collections.Counter()
    for word, count in word_counts.iteritems():
        words_with_count[count] += 1
    
    # Copy over features
    for count in xrange(1, 11):
        features[u"hapax:" + str(count)] = words_with_count[count]
    
        if normalize:    
            # Normalize
            features[u"hapax:" + str(count)] /= float(len(words))
    
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
        if word in function_words.function_words:
            function_counts["function:" + word] += 1
    
    for key, count in function_counts.iteritems():
        features[key] = count
        if normalize:
            features[key] /= float(len(words))
     
    if parse:
        # Include parent-child PoS edge frequencies
        # Parsing edge features
        # Parse all the sentences
        trees = list(stanford_parse(comment))
        
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
    else:
        # Just use part-of-speech frequencies
        pos_counts = collections.Counter()
        
        for (word, pos) in pos_tagged:
            pos_counts[u"POS:" + pos] += 1
        
        # Copy over, normalizing if needed
        for key, count in pos_counts.iteritems():
            features[key] = count
            if normalize:
                features[key] /= float(len(pos_tagged))
    
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
    
    # How many comments have we processed?
    comments_done = 0
    
    if job_server is None:
        # Do comments in serial
        # Use the passed feature extraction function to get dicts from comments
        for (user_number, (user_name, comments)) in enumerate(
            user_index.iteritems()):
        
        
            for comment in comments:
                # Strip out special characters
                # Kind of defeats the point of unicode everywhere else...
                comment_string = unicode(comment[1].encode("ascii", "ignore"))
            
                # Get the features
                feature_dicts.append(model_function(comment_string))
            
                # Store the label in the corresponding position in the labels
                # list
                labels.append(user_number)
                
                comments_done += 1
                if comments_done % 100 == 0:
                    sys.stdout.write(".")
                    sys.stdout.flush()
        sys.stdout.write("\n")
            
    else:
        # Do comments in parallel
        
        # Jobs in flight
        pp_jobs = []
        
        # Use the passed feature extraction function to get dicts from comments
        for (user_number, (user_name, comments)) in enumerate(
            user_index.iteritems()):
            
            for comment in comments:
                # Strip out special characters
                # Kind of defeats the point of unicode everywhere else...
                comment_string = unicode(comment[1].encode("ascii", "ignore"))
            
                # Queue extracting comment features
                # Uses lots of functions and modules
                pp_jobs.append(job_server.submit(model_function, 
                (comment_string,), 
                (content_free_features,),
                ("re", "nltk", "numpy", "itertools", "collections", "sys",
                "function_words")))
                
                # Store the label in the corresponding position in the labels
                # list
                labels.append(user_number)
                
        # Wait for all the jobs to be done and put the resulting dicts in a list
        comments_done = 0
        for i, job in enumerate(pp_jobs):
            feature_dicts.append(job())
            comments_done += 1
            if comments_done % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()
        sys.stdout.write("\n")
            
    if vectorizer is None:
        # This is the DictVectorizer that we will use to vectorize the feature dicts
        # for each comment
        vectorizer = DictVectorizer()
        
        # Train on this data
        vectorizer.fit(feature_dicts)
            
    # Transform dicts into vectors
    feature_matrix = vectorizer.transform(feature_dicts)
    
    return feature_matrix, labels, vectorizer
            
    
def main(args):
    """
    Parses command line arguments and download comments.
    "args" specifies the program arguments, with args[0] being the executable
    name. The return value should be used as the program's exit code.
    
    If job_server is set, uses Parallel Python to speed things up.
    DOES NOT WORK if trying to use both pp and the Stanford parser.
    
    Based on http://streamhacker.com/2010/05/10/text-classification-sentiment-analysis-naive-bayes-classifier/
    """
    
    # We may need to write to this
    global job_server
    
    options = parse_args(args) # This holds the nicely-parsed options object
    
    if options.pp:
        # Make the job server
        job_server = pp.Server(secret="".join(
            [random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for i in xrange(20)]))


    
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
    
    sys.stdout.flush()
    
    # Get the test and training sets
    training_index, test_index = split_user_index(user_index)
    
    # Feature models to try
    feature_models = {
        "Content-Free (counts)": noparse_content_free_features,
        "Content-Free (frequencies)": raw_content_free_features,
        "Bag-of-Words": bag_of_words_features
    }
    
    # Classifiers to try
    classifiers = {
        "Naive Bayes": MultinomialNB,
        "Nearest Neighbor": lambda: KNeighborsClassifier(n_neighbors=1),
        # non-linear SVC is one vs one by default, which is O(N^2) and too slow
        "Linear SVM": LinearSVC
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
            # Set probability=True to compute probability info
            classifier.fit(training_features, training_labels)
            
            # Calculate the accuracy on the test set
            print "Computing accuracy..."
            sys.stdout.flush()
            
            accuracy = classifier.score(test_features, test_labels)
            
            # Report the accuracy and most informative features
            print "{}/{} Accuracy: {}".format(classifier_name, model_name, 
                accuracy)
            
            if hasattr(classifier, "predict_proba"):
                # Our classifier supports this
                # Get how often correct answer is in top 5
                # This is a test points by classes matrix of log probs
                predictions = classifier.predict_proba(test_features)
                
                # Count correct-within-top-n predictions
                num_correct = 0
                for prediction, label in itertools.izip(predictions, 
                    test_labels):
                    
                    # Now we have a vector of probs
                    # Make a list of (prob, class) tuples
                    prob_class_tuples = []
                    for classification, prob in enumerate(prediction):
                        prob_class_tuples.append((prob, classification))
                        
                    prob_class_tuples.sort(reverse=True)
                    
                    top_classes = [item[1] 
                        for item in prob_class_tuples[:options.top_predictions]]
                    
                    if label in top_classes:
                        num_correct += 1
                        
                # Print the more lenient score    
                print "{}/{} in top {}: {}".format(classifier_name, model_name, 
                    options.top_predictions, 
                    num_correct / float(len(predictions)))    
            else:
                print "Predicting probabilities not supported by {}".format(
                    classifier_name)   
                    
            sys.stdout.flush()
    
    print "Done!"        
    return 0
    
if __name__ == "__main__" :
    # No error catching because stack traces are important
    # And I don't want to mess around with a module to get them
    sys.exit(main(sys.argv)) 
