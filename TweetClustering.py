import csv, json, math, decimal, random, copy, sys, re

# A class that represents a tweet
class Tweet(object):

	# Required args for a node:
    #   id : id of the tweet
	#	text : the tweet text
    #   words : words in the tweet
    def __init__(self, args):
        self.id = str(args.get('id', ''))
    	self.text = args.get('text', '')
        # Get rid of re-tweets tags so that re-tweets have the same Jaccard distance
        self.text = re.sub('^(RT @.+?: )+', '', self.text)
        # Get rid of hash tags
        self.text = re.sub('#\w+', '', self.text)
        # Remove unwanted characters i.e anything that is not a letter, number or space 
        self.text = re.sub("[^\w ]", '', self.text)
        #print "%s : %s" % (self.id, self.text)
        self.words = self.text.split()

    # __repr__ is Python's internal function that gets called when printing
    # an object. Modifying this so that 'print point' prints a point
    # decision tree in the below format.
    def __repr__(self):
        return self.id + " : " + self.text

# A class that represents a cluster
class Cluster(object):
    # Centroid : A Tweet
    # Points : array of Tweet objects that are within the cluster
    def __init__(self, args):
        self.centroid = args.get('centroid', None)
        self.points = args.get('points', None)

    def setPoints(self, points):
        self.points = points

    def computeNewCentroid(self):
        # compute mean of x & y co-ordinates and make it the new centroid
        totalPoints = len(self.points)
        if totalPoints != 0:
            minSum, tweetWithMinSum = decimal.Decimal('Infinity'), None

            for i in range(totalPoints):
                distanceSum = 0.0
                tweetA = self.points[i]
                for j in range(totalPoints):
                    distanceSum += computeJaccardDistance(tweetA, self.points[j])

                if distanceSum < minSum:
                    minSum = distanceSum
                    tweetWithMinSum = tweetA

            self.centroid = tweetWithMinSum

    # Prints the cluster's points
    def __repr__(self):
        return ",".join( str(tweet.id) for tweet in self.points )

class KMeans(object):

    # Initializes the value of K, data points required for the 
    # clustering and the max iterations(to limit the computation if it doesn't converge by then)
    def __init__(self, args):
        self.K = int(args.get('K', 0))
        self.data = args.get('data', None)
        self.seedsfile = args.get('seedsfile', None)
        self.maxIterations = args.get('maxiterations', 25)
        self.convergenceCutOff = args.get('cutoff', 0.0)
        self.clusters = []

    # Runs the k-means algorithm on the given data points
    def run(self):
        # Choose random k points as centroids for the clusters
        self.loadCentroids()

        iterationCount, maxChange = 0, decimal.Decimal('Infinity')
        while self.hasConverged(iterationCount, maxChange) != True:

            # Create empty lists for each clusters so that we could
            # store their respective points
            clusterLists = [ [] for i in self.clusters ]

            # Iterate over all all points and assign it to the cluster
            # with minimum Euclidean distance
            for tweet in self.data:
                clusterIndex = -1
                minDistance = decimal.Decimal('Infinity')

                for i in range(len(self.clusters)):
                    distance = computeJaccardDistance(tweet, self.clusters[i].centroid)
                    if minDistance > distance:
                        minDistance = distance
                        clusterIndex = i

                if clusterIndex != -1:
                    clusterLists[clusterIndex].append(tweet)

            maxChange = 0.0
            # Update clusters with the new set of points and compute
            # the new centroid for them.
            for i in range(len(self.clusters)):
                self.clusters[i].setPoints(clusterLists[i])
                
                oldCentroid = self.clusters[i].centroid
                self.clusters[i].computeNewCentroid()

                maxChange = max( maxChange, computeJaccardDistance(oldCentroid, self.clusters[i].centroid) )

            iterationCount += 1

    # If there was change below or equal to the cutoff range specified or if the
    # number of iterations have reached maxIterations specified, this returns true
    # and false otherwise.
    def hasConverged(self, iterationCount, maxChange):
        if maxChange <= self.convergenceCutOff:
            #print "convergence achieved at %d iterations." % (iterationCount)
            return True

        if iterationCount > self.maxIterations:
            return True

        return False

    # Loads the centroids given in the seeds file as the initial
    # centroids for the 'K' clusters.
    def loadCentroids(self):
        tweetIDList = []
        for tweetID in open(self.seedsfile, 'r'):
            tweetID = re.sub('[\s,]+', '', tweetID)
            tweetIDList.append(tweetID)

        kSamples = random.sample(tweetIDList, self.K)
        for sample in kSamples:
            tCluster = Cluster({
                'centroid' : self.getData(sample)
            })
            self.clusters.append(tCluster)

    # Returns the Tweet object for the given tweetID
    def getData(self, tweetID):
        for tweet in self.data:
            if tweet.id == tweetID:
                return tweet

        return None

    # Prints clusters in the below format
    # Cluster-id    csv of points
    def printClusters(self, outputFileHandle=None):
        clusterIndex = 1
        for cluster in self.clusters:
            if outputFileHandle == None:
                #print "%d\t%s" % (clusterIndex, cluster)
                print "%d:" % clusterIndex
                for point in cluster.points:
                    print point.text
            else:
                outputFileHandle.write( "%d\t%s\n" % (clusterIndex, cluster) )
            clusterIndex += 1

    # Returns the Sum of squared error for the clusters
    def computeSSE(self):
        sse = 0.0
        for i in range(self.K):
            for point in self.clusters[i].points:
                sse += computeJaccardDistance(self.clusters[i].centroid, point)**2
        return sse

# Computes Jaccard distance between two given sets
def computeJaccardDistance(A, B):
    wordsOfA = set(A.words)
    wordsofB = set(B.words)
    return 1 - ( len(wordsOfA.intersection(wordsofB)) / float( len(wordsOfA.union(wordsofB)) ) ) 


# Helper Subroutines

# Reads a given data set file and returns a list of the data points
def readTweets(file):
    dataset = []
    for line in open(file, 'r'):
        data = json.loads(line)
        filteredData = dict( (key, value) for key, value in data.items() if key in ('text', 'id') )
        tweet = Tweet({
            'id' : filteredData['id'],
            'text' : filteredData['text'],
        })
        dataset.append(tweet)

    return dataset


# Main Subroutine

K, seedsfile, tweetsfile, outputfile = sys.argv[1:]

# Create output file
outputFileHandle = open(outputfile, 'w')

kmeans = KMeans({
    'K' : K,
    'data' : readTweets(tweetsfile),
    'seedsfile' : seedsfile
})

kmeans.run()
kmeans.printClusters(outputFileHandle)

# Print cluster stuff to stdout
#kmeans.printClusters()

# Print SSE to file
#outputFileHandle.write("SSE : %f\n"  % kmeans.computeSSE())

# Print SSE to stdout
print "SSE : %f"  % kmeans.computeSSE()

