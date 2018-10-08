## Value:

  Market Cap: Stock price * # of stocks available

  Book Value: Tangible assests - intagible assests - liabilities

  Intrinsic Value: Dividends / discount rate


========================================
# Technical Analysis
    Historical price and volume only

    Compute statistics called indicators

    Rules of Thumb for using:

        1-Individual indicators are weak, use combinations (3-5)
        2-Look for contrasts in indicators (stock vs stock or stock vs market)
        3-Use over short time periods

## Indicators
    -Momentum: price / (price n days earlier) - 1

    -Simple moving average: n-day window of average
    SMA[t] = price[t] / price[t-n:t].mean() - 1
        Can be a proxy for value
        Graph SMA and see if it deviates from stock graph greatly
        Can combine with momentum and check where price graph crosses SMA graph

    -Bollinger Band
    Look for the crossing of price graph 'inside' BB

## Normalization
    Normalize indicators so one doesn't dominate the other (ex. momentum vs PE ratio)

    Done with: value - mean / value.std()

===============================
# Dealing with data
    -open
    -high
    -low
    -close
    -vol
    Each can apply to minutes, hours days etc

=========================
#Supervised Regression Learning
    Supervised: Provide examples: given x, expect y
    Regression: Numerical Prediction
    Learning: Provided Data

##Techniques
    -Linear regression (is parametric: i.e. it finds parameters)

    -k nearest neighbor (KNN) (instance-based)
        Saves all the data

    -decision tree

    -decision forest: A group of decision trees 

    Train on earlier data, test on later data. Never train and test on the same data.

##API Structure
    -Linear Regression ML must have:
        1) learner = LinRegLearner()
            A constructor that creates a learner instance
        2) learner.train(Xtrain, Ytrain)
            A method called train that takes training data to train
        3) y = learner.query(Xtest)
            A query function that takes a list of X values to test and returns the predicted y values
    
    -KNN
        1) learner = KNNLearner(k=3)
            A constructor that creates a learner instance.
            k = how many neighbors you want
        2) learner.train(Xtrain, Ytrain)
            A method called train that takes training data to train
        3) y = learner.query(Xtest)
            A query function that takes a list of X values to test and returns the predicted y values
    
    ex.:
    class LinRegLearner::
        def__init__():
            pass #Don't have to do anything for l.r.

        def train(x,y):
            self.m, self.b = favorite_linreg(x,y) # Use w/e l.r. algorith you want

        def query(x)
        y = self.m * x + self.b
        return y

===============================
#Assessing the Learning Algorithm
    -KNN results in a horizontal line before data begins and after it ends 
   
    -Decreasing K increases chance of overfit
     Increasing polynomial degree increases chance of overfit

    -Overfitting: The point where in-sampling error is decreasing but out-of-sample error is increaing

    -One method of evaluating is Root Mean Square: sqrt(sum of (ytest-ypredict)^2 / N)

    -Another method is correlation: np.corrcoef() (answer is between 1 and -1)

##Cross Validation
    - Can train on 60% of the data and test on 40%

    - If data set is small, you can split into 5 equal chunks. Train on the first 80%, test on 20%, then train on last 80%, test on first 20% etc for 5 trials in total.

=============================
#Ensembles 
    -Reduces bias

    -Bootstrap Aggregating Bagging:
        Train with 60% of the data. Randomly sample the data with replacement (i.e. each datum may be reselected into the same bag) into m bags for a total of n' samples. n is the number of data points from the training sector. In general n'<n, around 60% of n.

        Train each bag. Query each resulting model and take the mean of the resulting y's.
    
    -Ada Boost for Bagging:
        In subsequent baggings, weigh data points with errors more heavily depending on the magnitude

============================
#Markov Decision 
    -Set of states S
    -Set of actions A
    -Transition function T[s,a,s']
        -s' is the resulting state from a action in s state
        -Sum of probablities must equal 1
    -Reward function R[s,a]

    -The problem is to policy π(s) that will maximize reward. This is referred to as π*(s)
        -Done two ways: value iteration and policy iteration.

## Transition and Reward Function Don't Exist for Trading
    -Create an Experience Tupe: <s,a,s',r>
    -Repeat various times, making old s' the new s

    -From here there are the two possibilities for a solution:
        -Model-Based: Build model of T[s,a,s'] and R[s,a] from analyzing the list of tuples statistically. Then use value/policy iteration to solve the problem

        -Model-Free: Create a policy by directly looking at tuple list


===========================================
#Q-Learning
    -Q[s,a]
        -Q is the value of taking action a in state s
        -Equal to the immediate reward + discounted reward
        -Discounted reward is the reward from future actions
    
    -π(s) = argmaxa(Q[s,a])
        -π is the policy/action to take in state s
        -The function finds the a that maximizes the Q function.
        -We converge on the optimal solution π*(s), optimal Q table is Q*[s,a]
    
    Steps:
        1) Select training data
        2) Iteratate over time 
            a)Set starttime, initQ[]
            b) Compute s
            c)Select a
            d) Observe r,s'
            e)Update Q
        3) Test policy π
        4) Repeat until convergance (No more improvement on subsequent iterations)

##Update Rule

    -Q'[s,a] = (1-α)Q[s,a] + α * improved estimate
    -Q'[s,a] = (1-α)Q[s,a] + α * (r + λ * Q[s',argmaxa(Q[s',a'])])
        α: learning rate, ranges from 0 to 1.0. Set to 0.2
        λ: discount rate ranges from 0 to 1.0. Low = value future returns less
        r: reward

##Update Rule (Notes from Lecture)

    The formula for computing Q for any state-action pair <s, a>, given an  experience tuple <s, a, s', r>, is:
    Q'[s, a] = (1 - α) · Q[s, a] + α · (r + γ · Q[s', argmaxa'(Q[s', a'])])

    Here:

        r = R[s, a] is the immediate reward for taking action a in state s,
        γ ∈ [0, 1] (gamma) is the discount factor used to progressively reduce  the value of future rewards,
        s' is the resulting next state,
        argmaxa'(Q[s', a']) is the action that maximizes the Q-value among all  possible actions a' from s', and,
        α ∈ [0, 1] (alpha) is the learning rate used to vary the weight given to    new experiences compared with past Q-values.

##Finer Points
    -Success depends on exploraion
    -Choose random action with prob C
    -Set c = 0.3 at the beginning, decrease slowly to 0 over subsequent iteration.
