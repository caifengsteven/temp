Quantitative Finance, 2022
Vol. 22, No. 2, 349–366, https://doi.org/10.1080/14697688.2021.1962539

Sparse index clones via the sorted (cid:2)1-Norm

PHILIPP J. KREMER †, DAMIAN BRZYSKI‡, MAŁGORZATA BOGDAN §¶ and
SANDRA PATERLINI *(cid:2)

†Department of Finance and Accounting, EBS Universität für Wirtschaft und Recht, Wiesbaden, Germany
‡Faculty of Pure and Applied Mathematics, Wroclaw University of Science and Technology, Wroclaw, Poland
§Department of Mathematics, University of Wroclaw, Wroclaw, Poland
¶Department of Statistics, Lund University, Lund, Sweden
(cid:2)Department of Economics and Management, University of Trento, Trento, Italy

(Received 10 December 2020; accepted 26 July 2021; published online 15 September 2021 )

Index tracking and hedge fund replication aim at cloning the return time series properties of a given
benchmark, by either using only a subset of its original constituents or by a set of risk factors. In
this paper, we propose a model that relies on the Sorted (cid:2)1 Penalized Estimator, called SLOPE,
for index tracking and hedge fund replication. We show that SLOPE is capable of not only provid-
ing sparsity, but also to form groups among assets depending on their partial correlation with the
index or the hedge fund return times series. The grouping structure can then be exploited to cre-
ate individual investment strategies that allow building portfolios with a smaller number of active
positions, but still comparable tracking properties. Considering equity index data and hedge fund
returns, we discuss the real-world properties of SLOPE based approaches with respect to state-of-the
art approaches.

Keywords: Index tracking; Hedge fund clones; Regularization; SLOPE

1. Introduction

Passive replication models are an increasingly popular
method to gain exposure to the risk- and return properties of a
given benchmark, which could either be a broad market index
or an alternative investment vehicle, like a hedge fund.

For replicating equity indices, such models are largely
rooted in the empirical evidence that it is not possible to
earn a return that is above the average market yield (see
i.e. Malkiel 1995, Sorenson et al. 1998, Frino and Gal-
lagher 2001). As such, the investor is best advised to just
mimic a broad market index, which he could achieve by
just buying all of its constituents in the respective relative
amounts. However, such a full replication strategy is not only
costly, as the manager needs to monitor and frequently rebal-
ance a high number of stocks, but is sometimes not even
possible, as small illiquid securities might not be allowed
to be traded in large volumes. Furthermore, managerial con-
straints might prevent the investor from gaining positions in
all constituents (Canakgoz and Beasley 2009, Chiam et al.
2013).

Hedge funds, on the other hand, have always attracted the
attention of investors, as they are outside of most regulatory

∗Corresponding author. Email: sandra.paterlini@unitn.it

frameworks, which enables them to create complex and
dynamic trading strategies. As a result, these investments
show low and even negative correlations with traditional asset
classes, like equities or bonds, and are hence of special interest
to an investor that seeks risk diversiﬁcation. Still, investments
into hedge funds often demand an initial high endowment or
are subject to restrictions, i.e. the fund is not open to new
investors (Giamouridis and Paterlini 2010). Consequently, in
the case of a broad equity index or a given hedge fund, an
alternative approach for the investor is a replicating model
that constructs a sparse, i.e. with only a few active posi-
tions, and a stable, i.e. with a low turnover, portfolio that best
mimics the risk-and return distribution of the given bench-
mark. In what follows, we will refer to both approaches as
the ‘Index Tracking (IT)’ framework, and to the clone as the
‘tracking portfolio’. We explicitly distinguish between the two
approaches where necessary.

To construct a tracking portfolio, the literature has mainly
focused on minimizing a so called tracking error measure, for
example given by the squared deviations between the bench-
mark returns and its constituents (Rudolf et al. 1999). The
underlying idea is that the benchmark can be represented by a
linear combination of either the equity index constituents or,

© 2021 Informa UK Limited, trading as Taylor & Francis Group

350

P. J. Kremer et al.

for the hedge fund replication, by a set of chosen risk factors.†
Finding the optimal portfolio then reduces to a simple linear
regression framework, often solved by using ordinary least
squares (OLS). However, OLS usually leads to allocations
that have an exposure to all constituents of the replicating
universe, and is thus costly to implement (Giuzio et al. 2018).
An alternative way to obtain a sparse coefﬁcient vector is
then to impose a so called cardinality constraint, which lim-
its the number of assets in the tracking portfolio. However,
including this constraint leads to optimizations, which are NP
hard with local optima and discontinuous search spaces. The
investor then needs to rely on search heuristics to solve the IT
framework, and research in this area has focused on ﬁnding
new and more efﬁcient algorithms to solve such optimiza-
tion problems (see e.g. Gilli and Kellezi 2009, Gilli and
Winker 2009, Krink et al. 2009, Fastrich et al. 2014).

In light of dealing with estimation errors, regularization
methods, taking the form of a constraint on the norm‡ of
the weight vector, have found widespread attention in port-
folio selection in the last 10 years, to create sparse and
stable allocations (see e.g. Brodie et al. 2009, DeMiguel
et al. 2009, Fastrich et al. 2015).§ In the convex case, one of
the most prominent penalties is the Least Absolute Shrinkage
and Selection Operator (LASSO), introduced to the statistical
literature by Tibshirani 1996. The LASSO penalty consists
of an (cid:2)1-Norm, whose unit ball takes the shape of a cross-
polytope, with singularities at the coordinate axes. This makes
the penalty a desirable regularization technique, as it achieves
in a single step the two tasks of portfolio selection, which
are: (1) variable selection, i.e. choosing a subset of assets in
which to invest, and (2) parameter estimation, i.e. selecting
how much to invest in the chosen assets. With regard to index
replication, Brodie et al. (2009) state that it can be used to
create sparse replicas and at the same time account for trans-
action costs. Furthermore, Giamouridis and Paterlini (2010)
show that by adding the (cid:2)1-Norm to the IT model, it is possible
to construct sparse and stable hedge fund clones.

it suffers from various disadvantages,

Although the LASSO penalty has gained widespread atten-
tion,
including a
reduced recovery of sparse signals when applied to highly
dependent data (Fan and Li 2001), selecting at random from
equally correlated assets (Bondell and Reich 2008), and being
stuck in the no short-sale area, given an imposed budget con-
j=1 wj = 1, where wj represents the jth asset
straint (i.e.
weight, see DeMiguel et al. 2009). The latter is of special

(cid:2)

K

† Alternative methods for benchmark replication include, e.g.
moment matching or payoff distribution approaches that aim at
cloning the return distribution of a given index or hedge fund, by ﬁrst
replicating its quantiles, then pricing a payoff function and ﬁnally
converting it into a portfolio holding strategy. As we focus on the lin-
ear case, we do not employ these methods here and refer to Amenc
et al. (2010) for an overview.
‡ Deﬁne the (cid:2)q-Norm as: (cid:2)w(cid:2)q = (
If q = 1, then (cid:2)1 =
(cid:2)
k
(cid:2)w(cid:2)2 =
i=1 w2
but a quasi norm.
§ There is an extensive literature dealing with estimation errors in
the mean-variance setting, which Brodie et al. (2009) show can
be rewritten as a regression problem. The interested reader is also
referred to the works of Branger et al. (2019), Golosnoy et al. (2019),
or Mainik et al. (2015).

|wi|q) 1
q , with 0 < q ≤ 1.
|wi| (LASSO), while for q = 2 we have
i (RIDGE). Note that (cid:2)q with 0 < q < 1 is not a norm

(cid:2)
k
i=1

k
i=1

(cid:2)

interest when replicating equity indices, as for such optimiza-
tions a long-only constraint (i.e. wi ≥ 0, where wi is the ith
asset weight) is typically considered.

To circumvent some of these problems the literature has
focused on non-convex penalties like the Logarithmic Penalty
(LOG), Smoothly Clipped Absolute Deviation (SCAD) or
the (cid:2)q -Norm, such as in Fastrich et al. (2014) and Giuzio
et al. (2018).¶ While non-convex penalties are able to produce
clones with a smaller number of active positions than LASSO,
they suffer from serious numerical issues. It turns out that
state-of-the-art interior point or Coordinate Descend (CoDe)
methods easily get stuck in local optima and the investor
is better advised to resort to heuristic optimization methods
(Giuzio 2017). However, none of these methods guarantee
convergence to the global minimum.

Given the shortcomings of LASSO and the optimization
burdens of non-convex approaches, we aim to extend the
literature on IT in the following ways:

First, we introduce the Sorted (cid:2)1 Penalized Estimator,
called SLOPE, to the IT framework. SLOPE was recently
proposed in statistics by Bogdan et al. (2013) and Bogdan
et al. (2015) and its penalty takes the form of a sorted (cid:2)1 -
Norm, using a sequence of tuning parameters and penalizing
the assets according to their rank magnitude. At the same time,
the SLOPE penalty is still convex and singular at the coordi-
nate axes and thus promotes sparse solutions. Furthermore,
SLOPE’s unit ball takes the form of a polyhedron, which
allows for additional reduction of dimension, by assigning the
same weights to constituents with similar partial correlations
with the time series of benchmark returns. Early investiga-
tions of SLOPE mostly focused on the control of the False
Discovery Rate for orthogonal and independent predictors
(see e.g. Bogdan et al. 2013, 2015, Brzyski et al. 2018) and the
estimation and prediction properties in the context of sparse
multiple regression (see e.g. Su and Candès 2016, Bellec
et al. 2016, 2018), where it was shown that for some speciﬁc
decreasing sequences of tuning parameters SLOPE is asymp-
totically minimax in a variety of settings. The clustering prop-
erties of SLOPE and its predecessor, the Octangle Shrinkage
and Selection Operator (OSCAR, Bondell and Reich 2008),
are discussed for example in Bondell and Reich (2008) and
Figueiredo and Nowak (2014), who show that these meth-
ods are able to cluster strongly correlated features. Kremer
et al. (2020) apply, to our knowledge as one of the ﬁrst,
the SLOPE penalty in a mean-variance framework, provid-
ing a simulation study and extensive empirical evidence using
equity data. The authors conclude that SLOPE is able to span
the entire efﬁcient frontier, while at the same time exploiting
grouping structures among assets, to reduce overall turnover
and to provide improved risk- and weight diversiﬁcation
measures, relative to other state-of-the-art approaches.

In this article, we provide new theoretical results (see theo-
rems 2.1 and 2.2), which show that the clustering provided by
SLOPE is not merely due to the strong correlation between the
assets, but it is driven by the similarity of partial correlations
of different assets with the respective index or dependent vari-
able. Subsequently, we illustrate these theoretical properties

¶ Please refer to Appendix 1 for a description of the LOG and SCAD
penalties.

Sparse index clones via the sorted (cid:2)1-Norm

1(cid:6)w = 1,

351

(1)

in a simulated environment, in which assets belonging to one
group exhibit the same partial correlation with the dependent
variable, while being uncorrelated to the remaining universe.
In such situations, SLOPE is indeed able to identify and
assign the assets to their respective groups and is further able
to outperform LASSO with regard to Mean Squared Error
(MSE). Furthermore, the procedure assigns higher weights
to the groups of assets with larger partial correlation to the
dependent variable. Second, we consider real-world data and
use the grouping feature to construct tracking strategies, based
on selecting only the most important groups of approxi-
mately equally important constituents. The resulting analysis
conﬁrms that our theoretical results hold in real-life. The con-
sidered investment strategies are outlined in more detail in
section 4.1.

Our empirical analysis considers a rolling window
scheme and compares SLOPE to other convex and non-
convex penalties, including LASSO, LOG and SCAD. We
the equity
thereby aim to replicate, on the one hand,
indices of the S&P 100 (SP100), S&P 200 (SP200) and
S&P 500 (SP500), and on the other, 26 hedge fund
indices.

Our empirical results show that by using the grouping prop-
erties of the SLOPE penalty, we can create tracking strategies
that lead to sparse replicating portfolios with low turnover
and improved tracking statistics, which frequently outperform
strategies based on the non-convex SCAD and LOG penalties.
Speciﬁcally, we show that by selecting the relevant groups
based on a maximum value of the average partial correla-
tion with the respective index, the resulting allocation often
leads to lower tracking error volatility (TEV) and compara-
ble sparsity with regard to current state-of-art regularization
techniques.

The paper is structured as follows: section 1 introduces
the sparse index tracking model and the theoretical proper-
ties of the SLOPE penalty. Section 2 provides insights on
the theoretical properties within a simulated environment,
while section 3 presents the empirical results on real-world
data. Section 4 concludes. All proofs are reported in
Appendix 2.

2. Sparse index tracking models

1, rj

2, . . . rj

In IT, the investor wants to create a sparse replicating portfolio
that best mimics the return time series of a given benchmark.
Let Y = [Y1, Y2, . . . , YT ](cid:6) be the T × 1 vector of benchmark
returns, and R = [r1, . . . .rK] be theT × K return matrix of
benchmark constituents, with columns rj = [rj
T ](cid:6),
representing the T × 1 vector of time series returns for the
jth asset. Then, the investor wants to ﬁnd the optimal K × 1
weight vector w = [w1, . . . wK](cid:6) for the replicating portfo-
lio that minimizes the TEV, given by the squared 2-norm
((cid:2) · (cid:2)2
2) of the difference of the benchmark and the IT portfolio
returns. The weight vector w is assumed to be sparse—with
not all K weights different from zero. To introduce sparsity
in IT, we consider a regression framework and minimize the
following penalized tracking error measure, together with the
budget constraint:

arg minw∈RK

(cid:2)Y − Rw(cid:2)2
2

+ ρλ(w)

where ρλ(w) is a penalty function, whose intensity is con-
trolled by a tuning parameter λ and 1(cid:6) is a K × 1 unit vector.
Following, we shortly outline the LASSO penalty, before
introducing our new version of the Sorted (cid:2)1-Norm Penalized
Estimator (SLOPE). A detailed description of the remain-
ing considered penalties for comparison, can be found in
Appendix 1.

2.1. Least absolute shrinkage and selection operator
The combination of the (cid:2)2 loss function with the (cid:2)1-Norm
penalty:

ρλ(w) = λ × (cid:2)w(cid:2)1 = λ ×

K(cid:3)

i=1

|wi|

(2)

was ﬁrstly used in Santosa and Symes (1986) in the context of
signal processing. Then the procedure was reinvented in Chen
and Donoho (1994) and Tibshirani (1996) and introduced to
the statistical literature as LASSO (Tibshirani 1996).

The (cid:2)1 penalty is convex and singular at the origin, and thus
promotes sparsity in the portfolio. The penalty has already
found widespread attention in index tracking (see e.g. Brodie
et al. 2009, Giamouridis and Paterlini 2010, Giuzio
et al. 2018) and portfolio selection (see e.g. Brodie
et al. 2009, DeMiguel et al. 2009, Fan et al. 2012, Car-
rasco and Noumon 2012, Yen and Yen 2014, Fastrich
et al. 2015, Yen 2015). Due to its computational simplicity
it is typically considered a benchmark, against which new
regularization methods are tested (Xing et al. 2014).

From an economic perspective, DeMiguel et al. (2009)
show that given the budget constraint, the LASSO regulates
the total amount of short-sales in the portfolio, whereas this
amount can arbitrarily be distributed across all active com-
ponents, as opposed to being individually assigned to some
weights. The choice, of which assets are ﬁnally penalized or
sold short, however, depends on the underlying dependence
structure (Fastrich et al. 2015).

Still, the LASSO suffers from known shortcomings, such as
selecting at random from equally correlated assets (Bondell
and Reich 2008), of being biased for large coefﬁcient val-
ues, especially in the presence of multicollinearity (Fan and
Li 2001, Giuzio and Paterlini 2018), and of being ineffective
in the no-short-sale area, i.e. wi ≥ 0 (DeMiguel et al. 2009).

2.2. Sorted (cid:2)1 penalized estimator
In this paper, we introduce the Sorted (cid:2)1 Penalized Estimator
(SLOPE) to the IT framework. SLOPE is a convex optimiza-
tion procedure, which, contrary to LASSO, groups correlated
assets, instead of selecting from them at random. Moreover,
and again contrary to LASSO, SLOPE is active in the no-
short-sale area. The SLOPE penalty takes the form of a sorted
(cid:2)1-Norm such that:

ρλ(w) =

K(cid:3)

i=1

λi|w|(i) = λ1|w|(1) + λ2|w|(2) + · · · + λK|w|(K)

352

P. J. Kremer et al.

Figure 1. Shapes of SLOPE for a three asset universe.
Notes: The ﬁgure plots from left to right, the shapes of unit balls for different SLOPE penalties for a three asset universe, and when
considering different setups for the sequence of lambda parameters (λ). This includes in panel (a) the shape of the (cid:2)∞-Norm, with λ = [2 0 0],
in panel (b) the shape of the LASSO, with λ = [2 2 2], and in panel (c) the shape of SLOPE with λ = [3 2 1]. Point A (Point B) in panel
(b) (c) represents a solution in which w1 = w2, and for which SLOPE groups assets in two groups, depending on the value of wi. among the
assets in the universe.

s.t. λ1 ≥ λ2 ≥ · · ·λ K ≥ 0 and |w|(1) ≥ |w|(2) ≥ · · · |w|(K)
(3)

and assume that ˆw is an unique solution to SLOPE optimiza-
tion problem,

where λSLOPE = [λ1, λ2, . . . , λK] is a non-increasing sequence
of tuning parameters and |w|(i) denotes the ith largest entry of
the weight vector w in absolute value. Note that for λ1 = λ2 =
· · · = λK, SLOPE is equivalent to LASSO.

SLOPE requires to choose a sequence of decreasing λ
parameters, instead of a single one, such as for LASSO, LOG
and many other penalties. Depending on the tuning parameter
vector, SLOPE penalties can take many different forms.

Panel (a) in ﬁgure 1 shows that when we choose λ1 >
λ2 = λ3 = 0, the unit ball of SLOPE takes the shape of a
cube, corresponding to the (cid:2)∞-Norm. This norm promotes
the grouping of assets, i.e. it encourages solutions when two
or more assets have the same coefﬁcient value. For exam-
ple, let w = [0.2 0.6 0.2](cid:6) be the estimated coefﬁcient vector,
then we have three different coefﬁcients, but only two groups,
as the value of 0.2 is assigned to two assets. On the other
hand, choosing the same value for each lambda parameter, i.e.
λ1 = λ2 = λ3 > 0, leads to the well known diamond shape of
the unit ball of LASSO, given in panel (b) of ﬁgure 1. Here
point A represents a singular point of the LASSO octahedron,
in which the portfolio allocation is sparse, with w1 and w3
being equal to zero.

Finally, considering a decreasing sequence of lambda
parameters λ1 > λ2 > λ3 > 0, leads to the shape of SLOPE,
which in panel (c) takes the form of a regular polyhedron in
the three dimensional space. Hence, the penalty combines the
properties of the (cid:2)∞- and (cid:2)1-Norm, thereby promoting sparse
solutions and grouping of assets. In panel (c), this feature is
highlighted by point B, in which w1 = w2 and w3 = 0. In fact,
as the following theorem 2.1 shows, the number of different
non-zero coefﬁcients depends on the rank of the return matrix.
Theorem 2.1 Let R be T × K matrix of benchmark con-
stituents, Y be T-dimensional vector of benchmark returns

arg min
w

(cid:4)
(cid:4)Y − Rw

(cid:4)
(cid:4)2
2

1
2

+ ρλ(w),

(4)

with a non-increasing sequence λ1 ≥ · · · ≥ λK ≥ 0. If R is of
rank k ≤ K, then the vector | ˆw| contains at most k different
non-zero values.

Remark 2.1 In some very rare situations the objective func-
tion of SLOPE might not be strictly convex and there might
exist inﬁnitely many solutions to the SLOPE optimization
problem, with the same value of the objective function. In
this case at least one of these solutions satisﬁes the thesis of
theorem 2.1.

Kremer et al. (2020) conduct an extensive simulation study,
showing that SLOPE is able to identify assets, with the same
underlying risk factor exposure, out of a large investment uni-
verse, and assigns the same coefﬁcient value to them. This
is different to the LASSO, which selects the active weights
at random from equally correlated assets. Here, we further
formalize and generalize the empirical observations made by
Kremer et al. (2020). Theorem 2.2 characterizes the SLOPE
property of generating groups of assets by revealing its con-
nection with λ sequence and correlations between the columns
of R and residuals.
Theorem 2.2 Let ˆw be the SLOPE estimate for T × K matrix
of benchmark constituents, R, and T-dimensional vector of
benchmark returns, Y . Moreover, suppose that all columns of
R have been standardized to the same norm, namely it holds
d := (cid:2)R1(cid:2) = · · · = (cid:2)RK(cid:2) and that the solution satisﬁes ˆw1 ≥
· · · ≥ ˆwK ≥ 0 (this can always be achieved by permuting
columns of R and changing their signs).

Then, for any i ∈ {1, . . . , K − 1}, it holds

ˆwi > ˆwi+1 =⇒ RT

i rP − RT

i+1rP ≥ λi − λi+1, where

rP : = Y − R\i,i+1 ˆw\i,i+1

(5)

Sparse index clones via the sorted (cid:2)1-Norm

353

and R\i,i+1 and ˆw\i,i+1 are obtained by removing ith and
i + 1st columns of R and elements of ˆw.

The quantity RT

i rP and RT

i rP is similar to the classical notion of partial
correlation. It measures the correlation between the ith asset
and the residual, obtained by subtracting from the vector of
benchmark returns, Y, the SLOPE’s prediction based on assets
other than Ri and Ri+1. Similarity of RT
i+1rP suggests
that these two assets have a similar impact on Y when all other
assets are included in the model. Thus, theorem 2.2 says that
assets having a similar prediction power with respect to the
index are grouped together. This grouping feature of SLOPE
will enable the investor to incorporate views into the port-
folio selection process, by choosing only the most important
groups of assets to track the respective index and assigning
the same weights to the assets from the same group. This
approach allows to obtain sparse and stable predictive models
and will be used in section 2 to create the new index tracking
strategy, SLOPE-SLC (see section 4.1).

3. Simulation analysis

Our simulation aims to illustrate the theoretical insights from
section 1 by investigating the selection and tracking behav-
ior of SLOPE when the underlying data generating process
exhibits a grouping property. Consequently, we analyse, if
SLOPE is able to create a tracking portfolio, with the num-
ber of groups not larger than the rank of the design matrix
(as stated by theorem 2.1), while assigning coefﬁcient val-
ues according to theorem 2.2. The latter would result in the
same coefﬁcients for assets belonging to the same group and
larger coefﬁcient values for those assets, with a larger partial
correlation with the dependent variable.

The simulation design follows the set-up from section 1, in
which the vector of optimal portfolio weights w is obtained,
by solving the regression problem given in (1). Our simulation
assumes that the rows of the T × K return matrix R is obtained
by considering an underlying risk factor structure, in which
assets belonging to one group have the same exposure to a
subset of the factors in the universe. In detail, assume that the
universe consists of a total of S risk factors and each asset is
represented by a linear combination of those factors. Further-
more, let T be the number of observations, K be the number
of assets, and FT×S = [f 1 f 2
. . . f S], where f i is the T × 1
vector of returns of the ith risk factor. Moreover, let BS×K be
the loading matrix for the individual risk factors. Then, the
T × K matrix of asset returns from the Hidden Factor Model
(i.e. R) can be constructed as:

R = F × B + (cid:4)

(6)

where (cid:4) is a T × K matrix of normally distributed error
terms. In what follows, we consider the following values for
generating the return matrix R :

• T = 500, K = 99, S = 3,
• the risk factors f1, . . . , fS are independent from the
multivariate standard normal N(0, IS×S) distribu-
tion, with IS×S being an identity matrix,

• the vectors of error terms (cid:5)i, i = 1, . . . , K, for each
asset are independent from each other, as well as
from each of the risk factors and come from the
multivariate normal distribution N(0, 0.05 × IK×K)
• the loadings matrix BS×K is made of exactly
33 copies of each of the following columns:
[0.77 0.64 0](cid:6), [0.9 0 0.42](cid:6) and [0 0.31 0.64](cid:6).†
• each column of R is normalized to have norm equal

to one.

Finally, given R, the data generating process for the index

that we aim to replicate is as followed:

Y = Rw + ν

(7)

where ν is a T × 1 matrix of normally distributed error terms,
i.e. ν ∼ σ × N(0, 1), with σ = 0.0015.

To obtain further insights into the grouping and tracking

ability of our SLOPE parameter, we consider two scenarios:

• Scenario 1: w is chosen such that the assets belong-
ing to group 2 have coefﬁcients 2, those belonging
to group 3 have coefﬁcients 3, and zero otherwise,
i.e.

w = [0 0 · · · 0 0
(cid:8)

(cid:5)

(cid:6)(cid:7)
Group1

2 2 · · · 2 2
(cid:5)
(cid:8)
(cid:6)(cid:7)
Group2

](cid:6)
3 3 · · · 3 3
(cid:8)
(cid:6)(cid:7)
(cid:5)
Group3

• Scenario 2: w is chosen, such that all assets in
group one have a coefﬁcients of zero, all assets
from group 3 have a coefﬁcients of 3, while the ﬁrst
half of assets belonging to group 2 have a coefﬁ-
cients 1 and the other half of assets from group 2
have values equal to 2. That is:

1 1 · · · 1 1
(cid:5)
(cid:8)
(cid:6)(cid:7)
PartI−Group2

2 2 · · · 2 2
(cid:5)
(cid:8)
(cid:6)(cid:7)
PartII−Group2

w = [0 0 · · · 0 0
(cid:8)

(cid:5)

(cid:6)(cid:7)
Group1
](cid:6)
3 3 · · · 3 3
(cid:5)
(cid:8)
(cid:6)(cid:7)
Group3

To understand, why we have chosen the weight vectors
according to scenarios 1 and 2, it is ﬁrst important to note that
the solution to the SLOPE index tracking problem depends on
choosing a sequence of tuning parameters that trade-off the
minimum tracking error volatility and the number of active
weights. For our simulations, we follow Bogdan et al. (2013)
and choose each component of the sequence of tuning param-
eters, by setting λi = α(cid:9)−1(1 − qi), ∀i = 1, . . . , k, where (cid:9)
is the standard normal cumulative distribution function and
where qi = i × θ/2k, with θ = 0.1, regulates how fast the
sequence of lambda parameters is decreasing. This choice of
the sequence of the tuning parameters is motivated by the the-
oretical results on the asymptotic optimality of the related
versions of SLOPE reported in Su and Candès (2016) and
Bellec et al. (2018) and by the empirical results illustrating
their superior prediction properties reported in Bogdan and
Frommlet (2021). According to the asymptotic results of Su

† These underlying factor exposures have been chosen speciﬁcally to
model a block correlation structure among the resulting assets, while
ensuring that the determinant of the covariance matrix is non-zero.

354

P. J. Kremer et al.

Figure 2. Partial correlation matrix and lambda sequence. (a) Lambda Sequences. (b) Scenario 1 and (c) Scenario 2.
Notes: The ﬁgure plots in panel
(a)
the matrix of absolute differences between SLOPE ‘partial correlations’ of
i rP − RT
M = |RT
and scenario 2, respectively.

lambda parameters, while panels (b) and (c) show
the quantity
i+1rP|, whererP := Y − R\i,i+1 ˆw\i,i+1, for lambda sequence number seven, and when we choose w according to scenario 1

the 12 different sequences of

the respective pairs of assets,

i.e.

and Candès (2016) and Bellec et al. (2018), α should be
slightly larger than σ , while in Bogdan and Frommlet (2021)
it is observed that for real-life data sizes it is usually bene-
ﬁcial to select α < σ . In our simulation study σ = 0.0015,
therefore we vary α on a grid of 12 log-spaced points between
10−3.5 ∼ 0.0003 and 10−1.7 ∼ 0.02. Finally, we consider as a
starting point, a sequence with no difference in consecutive
lambda parameters, i.e. one in which λ1 = λ2 = · · · = λK,
the SLOPE penalty is then equal to the LASSO. Panel (a) of
ﬁgure 2 shows the resulting 12 different lambda sequences.

Given our chosen set-up and the loading matrix BS×K, we
explicitly model a three group block correlation structure for
the return matrix R, in which assets are allocated in groups,
and those belonging to the same group are exposed to exactly
two out of the three risk factors. Consequently, assets from the
same group are not only highly correlated with each other, but
maintaining a low correlation to all other assets. Figure 2(a)
shows the resulting block correlation matrix among the assets.
Furthermore, given the choice of the weight vector w, each
group has a speciﬁc partial correlation with the index. To illus-
trate this characteristic and for each of the two scenarios of w,
ﬁgure 2(b,c) plots the quantity M given as:

M = |RT

i rP − RT

i+1rP|, where rP := Y − R\i,i+1 ˆw\i,i+1

(8)
As explained in theorem 2.2, the quantity RT
i rP is similar
to the classical notion of partial correlation. Consequently,
M measures the absolute difference of partial correlations
between the assets: the lower the difference, the more sim-
ilar are the assets in terms of partial correlation with the
index. Panels (b) and (c) of ﬁgure 2 thus show that, given
Lambda Sequence Number 7, we have chosen the values of
w for scenarios 1 and 2 such that we explicitly model a par-
tial correlation block structure with the index, whereas assets
belonging to the same group having the same partial corre-
lation with the index and thus a value of M, which is equal
to zero.† On the other hand, assets belonging to different
groups, exhibit a non-zero value of M. Moreover, in our sec-
ond scenario we have chosen w, in such a way that we further

differentiate between the assets from the second group. In
fact, half of the assets in group 2 have asset weights equal
to 1 and the other half equal to 2. That is, as depicted in
ﬁgure 2(c), we create a subclass of assets belonging to group
2 with regard to partial correlation and the obtained values
of M for those assets. From our insights of theorem 2.2, we
thus expect SLOPE to not only group assets with small differ-
ences in partial correlation, but also assign higher weights to
those assets, for which the partial correlation with the index is
higher.

Considering scenario 1, we perform 1000 iterations and
report, for each of the 12 lambda sequences, the number of
non-zero coefﬁcient and the number of groups, as identiﬁed
by SLOPE. Furthermore, as a measure of effectiveness, we
also compute the Mean Squared Error (MSE) and the Mean
Squared Prediction Error (MSPE), given as:

MSE( ˆw) = E[(cid:2)w − ˆw(cid:2)2
2]

MSPE( ˆw) = E[(cid:2)Rw − R ˆw(cid:2)2
2]

(9)

(10)

=

(cid:2)

K
i=1

where w and ˆw are equal to the true and estimated portfo-
(wi − ˆwi)2.
lio weights, respectively, and (cid:2)w − ˆw(cid:2)2
2
Finally, we consider boxplots of group-based partial correla-
tions with the index. That is, over the 1000 iterations, and for
a given lambda sequence, we boxplot the partial correlations
of those assets which have been assigned the same coefﬁcient
value by SLOPE. Here, the SLOPE residual rP for ith vari-
able is calculated after eliminating all variables from the same
cluster.

For the set of lambda sequences (depicted on the x-axis),
ﬁgure 3 depicts our simulation results, including in panels
(a) and (b), the number of non-zero groups, the number of
non-zero coefﬁcients, as well as the MSE and the MSPE,
respectively. Furthermore panels (c) and (d) show for lambda
sequence number seven, i.e. a point that trades of a low
MSE and MSPE, the boxplots for scenario 1 and scenario 2,
respectively.‡

† These results also hold for the other lambda sequences, and are
available from the authors upon request. We here focus on lambda
sequence seven, as the higher tuning parameter sequence promotes
the grouping of coefﬁcients.

‡ For the sake of brevity, we do not report the results for the Number
of non-zero groups, the Number of SLOPE non-zero coefﬁcients, as
well as the MSE, and the MSPE for scenario 2 here, but make them
available upon request.

Sparse index clones via the sorted (cid:2)1-Norm

355

Figure 3. Simulation results. (a) Non-zero groups, (b) MSE and MSPE—scenario 1, (c) boxplots—scenario 1 and (d) boxplots—scenario 2.
Notes: The ﬁgure plots for the 12 different lambda sequences in panel (a) the number of non-zero coefﬁcients and the non-zero groups,
while in panel (b) the MSE and the MSPE. Furthermore, panels (c) and (d) depict, for the lambda sequence number seven, the boxplots of
the group based partial coefﬁcients, for those assets which have been assigned the same coefﬁcient value by SLOPE for scenarios 1 and 2,
respectively. To create the ﬁgures (c) and (d) we round the estimated coefﬁcients of SLOPE to one digit after the comma. All computations
are based on 1000 iterations.

From panels (a) and (b) of ﬁgure 3, we can observe that
as the difference among the consecutive lambda parameters
increases, SLOPE starts to form groups among the non-zero
coefﬁcients, while the MSE and MSPE are decreasing. Begin-
ning with no difference among the Lambda parameters, a
point in which SLOPE is equal to LASSO, solving prob-
lem (4) leads to the number of coefﬁcients being equal to the
number of groups. In fact, at this point we have no grouping
and the procedure results in a randomly sparse weight vector.
The latter is a typical behavior of LASSO. As we increase the
difference among consecutive lambda values (i.e. we move to
the right on the x-axis), SLOPE starts to form groups among
the non-zero coefﬁcients. The procedure starts to disentan-
gle the purposely modeled grouping structure, and the MSE
and MSPE decrease. As we continue to increase the gaps
between consecutive lambda parameters, the octagonal shape
of the penalty starts to have a detrimental impact on the esti-
mation, and starts to group all assets together. Consequently,
the MSE and MSPE increase. Looking at panels (c) and (d),
we boxplot the partial correlation of assets which has been
assigned the same coefﬁcient value across all 1000 iterations

and considering lambda sequence number seven. As the lat-
ter trades-off a low MSE and MSPE, we observe that SLOPE
forms groups exactly around the true coefﬁcient values, w,
considered in scenario 1. Moreover, in scenario 2, SLOPE not
only disentangles between the three block type structure that
we modeled for R, but also disentangles the inﬂuence that
each of the assets has on Y , which conﬁrms the theoretical
ﬁndings of theorems 2.1 and 2.2.

4. Empirical analysis

4.1. Set up and data

Our empirical analysis investigates the out-of-sample equity
index tracking and hedge fund replication ability of the newly
introduced SLOPE penalty and compares it to other state-
of-art regularization methods, such as LASSO, the SCAD
and the LOG penalties. Our ultimate goal is to construct a
sparse clone that best tracks the performance of the given

356

P. J. Kremer et al.

benchmark. In addition to the penalty functions, we add the
following constraints to the IT model in (1): (a) for track-
ing the equity indices, the constraint that the asset weights
are non-negative (i.e. wi ≥ 0, ∀ i = 1, . . . , K), and (b) for
replicating hedge fund returns, which result from complex
investment strategies including long and short positions, we
restrict the weights to be in the interval [−1, 1]. While the
equity index replication only considers long-only solutions,
we explicitly differentiate between SLOPE and SLOPE-
LO for the hedge fund replication, in which SLOPE-LO is
SLOPE, together with an added long-only constraint (i.e.
wi ≥ 0).

Furthermore, in both frameworks we implement a trading
strategy, SLOPE-SLC,† in which we utilize the grouping abil-
ity of SLOPE, by ﬁrst solving the IT problem with SLOPE
and then keeping only the most important groups of active
coefﬁcients. As theorem 2.2 shows, SLOPE groups assets
according to their partial correlation with the index. In our
search strategy we compute for each group the median par-
tial correlation of the constituents included and keep only
those groups active, which have a median partial correla-
tion value above the 75th percentile for the equity indices
and the 25th percentile for the hedge fund indices.‡ Then
we rescale SLOPE’s estimates so that the weights still sum
up to 1. The rescaling preserves the group structure, i.e.
the assets in the same group still have the same weights.
When solving the IT at each t for the hedge fund case, we
again distinguish between long-only and short-sales strate-
gies, leading to SLOPE-SLC, and SLOPE-LO-SLC, respec-
tively.§ Finally, and to guarantee that our clones can also
be implemented in practice, we impose a threshold and set
weights that are smaller in absolute value than 0.05%, to
zero.

For our equity index tracking analysis, we focus on recon-
structing the daily return observations of the SP100, SP200,
and SP500. The data is obtained from Datastream and cov-
ers the period from 31 December 2004 to 29 January 2016
(T = 2890 daily return observations). Stocks with missing
values are dropped from the dataset. Table 1 shows summary
statistics for the four equity indices, conﬁrming the typical
return time series characteristics of fat tails and light negative
asymmetry.

For our hedge fund replication study, we obtain the monthly
net of fee returns for 26 Hedge Fund Research (HFR) Indices,
over the period from 30 June 1994 to 31 July 2017 (T = 278
return observations). The data is obtained from the Hedge
Fund Research Database that is considered to be an indus-
try standard in benchmarking the performance of hedge fund
strategies.¶ The selected indices follow six broad hedge

† In the following, ‘SLC’ is used as an acronym for ‘select’.
‡ We choose a lower percentile value for the hedge fund indices, as
the considered universe is smaller, as compared to the equity indices.
For the former, using the 75th percentile would result in a too sparse
representations. We make these results available upon request.
§ Alternative investment strategies, exploiting the grouping prop-
erties could be set up. Here, we focus on one of the simplest
one.
¶ The data and more details on the index construction methodology
can be found at www.hfr.com.

fund strategy dimensions, including Fund of Fund-, Event
Driven-, Equity Hedge-, Emerging Markets-, Total Macro-
and Relative Value Strategies, as well as a Fund Weighted
Composite. The latter is a collection of over 2000 individ-
ual funds, excluding Fund of Funds, serving as an overall
benchmark for the hedge fund industry. Here table 2 reports
summary statistics for these seven broad hedge fund strat-
egy dimensions. A complete table with summary statistics for
all 26 different hedge fund indices is available in the online
appendix of the paper.

We can observe that all funds provide a positive return over
the considered period, whereas Equity Hedge and Emerging
Market Funds are among those strategies with the highest
return and risk. The high risk of those strategies is also
reﬂected in the highest minimum monthly return. Macro Total
strategies pose another interesting picture, as they show a rel-
atively small standard deviation and also the lowest minimum
return. This strategy dimension is also the only one, who has
a positive skewness. Finally, all strategies show a leptokurtic
distribution.

To replicate the performance of the chosen indices, we
select a total of 17 risk factors across ﬁve asset classes.
The monthly returns for the style factors are obtained from
Bloomberg and cover the same period as the hedge fund
indices, which is from 30 June 1994 to 31 July 2017. While
the selection of the replicating factors in the index track-
ing problem is naturally the constituents of the index, the
selection of factors for hedge fund replication is a crucial
step. Our chosen factors represent a subset of those uti-
lized by Giuzio et al. (2018), and which ultimately have
been selected based on the insights from previous replication
studies.

Panel (a) of table 3 provides an overview of the 17 style
factors, while panel (b) displays the associated correlation
matrix. From the matrix, we observe that factors in the equity
asset class are highly correlated, while the remaining con-
stituents have low and even up to negative correlation values.
Given that SLOPE is able to group assets with similar partial
correlations with the index, we might be able to identify those
important assets and to further modify the solution according
to any of the strategies introduced above.

To investigate the performance of our regularized repli-
cation strategies, we employ a rolling window approach,
considering a window size of τ = 750 daily (τ = 60 monthly)
observations for the index tracking (hedge fund replication)
problem. The rolling window approach works as followed: at
time t we use the ﬁrst τ observations of the benchmark and the
replicating constituents to estimate the weights wt, by mini-
mizing the in-sample tracking error subject to the respective
penalty and budget constraints. Given our optimal weights,
we then compute the out-of-sample excess return between
the benchmark and the tracking portfolio as: (Y t+1 − Rt+1wt).
Finally, we roll the estimation window forward, dropping the
last and adding the most recent n observations. The process
is then repeated until we reach the end of the data set. For
our daily return observations for the S&P Indices, we choose
n = 21, such that we rebalance the portfolio monthly and
obtain a total of M = 102 out-of-sample observations. For

Sparse index clones via the sorted (cid:2)1-Norm

Table 1. Descriptive statistics for S&P indices.

357

Index

k

ˆμ (%)

ˆσ (%) (cid:9)med (%) (cid:9)min (%) (cid:9)max (%) (cid:2)skew

(cid:9)kurt

SP100
SP200
SP500

93
134
443

0.023
0.007
0.023

1.30
1.10
1.40

0.047
0.009
0.046

− 9.80
− 8.70
− 10.70

11.60
5.60
10.90

− 0.240
− 0.427
− 0.418

14.816
7.839
13.234

Notes: The table reports descriptive summary statistics for the S&P 100, the S&P 200, and the
S&P 500 data set, respectively. Reported are the number of constituents (k), the daily mean ( ˆμ),
the daily standard deviation ( ˆσ ), the daily median ( (cid:9)med), the daily minimum ((cid:9)min), the daily
maximum ( (cid:9)max), the skewness ((cid:2)skew) and the kurtosis ((cid:9)kurt).

Table 2. Summary statistics for hedge fund strategy dimensions.

HFRI index

ˆμ (%)

ˆσ (%) (cid:9)med (%) (cid:9)min (%) (cid:9)max (%) (cid:2)skew (cid:9)kurt

Fund weighted composite
Fund of funds composite
Event driven
Equity hedge
Emerging markets
Macro total
Relative value total

0.67
0.44
0.76
0.79
0.70
0.59
0.63

1.92
1.60
1.87
2.54
3.78
1.73
1.17

7.65
6.85
5.13
10.88
14.80
6.82
3.93

− 8.70
− 7.47
− 8.90
− 9.46
− 21.02
− 3.77
− 8.03

0.80
0.65
1.01
0.92
1.24
0.42
0.76

− 0.59
− 0.69
− 1.21
− 0.22
− 0.91
0.59
− 2.75

5.99
7.49
7.16
5.31
7.61
3.91
19.02

Notes: The table reports descriptive summary statistics for the seven hedge fund index strategy dimen-
sions, respectively. Reported are, the monthly mean ( ˆμ (%)), the monthly standard deviation ( ˆσ (%)),
the monthly median ( (cid:9)med (%)), the monthly minimum ((cid:9)min (%)), the monthly maximum ( (cid:9)max (%)), the
skewness ((cid:2)skew) and the kurtosis ((cid:9)kurt). The values are computed considering the complete time horizon
from 30 June 1994 to 31 July 2017.

Table 3. Overview of style factors.

Notes: Panel (a) provides an overview of the 17 style factors. Panel (b) shows the correlation between the style factors returns
in the period from June 1994 to July 2017. Numbering at the axes corresponds to the numbers in panel (a).

our monthly hedge fund returns, we also rebalance our port-
folio monthly and choose n = 1 accordingly, leaving us with
a total of M = 218 out-of-sample observations.†

As for our simulation study, we again need to choose a tun-
ing parameter that trades-off minimum tracking error and the
number of active weights, and thus inﬂuences the solution
to the constrained tracking error problem. In our empirical

† Hedge funds only report their returns on a monthly basis. Further-
more, equity indices only change their composition at the end of a
month, at which time stocks are either removed or added to the index.
To account for these to effects, and to also ensure a consistent treat-
ment of the two datasets, we choose a monthly rebalancing frequency
for both of them.

analysis, we consider for all our penalty functions, a grid of
100 linearly spaced lambda values, while again in the case
of SLOPE, these values represent the starting point, λSlope,1,
of the decreasing sequence of lambda parameters. As before,
we follow Bogdan et al. (2013) and choose each compo-
nent of the sequence of tuning parameters, by setting λi =
α(cid:9)−1(1 − qi), ∀i = 1, . . . , k, where (cid:9) is the standard normal
cumulative distribution function and where qi = i × q/2k,
with q = 0.1, regulates how fast the sequence of lambda
parameters is decreasing. For our empirical study, we vary
the scaling parameter α to consider a grid of starting points
λ1 = α(cid:9)−1(1 − q1), such that λ1 = λLASSO. That
the
is the same as for
shrinkage effect for the ﬁrst asset

is,

358

the LASSO, but
securities.

is subsequently lower

for

the k − 1

P. J. Kremer et al.

To select the optimal tuning parameter from the set of
possible values, the literature has resorted to either using
cross-validation techniques or information criteria, like the
Akaike Information Criteria (AIC) or the Bayesian Informa-
tion Criteria (BIC). In this study, we focus on a criterion,
inspired by the BIC, to balance the trade-off between the
tracking error volatility and the number of active weights and
to select the optimal tuning parameters. Using such criteria
is preferred to e.g. cross-validation procedures, as it has a
lower computational burden (Hastie et al. 2001). Therefore,
we choose the optimal tuning parameter that minimizes:
(cid:10) (cid:2)τ

(cid:11)

SC = −2 × log

t=1

(Y t+1 − Rt+1 ˆwt)2

τ

+ log(τ ) ×

k(cid:3)

i

1(wi (cid:12)= 0)

(11)

where 1 represents the indicator function and τ is the win-
dowsize. After obtaining our optimal lambda parameter, we
evaluate the tracking ability of our clones, by computing the
out-of-sample TEV and tracking error (TE) given by:

TEV = 1
M

TE = 1
M

M(cid:3)

(Y t+1 − Rt+1 ˆwt)2

t=1
M(cid:3)

(Y t+1 − Rt+1 ˆwt)

t=1

(12)

(13)

where Y t+1 is the return of the benchmark and Rt+1 are
the returns of the benchmark constituents at time t + 1,
respectively, while ˆwt is the estimated weight vector obtained
from (1). For an ideal tracking portfolio both the tracking
error volatility and the tracking error should be close to zero
(Giuzio 2017).† Consequently, the optimization problem is set
up to minimize the tracking error volatility. Furthermore, we
compute the Information Ratio, given by the ratio of the track-
ing error to the tracking error volatility. As we are interested
in a sparse and cost efﬁcient replication of our benchmark, we
include statistics on the total number of active positions (AP)
and the average total turnover (TO), both given by:

AP = 1
M

TO = 1
M

M(cid:3)

K(cid:3)

t=1
M(cid:3)

i=1
K(cid:3)

t=1

i=1

1(wi,t (cid:12)= 0)

|wi,t−1 − wi,t|

(14)

(15)

As we want to create tracking portfolios that perform well
out-of-sample, we analyse the predictive abilities of our mod-
els, by following Gu et al. (2020) and compute the predictive
R2
OOS, as well as the out-of-sample Maximum Drawdown. The

† The tracking error can be both, positive and negative. A posi-
tive (negative) tracking error would indicate that the target index or
strategy outperforms (underperforms) the respective tracking clone.

predictive R2

OOS is given as:
(cid:2)τ

R2

OOS

= 1 −

t=1

(Y t+1 − Rt+1 ˆwt)2
(cid:2)τ

t=1 Y 2

t+1

,

(16)

whereas −∞ ≤ R2
Drawdown is computed as:

OOS

≤ 1. The out-of-sample Maximum

MaxDD = max

0≤t1≤t2≤τ

( ˆYt1

− ˆYt2

),

(17)

with ˆYti

= ( ˆwti−1Rti

) and ˆY0 = 100.

Finally, we also test for differences in the out-of-sample
predictive accuracy between two models, according to
Diebold and Mariano (1995), and calculate the test statis-
tic DM = ¯d
∼ N(0, 1), where dt+1 = (ˆe
)2, with
)2 − (ˆe
ˆσd
(cid:2)τ
(m)
) is the prediction error at time
(Y t+1 − Rt+1 ˆw
=
ˆe
t+1
t + 1 for model m, and ˆw
is the estimated weight vector
at time t of that model. Then, ¯d, and ˆσd denote the mean and
the Newey-West standard error of d over the testing period,
respectively.

(m)
t
(m)
t

(1)
t+1

(2)
t+1

t=1

We aim to obtain a portfolio with a low value of tracking
error volatility, which is ideally achieved by investing in a
small number of active positions. In this context, we do not
control for the turnover, whereas this could be improved by
using a method similar to Kremer et al. (2018), in which the
update of the portfolio weights is performed only in cases, in
which two consecutive window estimates of the asset’s corre-
lation or partial correlation matrices exhibit large variability.‡
In such case we believe the stability of the weight estimates
and thereby the turnover could be even further improved.

4.2. Empirical results

S&P Indices. Table 4 reports the SP100, SP200 and SP500
indices (by row), reporting, from top left to bottom right, the
annualized tracking error volatility, the annualized tracking
error, the Information Ratio, the number of active positions,
the turnover, the correlation between the replicating portfolio
and the given index, as well as the predicted R2 and the max-
imum drawdown. Furthermore, we use a t-test to investigate,
whether the IR is statistically signiﬁcantly different from zero
at the 1%, 5% and 10% level and report the signiﬁcance at
the IR values. Finally, we report the statistical signiﬁcance
for the Diebold and Mariano (1995) Test at the values for
the tracking error.§ All methods are reported in columns that
is, from left to right: LASSO, SLOPE-LO, SLOPE-LO-SLC,
LOG and SCAD.¶

As described in the previous section, we are interested in
constructing a sparse and cost efﬁcient clone that best reﬂects
the performance of the given benchmark. Therefore, the ideal
optimal strategy should be based on investing only in a sub-
set of the index constituents (i.e. a small number of active

‡ Kremer et al. 2018 restrict the rebalancing of the portfolio to
instances, in which a Chow-Test indicates that the covariance matrix
between rebalancing dates has signiﬁcantly changed.
§ The test statistic for the IR t-test is computed as: IR ×
¶ In the table, SLOPE-LO and SLOPE-LO-SLC are displayed as S-
LO and S-LO-SLC, respectively.

M − 1.

√

Sparse index clones via the sorted (cid:2)1-Norm

359

.
s
e
c
i
d
n
i
P
&
S
r
o
f

s
c
i
t
s
i
t
a
t
s

g
n
i
k
c
a
r
T

.
4

e
l
b
a
T

D
A
C
S

G
O
L

C
L
S
-
O
L
-
S

O
L
-
S

O
S
S
A
L

D
A
C
S

G
O
L

C
L
S
-
O
L
-
S

O
L
-
S

O
S
S
A
L

D
A
C
S

G
O
L

C
L
S
-
O
L
-
S

O
L
-
S

O
S
S
A
L

D
A
C
S

G
O
L

C
L
S
-
O
L
-
S

O
L
-
S

O
S
S
A
L

s
n
o
i
t
i
s
o
p

e
v
i
t
c
A

o
i
t
a
r

n
o
i
t
a
m
r
o
f
n
I

)

%
n
i
(

r
o
r
r
e

g
n
i
k
c
a
r
T

)

%
n
i
(

y
t
i
l
i
t
a
l
o
v

r
o
r
r
e

g
n
i
k
c
a
r
T

)

%
n
i
(

n
w
o
d
w
a
r
D
m
u
m
i
x
a
M

S
O
2O
R
d
e
t
c
i
d
e
r
P

n
o
i
t
a
l
e
r
r
o
C

)

%
n
i
(

r
e
v
o
n
r
u
T

D
A
C
S

G
O
L

C
L
S
-
O
L
-
S

O
L
-
S

O
S
S
A
L

D
A
C
S

G
O
L

C
L
S
-
O
L
-
S

O
L
-
S

O
S
S
A
L

D
A
C
S

G
O
L

C
L
S
-
O
L
-
S

O
L
-
S

O
S
S
A
L

D
A
C
S

G
O
L

C
L
S
-
O
L
-
S

O
L
-
S

O
S
S
A
L

0
5

3
3

4
7

8
4

5
2

5
7

8
2

6
2

6
6

4
8

4
0
1

2
6
2

3
8

3
0
1

8
3
2

∗
∗
∗

3
7
.
0

∗
∗
∗

5
1
.
1

∗
∗
∗

1
4
.
0

∗
∗
∗

4
5
.
0

∗
∗
∗

6
3
.
1

∗
∗
∗

1
6
.
0

∗
∗
∗

6
1
.
1

∗
∗
∗

8
2
.
1

∗
∗
∗

6
1
.
1

∗
∗
∗

2
0
7
.
0

∗
∗
∗

2
4
.
1

∗
∗
∗

2
1
.
1

∗
∗
∗

9
5
.
0

∗
∗
∗

2
4
.
1

∗
∗
∗

5
1
.
1

∗
∗
∗

5
6
.
2

∗
∗
∗

0
5
.
3

∗
∗
∗

5
9
.
5

∗
∗

8
1
.
2

∗
∗

6
7
.
3

∗
∗
∗

5
5
.
4

∗
∗

6
4
.
4

∗
∗
∗

9
1
.
3

1
7
.
2

9
2
.
1

3
4
.
2

9
6
.
1

7
0
.
1

0
4
.
2

6
7
.
1

1
6
.
3

5
0
.
3

0
4
.
4
1

3
0
.
4

7
7
.
2

7
4
.
7

3
8
.
3

2
1
.
2

4
7
.
2

3
8
.
1

1
7
.
1

1
5
.
1

0
8
1

.

9
6
1

.

4
5
1

.

0
0
1
P
S

0
0
2
P
S

0
0
5
P
S

0
7
.
1
6

5
2
.
6
5

6
3
.
5
7

9
5
.
0
6

3
7
.
4
5

7
9
.
6
6

3
4
.
9
5

1
2
.
3
5

9
9
.
6
5

7
0
.
5
5

9
3
.
5
5

5
4
.
5
5

3
0
.
5
5

9
2
.
5
5

5
5
.
5
5

4
9
.
0

6
9
.
0

8
1
.
0

3
9
.
0

6
9
.
0

7
7
.
0

3
9
.
0

8
9
.
0

7
9
.
0

9
9
.
0

9
9
.
0

9
9
.
0

9
9
.
0

9
9
.
0

9
9
.
0

8
9
.
0

9
9
.
0

4
9
.
0

8
9
.
0

9
9
.
0

7
9
.
0

8
9
.
0

9
9
.
0

9
9
.
0

9
9
.
0

0
0
.
1

0
0
.
1

9
9
.
0

0
0
.
1

0
0
.
1

8
2
.
0

3
1
.
0

9
0
.
0

9
2
.
0

4
1
.
0

8
0
.
0

9
2
.
0

5
0
.
0

4
0
.
0

3
1
.
0

5
0
.
0

3
0
.
0

8
0
0

.

7
0
0

.

4
0
0

.

0
0
1
P
S

0
0
2
P
S

0
0
5
P
S

e
h
t

,
o
i
t
a
R
n
o
i
t
a
m
r
o
f
n
I

e
h
t

,
)

%
n
i
(

r
o
r
r
e

g
n
i
k
c
a
r
t

e
g
a
t
n
e
c
r
e
p

e
g
a
r
e
v
a

l
a
u
n
n
a

e
h
t

,
)
)

%
n
i
(

y
t
i
l
i
t
a
l
o
v

r
o
r
r
e

g
n
i
k
c
a
r
t

l
a
u
n
n
a

e
h
t

e
r
a

d
e
t
r
o
p
e
R

.
y
l
e
v
i
t
c
e
p
s
e
r

,
0
0
5
P
S

d
n
a

,
0
0
2
P
S

,
0
0
1
P
S

e
h
t

r
o
f

s
c
i
t
s
i
t
a
t
s

g
n
i
k
c
a
r
t

e
l
p
m
a
s
-
f
o
-
t
u
o

e
h
t

s
t
r
o
p
e
r

e
l
b
a
t

e
h
T

:
s
e
t
o
N

∗

∗
∗

∗
∗
∗

,
s
i
s
y
l
a
n
a
w
o
d
n
i
w
g
n
i
l
l
o
r

a

n
o

d
e
s
a
b

e
r
a

s
e
u
l
a
v

l
l

A

.
2
R
d
e
t
c
i
d
e
r
p

e
h
t

s
a

l
l
e
w
s
a

,
)

%
n
i
(

n
w
o
d
w
a
r
D
m
u
m
i
x
a
M

e
h
t

,
o
i
l
o
f
t
r
o
p

g
n
i
t
a
c
i
l
p
e
r

e
h
t

d
n
a

x
e
d
n
i

e
v
i
t
c
e
p
s
e
r

h
c
a
e

n
e
e
w
t
e
b

n
o
i
t
a
l
e
r
r
o
c

e
h
t

,
)
)

%
n
i
(

r
e
v
o
n
r
u
t

g
n
i
k
c
a
r
T
e
h
t

t
a

e
c
n
o

,
l
e
v
e
l

%
0
1

d
n
a
%
5

,

%
1

e
h
t

t
a

e
c
n
a
c
ﬁ
i
n
g
i
s

e
t
a
c
i
d
n
i

o
t

d
n
a

,

,

e
s
u

e
w

,
y
l
l
a
n
i
F

.
y
l
h
t
n
o
m
o
i
l
o
f
t
r
o
p

e
h
t

g
n
i
c
n
a
l
a
b
e
r

d
n
a

,
6
1
0
2

y
r
a
u
n
a
J

d
n
a

4
0
0
2

r
e
b
m
e
c
e
D
n
e
e
w
t
e
b

,
s
n
o
i
t
a
v
r
e
s
b
o

y
l
i
a
d

e
g
a
r
e
v
a

0
5
7
=

τ

e
h
t

,
s
n
o
i
t
i
s
o
p

e
v
i
t
c
a

f
o

r
e
b
m
u
n

f
o

e
z
i
s
w
o
d
n
i
w
a

g
n
i
r
e
d
i
s
n
o
c

positions) with low turnover while having good tracking capa-
bilities with respect to the index, quantiﬁed by tracking error
volatility, tracking error, correlation with respect to the index,
and predicted R2
OOS. From a performance perspective, invest-
ment strategies that can track the index and deliver high
information ratios and low values of the maximum drawdown
would also be attractive.

Focusing on SLOPE-LO, we ﬁnd that the strategy performs
best in terms of tracking error volatility and tracking error
for the SP500, while LASSO slightly outperforms for the
SP100 and SP200. Nevertheless, both LASSO and SLOPE-
LO show the same values of correlation and predicted R2
OOS.
In terms of cost efﬁciency, SLOPE-LO has lower turnover
values for SP200 and SP500, achieved with comparable num-
ber of active positions for SP100 and SP200, while a larger
value for SP500 (i.e. 262 vs. 238). Despite excellent track-
ing capabilities, both LASSO and SLOPE-LO tend to invest
on a much larger number of active positions than SLOPE-
LO-SLC, LOG and SCAD. In fact, they invest in about 80%
of constituents for SP100 and in about 50% for SP200 and
SP500. To improve the cost efﬁciency of the tracking portfo-
lio, we must either resort to the non-convex penalties, such
as SCAD or LOG, or consider the new strategy SLOPE-
LO-SLC, which exploits the grouping properties of SLOPE.
While SCAD and LOG penalties lead to optimization prob-
lems that are NP hard and may have multiple local optima,
SLOPE-LO-SLC requires solving a convex problem in a fast
and efﬁcient way and then selects only active groups with
median partial correlation value above the 75th percentile. In
addition to the computational complexity beneﬁts, SLOPE-
LO-SLC allows us to identify very sparse portfolios, even
sparser than SCAD and LOG for SP100 and SP500, by paying
only a small price in terms of the tracking error volatility. In
particular, we ﬁnd that the tracking error volatility of SLOPE-
LO-SLC is much smaller for the large problem size in SP500
than for LOG and SCAD (i.e. 2.74 vs. 7.47 and 14.40), while
it relies on a smaller number of active positions on average
(i.e. 66 vs. 75 and 74). In this case, SLOPE-LO-SLC also has
a lower turnover rate (i.e. 0.04), which is even comparable to
LASSO (i.e. 0.04) and only slightly larger than SLOPE-LO
(i.e. 0.03). When considering the correlation values, SLOPE-
LO-SLC performs slightly worse compared to LASSO and
SLOPE-LO, but always reaches equal or greater values than
LOG and SCAD. In terms of predicted R2
OOS, SLOPE-LO-
SLC reports higher values than LOG and SCAD for SP200
and SP500, conﬁrming its properties of providing a sparse
and cheap investment strategy, especially for larger prob-
lem size, that pays a rather small price in terms of tracking
ability.

To give more intuition behind our new SLOPE procedure,
ﬁgure 4, for SLOPE-LO and for an increasing sequence of
lambda parameters, shows in panel (a) the number of groups
formed by SLOPE-LO, in panel (b) the number of active posi-
tions, in panel (c) the maximum weight, and in panel (d) the
group-based median partial correlation with the index. All cal-
culations are based on the ﬁrst windows of size τ = 750 from
daily observations of SP100. In panel (a), one can observe
that as the lambda parameter increases, moving from left to
right on the x-axis , SLOPE-LO starts to form an increasing
number of groups among the assets in the universe. Then, the

.
o
r
e
z
m
o
r
f

t
n
e
r
e
f
f
i
d

y
l
t
n
a
c
ﬁ
i
n
g
i
s

s
i

o
i
t
a
R
n
o
i
t
a
m
r
o
f
n
I

e
h
t

r
e
h
t
e
h
w
e
t
a
c
i
d
n
i

o
t

R

I

e
h
t

t
a

e
c
n
o

d
n
a

,
t
s
e
T
)
5
9
9
1
(

o
n
a
i
r
a

M
d
n
a

d
l
o
b
e
i
D
e
h
t

o
t

g
n
i
d
r
o
c
c
a

y
g
e
t
a
r
t
s

r
e
h
t
o
y
n
a

o
t

t
n
e
r
e
f
f
i
d
s
i

y
g
e
t
a
r
t
s
O
L
-
E
P
O
L
S
e
h
t

r
e
h
t
e
h
w
e
t
a
c
i
d
n
i

o
t

r
o
r
r
E

360

P. J. Kremer et al.

Figure 4. Median partial correlations for SP100. (a) Number of groups, (b) number of active positions, (c) maximum weight and (d) median
partial correlation with the index.
Notes: The ﬁgure shows for the SP100, using the SLOPE-LO method, considering the ﬁrst window of size τ = 750 daily observations and
varying the lambda parameter between 10−4.5 and 10−2: in panel (a) the number of groups that SLOPE-LO identiﬁes, in panel (b) the number
of active positions that SLOPE-LO identiﬁes, in panel (c) the maximum weight and in panel (d) the median partial correlation across all
assets belonging to one group, that is SLOPE-LO has assigned the same coefﬁcient value to them.

number of groups begins to decrease as the octagonal shape of
the penalty pushes the solutions toward the equally weighted
portfolio, forming larger groups of assets with the same coef-
ﬁcient values until they converge to a single group formed by
the equally weighted portfolio for the last value of lambda.
Panel (c) shows that for about ﬁrst 30 values of lambda,
the maximum weight values remain approximately constant,
while as the value of lambda increases, the maximum weight
value begins to decrease toward the weight value of the
equally weighted portfolio. Panel (b) consistently shows that
the number of active positions continues to grow up to the
equally weighted portfolio as lambda increases.† Given the
behavior of SLOPE-LO, panel (d) now represents the group-
based median partial correlation with the index. That is, for
each value of the lambda coefﬁcient, we compute the median
partial correlation with the index across all assets assigned
the same coefﬁcient value from SLOPE-LO. Our goal is to
provide empirical illustration of the theoretical explanations
in theorems (1) and (2). To this end, panel (d) not only plots
the median partial correlation of all assets belonging to the
same group, but also sorts the weights in ascending order on
the y-axis . For example, with the smallest lambda value, we
obtain 84 distinct non-zero weights. The maximum value is
close to 0.06 and the corresponding asset has a partial corre-
lation with the index greater than 0.6. This weight is therefore
plotted in yellow near the bottom of panel (d). As the lambda
parameter increases and the grouping property of SLOPE-LO
becomes stronger, higher coefﬁcient weights initially retain
the high partial correlation even in their groups, but then the

† Note that for the solutions of SLOPE-LO the number of active
positions increases as we impose a larger tuning parameter. This
results from the added long-only constraint, which serves as the ini-
tial shrinkage term, and the fact that as lambda increases we move
towards the equally weighted portfolio, where all assets are equally
weighted.

median partial correlations decrease as the penalty pushes the
portfolio toward the equally weighted solutions by assigning
the same coefﬁcient value to all assets. Therefore, the investor
is best advised to choose a moderate value of lambda and
keep only the most important groups from these solutions.
SLOPE-LO-SLC is then constructed to retain only the assets
within the groups with a median partial correlation above
the 75th percentile. This results in a sparser solution with
good turnover properties and still reasonable tracking per-
formance compared to LASSO and SLOPE-LO and is able
to outperform the non-convex methods in terms of tracking
error volatility and tracking error for the large dimensional
problems (i.e. SP200, SP500).

Hedge Funds Replication. Hedge fund replication is char-
acterized by portfolios that start from a much smaller number
of factors than the number of the index constituents for equity
index tracking replication. In fact, in the previous section,
optimal portfolio could select among 93, 134 and 443 assets
for SP100, SP200 and SP500, respectively. Here, instead we
consider 17 potential factors to be used to replicate the returns
of 26 different hedge fund indexes. Therefore, sparsity is not
then the main relevant aspect in this case, but rather to iden-
tify the portfolio that best tracks the hedge fund return and
ideally also exhibits some other properties. Table 5 reports
the results for the 26 hedge fund indices, including the annu-
alized tracking error Volatility, the annualized tracking error,
the Information Ratio, the number of active positions, the
turnover, the correlation between the replicating portfolio and
the given index, as well as the predicted R2 and the maxi-
mum drawdown. Furthermore, we use a t-test to investigate,
whether the IR is statistically signiﬁcantly different from zero
at the 1%, 5% and 10% level and report the signiﬁcance at
the IR values. Similarly, we report the statistical signiﬁcance
for the Diebold and Mariano (1995) Test at the values for the
tracking error. As in the equity index tracking analysis, we

Table 5. Tracking statistics for hfri hedge fund strategies.

Tracking error volatility (in %)

Tracking error (in %)

Information ratio

Active positions

LASSO SLOPE

S-SLC S-LO-SLC

LOG SCAD LASSO SLOPE

S-SLC

S-LO-SLC

LOG

SCAD

LASSO SLOPE

S-SLC

S-LO-SLC

LOG

SCAD

LASSO SLOPE

S-SLC S-LO-SLC LOG SCAD

Fund weighted
composite

Fund of funds
composite

FOF conservative

FOF diversiﬁed

FOF market
defensive

FOF strategic

Event driven

ED distressed
restructuring

ED merger
arbitrage

Equity hedge

EH equity market

neutral

EH quantitative
directional

EH sector

technology
healthcare

Emerging markets

Emerging markets
Asia ex Japan

3.17

3.14

3.60

4.42

4.15

3.95

1.98

2.21∗∗∗

2.61∗∗∗

1.19∗∗∗

1.98∗∗∗

1.74∗∗∗

0.63∗∗∗

0.63∗∗∗

0.32∗∗∗

0.5∗∗∗

0.47∗∗∗

0.58∗∗∗

3.59

3.54

3.99

5.32

4.78

4.91

0.60∗

0.76∗∗

1.27∗∗∗

− 1.02∗∗∗

2.32∗∗∗

− 0.37∗∗∗

0.15∗∗

0.19∗∗∗

− 0.05

− 0.09

0.46∗∗∗

0.25∗∗∗

2.85

3.83

5.09

4.64

3.47

4.34

2.83

3.80

5.14

4.59

3.47

4.31

3.17

4.26

5.26

4.94

3.95

4.52

7.71

5.48

6.56

5.09

4.37

6.36

3.76

4.69

5.44

6.51

4.36

5.83

4.67

5.15

6.69

5.64

4.06

5.66

0.30

0.54

1.91

0.87
2.05∗
2.19

0.39∗∗∗
0.68∗∗∗
1.40

1.00∗∗∗
2.28∗∗∗
2.64∗∗

0.08∗∗
0.69∗∗
1.01∗

0.22∗∗∗
2.21∗∗
2.61∗∗

− 2.09∗∗∗
− 0.3∗∗∗
1.02

0.13∗∗∗
2.11∗∗
1.58∗∗∗

1.99∗∗∗
0.91∗∗∗
2.15∗∗∗

1.06∗∗∗
1.24∗
3.21∗∗∗

− 0.24∗∗∗
0.22∗∗∗
1.17∗∗∗

0.94∗∗∗
1.63∗∗∗
2.71∗∗∗

0.12∗
0.13∗
0.37∗∗∗

0.19∗∗∗
0.6∗∗∗
0.52∗∗∗

0.12∗
0.16∗∗
0.28∗∗∗

0.23∗∗∗
0.72∗∗∗
0.65∗∗∗

− 0.08
− 0.03
0.37∗∗∗

− 0.02
0.58∗∗∗
0.45∗∗∗

− 0.17∗∗
− 0.05
0.34∗∗∗

0.05
0.61∗∗∗
0.56∗∗∗

0.55∗∗∗
0.21∗∗∗
0.41∗∗∗

0.1
0.32∗∗∗
0.55∗∗∗

0.11
0.16∗∗
0.23∗∗∗

0.18∗∗
0.38∗∗∗
0.56∗∗∗

2.60

2.65

2.89

5.22

3.34

3.59

1.30∗

1.26∗∗∗

1.75∗∗

1.27∗∗

1.51∗

1.41∗∗∗

0.50∗∗∗

0.48∗∗∗

0.28∗∗∗

0.23∗∗∗

0.48∗∗∗

0.55∗∗∗

4.33

2.56

4.25

2.61

4.38

3.80

4.89

5.92

5.22

3.07

5.11

5.23

1.76

1.05

2.01∗∗∗
0.70∗∗

2.27∗∗∗
1.24∗∗∗

2.17∗∗∗
0.04∗∗

1.99∗∗∗
0.35∗

1.36∗∗∗
0.30∗∗∗

0.41∗∗∗
0.39∗∗∗

0.5∗∗∗
0.16∗∗

0.46∗∗∗
0.05

0.34∗∗∗
0.29∗∗∗

0.35∗∗∗
0.13∗

0.28∗∗∗
0.46∗∗∗

5.18

5.11

5.65

7.05

6.71

6.87

3.42

3.25∗∗∗

3.58∗∗∗

3.55∗∗∗

5.30∗∗∗

3.00∗∗∗

0.67∗∗∗

0.61∗∗∗

0.46∗∗∗

0.48∗∗∗

0.72∗∗∗

0.53∗∗∗

10.07

9.84

9.69

10.44

10.83

10.44

3.42

3.82

3.79

3.60∗∗

3.89

3.08∗∗∗

0.34∗∗∗

0.39∗∗∗

0.38∗∗∗

0.31∗∗∗

0.36∗∗∗

0.34∗∗∗

6.64

8.39

6.59

8.43

6.69

8.40

7.15

9.05

7.94

9.79

7.86

9.12

4.63

4.20

4.72

4.25

4.47∗∗
4.06

5.17∗∗∗
4.58∗∗∗

5.55∗∗∗
5.07∗∗∗

4.62∗∗∗
3.38∗∗∗

0.70∗∗∗
0.51∗∗∗

0.72∗∗∗
0.52∗∗∗

0.57∗∗∗
0.55∗∗∗

0.69∗∗∗
0.49∗∗∗

0.69∗∗∗
0.50∗∗∗

0.56∗∗∗
0.46∗∗∗

Emerging markets

5.71

5.65

5.75

5.92

7.24

6.58

2.73

2.95∗∗

3.20∗∗

3.15∗∗∗

4.33∗

4.35∗∗∗

0.48∗∗∗

0.53∗∗∗

0.42∗∗∗

0.46∗∗∗

0.58∗∗∗

0.71∗∗∗

global

Emerging markets
Latin America

Emerging markets
Russia Eastern
Europe

Macro total

Macro systematic
diversiﬁed

Relative value total

RV ﬁxed income
asset backed

RV ﬁxed income
convertible
arbitrage

9.20

9.19

9.18

9.48

9.64

10.27

1.80

1.82

1.67

2.03

− 0.01∗∗

2.13∗∗∗

0.20∗∗∗

0.22∗∗∗

0.20∗∗∗

0.21∗∗∗

0

0.15∗∗

13.87

13.65

13.70

13.58

14.46

13.84

11.69

11.82

11.78

12.15∗∗

11.64

14.14∗∗∗

0.84∗∗∗

0.83∗∗∗

0.86∗∗∗

0.86∗∗∗

0.83∗∗∗

0.91∗∗∗

5.03

7.60

2.45

3.56

5.08

7.67

2.46

3.50

5.38

7.36

2.70

3.66

6.08

7.97

5.00

7.10

5.48

6.92

3.48

4.18

6.30

8.47

4.35

5.39

1.86

3.08

1.54∗
2.8

1.50∗∗
2.79

1.39

2.79

1.29

1.54

1.66∗∗∗
3.14

1.52∗∗
3.51

2.14∗∗∗
2.53∗∗∗

4.31

4.36

4.31

6.03

5.16

5.18

0.34

0.44∗∗

0.52

2.7∗∗

2.70∗∗∗
1.12

1.77∗∗∗
2.49∗∗∗

− 1.32

0.44∗∗∗
0.83∗∗∗

1.93∗∗∗
2.15∗∗∗

1.87∗∗∗

0.37∗∗∗
0.41∗∗∗

0.63∗∗∗
0.79∗∗∗

0.31∗∗∗
0.44∗∗∗

0.80∗∗∗
0.92∗∗∗

0.31∗∗∗
0.49∗∗∗

0.36∗∗∗
0.27∗∗∗

0.33∗∗∗
0.39∗∗∗

0.55∗∗∗
0.72∗∗∗

0.50∗∗∗
0.17∗∗

0.59∗∗∗
0.63∗∗∗

0.25∗∗∗
0.15∗∗

0.33∗∗∗
0.48∗∗∗

0.08

0.09

0.15∗∗

0.15∗∗

− 0.25∗∗∗

0.26∗∗∗

RV ﬁxed income

3.41

3.41

3.49

5.42

4.69

4.99

0.57

0.69∗∗∗

0.17∗∗∗

− 0.44∗∗

0.25∗∗

− 0.15∗∗∗

0.16∗∗

0.18∗∗

0.01

0.02

0.03

0.06

corporate

RV multi strategy

RV yield

alternatives

2.67

6.30

2.66

6.23

2.76

6.17

5.66

6.56

4.54

7.86

4.55

8.27

0.33

1.54

0.51∗∗
1.62

0.47

1.58

0.27∗∗∗
2.38∗∗

− 0.36∗∗∗
1.70∗

− 0.52∗∗∗
2.51∗∗∗

0.12∗
0.24∗∗∗

0.22∗∗∗
0.22∗∗∗

0.10
0.21∗∗∗

0.11
0.26∗∗∗

− 0.09
0.19∗∗∗

− 0.14∗
0.16∗∗

8

8

7

7

6

7

7

7

8

8

7

6

6

6

6

6

6

4

7

6

7

6

6

6

7

6

12

11

11

11

11

11

11

10

11

11

10

9

9

9

9

9

8

6

11

10

11

9

9

10

10

10

10

10

9

10

9

10

9

9

10

10

9

8

8

8

7

8

7

6

9

9

9

8

8

8

9

9

16

15

12

14

12

15

16

14

14

14

14

12

10

13

10

13

9

5

12

8

15

13

16

13

14

12

14

10

10

10

8

8

9

6

13

11

10

8

7

3

2

5

2

2

9

7

11

8

10

6

9

5

12

10

11

10

7

8

9

7

11

10

10

7

7

3

3

5

3

3

8

7

10

8

10

8

9

5

S
p
a
r
s
e

i
n
d
e
x

c
l
o
n
e
s

v
i
a
t
h
e

s
o
r
t
e
d

(cid:2)

1
-

N
o
r
m

3
6
1

Fund weighted
composite

Fund of funds
composite

FOF conservative

FOF diversiﬁed

FOF market
defensive

FOF strategic

Event driven

ED distressed
restructuring

ED merger
arbitrage

Equity hedge

EH equity market

neutral

EH quantitative
directional

EH sector

technology
healthcare

Emerging markets

EM Asia ex Japan

EM global

EM Latin America

EM Russia Eastern

Europe

Macro total

Macro systematic
diversiﬁed

Relative value total

RV ﬁxed income
asset backed

RV ﬁxed income
convertible
arbitrage

RV ﬁxed income

corporate

RV multi strategy

RV yield

alternatives

0.63

1.09

3.41

1.31

1.70

3.27

0.87

0.87

0.82

0.85

0.78

0.80

0.76

0.76

0.83

1.09

2.36

1.27

2.75

3.77

0.75

0.76

0.71

0.70

0.63

0.65

0.55

0.58

0.75

0.86

0.88

0.70

0.67

1.01

1.18

1.09

1.01

1.56

1.11

1.21

2.42

2.49

1.80

2.23

2.26

2.45

1.42

1.51

1.22

1.21

1.22

1.09

3.98

3.15

3.32

1.84

2.05

2.50

4.75

3.89

4.51

2.76

4.04

3.63

0.69

0.71

0.32

0.79

0.84

0.71

0.69

0.72

0.31

0.79

0.84

0.72

0.59

0.64

0.27

0.76

0.80

0.67

0.57

0.66

0.31

0.76

0.81

0.68

0.52

0.61

0.22

0.68

0.78

0.51

0.58

0.61

0.19

0.71

0.80

0.57

0.48

0.50

0.03

0.61

0.72

0.54

0.48

0.51

0.01

0.62

0.71

0.54

0.41

0.28

− 0.12

0.00
− 0.27

0.46

0.37
− 0.14

0.68

1.14

2.00

1.45

3.43

4.12

0.64

0.62

0.57

0.58

0.51

0.61

0.42

0.39

0.09

0.63

0.26

0.01

0.21
− 0.04

0.46

0.48

0.25

0.22

0.69

0.79

1.44

1.14

2.74

1.77

1.00

1.47

2.41

3.74

2.95

4.54

0.87

0.48

0.88

0.46

0.84

0.35

0.86

0.40

0.82

0.18

0.82

0.27

0.76

0.31

0.77

0.28

0.54
− 0.93

0.71
− 0.33

0.62

1.03

1.60

0.99

1.95

2.90

0.89

0.89

0.89

0.89

0.86

0.86

0.78

0.79

1.06

1.37

1.76

1.01

3.20

3.90

0.71

0.72

0.71

0.71

0.67

0.70

0.51

0.52

0.55

0.83

0.61

0.71

0.74

0.84

0.83

0.68

0.73

1.01

0.96

1.00

0.96

1.06

1.16

1.89

1.19

0.80

4.47

1.60

1.78

1.33

1.14

2.16

2.59

2.40

2.36

0.83

0.96

1.11

0.87

0.78

1.14

1.02

1.27

1.20

0.77

1.86

1.07

1.10

0.82

3.72

2.77

2.91

4.69

2.31

2.67

2.85

2.27

1.15

4.65

3.79

3.72

5.50

0.83

0.76

0.81

0.77

0.71

0.45

0.38

0.80

0.25

0.84

0.76

0.82

0.76

0.72

0.44

0.38

0.79

0.25

0.61

0.76

0.80

0.75

0.72

0.42

0.37

0.70

0.21

0.83

0.76

0.80

0.76

0.72

0.44

0.40

0.78

0.32

0.78

0.69

0.76

0.74

0.68

0.30

0.45

0.65

0.08

0.80

0.72

0.79

0.71

0.72

0.39

0.32

0.69

0.04

0.69

0.58

0.66

0.59

0.50

0.16

0.05

0.69

0.23

0.70

0.57

0.66

0.58

0.51

0.14

0.03

0.67

0.24

0.77

0.50

0.01

0.57

0.61

0.57

0.51

0.05
− 0.05

0.24
− 1.70

0.76

0.51

0.68

0.57

0.62

0.58

0.51

0.12

0.06

0.62
− 0.24

0.60

0.23

0.07

0.25
− 0.08

0.39

0.58

0.17

0.10

0.65

0.03

0.66

0.44

0.56

0.42

0.45

0.55

0.45

− 0.01

0.22

0.41
− 0.02

0.62

0.29

0.18

0.17
− 0.44

0.43

0.64

0.21

0.30

25.67

25.74

31.98

27.53

26.58

25.73

22.24

23.10

24.79

23.43

25.52

25.10

16.51

22.25

14.30

29.76

26.49

22.76

17.04

18.97

16.86

19.65

17.13

23.36

28.07

23.66

22.68

26.55

14.68

15.12

14.98

12.42

17.77

29.94

33.46

31.71

28.10

28.10

27.45

37.95

41.56

24.85

25.15

23.80

47.14

39.05

22.98

21.24

14.47

14.48

14.30

14.48

17.69

14.50

0.67
− 1.36

33.30

7.96

32.87

44.40

35.63

34.25

31.18

6.60

17.07

16.04

1.87

20.38

0.69

0.49

0.63

0.49

0.56

0.51

0.49

− 0.02
− 0.09

0.32
− 0.33

46.04

44.13

45.12

47.44

53.73

52.11

42.53

42.26

42.26

41.78

44.46

44.01

44.05

45.66

35.33

38.94

56.16

14.50

25.33

16.09

7.25

43.71

46.89

45.03

58.18

47.62

45.11

45.49

45.97

56.57

52.32

35.28

36.82

37.84

54.81

46.75

37.99

40.23

38.94

36.21

32.50

55.25

55.25

57.01

61.20

64.35

14.66

16.22

15.11

8.24

15.07

27.78

31.06

24.77

10.21

19.68

16.41

16.83

16.27

21.50

14.85

6.72

21.75

22.79

4.09

11.75

0.58

1.07

1.67

0.97

1.87

3.13

0.80

0.78

0.75

0.81

0.72

0.73

0.66

0.64

0.59

0.66

0.50

0.56

23.07

21.54

22.32

27.42

18.49

21.65

0.83

0.95

1.77

0.95

2.99

3.39

0.78

0.79

0.78

0.77

0.62

0.67

0.62

0.64

0.80

0.84

0.69

1.23

1.99

2.12

1.58

1.07

3.82

2.78

4.34

3.37

0.79

0.63

0.79

0.63

0.70

0.64

0.80

0.63

0.44

0.48

0.61

0.50

0.65

0.41

0.65

0.42

0.12

0.28

0.39

0.45

0.60

0.41

0.31

0.00

0.09

0.36

0.19

0.07

22.81

22.80

41.54

30.26

18.46

20.45

16.69

24.22

16.31

21.33

19.87

14.38

16.09

22.70

26.14

26.36

30.08

28.77

Notes: The table reports the tracking statistics for the 26 Hedge Fund Indices, covering 6 broad hedge fund strategy dimensions, including Fund of Fund-, Event Driven-, Equity Hedge-, Emerging Markets-, Total Macro- and Relative Value Strategies, as well as a Fund Weighted Composite. Reported
are the annual Tracking Error Volatility (in %)), the annual average percentage Tracking Error (in %), the Information Ratio, the number of active positions, the average turnover (in %)), the correlation between each respective index and the replicating portfolio, the Maximum Drawdown (in %), as
well as the predicted R2. All values are based on a rolling window analysis with monthly rebalancing, considering a window size of τ = 60 monthly observations, between June 1994 and July 2017. Finally, we use ∗∗∗, ∗∗, and ∗ to indicate signiﬁcance at the 1%, 5% and 10% level, once at the TE
to indicate whether the SLOPE-LO strategy is different to any other strategy according to the Diebold and Mariano (1995) Test, and once at the IR to indicate whether the Information Ratio is signiﬁcantly different from zero.

3
6
2

P.
J
.

K
r
e
m
e
r

e
t

a
l
.

Sparse index clones via the sorted (cid:2)1-Norm

363

report the indices in the rows and for each computed measure
the methods in columns.†

Finally, in the case of hedge fund replication, we not only
impose the budget constraint, but also constraint the weights
in such a way that they are in the interval [−1, 1]. Differ-
ent to the equity index tracking, we thus explicitly allow
short-sales in our replicating portfolios. As outlined by Kre-
mer et al. (2020), SLOPE, compared to LASSO, has the
desiring property of still being active in the no-short-sale
area, where the grouping property of SLOPE is especially
dominant and under the presence of a budget constraint. As
such, we distinguish, for our newly created tracking strategy,
between a short-sale allowed (SLOPE-SLC) and a long-only
(SLOPE-LO-SLC) strategy.

Looking at the results from table 5, LASSO and SLOPE
reach consistently the lowest out of sample value of track-
ing error volatility and tracking error among all strategies
and considered indices. Given that the available universe
of risk factors is small (i.e. 17 in total), as compared
to the Equity Index Tracking framework, both strategies
achieve this performance additionally with only a small
number of active positions. SCAD and LOG penalties do
not always converge to sparser solutions than LASSO and
SLOPE, while still underperforming in terms of tracking
error Volatility. These observations conﬁrm the ﬁndings of
Giuzio et al. (2018), who also found that LOG clones might
not always dominate LASSO clones with regard to track-
ing error volatility, tracking error and turnover. The highest
values for the tracking error volatility and tracking error inde-
pendent of the underlying cloning procedure are reported
for the Emerging Market Indices. Those observations can
result from changes in the underlying risk exposures and/or
structural breaks in managers behavior (see i.e. Fung and
Hsieh 2007, Amenc et al. 2008). As Emerging Market Funds
also make use of macroeconomic changes, this observation
is also in line with the ﬁndings of Giuzio et al. (2018),
who show that in an unconstrained regression framework,
Global Macro Manager reports the highest turnover, indicat-
ing that they frequently change the exposure to different risk
factors.

Turning to the ‘SLOPE-SLC’—strategies and given the
small risk factor universe, the grouping feature does not lead
to increased sparsity, as the number of active positions for
SLOPE-SLC compared to the LASSO or SLOPE only dif-
fer by 1–2 factors. Still, the SLOPE-LO-SLC strategy poses
an interesting case. The number of active positions is among
the largest across all strategies and indices. Nevertheless, the
strategy achieves to reduce the turnover of the overall portfo-
lio and even outperforms the SLOPE and LASSO strategies
for some of the Equity Hedge Indices with regard to the
tracking error volatility and tracking error. One most notably
result are those for the Equity Market Neutral in which the
tracking error is only 0.04%, also being signiﬁcant at the 5%
level. The explanation might be that SLOPE in the long-only
area pushes the solution to the equally weighted portfolio,
thus leading to a higher number of active positions, but a

† As before, in each table, we denote SLOPE-SLC and SLOPE-LO-
SLC as SLOPE-SLC and SLOPE-LO-SLC, respectively.

more stable allocation with regard to the gross exposure. Fur-
thermore, given that the risk factor universe consists of a
set of six equity risk factors, a broader allocation to those
factors might allow for enhanced tracking abilities. Conse-
quently, the fact that SLOPE is still active in the long-only
area and pushes the solution towards an allocation with more
active positions, might allow the index tracker to capture
more return streams and thus allows to improve the tracking
ability with a desirable low turnover. As discussed previ-
ously, LASSO would be stuck in the long-only case and as
it is also evident from the results of table 5 it would then
lead to sparser, but not necessarily better hedge fund clones.
Notice in fact that the only cases for a negative tracking errors
and therefore outperforming the benchmarks are for SLOPE-
LO-SLC, which also always exhibit a lower turnover than
SLOPE-SLC.

Following the result of table 5, we can observe that SLOPE
and LASSO are again able to outperform the more complex,
non-convex methods. When considering a limited number of
risk factors, as in this investigation, such penalties turn out to
be possibly the most appealing. At the same time, SLOPE-
SLC procedure is well equipped to perform in line with
state-of-the-art tracking portfolios. Considering the idiosyn-
cratic nature of each hedge fund strategy, the grouping feature,
and especially the fact that SLOPE is still active in the long-
only area might come in handy for index trackers, as it
allows to gain a larger exposure to a broader set of under-
lying risk factors that allow to capture more variation of
returns.

5. Conclusion

Index tracking aims at constructing sparse and stable replicat-
ing portfolios that best mimics the risk and return time series
pattern of a given benchmark, which could either be a broad
equity market index or the performance of an alternative
investment vehicle, like a hedge fund.

This paper introduces the Sorted (cid:2)1 Penalized Estimator,
called SLOPE, to the index tracking and hedge fund repli-
cation framework, and compares its performance to current
state-of-the-art convex and non-convex penalty functions. We
provide new theoretical insight that SLOPE’s grouping ability
is based on the difference among the partial correlations of the
constituents and that it assigns higher weights to assets which
have a larger partial correlation with the respective index.
We show these ﬁndings in both a simulated and a real-world
environment. We ﬁnd that SLOPE has the desired feature
of grouping assets together, enabling us to pick individual
constituents from them and hence to create new tracking
strategies, such as SLOPE-SLC, that lead to sparse replicat-
ing portfolios with good tracking ability, especially for equity
index tracking, when the problem typically requires choosing
from a large pool of index constituents.

New investment strategies, using the grouping feature can
then be developed, as well as testing alternative lambda
sequences to improve SLOPE shrinkage and model selection
properties. Such extensions are high on our agenda.

364

Disclosure statement

P. J. Kremer et al.

No potential conﬂict of interest was reported by the authors.

Funding

The research of Malgorzata Bogdan was funded by the NCN
[grant number 2016/23/B/ST1/00454]. Research support for
Damian Brzyski was provided by the National Institutes
of Health [grant number R01MH108467]. Finally, Sandra
Paterlini and Malgorzata Bogdan acknowledge support from
EU-ICT CRONOS ACTION.

Supplemental data

Supplemental data for
https://doi.org/10.1080/14697688.2021.1962539.

this article can be accessed at

ORCID

Philipp J. Kremer
Małgorzata Bogdan
4342
Sandra Paterlini

http://orcid.org/0000-0002-2618-6523
http://orcid.org/0000-0002-0657-

http://orcid.org/0000-0003-4269-4496

References

Amenc, N., Géhin, W.M. and Meyfredi, J.-C., Passive hedge fund
replication: A critical assessment of existing techniques. J. Altern.
Invest., 2008, 11(2), 69–83.

Amenc, N., Martellini, L., Meyfredi, J.-C. and Ziemann, V., Pas-
sive hedge fund replication – Beyond the linear case. Eur. Financ.
Manage., March 2010, 16(2), 191–210.

Bellec, P.C., Lecué, G. and Tysbakov, A.B., Bounds on the prediction
error of penalized least squares estimators with convex penalty. In
Modern Problems of Stochastic Analysis and Statistics, Festschrift
in honor of Valentin Konakov, edited by V. Panov, 2016.

Bellec, P.C., Lecué, G. and Tsybakov, A.B., Slope meets lasso:
Improved oracle bounds and optimality. Ann. Stat., December
2018, 46(6B), 3603–3642. https://doi.org/10.1214/17-AOS1670
Bogdan, M. and Frommlet, F., Identifying important predictors in
large data bases – Multiple testing and model selection. In Hand-
book of Multiple Comparisons, edited by X. Cui, T. Dickhaus,
Y. Ding, and J.C. Hsu, 2021 (Chapman & Hall/CRC). Available
online at: https://arxiv.org/abs/2011.12154.

Bogdan, M., van den Berg, E., Su, W. and Candès, E.J., Statistical
estimation and testing via the ordered (cid:2)1 norm. arXiv:1310.1969,
pp. 1–46, 2013.

Bogdan, M., van den Berg, E., Sabatti, C., Su, W. and Candes, E.J.,
SLOPE – Adaptive variable selection via convex optimization.
Ann. Appl. Statist., 2015, 9(3), 1103–1140.

Bondell, H. and Reich, B., Simultaneous regression shrinkage,
variable selection, and supervised clustering of predictors with
OSCAR. Biometrics, 2008, 64(1), 115–123.

Branger, N., Lucivjanska, K. and Weissensteiner, A., Optimal gran-
ularity for portfolio choice. J. Empir. Finance, 2019, 50(C),
125–146.

Brodie, J., Daubechies, I., DeMol, C., Giannone, D. and Loris, D.,
Sparse and stable Markowitz portfolios. Proc. Nat. Acad. Sci.,
2009, 106(30), 12267–12272.

Brzyski, D., Gossmann, A., Su, W. and Bogdan, M., Group slope
– Adaptive selection of groups of predictors. J. Am. Stat. Assoc.,
2018, 114(525), 419–433. https://doi.org/10.1080/01621459.2017.
1411269

Canakgoz, N.A. and Beasley, J.E., Mixed-integer programming
approaches for index tracking and enhanced indexation. Eur. J.
Oper. Res., 2009, 196(1), 384–399.

Candes, E., Waking, M.B. and Boyd, S.P., Enhencing sparsity by
reweighted (cid:2)1 mminimization. J. Fourier Anal. Appl., 2008, 14(5),
877–905.

Carrasco, M. and Noumon, N., Optimal portfolio selection using
regularization. Working Paper University of Montreal, pp. 1–52,
2012.

Chen, S. and Donoho, D., Basis pursuit. In Proceedings of 1994 28th
Asilomar Conference on Signals, Systems and Computers, Vol. 1,
pp. 41–44, 1994 (IEEE).

Chen, C., Li, X., Tolman, C., Wang, S. and Ye, Y., Sparse port-
folio selection via quasi-norm regularization. Papers 1312.6350,
arXiv.org, 2013.

Chiam, S.C., Tan, K.C. and Al Mamun, A., Dynamic index tracking
via multi-objective evolutionary algorithm. Appl. Soft. Comput.,
2013, 13(7), 3392–3408.

DeMiguel, V., Garlappi, L., Nogales, F. and Uppal, R., A generalized
approach to portfolio optimization: Improving performance by
constraining portfolio norm. Manage. Sci., 2009, 55(5), 798–812.
Diebold, F.X. and Mariano, R.S., Comparing predictive accuracy.

J. Bus. Econ. Stat., 1995, 13, 134–144.

Fan, J. and Li, R., Variable selection via nonconcave penalized like-
lihood and its oracle properties. J. Am. Stat. Assoc., 2001, 96(456),
1348–1360.

Fan, J., Zhang, J. and You, K., Vast portfolio selection with gross-
exposure constraint. J. Am. Stat. Assoc., 2012, 107(498), 592–606.
Fastrich, B., Paterlini, S. and Winker, P., Cardinality versus q-norm
constraints for index tracking. Quant. Finance, 2014, 14(11),
2019–2032.

Fastrich, B., Paterlini, S. and Winker, P., Constructing optimal sparse
portfolios using regularization methods. Comput. Manage. Sci.,
2015, 12(3), 417–434.

Fernholtz, R., Garvy, R. and Hannon, J., Diversity-weighted index-

ing. J. Portfolio Manage., 1998, 4(2), 74–82.

Figueiredo, M. and Nowak, R., Sparse estimation with strongly
correlated variables using ordered weighted (cid:2)1 regularization.
arXiv:1409.4005, Working Paper, pp. 1–15, 2014.

Frino, A. and Gallagher, D., Tracking S&P Index funds. J. Portfolio

Manage., 2001, 28(1), 44–55.

Fung, W. and Hsieh, D.A., Will hedge funds regress towards index-

like products? J. Invest. Manage., 2007, 5(32), 46–65.

Giamouridis, D. and Paterlini, S., Regular(ized) hedge fund clones.

J. Financ. Res., 2010, 33(3), 223–247.

Gilli, M. and Kellezi, E., The threshold accepting heuristic for index
tracking. In Financial Engineering, E-Commerce, and Supply
Chain, pp. 1–18, 2009 (Kluwer: Dordrecht).

Gilli, M. and Winker, P., Heuristic optimization methods in econo-
metrics. In Handbook of Computational Econometrics, edited
by D. Beasley and E. Kontoghiorghes, pp. 81–120, 2009
(Wiley).

Giuzio, M., Genetic algorithm versus classical methods in sparse
index tracking. Decis. Econ. Finance, 2017, 40(1-2), 243–256.
Giuzio, M and Paterlini, S., Un-diversifying during crises: Is it a
good idea?. Comput. Manage. Sci., 2018. https://doi.org/10.1007/
s10287-018-0340-y

Giuzio, M., Eichhorn-Schott, K., Paterlini, S. and Weber, V., Track-
ing hedge funds using sparse clones. Ann. Oper. Res., 2018,
266(1–2), 349–371.

Golosnoy, V., Gribisch, B. and Seifert, M.I., Exponential smoothing
of realized portfolio weights. J. Empir. Finance, 2019, 53, 222–
237.

Sparse index clones via the sorted (cid:2)1-Norm

365

Gu, S., Kelly, B. and Xiu, D., Empirical asset pricing via
machine learning. Rev. Financ. Stud., 2020, 33, 2223–2273.
https://doi.org/10.1093/rfs/hhaa009

Hastie, T., Tibshirani, R. and Friedman, J., The Elements of Statis-
tical Learning - Data Mining,Inference and Prediction, 2nd ed.
2001 (Springer: Stanford, CA).

Kremer, P.J., Talmaciu, A. and Paterlini, S., Risk minimization in
multi-factor portfolios: What is the best strategy?. Ann. Oper. Res.,
2018, 266(1–2), 255–291.

Kremer, P.J., Lee, S., Bogdan, M. and Paterlini, S., Sparse portfolio
selection via the sorted (cid:2)1-norm. J. Bank. Finance, 2020, 110(6B),
1–41.

Krink, T., Mittnik, S. and Paterlini, S., Differential evolution and
combinatorial search for constrained index tracking. Ann. Oper.
Res., 2009, 172(1), 153–176.

Mainik, G., Mitov, G. and Rüschendorf, L., Portfolio optimization for
heavy-tailed assets: Extreme risk index vs. markowitz. J. Empir.
Finance, 2015, 32(2), 115–134.

Malkiel, B.G., Returns from investing in equity mutual funds 1971

to 1991. J. Finance., 1995, 50(2), 549–572.

Rudolf, M., Wolter, H.-J. and Zimmermann, H., A linear model for
tracking error minimization. J. Bank. Finance, 1999, 23(1), 85–
103.

Santosa, F. and Symes, W.W., Linear inversion of band-limited
reﬂection seismograms. SIAM J. Sci. Stat. Comput., 1986, 7(4),
1307–1330.

Sorenson, E., Miller, K. and Samak, V., Allocating between active
and passive management. Financ. Anal. J., 1998, 54(5), 18–31.
Su, W. and Candès, E.J., SLOPE is adaptive to unknown sparsity and
asymptotically minimax. Ann. Stat., 2016, 44(3), 1038–1068.
Tibshirani, R., Regression shrinkage and selection via the LASSO.

R. Stat. Soc., 1996, 58(1), 267–288.

Weston, J., Elisseeff, A. and Schoelkopf, B., Use of the zero-norm
with linear models and kernel methods. J. Mach. Learn. Res.,
2003, 3, 1439–1461.

Xing, X., Hub, J. and Yang, Y., Robust minimum variance portfolio

with (cid:2)∞ constraints. J. Bank. Finance, 2014, 46, 107–117.

Yen, Y.-M., Sparse weighted norm minimum variance portfolio. Rev.

Financ., 2015, 20(3), 1259–1287.

Yen, Y.-M. and Yen, T.-J., Solving norm constrained portfolio opti-
mization via coordinate-wise descent algorithms. Comput. Stat.
Data Anal., 2014, 76(1), 737–759.

Appendices

Appendix 1. Alternative penalty functions

A.1. Smoothly clipped absolute deviation

To resolve the problem of large biased coefﬁcient values of the
LASSO, Fan and Li (2001) developed the non-convex smoothly
clipped absolute deviation (SCAD) penalty, given by:

closer to the LASSO. On the other hand, given a large estimated
coefﬁcient, that is |wi| > aλ, the SCAD imposes an upper bound
on the value of the penalty function. Even if the estimated coef-
ﬁcient value increases past this point, it will no longer inﬂate the
penalty function. Consequently, the SCAD has the tendency to pro-
duce extreme positive and negative weights, when compared to the
LASSO. Still, the maximum attainable value of such coefﬁcients is
limited by the added budget constraint, as a larger weight for asset
i only goes in hand with a lower weight for asset j. In between the
two extreme points, and when λ <|w i| ≤aλ, the penalty ‘smoothly
clips’ the estimated parameters.

From an economic perspective and given the correlation struc-
ture, the SCAD penalizes those assets which receive a small weight
and thus probably have less explanatory power. On the other hand,
weights which exceed the threshold of aλ are considered to be impor-
tant predictors and are not intended to be further penalized (Fastrich
et al. 2015).†

A.2. Logarithmic penalty

Besides the SCAD, we also consider the non-convex logarithmic
penalty function (LOG) (see e.g. Weston et al. 2003), given by:

ρλ(w) = λ ×

K(cid:3)

i=1

(log(|wi| +γ ) − log(γ ))

(A2)

where 0 < γ <1 is a constant to avoid the occurrence of an unde-
ﬁned logarithm, when wi = 0. The LOG penalty can be considered
as an approximation to the (cid:2)0 or cardinality constraint. Further-
more, it closely approximates the behavior of the (cid:2)q-Norm with
0<q<1,‡ which even intensiﬁes when q → 0. Compared to the (cid:2)q
-Norm, however the LOG penalty leads to sparser solutions and has
shown to possess good sparsity recovery properties (see i.e. Candes
et al. 2008, Giuzio et al. 2018).

From an economic perspective, an increase of a small weight
is penalized more heavily, than an associated increase of a large
weights. Based on the correlation structure, it thus promotes select-
ing only a few large coefﬁcients with high explanatory power, while
disregarding small and unnecessary coefﬁcients.

Appendix 2. Proof of theorems

A.3. Proof of theorem 2.1

Proof Clearly, permuting columns of R and switching their signs
correspond, respectively, to permuting and switching signs of the
coefﬁcients of the solution. Let (cid:14)R denote the matrix R after multi-
plying its columns by the signs of corresponding coefﬁcients of ˆw
and after permuting them with respect to the order of magnitudes of
ˆw’s coefﬁcients. It holds rank((cid:14)R) = k and it is enough to prove that
the solution, (cid:14)w, to the problem with modiﬁed design matrix satisﬁes
the claim. We have (cid:14)w1 ≥ · · · ≥ (cid:14)wK ≥ 0 and hence (cid:14)w solves

(cid:12)

λ|wi| × 1(|wi| ≤λ) +

ρλ(w) =

K(cid:3)

i=1

−w2
i

+ 2aλ|wi| − λ2
2(a − 1)

arg minw

(cid:4)
(cid:4)Y − ˜Rw

(cid:4)
(cid:4)2
2

1
2

+ λTw s.t. w1 ≥ · · · ≥ wK ≥ 0 (A3)

× 1(λ < |wi| ≤aλ) +

(a + 1)λ2
2

× 1(aλ <|w i|)

where a is a threshold parameter, and 1(·) is the indicator function,
which is equal to one when the argument in the parenthesis is true
and zero otherwise.

The SCAD consists of three regions that, depending on the esti-
mated coefﬁcients, determine the value of ρλ(w). As long as |wi| <
λ, the SCAD increases linearly and thus has the same shrinkage
abilities as the LASSO. Furthermore, as we start to impose a larger
lambda parameter, this linear region will expand and the penalty gets

(cid:13)

(A1)

The claim is true for k = K. Suppose that k < K and denote by N
the null space of (cid:14)R. Then we have dim(N) = K − k > 0. We need to
show that at least K − k inequalities deﬁning the feasible set in (A3)
become equalities for (cid:14)w. Suppose that this is not true and that we
have a partition of {1, . . ., K} into two sets of indices, Ieq and Iineq,
such that

i ∈ Ieq =⇒ ˜wi = ˜wi+1,

i ∈ Iineq =⇒ ˜wi > ˜wi+1,

† For our empirical investigations, we choose a = 3.
‡ For applications of this norm in the area of index tacking,
see i.e. Fernholtz et al. (1998), Fastrich et al. (2014), Chen
et al. (2013), Giuzio (2017).

366

P. J. Kremer et al.

|Ieq| < K − k,

(A4)

with a convention that (cid:14)wK+1 := 0. Now, consider the vector sub-
space H, deﬁned as H := {w ∈ RK |wi = wi+1, i ∈ Ieq}, once again
with the convention that wK+1 := 0. Since we have less than K − k
linear equations deﬁning the subspace, it holds dim(H) > k. There-
fore, there exists the non-zero vector, d, such asd ∈ N ∩ H and
without the loss of generality we can assume that λTd ≤ 0. We
can also ﬁnd δ > 0 such that for all i ∈ Iineq it holds (cid:14)wi + δ · di >
(cid:14)wi+1 + δ · di+1. Consider the vector deﬁned as c := (cid:14)w + δ · d. Since
c ∈ H, we havec i = ci+1, for i ∈ Ieq, and the construction of δ yields
ci > ci+1, fori ∈ Iineq. Consequently, c is a feasible point in the
optimization problem (A3). Now,

(cid:4)
(cid:4)Y − ˜Rc

(cid:4)
(cid:4)2
2

1
2

+ λTc = 1
2
≤ 1
2

(cid:4)
(cid:4)Y − ˜R ˜w − δ ˜Rd
(cid:4)
(cid:4)
(cid:4)2
(cid:4)Y − ˜R ˜w
2

+ λT ˜w,

(cid:4)
(cid:4)2
2

+ λT ˜w + δ · λTd

(A5)

By the construction we have c (cid:12)= (cid:14)w, hence the last inequality contra-
dicts either the optimality of (cid:14)w (if ‘ < ’ holds) or the uniqueness of
(cid:2)
the solution (if ‘ = ’ holds).

A.4. Proof of theorem 2.2
Proof Suppose that ˆwi > ˆwi+1 and deﬁne ψ := ei+1 − ei, where
ej is an indicator vector of index j. There exists ε > 0 such that
the vector cδ := ˆw + δ · ψ is feasible for any δ ∈ (0, ε). Now, ˆw is
the solution to the minimization problem with the objective f (w) :=
1
2 and under the constraints w1 ≥ · · · ≥ wK ≥ 0. Since
2

(cid:2)Y − Rw(cid:2)2

is differentiable, the directional derivative of f exists for any

f
argument and any direction. We have

∇ψ f ( ˆw) := lim
δ→0

f ( ˆw + δ · ψ) − f ( ˆw)
δ

f (cδ) − f ( ˆw)
δ

≥ 0,

= lim
δ→0+
δ∈(0,ε)

where the inequality follows from the optimality of
construction of cδ. Now, deriving the gradient of f in ˆw yields

ˆw and the

∇f ( ˆw) = −RT(Y − R ˆw) + λ

and, since it always holds ∇ψ f ( ˆw) = ψ T∇f ( ˆw), we get (ei −
ei+1)TRT(Y − R ˆw) ≥ (ei − ei+1)Tλ. Now, the condition ˆwi > ˆwi+1
implies

λi − λi+1 ≤

=

=

=

≤

(cid:15)

Ri − Ri+1
(cid:15)

Ri − Ri+1
(cid:15)

(cid:16)

(cid:15)

(cid:16)
T

(cid:16)
T

Y − R ˆw
(cid:15)

rP − Ri ˆwi − Ri+1 ˆwi+1

(cid:16)

Ri − Ri+1
(cid:15)

(cid:16)
TrP
i Ri+1 ˆwi+1 − RT
d2 ˆwi + RT
(cid:16)
TrP −
d2 − RT
(cid:16)
TrP,

Ri − Ri+1

Ri − Ri+1

(cid:15)

−
(cid:15)

(cid:15)

i+1Ri ˆwi − d2 ˆwi+1
(cid:16)
ˆwi − ˆwi+1

(cid:16)(cid:15)

i Ri+1

(cid:16)

(A6)

after applying the Cauchy-Schwarz inequality, which ends the proof.
(cid:2)

