## c:/Dropbox/Rpackages/ipopCpp/tests/testthat/ipopCpp.R

##    Chandler Lutz
##    Questions/comments: cl.eco@cbs.dk
##    $Revisions:      1.0.0     $Date:  2016-11-16

context("ipopCpp")

library(Rcpp);
library(RcppArmadillo);
library(kernlab);
library(microbenchmark)

##source("R/ipop_kernlab.R")

test_that("pmax_arma is equal to pmax()", {
    expect_equal(pmax(1:10, 4), as.numeric(pmax_arma(1:10, 4)))
    expect_equal(pmax(10:1, 4), as.numeric(pmax_arma(10:1, 4)))
})


set.seed(1234)

## -- Small tests for comparing Cpp to r output -- ##

## solve the Support Vector Machine optimization problem
data(spam)
## sample a scaled part (500 points) of the spam data set
m <- 500
set <- sample(1:dim(spam)[1],m)
x <- scale(as.matrix(spam[,-58]))[set,]
y <- as.integer(spam[set,58])
y[y==2] <- -1
##set C parameter and kernel
C <- 5
rbf <- rbfdot(sigma = 0.1)
## create H matrix etc.
H <- kernelPol(rbf,x,,y)
c <- matrix(rep(-1,m))
A <- t(y)
b <- matrix(0)
l <- matrix(rep(0,m))
u <- matrix(rep(C,m))
r <- matrix(0)

##Run test
test_that("kernlab::ipop and ipopCpp are equal", {
    expect_equal(ipop(c,H,A,b,l,u,r)@primal, as.numeric(ipopCpp(c,H,A,b,l,u,r)$primal))
    expect_equal(ipop(c,H,A,b,l,u,r)@dual, as.numeric(ipopCpp(c,H,A,b,l,u,r)$dual))
})

##########################
##### Synth Examples #####
##########################

tol <- 1e-01

##source("../scratch/ipop_kernlab.r")

##Run test using synth
data(solution_v_forc_example)
data(X_scaled_forc_example)

## -- Check for i == 9, a previous bug -- ##

solution.v <- rep(1 / 12, 12)
treated <- 9

X0.scaled <- X_scaled_forc_example[, -treated]
X1.scaled <- X_scaled_forc_example[, treated]
##solution.v <- solution_v_forc_example
solution.v <- rep(1 / 12, 12)
nvarsV = length(solution.v)

V <- diag(x=as.numeric(solution.v),nrow=nvarsV,ncol=nvarsV)
H <- t(X0.scaled) %*% V %*% (X0.scaled)
a <- X1.scaled
c <- -1*c(t(a) %*% V %*% (X0.scaled) )
A <- t(rep(1, length(c)))
b <- 1
l <- rep(0, length(c))
u <- rep(1, length(c))
r <- 0
res <- ipop(c = c, H = H, A = A, b = b, l = l, u = u, r = r,
            margin = 0.0005, maxiter = 1000, sigf = 5, bound = 10)
res2 <- ipopCpp(c = c, H = H, A = A, b = b, l = l, u = u, r = r,
                margin = 0.0005, maxiter = 1000, sigf = 5, bound = 10)
test_that(paste0("kernlab and ipopCPP are equal for synth examples with equal weights for starting values for i == 9"), {
    expect_equal(as.numeric(res@primal), as.numeric(ipopCpp(c,H,A,b,l,u,r)$primal),
                 tolerance = tol)
    expect_equal(res@dual, as.numeric(ipopCpp(c,H,A,b,l,u,r)$dual),
                 tolerance = tol)

})


## -- Use equal weights for starting value -- ##

solution.v <- rep(1 / 12, 12)



for (i in 1:ncol(X_scaled_forc_example)) {
    treated <- i

    X0.scaled <- X_scaled_forc_example[, -treated]
    X1.scaled <- X_scaled_forc_example[, treated]
    ##solution.v <- solution_v_forc_example
    solution.v <- rep(1 / 12, 12)
    nvarsV = length(solution.v)

    V <- diag(x=as.numeric(solution.v),nrow=nvarsV,ncol=nvarsV)
    H <- t(X0.scaled) %*% V %*% (X0.scaled)
    a <- X1.scaled
    c <- -1*c(t(a) %*% V %*% (X0.scaled) )
    A <- t(rep(1, length(c)))
    b <- 1
    l <- rep(0, length(c))
    u <- rep(1, length(c))
    r <- 0
    res <- ipop(c = c, H = H, A = A, b = b, l = l, u = u, r = r,
                margin = 0.0005, maxiter = 1000, sigf = 5, bound = 10)
    res@dual
    res2 <- ipopCpp(c = c, H = H, A = A, b = b, l = l, u = u, r = r,
                    margin = 0.0005, maxiter = 1000, sigf = 5, bound = 10)
    test_that(paste0("kernlab and ipopCPP are equal for synth examples with equal weights for starting values for ", i), {
        expect_equal(res@primal, as.numeric(ipopCpp(c,H,A,b,l,u,r)$primal),
                     tolerance = tol)
        expect_equal(res@dual, as.numeric(ipopCpp(c,H,A,b,l,u,r)$dual),
                     tolerance = tol)

    })
}


## -- Use the optimal solution for starting values starting value -- ##

solution.v <- solution_v_forc_example

for (i in 1:ncol(X_scaled_forc_example)) {
    treated <- i

    X0.scaled <- X_scaled_forc_example[, -treated]
    X1.scaled <- X_scaled_forc_example[, treated]
    ##solution.v <- solution_v_forc_example
    solution.v <- rep(1 / 12, 12)
    nvarsV = length(solution.v)

    V <- diag(x=as.numeric(solution.v),nrow=nvarsV,ncol=nvarsV)
    H <- t(X0.scaled) %*% V %*% (X0.scaled)
    a <- X1.scaled
    c <- -1*c(t(a) %*% V %*% (X0.scaled) )
    A <- t(rep(1, length(c)))
    b <- 1
    l <- rep(0, length(c))
    u <- rep(1, length(c))
    r <- 0
    res <- ipop(c = c, H = H, A = A, b = b, l = l, u = u, r = r,
                margin = 0.0005, maxiter = 1000, sigf = 5, bound = 10)
    res@dual
    res2 <- ipopCpp(c = c, H = H, A = A, b = b, l = l, u = u, r = r,
                    margin = 0.0005, maxiter = 1000, sigf = 5, bound = 10)
    test_that(paste0("kernlab and ipopCPP are equal for synth examples with the optimal solution for starting values for ", i), {
        expect_equal(res@primal, as.numeric(ipopCpp(c,H,A,b,l,u,r)$primal),
                     tolerance = tol)
        expect_equal(res@dual, as.numeric(ipopCpp(c,H,A,b,l,u,r)$dual),
                     tolerance = tol)
    })
}


## -- Use random weights for starting values starting value -- ##

solution.v <- runif(12)
solution.v <- solution.v / sum(solution.v)

for (i in 1:ncol(X_scaled_forc_example)) {
    treated <- i

    X0.scaled <- X_scaled_forc_example[, -treated]
    X1.scaled <- X_scaled_forc_example[, treated]
    ##solution.v <- solution_v_forc_example
    solution.v <- rep(1 / 12, 12)
    nvarsV = length(solution.v)

    V <- diag(x=as.numeric(solution.v),nrow=nvarsV,ncol=nvarsV)
    H <- t(X0.scaled) %*% V %*% (X0.scaled)
    a <- X1.scaled
    c <- -1*c(t(a) %*% V %*% (X0.scaled) )
    A <- t(rep(1, length(c)))
    b <- 1
    l <- rep(0, length(c))
    u <- rep(1, length(c))
    r <- 0
    res <- ipop(c = c, H = H, A = A, b = b, l = l, u = u, r = r,
                margin = 0.0005, maxiter = 1000, sigf = 5, bound = 10)
    res@dual
    res2 <- ipopCpp(c = c, H = H, A = A, b = b, l = l, u = u, r = r,
                    margin = 0.0005, maxiter = 1000, sigf = 5, bound = 10)
    test_that(paste0("kernlab and ipopCPP are equal for synth examples with random weights for starting values for ", i), {
        expect_equal(res@primal, as.numeric(ipopCpp(c,H,A,b,l,u,r)$primal),
                     tolerance = tol)
        expect_equal(res@dual, as.numeric(ipopCpp(c,H,A,b,l,u,r)$dual),
                     tolerance = tol)
    })
}





##compare speed
## microbenchmark(
##     ipop(c,H,A,b,l,u,r, maxiter = 50, margin = 0.005, bound = 5),
##     ipopCpp(c,H,A,b,l,u,r, maxiter = 50, margin = 0.005, bound = 5)
## )
