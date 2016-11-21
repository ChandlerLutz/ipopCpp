Rcpp implementation of the `kernlab::ipop()` function

### Instillation:

    library(devtools)
    install_github("ChandlerLutz/ipopCpp")

### Examples:

    ##Comparison to kernlab::ipop()
    library(kernlab)
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
	sv <- ipop(c,H,A,b,l,u,r)
	sv2 <- ipopCpp(c,H,A,b,l,u,r)
	all.equal(ipop(c,H,A,b,l,u,r)@primal, as.numeric(ipopCpp(c,H,A,b,l,u,r)$primal))
	all.equal(ipop(c,H,A,b,l,u,r)@dual, as.numeric(ipopCpp(c,H,A,b,l,u,r)$dual))
