# Upper Quartile normalize a matrix
# https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0206312#:~:text=Upper%20Quartile%20(UQ)%3A%20Under,multiplied%20by%20the%20mean%20upper


upperQuartileNormalise <- function(exprs_matrix)
{
    # function to upper quartile normalise a matrix

    uqs <- apply(exprs_matrix, 2, function(x) quantile(x[x>0],0.75))

    uqn <- t(t(exprs_matrix) / uqs) * mean(uqs)

    return(uqn)

}
