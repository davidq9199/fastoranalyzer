# test_r_factanal.R

library(stats)
library(jsonlite)

set.seed(42)  

dir.create("tests/r_tests/output", showWarnings = FALSE, recursive = TRUE)

generate_sample_data <- function(n_samples, n_features) {
  matrix(rnorm(n_samples * n_features), nrow = n_samples)
}

run_factanal <- function(X, n_factors, rotation = "varimax", scores = "regression") {
  result <- factanal(X, factors = n_factors, rotation = rotation, scores = scores)
  
  list(
    loadings = as.matrix(unclass(result$loadings)),  # convert loadings to matrix
    uniquenesses = result$uniquenesses,
    correlation = result$correlation,
    criteria = result$criteria,
    factors = result$factors,
    dof = result$dof,
    method = result$method,
    STATISTIC = result$STATISTIC,
    PVAL = result$PVAL,
    scores = if(!is.null(result$scores)) as.matrix(result$scores) else NULL
  )
}

test_cases <- list(
  list(n_samples = 100, n_features = 5, n_factors = 2),
  list(n_samples = 1000, n_features = 10, n_factors = 3),
  list(n_samples = 500, n_features = 8, n_factors = 4)
)

results <- list()

for (i in seq_along(test_cases)) {
  case <- test_cases[[i]]
  X <- generate_sample_data(case$n_samples, case$n_features)
  
  write.csv(X, file = sprintf("tests/r_tests/output/test_data_case_%d.csv", i), row.names = FALSE)
  
  results[[sprintf("case_%d_varimax", i)]] <- run_factanal(X, case$n_factors, rotation = "varimax")
  results[[sprintf("case_%d_promax", i)]] <- run_factanal(X, case$n_factors, rotation = "promax")
  results[[sprintf("case_%d_none", i)]] <- run_factanal(X, case$n_factors, rotation = "none")
}

writeLines(toJSON(results, pretty = TRUE, digits = 10), "tests/r_tests/output/r_factanal_results.json")

print("R factanal tests done. Results saved at tests/r_tests/output/r_factanal_results.json")