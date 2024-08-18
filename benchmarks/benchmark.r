run_factanal_benchmark <- function(file_path, n_factors) {
  data <- read.csv(file_path, header = FALSE)
  
  run_factanal <- function() {
    factanal(data, factors = n_factors, rotation = "none")
  }
  
  run_factanal() # warm up
  
  n_runs <- 10
  times <- numeric(n_runs)
  memories <- numeric(n_runs)
  
  for (i in 1:n_runs) {
    gc(full = TRUE, reset = TRUE) 
    
    start_time <- Sys.time()
    run_factanal()
    end_time <- Sys.time()
    
    gc_result <- gc(full = TRUE)
    
    times[i] <- as.numeric(difftime(end_time, start_time, units = "secs"))
    memories[i] <- sum(gc_result[,6]) 
  }
  
  results <- data.frame(
    dataset = basename(file_path),
    method = "factanal",
    mean_time = mean(times),
    std_time = sd(times),
    median_time = median(times),
    min_time = min(times),
    max_time = max(times),
    mean_memory = mean(memories),
    std_memory = sd(memories),
    median_memory = median(memories),
    min_memory = min(memories),
    max_memory = max(memories)
  )
  
  return(results)
}

datasets <- c("1MB", "10MB", "50MB", "100MB")
n_factors <- 5

results <- data.frame()

for (dataset in datasets) {
  file_path <- file.path("benchmarks", "datasets", paste0("dataset_", dataset, "_50features.csv"))
  result <- run_factanal_benchmark(file_path, n_factors)
  results <- rbind(results, result)
}

write.csv(results, "r_factanal_results.csv", row.names = FALSE)
print("R benchmarks completed. Results saved to r_factanal_results.csv")