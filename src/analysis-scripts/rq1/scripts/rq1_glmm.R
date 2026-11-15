# RQ1: binomial mixed-effects model comparing FT-CodeLlama-13B and FT-HPC-Coder-6.7B.
# Input:  results/rq1_long.csv (written by scripts/rq1_analysis.py)
# Output: results/rq1_glmm.txt
# Usage:  Rscript scripts/rq1_glmm.R   (run from the rq1/ directory, after rq1_analysis.py)
suppressMessages({library(lme4); library(emmeans)})
d <- read.csv("results/rq1_long.csv")
d <- subset(d, translator %in% c("FT-CodeLlama-13B", "FT-HPC-Coder-6.7B"))
# Applications in which neither translator produces a correct translation carry no
# information about the difference and prevent the model from converging.
d <- subset(d, application %in% names(which(tapply(d$correct, d$application, sum) > 0)))
d$run <- factor(d$run)
m <- glmer(cbind(correct, translated - correct) ~ translator +
             (1 | application) + (1 | application:run) + (1 | application:translator),
           data = d, family = binomial, control = glmerControl(optimizer = "bobyqa"))
sink("results/rq1_glmm.txt")
cat("Applications in model:", length(unique(d$application)), "\n\n")
print(summary(m))
cat("\nOdds ratio (CodeLlama / HPC-Coder) with 95% CI:\n")
print(summary(pairs(emmeans(m, ~ translator), type = "response"), infer = TRUE))
sink()
cat(readLines("results/rq1_glmm.txt"), sep = "\n")
