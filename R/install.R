packages <- c("dplyr", "tidyr", "magrittr", "zoo", "arrow", "knitr")

install.packages(setdiff(
    packages,
    rownames(installed.packages())
), dependencies = TRUE)
