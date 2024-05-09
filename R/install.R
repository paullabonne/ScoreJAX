packages <- c(
    "dplyr", "tidyr", "zoo",
    "magrittr", "arrow", "knitr"
)

install.packages(setdiff(
    packages,
    rownames(installed.packages())
), dependencies = TRUE)
