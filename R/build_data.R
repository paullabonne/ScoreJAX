library(dplyr)
library(tidyr)
library(zoo)
library(magrittr)
library(arrow)
library(knitr)

# load the fred md vintages
load("data/vintages.Rdata")

# get the last vintage in tibble
last_vintage <- length(list_vintages)
df <- list_vintages[[last_vintage]]$df
df <- as_tibble(df)

# vars to select
selection_var <- c(
    "RPI", "INDPRO", "CUMFNS",
    "CE16OV", "UNRATE", "PAYEMS",
    "DPCERA3M086SBEA", "RETAILx",
    "S&P 500"
)

# clean
df %<>%
    mutate(date = yearmon(date)) %>%
    select(date, all_of(selection_var)) %>%
    relocate(date, INDPRO)

# standardising all variables
df %<>%
    pivot_longer(-date, names_to = "series", values_to = "values") %>%
    group_by(series) %>%
    mutate(
        values = values / sd(values, na.rm = T),
        values = values - mean(values, na.rm = T)
    ) %>%
    ungroup() %>%
    pivot_wider(names_from = series, values_from = values) %>%
    filter(date > "Jan 1959")

N <- nrow(df)
cat("First and last 5 rows of the dataframe:")
kable(df[c(1:5, (N - 4):N), ])

# save as a parquet file
df %<>% mutate(date = as.numeric(date))
write_parquet(df, "data/df.parquet")
