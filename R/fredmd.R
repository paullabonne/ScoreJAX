library(dplyr)
library(tidyr)
library(zoo)
library(arrow)
library(knitr)
library(magrittr)

## transformations to make series stationary (from the fredmd working paper)
fred_transf <- function(series, type) {
    if (type == 2) {
        series <- series - lag(series)
    } else if (type == 3) {
        series <- series - lag(series)
        series <- series - lag(series)
    } else if (type == 4) {
        series <- log(series)
    } else if (type %in% 5) {
        series <- log(series)
        series <- series - lag(series)
    } else if (type == 6) {
        series <- log(series)
        series <- series - lag(series)
        series <- series - lag(series)
    } else if (type == 7) {
        series <- series / lag(series) - 1
        series <- series - lag(series)
    }

    return(series)
}

# selected series
selected_var <- c(
    "RPI", "INDPRO", "CUMFNS",
    "CE16OV", "UNRATE", "PAYEMS",
    "DPCERA3M086SBEA", "RETAILx",
    "S&P 500"
)

# load the vintage
fredmd <- read.csv("data/fredmd/2024-04.csv")

# extract the transformation codes
transforms <- tibble(
    type = unlist(fredmd[1, -1]),
    series = colnames(fredmd)[-1]
)

# build the dataset
df_fredmd <- as_tibble(fredmd[-1, ]) %>%
    rename(date = sasdate) %>%
    mutate(date = as.yearmon(date, "%m/%d/%Y")) %>%
    pivot_longer(-date, names_to = "series", values_to = "values") %>%
    left_join(transforms, by = "series") %>%
    arrange(series, date) %>%
    group_by(series) %>%
    mutate(
        values = fred_transf(values, type[1]),
        values = values - mean(values, na.rm = T),
        values = values / sd(values, na.rm = T)
    ) %>%
    ungroup() %>%
    select(-type) %>%
    filter(
        series %in% selected_var,
        date > "Jan 1959"
    ) %>%
    pivot_wider(names_from = series, values_from = values) %>%
    relocate(date, INDPRO)

# preview
N <- nrow(df_fredmd)
cat("First and last 5 rows of the dataframe:")
kable(df_fredmd[c(1:5, (N - 4):N), ])

# save as a parquet file
df_fredmd %<>% mutate(date = as.numeric(date))
write_parquet(df_fredmd, "data/df_fredmd.parquet")
