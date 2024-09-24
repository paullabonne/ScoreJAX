# ScoreJAX

JAX (Bradbury et al. 2018) implementation of the score-driven model
featuring location, scale and shape common factors introduced in Labonne
P. (2024). “Asymmetric uncertainty: Nowcasting using skewness in
real-time data”. *International Journal of Forecasting*

JAX adds automatic differentiation and high-performance numerical
computing features to code written in Python.

#### First install all necessary python libraries

``` python
%%capture

! pip install -r requirements.txt
```

#### R code for building a dataframe from fred-md vintages. The dataframe is saved in the `arrow` `parquet` format for easy interoperability with Python.

``` python
%%capture

# first install all necessary packages
! Rscript "R/install.R"

# build the dataframe
! Rscript "R/fredmd.R"
```

#### Load the dataframe in `Python`

``` python
import pyarrow.parquet as pq

# load the data
df = pq.read_table("data/df_fredmd.parquet").to_pandas()
```

#### Estimation with maximum likelihood

``` python
import sys
sys.path.append('src') # to find sdfm
from sdfm import build_model
from sdfm import mle

import jax
import jax.numpy as jnp
from jax import random

# convert the data to a JAX type
df_np = df.to_numpy() # for easy use with JAX
y = jnp.array(df_np[:, 1:])

# build the model
slack_model = build_model(y)

# estimation with maximum likelihood
key = jax.random.PRNGKey(123)
mle_result = mle(model=slack_model, iter=100, pertu=0.25, key=key)

print("ML at", -mle_result.fun)
```

    100%|██████████| 100/100 [01:44<00:00,  1.04s/it]

    ML at -4149.11669921875

#### Run the filter with the estimated parameters

``` python

from sdfm import sd_filter
estimated_filter = sd_filter(mle_result.x, slack_model)
```

#### Plot of the estimated common factors

![](README_files/figure-commonmark/cell-7-output-1.png)

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-jax2018github" class="csl-entry">

Bradbury, James, Roy Frostig, Peter Hawkins, Matthew James Johnson,
Chris Leary, Dougal Maclaurin, George Necula, et al. 2018. “JAX:
Composable Transformations of Python+NumPy Programs.”
<http://github.com/google/jax>.

</div>

</div>
