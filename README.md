# jax_sumo
Importance-weighted ELBO and [SUMO estimator](https://openreview.net/pdf?id=SylkYeHtwr) implemented in Jax, applied to a toy problem of approximate sampling from [Neal's funnel density](https://projecteuclid.org/euclid.aos/1056562461). 

See [associated blog post](https://justin-tan.github.io/blog/2020/06/20/Intuitive-Importance-Weighted-ELBO-Bounds) for more details.

## Usage
```
pip install -r requirements.txt
python3 train.py -h
```

### Samples from surrogate
![Image](/assets/density_0.png)
![Image](/assets/density_20000.png)
![Image](/assets/density_50000.png)