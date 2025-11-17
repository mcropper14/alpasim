# Viewing performance results for Alpasim-runtime tests

Alpasim-runtime will generate a file, `alpasim-runtime.prom` for each test, capturing the performance of API requests over the period of that test.

To view a Grafana dashboard containing performance results for a given test, run the `view-metrics.sh` script in this repo, providing a `--metrics-dir` containing `alpasim-runtime.prom.

```sh
# Example: view metrics for local directory
./tools/metrics/view-metrics.sh --metrics-dir /tmp/tmp.yLBX41OwHZ
```
