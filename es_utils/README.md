# Elasticsearch Utils

We provide some scripts to manage the Elasticsearch indices.

To delete indices with specifice name:
```sh
python delete_index.py --index [index_name]
```

To check the existing indices:
```sh
python list_index.py
```

To use elasticsearch without verifying the certificate, please add below lines to elasticsearch/config/elasticsearch.yml:
```sh
xpack.security.enabled: false

xpack.security.enrollment.enabled: false

#----------------------- END SECURITY AUTO CONFIGURATION -------------------------
cluster.routing.allocation.disk.threshold_enabled: true
cluster.routing.allocation.disk.watermark.flood_stage: 5gb
cluster.routing.allocation.disk.watermark.low: 30gb
cluster.routing.allocation.disk.watermark.high: 20gb
```