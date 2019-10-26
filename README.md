# 6 GHz two RX channel FMCW radar design files.

For more information see: http://hforsten.com/third-version-of-homemade-6-ghz-fmcw-radar.html

## SAR image formation steps:

`pc/sar/process_sweeps.py` is used to slice the correct time from the measurement file.

`./process_sweeps.py <log file> <start time> <end time> [decimation factor]`

It writes `sweeps.p` file with the data in the format the image formation scripts know how to read.
To form the SAR image run `./backprojection_tf.py sweeps.p` or `./sar_tf_autofocus.py sweeps.p`.
Some variables related to the image formation can be found on the top of the image formation scripts.

Example data can be downloaded from: http://hforsten.com/fmcw3/parking_lot_sar.log

Use start and end times of 20 and 136 for it. Decimation factor should be 1 or
2 for backprojection and 2 for omega-k.
