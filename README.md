# POBR_Company-logo-recognition
The goal of this project is to implement processing pipeline for detecting logos in provided image. In this project I detect Microsft logo. All necessery algorithms are implemented from scratch.

## Pipline steps
* Load image to memory - OpenCV
* Scale down to reduce further computations
* Histogram equalization in HSV color space
* Low-pass filtration using convolution in RGB space
* Final conversion to HSV
* Separate thresholding for every color in the logo (red, green, blue, yellow)
* Segment detection using the flood fill algorithm. Calculation of descriptive features per segment.
* (Optional) Closing (dilation followed by erosion). Recalculation of descriptive features if necessary.
* Segments filtration based on descriptive features.
* Identification using topological correspondence.

The project was aimed at practicing and implementing typical operations in the image recognition processing pipeline. Accordingly, the implementation is rather straight-forward.

## Obtained results
<img src="/doc/ms_building_1_rec.png" width="400" />
<img src="/doc/ms_building_2_rec.png" width="400" />
<img src="/doc/ms_building_3_rec.png" width="400" />
<img src="/doc/ms_multi_logo_rec.png" width="400" />
<img src="/doc/ms_diods_rec.png" width="400" />
