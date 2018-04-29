# salient-maps

Various open source salient maps.

![](https://github.com/asilvas/salient-maps/blob/master/docs/images/maps.jpg)


# Developer Usage

Example using the Deep Gaze model.

```
const models = require('salient-maps');
const cv = require('opencv4nodejs');

const Deep = models.deep.load();
const deep = new Deep({ width: 200, height: 200 });
const salientMap = deep.computeSaliency(cv.imread('myimage.jpg'));
```

## Options

| Option | Type | Default | Info |
| --- | --- | --- | --- |
| width | `number` | `200` | Width of saliency map. It's not recommended to go above 300 or below 100. |
| height | `number` | `200` | Height of saliency map. It's not recommended to go above 300 or below 100. |


# What to do with salient map?

While it's entirely up to you how use these maps, the original intent of this project was to
be paired with the [salient-autofocus](https://github.com/asilvas/salient-autofocus) project
for providing fast image auto-focus capabilities.

![](https://github.com/asilvas/salient-maps/blob/master/docs/images/salient7.jpg)
![](https://github.com/asilvas/salient-maps/blob/master/docs/images/salient8.jpg)
![](https://github.com/asilvas/salient-maps/blob/master/docs/images/salient9.jpg)
![](https://github.com/asilvas/salient-maps/blob/master/docs/images/salient10.jpg)
![](https://github.com/asilvas/salient-maps/blob/master/docs/images/salient11.jpg)
![](https://github.com/asilvas/salient-maps/blob/master/docs/images/salient6.jpg)


# Models

| ID | Description | License | Usage |
| --- | --- | --- | --- |
| deep | [MIT](https://github.com/mpatacchiola/deepgaze/blob/master/deepgaze/saliency_map.py) | [Deep Gaze](https://github.com/mpatacchiola/deepgaze/blob/master/deepgaze/saliency_map.py) port of FASA (Fast, Accurate, and Size-Aware Salient Object Detection) algorithm | Recommended for most static usage where high accuracy is important, and near-realtime is sufficient performance (tunable by reducing map size). May not be ideal for video unless you drop map size to 150^2 or lower. |
| deep-rgb | [MIT](https://github.com/mpatacchiola/deepgaze/blob/master/deepgaze/saliency_map.py) | A varient of [Deep Gaze](https://github.com/mpatacchiola/deepgaze/blob/master/deepgaze/saliency_map.py) port but leveraging the RGB colour space instead of LAB. | Not recommended. Useful for comparison. Can perform better. |
| spectral | [BSD](https://github.com/opencv/opencv_contrib/blob/master/modules/saliency/src/staticSaliencySpectralResidual.cpp) | A port of the [Spectral Residual](https://github.com/opencv/opencv_contrib/blob/master/modules/saliency/src/staticSaliencySpectralResidual.cpp) model from [OpenCV Contributions](https://github.com/opencv/opencv_contrib). | Amazing performance, great for video, but at the cost of quality/accuracy. |
| fine | [BSD](https://github.com/opencv/opencv_contrib/blob/master/modules/saliency/src/staticSaliencyFineGrained.cpp) | A port of the [Fine Grained](https://github.com/opencv/opencv_contrib/blob/master/modules/saliency/src/staticSaliencyFineGrained.cpp) model from [OpenCV Contributions](https://github.com/opencv/opencv_contrib). | Interesting for testing but useless for realtime applications. |



# Want to contribute?

## Installation

Typical local setup.

```
git clone git@github.com:asilvas/salient-maps.git
cd salient-maps
npm i
```

## Import Assets

By default testing looks at `trainer/image-source`, so you can put any images you like there.
Or follow the below instructions to import a known dataset.

1. Download and extract [CAT2000](http://saliency.mit.edu/testSet.zip)
2. Run `node trainer/scripts/import-CAT2000.js {path-to-CAT2000}`

The benefit of using the above script is it'll seperate the truth maps into `trainer/image-truth`,
which are optional.


## Preview

You can run visual previews of the available saliency maps against the dataset via:

```
npm run preview
```


## Benchmark

If all you're interested in is determing performance data:

```
npm run benchmark
```


## Export

Also available is the ability to export the salient map data to `trainer/image-saliency` folder, broken
down by the saliency model. This permits review of maps from disk, in addition to being in a convenient
format for submission to the [mit saliency benchmark](http://saliency.mit.edu/submission.html) for
quality analysis against other models.

```
npm run export
```


# License

While this project falls under an MIT license, each of the models are subject to their own license.
See [Models](#models) for details.
