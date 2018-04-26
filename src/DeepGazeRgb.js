const cv = require('opencv4nodejs');
const { waitKey, Mat, Vec, Vec2, Size, Point, Point2, BORDER_DEFAULT, TermCriteria, KMEANS_RANDOM_CENTERS, CV_8U, CV_8UC1, CV_64FC2, CV_32F, CV_64F, threshold, THRESH_BINARY, THRESH_OTSU, BinaryMap, cvtColor, COLOR_BGR2GRAY, resize, merge, dft, split, cartToPolar, polarToCart, log, blur, exp, GaussianBlur, minMaxLoc, imshow, imshowWait, INTER_LINEAR_EXACT, DFT_INVERSE } = cv;
const DeepGaze = require('./DeepGaze');

module.exports = class SpectralResidual extends DeepGaze {
  constructor(options) {
    super(options);

    this.convertToLAB = false;
  }
}
