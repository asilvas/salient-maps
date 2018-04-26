const cv = require('opencv4nodejs');
const { Mat, termCriteria, Point2, TermCriteria, kmeans, KMEANS_RANDOM_CENTERS, CV_64F, CV_8U, threshold, THRESH_BINARY, THRESH_OTSU, BinaryMap } = cv;

module.exports = class Saliency {
  constructor({ width, height } = {}) {
    this.width = width || 200;
    this.height = height || 200;
  }

  computeSaliency() {
    throw new Error('computeSaliency not implemented');
  }
  
  computeBinaryMap(/* Mat */ saliencyMap)
  {
    //const labels = new Mat(saliencyMap.rows * saliencyMap.cols, 1, 1, 0);
    const samples = [];
    const terminationCriteria = new TermCriteria(termCriteria.COUNT + termCriteria.EPS /* type */, 1000 /* maxCount */, 0.2 /* epsilon */);
    let elemCounter = 0;
    for (let i = 0; i < saliencyMap.rows; i++)
    {
      for (let j = 0; j < saliencyMap.cols; j++)
      {
        samples.push(new Point2(saliencyMap.at(i, j), 0));
        elemCounter++;
      }
    }

    const { centers, labels } = kmeans(samples /* <cv::Point2f>data */, 5 /* k */, terminationCriteria, 5 /* attempts */, KMEANS_RANDOM_CENTERS /* flags */);

    let outputMat = new Mat(saliencyMap.rows, saliencyMap.cols, CV_64F, 0);
    let intCounter = 0;
    for (let x = 0; x < saliencyMap.rows; x++)
    {
      for (let y = 0; y < saliencyMap.cols; y++)
      {
        const center = centers[labels[intCounter]];
        outputMat.set(x, y, center.x);
        intCounter++;
      }
    }

    //Convert
    outputMat = outputMat.mul(255);
    outputMat = outputMat.convertTo(CV_8U);

    // adaptative thresholding using Otsu's method, to make saliency map binary
    return outputMat.threshold(0, 255, THRESH_BINARY | THRESH_OTSU);
  }
}