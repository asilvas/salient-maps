/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                           License Agreement
 //                For Open Source Computer Vision Library
 //
 // Copyright (C) 2014, OpenCV Foundation, all rights reserved.
 // Third party copyrights are property of their respective owners.
 //
 // Redistribution and use in source and binary forms, with or without modification,
 // are permitted provided that the following conditions are met:
 //
 //   * Redistribution's of source code must retain the above copyright notice,
 //     this list of conditions and the following disclaimer.
 //
 //   * Redistribution's in binary form must reproduce the above copyright notice,
 //     this list of conditions and the following disclaimer in the documentation
 //     and/or other materials provided with the distribution.
 //
 //   * The name of the copyright holders may not be used to endorse or promote products
 //     derived from this software without specific prior written permission.
 //
 // This software is provided by the copyright holders and contributors "as is" and
 // any express or implied warranties, including, but not limited to, the implied
 // warranties of merchantability and fitness for a particular purpose are disclaimed.
 // In no event shall the Intel Corporation or contributors be liable for any direct,
 // indirect, incidental, special, exemplary, or consequential damages
 // (including, but not limited to, procurement of substitute goods or services;
 // loss of use, data, or profits; or business interruption) however caused
 // and on any theory of liability, whether in contract, strict liability,
 // or tort (including negligence or otherwise) arising in any way out of
 // the use of this software, even if advised of the possibility of such damage.
 //
 //M*/

const cv = require('opencv4nodejs');
const { waitKey, Mat, Vec, Vec2, Size, Point, Point2, BORDER_DEFAULT, TermCriteria, KMEANS_RANDOM_CENTERS, CV_8U, CV_8UC1, CV_64FC2, CV_32F, CV_64F, threshold, THRESH_BINARY, THRESH_OTSU, BinaryMap, cvtColor, COLOR_BGR2GRAY, resize, merge, dft, split, cartToPolar, polarToCart, log, blur, exp, GaussianBlur, minMaxLoc, imshow, imshowWait, INTER_LINEAR_EXACT, DFT_INVERSE } = cv;
const Saliency = require('./Saliency');

const DEFAULT_CONFIDENCE = 7;
const FALSE_POSITIVE_THRESHOLD = 0.7;

const FACE_CLASSIFIERS = [
  /*{
    id: 'HAAR_EYE',
    classifier: new cv.CascadeClassifier(cv.HAAR_EYE),
    weight: 1.0
  },
  {
    id: 'HAAR_FRONTALCATFACE',
    classifier: new cv.CascadeClassifier(cv.HAAR_FRONTALCATFACE),
    weight: 1.0
  },
  {
    id: 'HAAR_FRONTALCATFACE_EXTENDED',
    classifier: new cv.CascadeClassifier(cv.HAAR_FRONTALCATFACE_EXTENDED),
    weight: 1.0
  },*/
  {
    id: 'HAAR_FRONTALFACE_ALT',
    classifier: new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT),
    minConfidence: DEFAULT_CONFIDENCE,
    weight: 1.0
  },
  {
    id: 'HAAR_FRONTALFACE_ALT2',
    classifier: new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2),
    minConfidence: DEFAULT_CONFIDENCE,
    weight: 1.0
  }/*,
  {
    id: 'HAAR_FRONTALFACE_DEFAULT',
    classifier: new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_DEFAULT),
    minConfidence: 8,
    weight: 1.0
  },
  {
    id: 'HAAR_LEFTEYE_2SPLITS',
    classifier: new cv.CascadeClassifier(cv.HAAR_LEFTEYE_2SPLITS),
    weight: 1.0
  },
  {
    id: 'HAAR_LOWERBODY',
    classifier: new cv.CascadeClassifier(cv.HAAR_LOWERBODY),
    minConfidence: 15,
    weight: 0.3
  }*/,
  {
    id: 'HAAR_PROFILEFACE',
    classifier: new cv.CascadeClassifier(cv.HAAR_PROFILEFACE),
    minConfidence: DEFAULT_CONFIDENCE,
    weight: 1.0
  }/*,
  {
    id: 'HAAR_UPPERBODY',
    classifier: new cv.CascadeClassifier(cv.HAAR_UPPERBODY),
    minConfidence: 18,
    weight: 0.5
  },
  {
    id: 'LBP_FRONTALCATFACE',
    classifier: new cv.CascadeClassifier(cv.LBP_FRONTALCATFACE),
    weight: 1.0
  },
  {
    id: 'LBP_FRONTALFACE',
    classifier: new cv.CascadeClassifier(cv.LBP_FRONTALFACE),
    minConfidence: 8,
    weight: 1.0
  },
  {
    id: 'LBP_FRONTALFACE_IMPROVED',
    classifier: new cv.CascadeClassifier(cv.LBP_FRONTALFACE_IMPROVED),
    minConfidence: 8,
    weight: 1.0
  },
  {
    id: 'LBP_PROFILEFACE',
    classifier: new cv.CascadeClassifier(cv.LBP_PROFILEFACE),
    minConfidence: 8,
    weight: 1.0
  }*/
];

module.exports = class SpectralResidual extends Saliency {
  computeSaliency(/* Mat */image, { imagePath, relPath } = {})
  {
    let gray;
    let mv = [];
    const size = new Size(this.width, this.height);

    let realImage = new Mat(size.height, size.width, CV_64F);
    const imaginaryImage = new Mat(size.height, size.width, CV_64F, 0);
    let combinedImage = new Mat(size.height, size.width, CV_64F, 0);
    let imageDFT;
    let logAmplitude_blur;
    let imageGR;

    const resized = image.resize(size, 0, 0, INTER_LINEAR_EXACT);

    if (image.channels === 3 || image.channels === 4)
    { // RGB(A)
      gray = resized.cvtColor(COLOR_BGR2GRAY);
    }
    else
    {
      gray = resized;
    }

    const maxY = Math.round(this.height * FALSE_POSITIVE_THRESHOLD);
    const minSize = new Size(Math.round(this.width * 0.03), Math.round(this.height * 0.03)); // 3%
    const maxSize = new Size(Math.round(this.width * 0.30), Math.round(this.height * 0.30)); // 30%
    let objects = [];
    const idealRegionSize = Math.round(this.width * 0.1) * Math.round(this.height * 0.1); // 10% x 10%
    FACE_CLASSIFIERS.forEach(({ classifier, id, minConfidence, weight }) => {
      const results = classifier.detectMultiScale(gray, 1.1, 1, 0/*, 0, minSize, maxSize*/);
      ((results.objects && results.objects) || results)
        .forEach((rect, idx) => {
          if ((rect.y + rect.height) > maxY) return;
          const confidence = (results.numDetections && results.numDetections[idx]) || (minConfidence * 2);
          if (confidence < minConfidence) return;
          //console.log(`confidence.${id}:`, confidence);
          objects.push({
            rect, id,
            weight: (weight * (confidence / 20) * (idealRegionSize / (rect.width * rect.height))),
            confidence
          });
        })
      ;
    });
    //objects.length > 0 && console.log('objects:', objects);

    realImage = gray.convertTo(CV_64F);

    mv.push(realImage);
    mv.push(imaginaryImage);

    combinedImage = new Mat(mv); // merge channels
    imageDFT = combinedImage.dft();
    
    mv = imageDFT.split(); // split channels
    let { magnitude, angle } = cartToPolar(mv[0], mv[1], false);
    
    const logAmplitude = magnitude.log();
    logAmplitude_blur = logAmplitude.blur(new Size(3, 3), new Point2(-1, -1), BORDER_DEFAULT);
    magnitude = logAmplitude.sub(logAmplitude_blur).exp();

    const { x, y } = polarToCart(magnitude, angle, false);

    imageDFT = new Mat([x, y]); // merge channels
    combinedImage = imageDFT.dft(DFT_INVERSE);
    mv = combinedImage.split(); // split channels

    let polar = cartToPolar(mv[0], mv[1], false);
    magnitude = polar.magnitude;
    angle = polar.angle;
    magnitude = magnitude.gaussianBlur(new Size(5, 5), 8, 0, BORDER_DEFAULT);
    magnitude = magnitude.hMul(magnitude);

    const { minVal, maxVal } = magnitude.minMaxLoc();

    objects.forEach(o => {
      intensifyRegion(magnitude, o.rect, 0.3/*o.weight*/, maxVal * 1.2);
    });
    magnitude = magnitude.div(maxVal);
    
    const saliencyMap = magnitude.convertTo(CV_32F);

    return {
      saliencyMap,
      objects
    };
  }
}

function intensifyRegion(mat, rect, weight, max) {
  let x, y, right, bottom, val;
  right = rect.x + rect.width;
  bottom = rect.y + rect.height;
  for (y = rect.y; y < bottom; y++) {
    for (x = rect.x; x < right; x++) {
      val = mat.at(y, x);
      val = Math.min(val + (weight * max), max);
      mat.set(y, x, val);
    }
  }
}