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
const { Mat, Vec, Vec2, Size, Point, Point2, BORDER_DEFAULT, TermCriteria, KMEANS_RANDOM_CENTERS, CV_8U, CV_8UC1, CV_64FC2, CV_32F, CV_64F, threshold, THRESH_BINARY, THRESH_OTSU, BinaryMap, cvtColor, COLOR_BGR2GRAY, resize, merge, dft, split, cartToPolar, polarToCart, log, blur, exp, GaussianBlur, minMaxLoc, imshow, imshowWait, INTER_LINEAR_EXACT, DFT_INVERSE } = cv;
const Saliency = require('./Saliency');

module.exports = class SpectralResidual extends Saliency {
  computeSaliency(/* Mat */image)
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

    if (image.channels === 3 || image.channels === 4)
    { // RGB(A)
      imageGR = image.cvtColor(COLOR_BGR2GRAY);
      gray = imageGR.resize(size, 0, 0, INTER_LINEAR_EXACT);
    }
    else
    {
      gray = image.resize(size, 0, 0, INTER_LINEAR_EXACT);
    }

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

    magnitude = magnitude.div(maxVal);

    const saliencyMap = magnitude.convertTo(CV_32F);

    return saliencyMap;
  }
}
