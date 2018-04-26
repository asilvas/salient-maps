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
const { Mat, Vec, Vec2, Size, Point, Point2, BORDER_DEFAULT, TermCriteria, KMEANS_RANDOM_CENTERS, CV_8U, CV_8UC1, CV_64FC2, CV_32F, CV_64F, CV_16UC1, threshold, THRESH_BINARY, THRESH_OTSU, BinaryMap, cvtColor, COLOR_BGR2GRAY, resize, merge, dft, split, cartToPolar, polarToCart, log, blur, exp, GaussianBlur, minMaxLoc, imshow, imshowWait, INTER_LINEAR_EXACT, DFT_INVERSE } = cv;
const logMat = require('./utils/log-mat');
const Saliency = require('./Saliency');

const neighborhoods = [3*4, 3*4*2, 3*4*2*2, 7*4, 7*4*2, 7*4*2*2];

module.exports = class FineGrained extends Saliency {
  computeSaliency(/* Mat */image)
  {
    const size = new Size(this.width, this.height);

    const numScales = 6;
    const intensityScaledOn = new Array(numScales);
    const intensityScaledOff = new Array(numScales);
    let gray;

    let integralImage;
    const intensityOn = new Mat(size.height, size.width, CV_8UC1, 0);
    const intensityOff = new Mat(size.height, size.width, CV_8UC1, 0);
    let i;

    for (i = 0; i < numScales; i++)
    {
      intensityScaledOn[i] = new Mat(size.height, size.width, CV_8UC1, 0);
      intensityScaledOff[i] = new Mat(size.height, size.width, CV_8UC1, 0);
    }

    if (image.channels === 3 || image.channels === 4)
    { // RGB(A)
      gray = image.cvtColor(COLOR_BGR2GRAY);
    } else {
      gray = image;
    }

    gray = gray.resize(size, 0, 0, INTER_LINEAR_EXACT).convertTo(CV_32F);

    // smooth pixels at least twice, as done by Frintrop and Itti
    gray = gray.gaussianBlur(new Size(3, 3), 0, 0);
    //gray = gray.gaussianBlur(new Size(3, 3), 0, 0);

    const integral = gray.integral(CV_32F);
    //console.log('integralImage', integral, CV_32F);
    integralImage = integral.sum;
    //console.log('integral');
    //logMat(integral.sum);
    //logMat(integral.sqsum);
    //logMat(integral.tilted);
    for (i = 0; i < numScales; i++)
    {
      getIntensityScaled(integralImage, gray, intensityScaledOn[i], intensityScaledOff[i], neighborhoods[i]);
    }

    mixScales(intensityScaledOn, intensityOn, intensityScaledOff, intensityOff, numScales);

    let intensity = mixOnOff(intensityOn, intensityOff);
    //console.log('intensity');
    //logMat(intensity);

    const { minVal, maxVal } = intensity.minMaxLoc();

    intensity = intensity.div(maxVal);

    const saliencyMap = intensity;
    
    //console.log('saliencyMap');
    //logMat(saliencyMap);

    return saliencyMap;
  }
}

function getIntensityScaled(/*Mat */integralImage, /*Mat */gray, /*Mat */intensityScaledOn, /*Mat */intensityScaledOff, /*int */neighborhood)
{
    let value, meanOn, meanOff;
    let x, y;
    //intensityScaledOn.setTo(Scalar::all(0));
    //intensityScaledOff.setTo(Scalar::all(0));

    for(y = 0; y < gray.rows; y++)
    {
        for(x = 0; x < gray.cols; x++)
        {
            value = getMean(integralImage, new Point(x, y), neighborhood, gray.at(y, x));

            meanOn = gray.at(y, x) - value;
            meanOff = value - gray.at(y, x);

            if(meanOn > 0)
                intensityScaledOn.set(y, x, meanOn);
            else
                intensityScaledOn.set(y, x, 0);

            if(meanOff > 0)
                intensityScaledOff.set(y, x, meanOff);
            else
                intensityScaledOff.set(y, x, 0);
        }
    }
}

function getMean(/*Mat */srcArg, /*Point2i */PixArg, /*int */neighbourhood, /*int */centerVal)
{
  const P1 = new Point(Math.min(Math.max(PixArg.x - neighbourhood + 1, 0), srcArg.cols - 1), Math.min(Math.max(PixArg.y - neighbourhood + 1, 0), srcArg.rows - 1));
  const P2 = new Point(Math.min(Math.max(PixArg.x + neighbourhood + 1, 0), srcArg.cols - 1), Math.min(Math.max(PixArg.y + neighbourhood + 1, 0), srcArg.rows - 1));
  let value;

  value = (
    (srcArg.at(P2.y, P2.x)) +
    (srcArg.at(P1.y, P1.x)) -
    (srcArg.at(P2.y, P1.x)) -
    (srcArg.at(P1.y, P2.x))
  );
  value = (value - centerVal) / (( (P2.x - P1.x) * (P2.y - P1.y)) - 1);

  return value;
}

function mixScales(/*Mat **/intensityScaledOn, /*Mat */intensityOn, /*Mat **/intensityScaledOff, /*Mat */intensityOff, /*const int */numScales)
{
  let i=0, x, y;
  let width = intensityScaledOn[0].cols;
  let height = intensityScaledOn[0].rows;
  let maxValOn = 0;
  let currValOn = 0;
  let maxValOff = 0;
  let currValOff = 0;
  let maxValSumOff = 0;
  let maxValSumOn = 0;
  let mixedValuesOn = new Mat(height, width, CV_16UC1, 0);
  let mixedValuesOff = new Mat(height, width, CV_16UC1, 0);

  //mixedValuesOn.setTo(Scalar::all(0));
  //mixedValuesOff.setTo(Scalar::all(0));

  for(i=0;i<numScales;i++)
  {
    for(y=0;y<height;y++) {
      for(x=0;x<width;x++)
      {
        currValOn = intensityScaledOn[i].at(y, x);
        if(currValOn > maxValOn)
          maxValOn = currValOn;

        currValOff = intensityScaledOff[i].at(y, x);
        if(currValOff > maxValOff)
          maxValOff = currValOff;

        mixedValuesOn.set(y, x, mixedValuesOn.at(y, x) + currValOn);
        mixedValuesOff.set(y, x, mixedValuesOff.at(y, x) + currValOff);
      }
    }
  }

  for(y = 0; y < height; y++) {
    for(x = 0; x < width; x++)
    {
      currValOn = mixedValuesOn.at(y, x);
      currValOff = mixedValuesOff.at(y, x);
      if(currValOff > maxValSumOff)
        maxValSumOff = currValOff;
      if(currValOn > maxValSumOn)
        maxValSumOn = currValOn;
    }
  }

  for(y = 0; y < height; y++) {
    for(x = 0; x < width; x++)
    {
      intensityOn.set(y, x, (255 * ((mixedValuesOn.at(y, x) / maxValSumOn))));
      intensityOff.set(y, x, (255 * ((mixedValuesOff.at(y, x) / maxValSumOff))));
    }
  }
}

function mixOnOff(/*Mat */intensityOn, /*Mat */intensityOff, /*Mat */intensityArg)
{
  let x, y;
  let width = intensityOn.cols;
  let height = intensityOn.rows;
  let maxVal = 0;

  let currValOn, currValOff, maxValSumOff, maxValSumOn;

  const intensity = new Mat(height, width, CV_8UC1, 0);

  maxValSumOff = 0;
  maxValSumOn = 0;

  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      currValOn = intensityOn.at(y, x);
      currValOff = intensityOff.at(y, x);
      if(currValOff > maxValSumOff)
          maxValSumOff = currValOff;
      if(currValOn > maxValSumOn)
          maxValSumOn = currValOn;
    }
  }

  if(maxValSumOn > maxValSumOff)
    maxVal = maxValSumOn;
  else
    maxVal = maxValSumOff;

  for(y=0;y<height;y++) {
    for(x=0;x<width;x++)
    {
        intensity.set(y, x, 255. * (intensityOn.at(y, x) + intensityOff.at(y, x)) / maxVal);
    }
  }

  return intensity;
}
