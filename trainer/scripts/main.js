const path = require('path');
const yargs = require('yargs');
const klawSync = require('klaw-sync')
const cv = require('opencv4nodejs');
const { Mat, Point, Vec, Rect, imshow, imreadAsync, destroyAllWindows, imread, imwrite, imshowWait, COLOR_GRAY2BGR, FILLED, INTER_LINEAR_EXACT, IMWRITE_PNG_COMPRESSION, IMWRITE_PNG_STRATEGY, IMWRITE_PNG_STRATEGY_DEFAULT, waitKey } = cv;
const models = require('../../src/models');
const { ensureDirSync } = require('fs-extra');
const autoFocus = require('salient-autofocus');
const shuffle = require('shuffle-array');
const fs = require('fs');
const readlineSync = require('readline-sync');

const argv = yargs
  .option('imageSource', {
    default: './trainer/image-source',
    type: 'string'
  })
  .option('imageSaliency', {
    default: './trainer/image-saliency',
    type: 'string'
  })
  .option('imageSaliencyTruth', {
    default: './trainer/image-saliency-truth',
    type: 'string'
  })
  .option('recursive', {
    default: true,
    type: 'boolean'
  })
  .option('save', {
    default: false,
    type: 'boolean'
  })
  .option('preview', {
    default: false,
    type: 'boolean'
  })
  .option('previewTimeout', {
    default: 3000,
    type: 'number'
  })
  .option('compare', {
    default: true,
    type: 'boolean'
  })
  .option('random', {
    default: true,
    type: 'boolean'
  })
  .option('top', {
    default: 0,
    type: 'number'
  })
  .option('models', {
    default: ['all'],
    type: 'array'
  })
  .option('width', {
    default: 300,
    type: 'number'
  })
  .option('height', {
    default: 300,
    type: 'number'
  })
  .argv
;

console.log('Querying available images...');

const imageSourcePath = path.resolve(argv.imageSource);
const imageSourceFilter = argv.recursive ? () => true : data => data.path.substr(imageSourcePath.length + 1).indexOf(path.sep) < 0;
let imageSource = klawSync(imageSourcePath, { nodir: true, filter: imageSourceFilter }).map(({ path }) => path);
if (argv.random) {
  imageSource = shuffle(imageSource);
}
if (argv.top > 0) {
  imageSource = imageSource.slice(0, argv.top);
}

const imageDestPath = path.resolve(argv.imageSaliency);

const imageTruthPath = path.resolve(argv.imageSaliencyTruth);
const imageTruthFilter = argv.recursive ? () => true : data => data.path.substr(imageTruthPath.length + 1).indexOf(path.sep) < 0;
const imageTruth = klawSync(imageTruthPath, { nodir: true, filter: imageTruthFilter }).map(({ path }) => path);

const filteredModels = Object.keys(models).filter(modelKey => {
  if (argv.models[0] === 'all') return true; // nothing more to do

  return argv.models.indexOf(modelKey) >= 0;
}).map(modelKey => { return { key: modelKey, title: models[modelKey].title, Model: models[modelKey].load() }; });

console.log(`Found ${imageSource.length} source images, ${imageTruth.length} truth images`);
/*
const facesPath = path.join(__dirname, 'faces.json');
const facesJSON = fs.readFileSync(facesPath);
const faces = JSON.parse(facesJSON);

console.log(`Found ${Object.keys(faces).length} faces.`);
*/
console.log(`Processing against ${filteredModels.length} models...`);

const POSITIVE_EXP = new RegExp('CAT2000\\\\(Action|Affective)');
let positiveMatches = 0;
let negativeMatches = 0;
let unknownMatches = 0;
let noMatches = 0;

function processImage(imagePath) {

  console.log(`Processing ${imagePath}...`);

  const relPath = imagePath.substr(imageSourcePath.length + 1);
  const friendlyName = path.basename(relPath);
  const truthPath = path.join(imageTruthPath, relPath);

  const image = imread(imagePath);
  let truthMap;
  try {
    truthMap = imread(truthPath);
  } catch (ex) {
    // MJ: JUST BEAT IT!
  }

  const results = filteredModels.map(({ key, Model }) => {
    const model = new Model({ width: argv.width, height: argv.height });
    const startTime = Date.now();
    const saliencyResult = model.computeSaliency(image);
    const saliencyMap = saliencyResult.saliencyMap || saliencyResult;
    const saliencyObjects = saliencyResult.objects;
    const timeElapsed = Date.now() - startTime;
    const saliencyMapOriginalSize = saliencyMap.resize(image.rows, image.cols, 0, 0, INTER_LINEAR_EXACT);
    //const binaryMap = model.computeBinaryMap(saliencyMap);
    const salientArray = saliencyMap.getDataAsArray();
    const salientMeta = autoFocus.getMetaFromSalientMatrix(salientArray);

    // TODO: compare with truthMap

    if (argv.save === true) {
      // write back the original size
      const dstPath = path.join(imageDestPath, key, relPath);
      ensureDirSync(path.dirname(dstPath));
      imwrite(dstPath, saliencyMapOriginalSize.convertTo(cv.CV_8UC3, 255.0));
    }

    console.log(`${key} took ${timeElapsed}ms`);
/* experimental stuff
    if (saliencyObjects && saliencyObjects.length > 0) {
      const face = faces[relPath];
      if (face === true || POSITIVE_EXP.test(relPath)) {
        positiveMatches++;
      } else if (face === false) {
        negativeMatches++;
      } else {
        unknownMatches++;
        faces[relPath] = null;
        console.log('UNKNOWN', relPath, imagePath);
        //imshow('resized', resized);
        //imshow('saliency', saliencyMap);
        //waitKey(3000);
      }
      console.log('positives:', positiveMatches, 'negatives:', negativeMatches, 'unknowns:', unknownMatches, ' non-matches:', noMatches);
    } else if (saliencyObjects) {
      noMatches++;
    }
*/
    return {
      key,
      saliencyMap,
      salientMeta,
      //binaryMap,
      saliencyMapOriginalSize,
      truthMap,
      timeElapsed
    };
  });

  let windowX = 0;
  let windowY = 0;
  const showNextWindow = (title, image) => {
    imshow(title, image);
    cv.moveWindow(title, windowX, windowY);
    windowX += argv.width;
    if (windowX + argv.width > 1200) {
      windowX = 0;
      windowY += argv.height + 45;
    }
  };

  if (argv.preview === true) {
    destroyAllWindows();
    let originalPreview = paintPreviewWithAutoFocus(image.resize(argv.height, argv.width, 0, 0, INTER_LINEAR_EXACT), results[0].salientMeta);
    showNextWindow(`Original: ${relPath}`, originalPreview);
    results.forEach(result => {
      //imshow(`BIN: ${result.key}`, result.binaryMap);
      showNextWindow(`SAL: ${result.key}`, paintPreviewWithAutoFocus(result.saliencyMap, result.salientMeta));
    });
    if (results[0].truthMap) showNextWindow('TRUTH', results[0].truthMap);
    //console.log('Press any key to continue.');
    //readlineSync.keyInPause();
    waitKey(argv.previewTimeout);
  }
}

function paintPreviewWithAutoFocus(image, meta) {
  const preview = image.channels === 1 ? image.convertTo(cv.CV_8UC3, 255.0).cvtColor(COLOR_GRAY2BGR) : image.copy();
  preview.drawCircle(new Point(meta.c.x * image.cols, meta.c.y * image.rows), 4, new Vec(255, 0, 0), -1, FILLED);
  meta.r25th && preview.drawRectangle(new Rect(meta.r25th.l * image.cols, meta.r25th.t * image.rows, meta.r25th.w * image.cols, meta.r25th.h * image.rows), new Vec(100, 100, 255));
  meta.r40th && preview.drawRectangle(new Rect(meta.r40th.l * image.cols, meta.r40th.t * image.rows, meta.r40th.w * image.cols, meta.r40th.h * image.rows), new Vec(150, 150, 255));
  meta.r50th && preview.drawRectangle(new Rect(meta.r50th.l * image.cols, meta.r50th.t * image.rows, meta.r50th.w * image.cols, meta.r50th.h * image.rows), new Vec(0, 0, 255));
  meta.r75th && preview.drawRectangle(new Rect(meta.r75th.l * image.cols, meta.r75th.t * image.rows, meta.r75th.w * image.cols, meta.r75th.h * image.rows), new Vec(0, 255, 255));
  meta.r90th && preview.drawRectangle(new Rect(meta.r90th.l * image.cols, meta.r90th.t * image.rows, meta.r90th.w * image.cols, meta.r90th.h * image.rows), new Vec(0, 255, 0));

  let salientRegion;

  salientRegion = autoFocus.getRegionFromMeta(meta, { imageWidth: preview.cols, imageHeight: preview.rows, regionWidth: Math.round(preview.cols / 1.5), regionHeight: Math.round(preview.cols / 1.5 * 0.6) });
  preview.drawRectangle(new Rect(salientRegion.left, salientRegion.top, salientRegion.width, salientRegion.height), new Vec(255, 255, 0));

  salientRegion = autoFocus.getRegionFromMeta(meta, { imageWidth: preview.cols, imageHeight: preview.rows, regionWidth: Math.round(preview.cols / 1.5 * 0.6), regionHeight: Math.round(preview.cols / 1.5) });
  preview.drawRectangle(new Rect(salientRegion.left, salientRegion.top, salientRegion.width, salientRegion.height), new Vec(255, 255, 0));
  
  return preview;
}

imageSource.forEach(imagePath => processImage(imagePath));
/*
console.log(`Writing ${Object.keys(faces).length} face results.`);

fs.writeFileSync(facesPath, JSON.stringify(faces, null, '  '));
*/