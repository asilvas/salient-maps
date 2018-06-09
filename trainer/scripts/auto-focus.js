const path = require('path');
const yargs = require('yargs');
const cv = require('opencv4nodejs');
const { Mat, Point, Vec, Rect, imshow, imreadAsync, destroyAllWindows, imread, imwrite, imdecode, imdecodeAsync, imshowWait, COLOR_GRAY2BGR, FILLED, INTER_LINEAR_EXACT, IMWRITE_PNG_COMPRESSION, IMWRITE_PNG_STRATEGY, IMWRITE_PNG_STRATEGY_DEFAULT, waitKey } = cv;
const models = require('../../src/models');
const { ensureDirSync } = require('fs-extra');
const autoFocus = require('salient-autofocus');
const shuffle = require('shuffle-array');
const fs = require('fs');
const readlineSync = require('readline-sync');
const async = require('async');
const md5 = require('md5');
const request = require('request');

const argv = yargs
  .option('source', {
    default: './trainer/auto-focus/stock.json',
    type: 'string'
  })
  .option('cache', {
    default: './trainer/auto-focus/cache',
    type: 'string'
  })
  .option('random', {
    default: false,
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
    default: 200,
    type: 'number'
  })
  .option('height', {
    default: 200,
    type: 'number'
  })
  .option('cropRatio', {
    default: 0.75,
    type: 'number'
  })
  .option('minPoints', {
    default: 1,
    type: 'number'
  })
  .option('parallelLimit', {
    default: 8,
    type: 'number'
  })
  .option('removeOutliers', {
    default: 0.20, // 20% of furthest from mean
    type: 'number'
  })
  .argv
;

const sourcePath = path.resolve(argv.source);
const cachePath = path.resolve(argv.cache);
ensureDirSync(cachePath);
const sourceHash = JSON.parse(fs.readFileSync(sourcePath, 'utf8'));
let source = Object.keys(sourceHash).map(url => ({ url, points: sourceHash[url] })).filter(({ points }) => points.length >= argv.minPoints);
if (argv.random) {
  source = shuffle(source);
}
if (argv.top > 0) {
  source = source.slice(0, argv.top);
}

const filteredModels = Object.keys(models).filter(modelKey => {
  if (argv.models[0] === 'all') return true; // nothing more to do

  return argv.models.indexOf(modelKey) >= 0;
}).map(modelKey => { return { key: modelKey, title: models[modelKey].title, Model: models[modelKey].load() }; });

console.log(`Found ${source.length} source images...`);

console.log(`Processing against ${filteredModels.length} models...`);

// scrub focal points
source.forEach(o => {
  let { url, points } = o;

  let sum = points.reduce((state, cur) => {
    state.x += cur.x;
    state.y += cur.y;
    return state;
  }, { x: 0, y: 0 });

  let mean = o.mean = { x: sum.x / points.length, y: sum.y / points.length };

  points.forEach(pt => {
    pt.distance = (Math.abs(mean.x - pt.x) + Math.abs(mean.y - pt.y)) / 2;
  });

  points.sort((a, b) => {
    return a.distance - b.distance;
  });
  
  const outliers = Math.round(points.length * argv.removeOutliers);
  if (outliers > 0) {
    // remove outliers
    o.points = points = points.slice(0, points.length - outliers);

    // re-compute sum
    sum = points.reduce((state, cur) => {
      state.x += cur.x;
      state.y += cur.y;
      return state;
    }, { x: 0, y: 0 });

    // re-compute mean
    mean = o.mean = { x: sum.x / points.length, y: sum.y / points.length };

    // we don't need to re-compute/re-order distance despite it being incorrect as it's served its purpose
  }

  // grab median
  const median = o.median =
    points.length < 3 ? points[0]
    : points[Math.round(points.length / 2)]
  ;
});

const accuracy = {
  models: {
    center: { sum: 0 } // auto-injected
  },
  results: 0,
  tasks: 0
};

// init accuracy.models
filteredModels.forEach(model => accuracy.models[model.key] = { sum: 0 });

function displayStatus() {
  console.log(`Processed ${accuracy.tasks} images (${((accuracy.results / sourceTasks.length) * 100).toFixed(2)}%) complete`);
  Object.keys(accuracy.models).forEach(key => {
    const percent = 100 - ((accuracy.models[key].sum / accuracy.results) * 100);

    console.log(`* Model[${key}].accuracy = ${percent.toFixed(3)}%`);
  });
}

function createImageTask({ url, mean }) {
  // use hash for path instead of url
  const imageCache = path.join(cachePath, md5(url) + '.jpg');
  return cb => {
    async.auto({
      readFromDisk: cb => fs.readFile(imageCache, (err, data) => cb(null, data)),
      readFromUrl: ['readFromDisk', (results, cb) => {
        if (results.readFromDisk) return cb();

        request({
          url: /^http\:/.test(url) ? '' : 'http:' + url,
          method: 'GET',
          encoding: null
        }, (err, res, body) => {
          //if (err) return cb(err);
          //if (res.statusCode !== 200) return cb(new Error(`GET ${url} returned ${res.statusCode}`));

          // writing error is desired as it'll result in creation of bad cache file to avoid future url hits
          cb(null, body || (err && err.message) || res.statusCode);
        });
      }],
      writeToDisk: ['readFromUrl', (results, cb) => {
        if (!results.readFromUrl) return cb();

        fs.writeFile(imageCache, results.readFromUrl, cb);
      }],
      processedModels: ['readFromDisk', 'readFromUrl', 'writeToDisk', (results, cb) => {
        const imageData = results.readFromDisk || results.readFromUrl;
        if (!imageData || imageData.length < 1000) return cb(new Error(`${url} is only ${(imageData && imageData.length) || 0}B`));
        let image;
        try {
          image = imdecode(imageData);
        } catch (ex) {
          cb(ex);
        }

        const modelResults = {
          center: (Math.abs(0.5 - mean.x) + Math.abs(0.5 - mean.y)) / 2
        };
        try {
          filteredModels.forEach(m => {
            const model = new m.Model({ width: argv.width, height: argv.height });
            const saliencyResult = model.computeSaliency(image);
            const saliencyMap = saliencyResult.saliencyMap || saliencyResult;
            const salientArray = saliencyMap.getDataAsArray();
            const regionWidth = Math.round(argv.width * argv.cropRatio);
            const regionHeight = Math.round(argv.height * argv.cropRatio);
            const region = autoFocus.getRegionFromSalientMatrix(salientArray, {
              imageWidth: argv.width,
              imageHeight: argv.height,
              regionWidth,
              regionHeight
            });
            const meanX = (region.left + Math.round(region.width / 2)) / argv.width;
            const meanY = (region.top + Math.round(region.height / 2)) / argv.height;
            const distance = (Math.abs(meanX - mean.x) + Math.abs(meanY - mean.y)) / 2;
            //console.log(`* model[${m.key}] x:${meanX}, y:${meanY}, distance:${distance}`);

            modelResults[m.key] = distance;
          });
        } catch (ex) {
          return cb(ex);
        }
//console.log(modelResults);
        cb(null, modelResults);
      }]
    }, (err, results) => {
      accuracy.tasks++;
      if (err) {
        //console.warn(`Failed to process ${url}:`, err.stack || err);
      } else {
        accuracy.results++;
        Object.keys(results.processedModels).forEach(key => {
          accuracy.models[key].sum += results.processedModels[key];
        });
      }

      if ((accuracy.tasks % 50) === 0) displayStatus();

      cb();
    })
  };
}

const sourceTasks = source.map(o => createImageTask(o));

async.parallelLimit(sourceTasks, argv.parallelLimit, (err, results) => {
  console.log('Final results:');
  displayStatus();
});
