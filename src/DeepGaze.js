/*
  Port from: https://github.com/mpatacchiola/deepgaze/blob/master/deepgaze/saliency_map.py
*/

const cv = require('opencv4nodejs');
const { Mat } = cv;
const Saliency = require('./Saliency');
const math = require('mathjs');

const MEAN_VECTOR = [0.5555, 0.6449, 0.0002, 0.0063];

const COVARIANCE_MATRIX_INVERSE = [
  [43.3777, 1.7633, -0.4059, 1.0997],
  [1.7633, 40.7221, -0.0165, 0.0447],
  [-0.4059, -0.0165, 87.0455, -3.2744],
  [1.0997, 0.0447, -3.2744, 125.1503]
];

module.exports = class DeepGaze extends Saliency {
  constructor(options) {
    super(options);

    this.convertToLAB = true;
    this.salientImage = new Mat(this.height, this.width, cv.CV_32F);
  }

  computeSaliency(/* Mat */image)
  {
    const resized = image.resize(this.height, this.width, 0, 0, cv.INTER_LINEAR_EXACT);

    // ~100ms @ 400x400
    // this step is very expensive, roughly ~25% of all cpu time, but verified that the majority of the time it improves accuracy (filters alot of noise)
    // ALL OK -- only first hit is expensive. Less than 2% for subsequent hits
    let source = this.convertToLAB ? resized.cvtColor(cv.COLOR_BGR2Lab) : resized;

    this.calculateHistogram(source, 8);

    this.precomputeParameters();

    this.bilateralFiltering();

    this.calculateProbability();

    this.computeSaliencyMap();

    let index, val;
    for (let y = 0; y < this.height; y++) {
      for (let x = 0; x < this.width; x++) {
        index = this.imageQuantized[y][x];
        index = this.map3d1d[index[0]][index[1]][index[2]];
        val = this.saliency[index];
        this.salientImage.set(y, x, val || 0);
      }
    }
    
    return this.salientImage;
  }

  calculateHistogram(/* Mat */image, totBins = 8) {
    let i;

    const channels = image.split();

    const { minVal: minL, maxVal: maxL } = channels[0].minMaxLoc();
    const { minVal: minA, maxVal: maxA } = channels[1].minMaxLoc();
    const { minVal: minB, maxVal: maxB } = channels[2].minMaxLoc();

    this.lRange = linspace(minL, maxL, totBins, { endpoint: false });
    this.aRange = linspace(minA, maxA, totBins, { endpoint: false });
    this.bRange = linspace(minB, maxB, totBins, { endpoint: false });

    const lChannel = channels[0].getDataAsArray();
    const aChannel = channels[1].getDataAsArray();
    const bChannel = channels[2].getDataAsArray();

    const lDig = digitize(/*math.flatten(*/lChannel/*)*/, this.lRange, false);
    const aDig = digitize(/*math.flatten(*/aChannel/*)*/, this.aRange, false);
    const bDig = digitize(/*math.flatten(*/bChannel/*)*/, this.bRange, false);

    this.imageQuantized = dstack([lDig, aDig, bDig]);

    this.imageQuantized = math.subtract(this.imageQuantized, 1);

    this.map3d1d = math.zeros(totBins, totBins, totBins).toArray();

    // if 0 range this will crash. hack workaround
    const histMat = cv.calcHist(image, [
      { channel: 0, bins: totBins, ranges: [minL, minL === maxL ? minL + 1 : maxL] },
      { channel: 1, bins: totBins, ranges: [minA, minA === maxA ? minA + 1 : maxA] },
      { channel: 2, bins: totBins, ranges: [minB, minB === maxB ? minB + 1 : maxB] }
    ]);

    // SLOW, but should be sufficient for this small histo cube
    this.histogram = histMat.getDataAsArray();

    const iqChannelL = new Array(this.height);
    const iqChannelA = new Array(this.height);
    const iqChannelB = new Array(this.height);
    let iqChannelLrow, iqChannelArow, iqChannelBrow;
    i = 0;
    for (let y = 0; y < this.height; y++) {
      iqChannelL[y] = iqChannelLrow = new Array(this.width);
      iqChannelA[y] = iqChannelArow = new Array(this.width);
      iqChannelB[y] = iqChannelBrow = new Array(this.width);
      for (let x = 0; x < this.width; x++) {
        iqChannelLrow[x] = this.imageQuantized[y][x][0];
        iqChannelArow[x] = this.imageQuantized[y][x][1];
        iqChannelBrow[x] = this.imageQuantized[y][x][2];
      }
    }

    const imageIndeces = vstack([math.flatten(iqChannelL), math.flatten(iqChannelA), math.flatten(iqChannelB)]);

    const imageLinear = ravel_multi_index(imageIndeces, [totBins, totBins, totBins]);

    const histIndex = nonzero(this.histogram);

    this.indexMatrix = transpose(histIndex);

    const uniqueColorLinear = ravel_multi_index(histIndex, [totBins, totBins, totBins]);

    this.numberOfColors = amax(shape(this.indexMatrix));

    this.centxMatrix = new Array(this.numberOfColors);
    this.centyMatrix = new Array(this.numberOfColors);
    this.centx2Matrix = new Array(this.numberOfColors);
    this.centy2Matrix = new Array(this.numberOfColors);

    uniqueColorLinear.forEach((linearColor, idx) => {
      const linearColorIndexes = findInArray(imageLinear, linearColor);
      if (!linearColorIndexes.length) {
        // todo: in very rare cases the lookup fails. Need to verify impact of this failure to determine best handling
        this.centxMatrix[idx] = 0;
        this.centyMatrix[idx] = 0;
        this.centx2Matrix[idx] = 0;
        this.centy2Matrix[idx] = 0;
        return;
      }

      const [whereY, whereX] = unravel_index(linearColorIndexes, [this.height, this.width]);
      this.centxMatrix[idx] = math.sum(whereX)
      this.centyMatrix[idx] = math.sum(whereY)
      this.centx2Matrix[idx] = math.sum(pow(whereX, 2))
      this.centy2Matrix[idx] = math.sum(pow(whereY, 2))
    });
  }

  precomputeParameters(sigmac = 16) {
    let i;

    const [L_centroid, A_centroid, B_centroid] = meshgrid(this.lRange, this.aRange, this.bRange)

    this.uniquePixels = math.zeros(this.numberOfColors, 3).toArray();
    let iIndex, iL, iA, iB;
    for (i = 0; i < this.numberOfColors; i++) {
      iIndex = this.indexMatrix[i];
      iL = L_centroid[iIndex[0]][iIndex[1]][iIndex[2]];
      iA = A_centroid[iIndex[0]][iIndex[1]][iIndex[2]];
      iB = B_centroid[iIndex[0]][iIndex[1]][iIndex[2]];
      this.uniquePixels[i] = [iL, iA, iB];
      this.map3d1d[iIndex[0]][iIndex[1]][iIndex[2]] = i;
    }

    const uniquePixelsNew = this.uniquePixels.map(row => ([row]));

    const uniquePixelsSub = subtract(uniquePixelsNew, this.uniquePixels);

    const uniquePixelsPow = pow(uniquePixelsSub, 2);

    const colorDifferenceMatrix = uniquePixelsPow.map(row => row.map(col => math.sum(col)));

    this.colorDistanceMatrix = math.sqrt(colorDifferenceMatrix);

    this.exponentialColorDistanceMatrix = math.exp(math.multiply(math.divide(colorDifferenceMatrix, (2 * sigmac * sigmac)), -1));
  }

  bilateralFiltering() {
    const histVal = math.flatten(this.histogram).filter(v => v > 0);

    this.contrast = this.colorDistanceMatrix.map(row => math.multiply(row, histVal));

    const normalization = this.exponentialColorDistanceMatrix.map(row => math.multiply(row, histVal));

    this.mx = this.exponentialColorDistanceMatrix.map(row => math.multiply(row, this.centxMatrix));
    this.my = this.exponentialColorDistanceMatrix.map(row => math.multiply(row, this.centyMatrix));

    let mx2 = this.exponentialColorDistanceMatrix.map(row => math.multiply(row, this.centx2Matrix));
    let my2 = this.exponentialColorDistanceMatrix.map(row => math.multiply(row, this.centy2Matrix));

    this.mx = this.mx.map((v, i) => v / normalization[i]);
    this.my = this.my.map((v, i) => v / normalization[i]);
   
    mx2 = mx2.map((v, i) => v / normalization[i]);
    my2 = my2.map((v, i) => v / normalization[i]);

    const mxPow = this.mx.map(v => v * v);
    const myPow = this.my.map(v => v * v);

    const mx2sub = mx2.map((v, i) => v - mxPow[i]);
    const my2sub = my2.map((v, i) => v - myPow[i]);
    
    this.Vx = math.abs(mx2sub);
    this.Vy = math.abs(my2sub);
  }

  calculateProbability() {
    const g = [
      this.Vx.map(v => Math.sqrt(12 * v) / this.width), // INCONSISTENT with python... why?!
      this.Vy.map(v => Math.sqrt(12 * v) / this.height), // correct
      this.mx.map(v => ((v - this.width / 2) / this.width)), // correct
      this.my.map(v => ((v - this.height / 2) / this.height)) // correct
    ];

    let X = transpose(g).map(row => math.subtract(row, MEAN_VECTOR));
    // ^ correct minus the Vx part that is wrong as noted above

    const Y = X;

    const result = X.map(row => math.sum(math.multiply(math.multiply(row, COVARIANCE_MATRIX_INVERSE), row)));

    this.shapeProbability = result.map(v => Math.exp(- v / 2));
  }

  computeSaliencyMap() {
    this.saliency =this.contrast.map((v, i) => v * this.shapeProbability[i]);

    const a1 = this.exponentialColorDistanceMatrix.map(row => math.dot(row, this.saliency));

    const a2 = this.exponentialColorDistanceMatrix.map(arr => math.sum(arr));

    this.saliency = a1.map((v, i) => v / a2[i]);

    const { minVal, maxVal } = this.saliency.reduce((state, v) => {
      state.minVal = Math.min(v, state.minVal);
      state.maxVal = Math.max(v, state.maxVal);
      return state;
    }, { minVal: this.saliency[0], maxVal: this.saliency[0] });

    this.saliency = math.subtract(this.saliency, minVal);
    
    // CUSTOM CORRECTION, MEASURE IMPACT
    this.saliency = math.divide(this.saliency, maxVal - minVal + 0.001);

    // ALTERNATIVE: ORIGINAL
    //const div = maxVal - minVal + 0.001;
    //this.saliency = this.saliency.map(v => ((v * 255) / div));
  }
}

function findInArray(arr, val) {
  const indexes = [];
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] === val) indexes.push(i);
  }
  return indexes;
}

function pow(arr, exp) {
  const result = new Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    result[i] = Array.isArray(arr[i]) ? pow(arr[i], exp) : Math.pow(arr[i], exp);
  }
  return result;
}

/* https://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
*/
function linspace(start, stop, num = 50, { endpoint=true/*, retstep=false, dtype=null*/ } = {}) {
  if (num < 2) {
    return num === 1 ? [start] : [];
  }
  const ret = new Array(num);
  const step = (stop - start) / (endpoint ? num-1 : num);
  for (let i = 0; i < num; i++) {
    ret[i] = start + (step * i);
  }
  return ret;
}

/* https://docs.scipy.org/doc/numpy/reference/generated/numpy.dstack.html
*/
function dstack(input) {
  if (!Array.isArray(input[0])) return [[input]]; // 1D
  
  const rows = input.length;
  
  const is3D = Array.isArray(input[0][0]);
  const cols = input[0].length;
  const depth = is3D && input[0][0].length;
  
  const arr = new Array(cols);
  let x, y, z, row, col;
  if (is3D) { // 3D
    for (x = 0; x < cols; x++) {
      arr[x] = row = new Array(depth);
      for (z = 0; z < depth; z++) {
        row[z] = col = new Array(rows);
        for (y = 0; y < rows; y++) {
          col[y] = input[y][x][z];
        }
      }
    }
    
    return arr;
  } else { // 2D
    for (x = 0; x < cols; x++) {
      arr[x] = row = new Array(rows);
      for (y = 0; y < rows; y++) {
        row[y] = input[y][x];
      } 
    }
    
    return [arr];
  }
}

/* https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html
*/
function vstack(arrays) {
  if (arrays[0][0].length !== arrays[1][0].length) throw new Error('Inputs must be of same size/shape');

  // 1D
  if (!Array.isArray(arrays[0][0])) return arrays;
  
  // 2D
  const rows = [];
  arrays.forEach(arr => arr.map(row => rows.push(row)));
 
  return rows;
}

/* https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.digitize.html#numpy.digitize
*/
function digitize(x, bins, { right = false } = {}) {
  let i, bin;
  const results = [];
  
  const inBin = right
    // bins[i-1] < x <= bins[i]
    ? (leftBin, x, rightBin) => leftBin < x && x <= rightBin
    // bins[i-1] <= x < bins[i]
    : (leftBin, x, rightBin) => leftBin <= x && x < rightBin
  ;

  x.forEach(val => {
    if (Array.isArray(val)) {
      return void results.push(digitize(val, bins, right));
    }
    bin = 1;
    for (i = 1; i < bins.length; i++) {
      if (inBin(bins[i-1], val, bins[i])) {
        bin = i;
        break;
      }
    }
    if (val >= bins[bins.length-1]) bin = right ? bins.length - 1 : bins.length;
    results.push(bin);
  });
  
  return results;
}

/*https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html
*/
function amax(input) {
  if (!Array.isArray(input)) return input;
  
  return input.reduce((state, el) => {
    const val = Array.isArray(el) ? amax(el) : el;
    return Math.max(val, state);
  }, -1);
}

/* https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel_multi_index.html
*/
function ravel_multi_index(multiIndex, dims/*, { mode='raise', order='C' } = {}*/) {
  // 1D
  
  // [[2, 1, 4], [4, 3, 6]] = (2*3 + 1)*6 + 4 = 46
  // 2*3=6+1=7*6=42+4=46
  // (((idx[0][0]*dim[1])+idx[0][1])*dim[2])+idx[0][2]
  
  // vs 2D
  
  // [[[3,6,6],[4,5,1]], [7,6]] = [22, 41, 37]
  // 3*6=18+4=22
  // (idx[0][0]*dim[1])+idx[1][0]
  // 6*6=36+5=41
  // (idx[0][1]*dim[1])+idx[1][1]
  // 6*6=36+1=37
  // (idx[0][2]*dim[1])+idx[1][2]
  
  const arrays = Array.isArray(multiIndex[0]) ? multiIndex : [multiIndex]; 
  const is1D = arrays.length === 1;
  
  // results are based on number of columns in each array, unless it's a 1D input
  const indexes = new Array(is1D ? 1 : arrays[0].length).fill(0);
  const cells = is1D ? arrays[0].length : arrays.length;
  let i, idx, dimV, elV, cell;
  
  for (i = 0; i < indexes.length; i++) {
    for (cell = 0; cell < cells; cell++) {
      dim = (cell + 1) % dims.length;
      dimV = (cell === (cells-1)) ? 1 : dims[dim];
      elV = is1D ? arrays[i][cell] : arrays[cell % arrays.length][i];
      indexes[i] = (indexes[i] + elV) * dimV;
    }
  }

  return indexes.length === 1 ? indexes[0] : indexes;
}

/* https://docs.scipy.org/doc/numpy/reference/generated/numpy.meshgrid.html
*/
function meshgrid(...arrays) {
  const lastArg = arrays[arrays.length-1];
  let opts = {};
  if (!Array.isArray(lastArg)) {
    opts = lastArg;
    arrays.pop();
  }
  const { sparse = false, copy = false, indexing = 'xy' } = opts;
  
  // 0-1D returns as is
  if (arrays.length <= 1) return arrays[0];
  
  const is3D = Array.isArray(arrays[2]);
  const is2D = !is3D;
  
  const cols = arrays[0].length;
  const rows = arrays[1].length;
  const depth = is3D && arrays[2].length;

  const results = [];
  
  function getResult(arr, rows, cols, depth, dimension) {
    const s = {};
    let row, col;
    const result = [];
    for (s.y = 0; s.y < rows; s.y++) {
      row = [];
      for (s.x = 0; s.x < cols; s.x++) {
        if (!depth) {
          row.push(arr[s[dimension]]);
        } else {
          col = [];
          for (s.z = 0; s.z < depth; s.z++) {
            col.push(arr[s[dimension]]);
          }
          row.push(col);
        }
      }
      result.push(row);
    }
    return result;
  }
  
  let x, y, z, row, col, result;
  if (is2D) {
    if (indexing === 'xy') {
      results.push(getResult(arrays[0], rows, cols, 0, 'x'));
      results.push(getResult(arrays[1], rows, cols, 0, 'y'));
    } else { // ij
      results.push(getResult(arrays[0], cols, rows, 0, 'y'));
      results.push(getResult(arrays[1], cols, rows, 0, 'x'));
    }
  } else { // 3D
    if (indexing === 'xy') {
      results.push(getResult(arrays[0], rows, cols, depth, 'x'));
      results.push(getResult(arrays[1], rows, cols, depth, 'y'));
      results.push(getResult(arrays[2], rows, cols, depth, 'z'));
    } else { // ij
      results.push(getResult(arrays[0], cols, rows, depth, 'y'));
      results.push(getResult(arrays[1], cols, rows, depth, 'x'));
      results.push(getResult(arrays[2], cols, rows, depth, 'z'));
    }
  }
  
  return results;
}

/*https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html
*/
function transpose(a/*, axes*/) {
  
  // [[[0, 1], [2, 3]]] = [[0, 2], [1, 3]]
  // [
  //   [0, 1],
  //   [2, 3]
  // ] = [
  //   [0, 2],
  //   [1, 3]
  // ]
  // result[0][0] = idx[0][0]
  // result[0][1] = idx[1][0]
  // result[1][0] = idx[0][1]
  // result[1][1] = idx[1][1]
  
  /* [[0, 1, 2], [3, 4, 5], [6, 7, 8]] = [[0, 3, 6], [1, 4, 7], [2, 5, 8]])
     [
       [0, 1, 2],
       [3, 4, 5],
       [6, 7, 8]
     ] = [
       [0, 3, 6],
       [1, 4, 7],
       [2, 5, 8]
     ]
     result[0][0] = idx[0][0]
     result[0][1] = idx[1][0]
     result[0][2] = idx[2][0]
     result[1][0] = idx[0][1]
     result[1][1] = idx[1][1]
     result[1][2] = idx[2][1]
     result[2][0] = idx[0][2]
     result[2][1] = idx[1][2]
     result[2][2] = idx[2][2]
  */

  /* [[[0, 1], [2, 3], [4, 5]]], [[0, 2, 4], [1, 3, 5]]);
     [
       [0, 1],
       [2, 3],
       [4, 5]
     ] = [
       [0, 2, 4],
       [1, 3, 5]
     ]
     result[0][0] = idx[0][0]
     result[0][1] = idx[1][0]
     result[0][2] = idx[2][0]
     result[1][0] = idx[0][1]
     result[1][1] = idx[1][1]
     result[1][2] = idx[2][1]
  */
  
  // 1D - cannot transpose flat array, return as-is
  if (!Array.isArray(a[0])) return a;

  // source rows x cols
  const rows = a.length;
  const cols = a[0].length;
  
  // output is transposed, cols x rows
  const arr = a[0].map(() => a.map(() => [])); // init matrix
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      arr[col][row] = a[row][col];
    }
  }

  return arr;
}

/* https://docs.scipy.org/doc/numpy/reference/generated/numpy.unravel_index.html#numpy.unravel_index
*/
function unravel_index(indices, dims/*, { order='C' } = {}*/) {
  indices = Array.isArray(indices) ? indices : [indices];
  const is1D = indices.length === 1;

  const cols = !is1D ? indices.length : dims.length;
  const rows = !is1D ? (cols * dims.length) / cols : 1;
  const results = [];

  let i, i2, row, col, val, v;
  for (i = 0; i < rows; i++) results.push([]);
  for (i = 0, row = rows - 1, col = cols - 1; i < indices.length; i++) {
    val = indices[i];
    for (i2 = dims.length-1; i2 >= 0; --i2) {
      v = Math.floor(val) % dims[i2];
      if (is1D) results[i][i2] = v;
      else results[i2 % rows][i] = v;
      col--;
      if (col < 0) {
        col = cols - 1;
        row--;
      }

      val /= dims[i2];
    }
  }

  return (!results.length || results.length > 1) ? results : results[0];
}

/* https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flat.html
*/
function flat(arr) {
  let result = [];

  arr.forEach(v => {
    if (Array.isArray(v)) {
      result = result.concat(flat(v));
    } else {
      result.push(v);
    }
  });
  
  return result;
}

/* https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.nditer.html
*/
class nditer {
  constructor(arr) {
    if (!(this instanceof nditer)) return new nditer(arr);

    this.arr = arr;
    this.y = 0;
    this.x = 0;
    this.z = 0;
    this.next = this.nextY;
  }
  
  [Symbol.iterator]() {
    return {
      next: () => this.next()
    };
  }
  
  nextZ() {
    this.next = this.nextZ;
    if (this.z >= this.arr[this.y][this.x].length) {
      this.x++;
      return this.nextX();
    }
     
    const v = this.arr[this.y][this.x][this.z];
    if (!Array.isArray(v)) {
      this.z++;
      return { done: false, value: v };
    }
    
    this.z++;
    return this.nextZ();
  }
  
  nextX() {
    this.next = this.nextX;
    if (this.x >= this.arr[this.y].length) {
      this.y++;
      return this.nextY();
    }
    
    const v = this.arr[this.y][this.x];
    if (!Array.isArray(v)) {
      this.x++;
      return { done: false, value: v };
    }
    
    this.z = 0;
    return this.nextZ();
  }
  
  nextY() {
    this.next = this.nextY;
    if (this.y >= this.arr.length) return { done: true };
    
    const v = this.arr[this.y];
    if (!Array.isArray(v)) {
      this.y++;
      return { done: false, value: v };
    }
    
    this.x = 0;
    return this.nextX();
  }
}

/* https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.nonzero.html
*/
function nonzero(arr) {
  const rows = [];
  const cols = [];
  const depth = [];
  
  let x, y, z, row, col;
  for (y = 0; y < arr.length; y++) {
    row = arr[y];
    for (x = 0; x < row.length; x++) {
      col = row[x];
      if (!Array.isArray(col)) {
        if (col > 0) {
          rows.push(y);
          cols.push(x);
        }
      } else {
        for (z = 0; z < col.length; z++) {
          if (col[z] > 0) {
            rows.push(y);
            cols.push(x);
            depth.push(z);
          }
        }
      }
    }
  }
  
  return depth.length > 0 ? [rows, cols, depth] : [rows, cols];
}

/* https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html
*/
function shape(arr) {
  function getSizeOfDimension(dim) {
    const first = dim[0];
    if (!Array.isArray(first)) return;
    return first.length;
  }  
  
  let dim = arr.length;
  let shape = [dim];
  while ((dim = getSizeOfDimension(arr)) > 0) {
    shape.push(dim++);
    arr = arr[0];
  }
  
  return shape;
}

/* https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.subtract.html
*/
function subtract(a, b) {
  function shape(arr) {
    if (!arr.length) return [];
    const dims = [arr.length]; // 1D
    const dim2d = arr[0].length && dims.push(arr[0].length); // 2D
    const dim3d = dim2d && arr[0][0].length && dims.push(arr[0][0].length); // 3D
    const dim4d = dim3d && arr[0][0][0].length && dims.push(arr[0][0][0].length); // 4D
    return dims;
  }
  
  const aShape = shape(a);
  const bShape = shape(b);

  const ret = aShape.length > 0 ? new Array(aShape[0]) : [];
  
  let x, y, y2, yIdx, z, row, row2, col;
  if (aShape.length === 1) { // 1D
    for (y = 0; y < aShape[0]; y++) {
      ret[y] = a[y] - b[y];
    }
  } else if (aShape.length === 2) { // 2D
    for (y = 0; y < aShape[0]; y++) {
      ret[y] = row = new Array(aShape[1]);
      for (x = 0; x < aShape[1]; x++) {
        row[x] = a[y][x] - b[y][x];
      }
    }
  } else if (aShape.length === 3 && bShape.length === 3) { // 3D
    for (y = 0; y < aShape[0]; y++) {
      ret[y] = row = new Array(aShape[1]);
      for (x = 0; x < aShape[1]; x++) {
        row[x] = col = new Array(aShape[2]);
        for (z = 0; z < aShape[2]; z++) {
          col[z] = a[y][x][z] - b[y][x][z];
        }
      }
    }
  } else if (aShape.length === 3 && bShape.length === 2 && aShape[2] === bShape[1]) { // 3D - 2D
    yIdx = 0;
    for (y = 0; y < aShape[0]; y++) {
      ret[y] = row = new Array(aShape[0]);
      for (y2 = 0; y2 < aShape[0]; y2++, yIdx++) {
        row[y2] = col = new Array(aShape[2]);
        for (z = 0; z < aShape[2]; z++) {
          col[z] = a[y][0][z] - b[y2][z];
        }
      }
    }
  }
  
  return ret;
}
