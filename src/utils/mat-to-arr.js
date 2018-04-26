module.exports = mat => {
  // doesn't work for 3D yet
  const matArr = mat.getDataAsArray();

  const [rows, cols, depth] = mat.sizes;
  return matArr;
  const arr = new Array(rows);
  let row, col, x, y, z;
  for (y = 0; y < rows; y++) {
    arr[y] = row = new Array(cols);
    for (x = 0; x < cols; x++) {
      if (!depth) {
        row[x] = mat.at(y, x);
      } else {
        row[x] = col = new Array(depth);
        for (z = 0; z < depth; z++) {
          col[z] = mat.at(y, x, z);
          //console.log('matToArr:', col[z]);
        }
      }
    }
  }

  return arr;
};
