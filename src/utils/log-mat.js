module.exports = function logMat(mat) {
  let str, row, col, z, val;
  const [rows, cols, depth] = mat.sizes;
  console.log('logMat:', rows, cols, depth);
  for (row = 0; row < rows; row++) {
    str = '';
    for (col = 0; col < cols; col++) {
      let val;
      if (!depth) {
        val =  mat.at(row, col);
        if (typeof val === 'object' && val.x !== undefined) {
          val = `{ x:${val.x}${val.y !== undefined ? `,y:${val.y}` : ''}${val.z !== undefined ? `,y:${val.z}` : ''} }`;
        }
      } else { // 3D
        for (z = 0; z < depth; z++) {
          val =  mat.at(row, col, z);
          str += val + ':';
        }
        val = '';
      }
      str += val + ',';
    }
    console.log(str);
  }
}
