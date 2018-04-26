module.exports = function logMat(arr) {
  let str, row, col, z;
  for (let y = 0; y < arr.length; y++) {
    str = '';
    row = arr[y];
    for (let x = 0; x < row.length; x++) {
      val = col = row[x];
      if (!col.length) {
        if (typeof val === 'object' && val.x !== undefined) {
          val = `{ x:${val.x}${val.y !== undefined ? `,y:${val.y}` : ''}${val.z !== undefined ? `,y:${val.z}` : ''} }`;
        }
      } else { // 3D
        for (z = 0; z < col.length; z++) {
          val =  col[z];
          str += val + ':';
        }
        val = '';
      }
      str += val + ',';
    }
    console.log(str);
  }
}
