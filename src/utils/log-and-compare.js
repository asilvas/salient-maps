const fs = require('fs');

module.exports = function logAndCompare(srcFilename, dstFilename, data) {
  // re-stringify is simplest hack due to encoding and other minor potential differences
  const srcJson = srcFilename && JSON.stringify(JSON.parse(fs.readFileSync(srcFilename)), null, '  ');
  const json = JSON.stringify(data, null, '  ');
  fs.writeFileSync(dstFilename, json);
  if (srcFilename && srcJson !== json) {
    console.log(`!!! SOURCE:${srcFilename} !== ${dstFilename}`);
  }
}
