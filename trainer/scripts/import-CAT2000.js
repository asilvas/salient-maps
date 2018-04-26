const { ensureDirSync, copySync } = require('fs-extra');
const klawSync = require('klaw-sync')
const path = require('path');

const importPath = process.argv[process.argv.length - 1];
const sourcePath = path.resolve('./trainer/image-source/CAT2000');
const truthPath = path.resolve('./trainer/image-saliency-truth/CAT2000');

console.log(`Importing ${importPath}...`);

console.log(`To source: ${sourcePath}...`);
ensureDirSync(sourcePath);
copySync(importPath, sourcePath, {
  overwrite: false,
  errorOnExist: false,
  filter: src => src.indexOf('Output') < 0 ? true : false
});

const outputPaths = klawSync(importPath, { nodir: true }).filter(data => data.path.indexOf('Output') > 0).map(data => data.path);

console.log(`To truth: ${truthPath}...`);
ensureDirSync(truthPath);
outputPaths.forEach(outputPath => {
  const relativePath = outputPath.substr(importPath.length + 1).replace(`Output${path.sep}`, '').replace('_SaliencyMap', '');
  const truthFilePath = path.join(truthPath, relativePath);
  ensureDirSync(path.dirname(truthFilePath));
  copySync(outputPath, truthFilePath);
});

console.log('Import complete!');
