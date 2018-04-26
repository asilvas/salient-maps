module.exports = {
  'deep': {
    title: 'Deep Gaze',
    load: () => require('./DeepGaze')
  },
  'deep-rgb': {
    title: 'Deep Gaze RGB',
    load: () => require('./DeepGazeRgb')
  },
  'spectral': {
    title: 'Spectral Residual',
    load: () => require('./SpectralResidual')
  }/*,
  'spectral-facial': {
    title: 'Spectral Residual /w Facial Detection',
    load: () => require('./SpectralResidualFacial')
  }*/,
  'fine': {
    title: 'Fine Grained',
    load: () => require('./FineGrained')
  }
};
