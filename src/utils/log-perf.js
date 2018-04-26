const state = {};

module.exports = function(title, { minPerf=5 } = {}) {
  const now = Date.now();

  if (!state.firstStart) {
    state.firstStart = now;
  }

  if (state.start) {
    const elapsed = now - state.start;
    if (elapsed >= minPerf) console.log(`perf> ${elapsed}ms : ${state.title} (${now - state.firstStart}ms total)`);
  }

  if (!title) {
    // reset
    console.log(`perf> complete time: ${now - state.firstStart}ms`);
    state.firstStart = null;
    state.start = null;
    return;
  }

  state.title = title;
  state.start = now;
};
