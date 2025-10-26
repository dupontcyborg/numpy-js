// Test CJS bundle
const np = require('./dist/numpy-ts.node.cjs');

console.log('CJS Bundle Test:');
console.log('typeof np:', typeof np);
console.log('Has array?', typeof np.array);
console.log('Has zeros?', typeof np.zeros);
console.log('Has add?', typeof np.add);

if (np.array) {
  try {
    const arr = np.array([1, 2, 3]);
    console.log('Created array:', arr.shape);
  } catch (e) {
    console.error('Error creating array:', e.message);
  }
}
