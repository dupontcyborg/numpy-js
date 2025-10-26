/**
 * Setup file for browser tests
 * Loads the IIFE bundle into the browser context
 */

// In a browser environment, dynamically load the IIFE bundle
if (typeof window !== 'undefined') {
  // Create a script tag that loads the built bundle
  const script = document.createElement('script');
  script.src = '/dist/numpy-ts.browser.js';
  script.type = 'text/javascript';

  // Wait for it to load
  await new Promise((resolve, reject) => {
    script.onload = resolve;
    script.onerror = reject;
    document.head.appendChild(script);
  });
}
