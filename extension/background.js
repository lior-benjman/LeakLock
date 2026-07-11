'use strict';

// LeakLock service worker — initialises default storage state on install.

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.get(['leaklockEnabled'], (data) => {
    if (data.leaklockEnabled === undefined) {
      chrome.storage.local.set({ leaklockEnabled: false });
    }
  });
});
