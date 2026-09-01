# Chrome Web Store Submission Notes

## Current Package Status

- Extension source: `extension/`
- Public API endpoint: `https://leaklock-api-799658247857.us-central1.run.app`
- Official homepage URL after GitHub Pages is enabled: `https://lior-benjman.github.io/LeakLock/`
- Privacy policy URL after GitHub Pages is enabled: `https://lior-benjman.github.io/LeakLock/privacy.html`
- Support URL after GitHub Pages is enabled: `https://lior-benjman.github.io/LeakLock/support.html`
- Store-ready zip should place `manifest.json` at the zip root.
- Do not upload the repository root or the old `extension.zip` if it contains a top-level `extension/` folder.

## Enable GitHub Pages

1. Commit and push the `docs/` changes to the `main` branch.
2. Open `https://github.com/lior-benjman/LeakLock/settings/pages`.
3. Under "Build and deployment", select "Deploy from a branch".
4. Set branch to `main` and folder to `/docs`.
5. Save.
6. Wait for GitHub Pages to finish deploying.
7. Open `https://lior-benjman.github.io/LeakLock/`.

If the repository is private, GitHub Pages availability depends on the account/organization plan. A public repository is the simplest setup for a public Chrome Web Store extension homepage.

## Listing Assets

Use these generated files in the Chrome Web Store listing form:

- Store icon: `dist/chrome-web-store-assets/store-icon-128.png`
- Screenshot: `dist/chrome-web-store-assets/screenshot-1280x800-protection-enabled.png`
- Screenshot: `dist/chrome-web-store-assets/screenshot-1280x800-risk-overlay.png`
- Screenshot: `dist/chrome-web-store-assets/screenshot-1280x800-blur-option.png`
- Small promo tile: `dist/chrome-web-store-assets/small-promo-tile-440x280.png`
- Marquee promo tile: `dist/chrome-web-store-assets/marquee-promo-tile-1400x560.png`
- Global promo video: optional. Leave it blank unless you have a YouTube demo video.

## Store Listing Draft

Name:
LeakLock

Short description:
Active image privacy protection that checks selected uploads for sensitive content before they are posted.

Single purpose:
LeakLock helps users review image files they select for upload by scanning them for sensitive visual content before allowing the upload to continue.

Detailed description:
LeakLock monitors image file selections on ordinary web pages when protection is enabled. Before an image upload continues, LeakLock sends the selected image to the LeakLock analysis API, checks for sensitive content such as faces, identity documents, documents, and license plates, and shows a risk result. Users can cancel the upload, continue, or blur flagged regions before continuing.

Homepage URL:
`https://lior-benjman.github.io/LeakLock/`

Support URL:
`https://lior-benjman.github.io/LeakLock/support.html`

## Permission Justifications

`storage`:
Stores whether LeakLock protection is enabled or disabled. Protection is off by default.

Host access for `https://leaklock-api-799658247857.us-central1.run.app/*`:
Sends selected image files to the LeakLock analysis API and receives risk results.

Content scripts on `http://*/*` and `https://*/*`:
Detects image file selections on upload forms so LeakLock can scan the selected image before the page processes the upload.

## Privacy Tab Draft

Privacy policy URL:
`https://lior-benjman.github.io/LeakLock/privacy.html`

Data collected:
- User-generated content: image files selected by the user for upload.
- Website content: only file input interactions needed to detect selected uploads. The extension does not send page URLs, browsing history, cookies, or form text to the API.

Data use:
- Images are used only to provide LeakLock's image-risk analysis and optional local redaction preview workflow.
- Images are not used for advertising, personalization, credit decisions, or sale to third parties.

Remote code:
No. The extension calls a remote API for analysis results, but it does not load or execute remote JavaScript.

## Test Instructions Draft

1. Install the extension from the submitted package.
2. Open the extension popup and enable Protection.
3. Visit any standard HTTP/HTTPS page with an image file input.
4. Select an image containing a visible face or license plate.
5. Confirm that the LeakLock overlay appears before upload completion.
6. Confirm the overlay displays risk score, detections, confidence, and available actions.
7. Test backend health at `https://leaklock-api-799658247857.us-central1.run.app/health`.

## Before Submit

- Create or choose a Chrome Web Store developer account.
- Publish a public privacy policy URL based on `docs/privacy-policy-draft.md`.
- Prepare at least one 1280x800 or 640x400 screenshot for the store listing.
- Upload the store-ready zip in Chrome Developer Dashboard.
- Fill Store Listing, Privacy, Distribution, and Test Instructions tabs.
- Submit for review.
