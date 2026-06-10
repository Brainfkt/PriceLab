# PriceLab Deployment

PriceLab is a Streamlit application. GitHub Pages cannot run the Python app directly because Pages is static hosting. The recommended setup is:

1. Deploy the Streamlit app on Streamlit Community Cloud.
2. Publish `docs/` on GitHub Pages with your personal domain.
3. Make the GitHub Pages URL redirect directly to the Streamlit app.

This gives you a clean public URL using your own domain, while Streamlit runs the interactive Python workload. The GitHub repository URL itself (`https://github.com/Brainfkt/PriceLab`) cannot be converted into an HTTP redirect; the redirect applies to the repository's GitHub Pages URL or custom domain.

## Streamlit Community Cloud

1. Push the repository to GitHub.
2. Open Streamlit Community Cloud and create a new app.
3. Select repository: `Brainfkt/PriceLab`.
4. Branch: `main`.
5. Main file path: `app.py`.
6. Python version: use Python `3.11` if offered.
7. Deploy.
8. Copy the resulting `https://...streamlit.app` URL.

The root `requirements.txt` is provided for Streamlit Cloud dependency installation.

## Custom Domain Link Via GitHub Pages

Replace placeholders in `docs/config.js`:

```js
window.PRICELAB_DEPLOYMENT = {
  streamlitAppUrl: "https://pricelab.streamlit.app",
  publicProjectUrl: "https://pricelab.YOUR-DOMAIN.com"
};
```

Then configure GitHub Pages:

1. In GitHub repository settings, open **Pages**.
2. Set source to the `docs/` folder on the `main` branch.
3. Set the custom domain, for example `pricelab.YOUR-DOMAIN.com`.
4. Enable **Enforce HTTPS** when GitHub allows it.

DNS for a subdomain:

```text
Type: CNAME
Name: pricelab
Value: Brainfkt.github.io
```

If you want the apex domain, use GitHub Pages `A` records from the GitHub documentation instead.

## Redirect Behavior

`docs/index.html` redirects directly to `streamlitAppUrl` from `docs/config.js`.

`docs/404.html` redirects unknown GitHub Pages paths to the same Streamlit app. This helps if someone opens a stale path under the custom domain.

If `docs/config.js` still contains placeholders, the page does not redirect and instead shows a short setup message.
