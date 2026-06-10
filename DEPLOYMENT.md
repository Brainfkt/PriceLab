# PriceLab Deployment

PriceLab is a Streamlit application. Cloudflare Pages cannot run the Python app directly because Pages is static hosting. The recommended setup is:

1. Deploy the Streamlit app on Streamlit Community Cloud.
2. Publish `docs/` on Cloudflare Pages from the GitHub repository.
3. Make the Cloudflare Pages URL redirect directly to the Streamlit app.

This gives you a clean public URL using Cloudflare or your own domain, while Streamlit runs the interactive Python workload. The GitHub repository URL itself (`https://github.com/Brainfkt/PriceLab`) cannot be converted into an HTTP redirect; the redirect applies to the Cloudflare Pages URL or custom domain.

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

## Cloudflare Pages

Current Cloudflare Pages settings:

- Project: `pricelab`
- Public URL: `https://pricelab-71i.pages.dev`
- Source repository: `Brainfkt/PriceLab`
- Production branch: `main`
- Root directory: `/`
- Build command: empty
- Build output directory: `docs`
- Automatic production deployments: enabled

`docs/config.js` controls the static redirect:

```js
window.PRICELAB_DEPLOYMENT = {
  streamlitAppUrl: "https://pricelab.streamlit.app",
  publicProjectUrl: "https://pricelab-71i.pages.dev"
};
```

## Custom Domain

To use a custom domain, add it to the Cloudflare Pages project:

1. Open Cloudflare dashboard > Workers & Pages > `pricelab`.
2. Open Custom domains.
3. Add a subdomain, for example `pricelab.YOUR-DOMAIN.com`.
4. Keep HTTPS enabled.

DNS for a subdomain:

```text
Type: CNAME
Name: pricelab
Value: pricelab-71i.pages.dev
```

If the domain is already managed by Cloudflare, Pages can create or validate the required DNS record from the dashboard.

## Redirect Behavior

`docs/index.html` redirects directly to `streamlitAppUrl` from `docs/config.js`.

`docs/404.html` redirects unknown Cloudflare Pages paths to the same Streamlit app. This helps if someone opens a stale path under the custom domain.

If `docs/config.js` still contains placeholders, the page does not redirect and instead shows a short setup message.
