
## Put it on GitLab Pages

1. Create a project on GitLab and push this folder to its default branch.
2. The first pipeline runs the tests, builds the data and publishes the site. The address
   is shown under **Deploy > Pages**.
3. Keep it fresh: **Build > Pipeline schedules > New schedule**, target the default branch,
   with a cron pattern such as `17 19 * * *` (daily, 06:17 Sydney time during daylight
   saving; schedules use UTC unless you set a time zone). The first run reads every
   repository's size from Hugging Face and takes several minutes; later runs reuse those
   answers and are much shorter, well inside the 400 free compute minutes a month on
   gitlab.com.
4. Optional, under **Settings > CI/CD > Variables** (masked):
   - `HF_TOKEN`: a read-only Hugging Face token. Not required, but it raises rate limits
     and lets the build read gated repositories you have accepted.
   - `AA_API_KEY`: a free Artificial Analysis API key. Without it that source is skipped.

`.gitlab-ci.yml` uses `pages: true`, which needs GitLab 17.6 or later. On an older
self-managed GitLab, delete that one line; the job name `pages` does the same job there.

## Or put it on GitHub Pages

The same build runs from `.github/workflows/pages.yml`.

1. Push this folder to a GitHub repository whose default branch is `main` (edit the
   branch name in the workflow if yours differs).
2. **Settings > Pages > Build and deployment > Source: GitHub Actions.**
3. Optional secrets under **Settings > Secrets and variables > Actions**: `HF_TOKEN`,
   `AA_API_KEY`.