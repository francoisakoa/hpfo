# Publication-tool references

Official documentation consulted on 10 October 2026:

- GitHub file/repository size limits and use of Release assets:
  https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github
- GitHub CLI release creation, draft handling and explicit tag verification:
  https://cli.github.com/manual/gh_release_create
- GitHub CLI release asset download:
  https://cli.github.com/manual/gh_release_download
- GitHub CLI repository authentication:
  https://cli.github.com/manual/gh_auth_login

The local package does not change the visibility of francoisakoa/hpfo and does not
claim that the current remote contents or reviewer access have been inspected.
The publication script checks the actual repository through the authenticated CLI
before writing, preserves the default branch, and never force-pushes a reference.
