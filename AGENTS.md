# Storage requirement

The user requires all files written for this project to stay physically under
`/data`. This includes temporary files, downloads, model and package caches,
environments, compiler caches, checkpoints, logs, screenshots, and test fixtures.
Do not use `/tmp`, `/var/tmp`, or home-directory caches for new task files.
Reading existing files elsewhere is allowed. Do not change `$HOME`.

Before running Python or other tools that create temporary/cache files, source
`scripts/hdepic_data_storage.sh` or explicitly set equivalent `/data` paths.
Check resolved symlink destinations and available disk space. If `/data` lacks
space, stop writes that need that space and report the blocker; never fall back
to another filesystem. Do not delete existing datasets, results, checkpoints,
or unrelated files to make room without authorization.

Some frozen HD-EPIC provenance files and source constants retain historical
`/tmp` path strings. Their existing aliases point into
`/data/yaxuanli/qwen38-storage`. Preserve those immutable identities; verify
the aliases still resolve there before using the old code. Never recreate an
ordinary directory under `/tmp` if an alias is missing.
