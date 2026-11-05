# Public source distribution

The public repository contains a reviewed source snapshot, usage documentation and generic
configuration examples. It does not mirror private development history or operational records.
Upstream license notices and contributor attribution in source files remain intact.

Before publishing an update:

1. Prepare an isolated checkout from the intended source version. Do not push the development
   checkout or its branches/tags directly to the public remote.
2. Remove private repository URLs, organization identifiers, deployment inventories, personal
   paths, private endpoints, runtime logs, caches, calibration captures and generated bundles.
3. Require operators to provide authorized external sources and model artifacts. Use reserved
   example domains and documentation networks for examples, never real deployment addresses.
4. Scan every tracked file, filenames, commit author/committer identities and messages. Review
   non-text files separately. Run the affected tests and inspect the exact publication diff.
5. Publish only the reviewed tree and public history, using a public contact identity. Do not
   merge private history back into the public branch. Inspect all public branches and tags.
6. Verify the remote refs and a fresh clone. Removing refs does not remove previously cached
   commits, pull-request references or copies; coordinate removal with the hosting provider
   when private information was previously exposed.

The source CLI can be installed with `pipx install ./clients/data-platform-cli`. Hosted
installer/update URLs, CA hashes and server addresses are deployment templates, and must be
set by the distribution owner before use. The network configuration uses a reserved
`192.0.2.0/24` example allowlist; replace it with your own authorized network policy.

This workflow changes source publication only. It does not deploy Server/Agent or publish CLI
release artifacts. Private operational records must stay outside the public repository.
