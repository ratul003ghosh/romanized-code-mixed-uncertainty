# How we work

## Git
- Nobody pushes to `main`. Work on a branch named `<member>/<short-topic>`, open a pull request, get one review, then merge.
- `configs/schema.json` is frozen. Changing it needs every member's approval, a version bump in `$id`, a new tag, and an entry in `docs/decisions.md`.
- Never commit data, tokens or checkpoints. `.gitignore` blocks `data/`; check your diff anyway.

## Kaggle
- Everyone runs their own copy of `kaggle_block1.ipynb` with their own secrets (`GITHUB_TOKEN`, `HF_TOKEN`). Secrets are never shared.
- Shared inputs live in private Kaggle Datasets owned by one member and shared with the group. Each upload is a new dataset version; record the version number you used.
- Long GPU jobs (silver generation, teacher scoring) are split into shards by index. Claim a shard in the issue before running it, so two people do not spend quota on the same shard.
- Every output file records: git commit, member, shard, seed, date.

## Annotation
- The two annotators on a gold item work independently and do not see each other's labels until adjudication.
- Annotators do not look at teacher outputs for gold items.
- Only members covered by the ethics approval open the BanglishRev real-PII slice.

## Decisions
- Anything that changes the method, data or evaluation goes in `docs/decisions.md` with the date and who agreed.
