# CLAUDE.md

## Git — do not commit or push

**Never run `git commit` or `git push`.** Leave finished work as uncommitted changes in the
working tree and say what changed; committing and pushing is the developer's call, not
Claude's. This holds even when the work is verified and even when a previous message in the
same session asked for a commit — ask again rather than assuming the permission carries
forward.

The same applies to anything that rewrites history or moves the branch: no `git reset`,
`git rebase`, `git merge`, `git stash`, `git checkout <branch>`, or `git checkout -- <path>`
without being asked for that specific command. `git stash` and `git checkout -- .` are the
dangerous ones — they silently discard uncommitted work, and in this repo the running system
reads code straight from the working tree (see below), so discarding it also changes what is
live.

Reading git is always fine: `status`, `log`, `diff`, `show`, `fetch`.

## Environment facts that are easy to get wrong

**There are two clones of this repo on this machine.** They are not interchangeable:

    /Users/samta/veda-platform          # Docker containers bind-mount THIS one; complete .env
    /Users/samta/Desktop/veda-platform  # .env is incomplete (missing ~32 keys)

Check which one you are in before concluding a change "isn't showing up". A change made in
one is invisible in the other's `git status`.

**Code reaches the containers through a bind mount, not the image.** `docker-compose.yml`
mounts `.:/app`, so editing a file changes what the running service will execute — after a
restart, since the Python process holds the old modules. `docker compose restart` does NOT
pick up `.env` changes (env is read at container *create* time); that needs `up -d`, which
recreates. The same is true of `restart:` policies.

**`.env` is gitignored, so its settings do not travel.** Anything set only there is local to
this machine. Two settings currently matter and exist nowhere else — without them the engine
asks the Ollama host for a model it will not serve (503, "server busy") and the same question
can produce a different intent on each run:

    SLM_MODEL_NAME=qwen2.5:7b-instruct
    SLM_TEMPERATURE=0

**Two separate Postgres servers are in play.** Do not confuse them:

    pgvector/pgvector:pg17 in Docker, host port 15432   -> veda, veda_engine (platform's own)
    homebrew postgresql@17 on the host, port 5432       -> homzhub (the source DB, a local copy)

`veda` holds the Django tables; `veda_engine` holds the engine's pgvector tables
(`doc_chunks`, `column_embeddings_v2`, `source_item_embeddings`, …). Code that queries an
engine table through Django's `connection` works on a single-database setup and silently
returns "relation does not exist" here — the error gets swallowed and the caller reports
zero rows. Reach engine tables through `VEDA_INTERNAL_*` instead.

Prod is on DigitalOcean and nothing here points at it: query execution resolves its
connection from the `sources_source` table, which names the local copy. `run_homzhub_query.sh`
is the exception — it exports `VEDA_SOURCE_*` at `homzhub_prod` on DigitalOcean, and
`veda/runtime.py::get_db_config` falls back to those env vars when no request context is set.
Do not run it casually.
