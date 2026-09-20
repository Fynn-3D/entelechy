# Migrate

State migration, the architecture plan's section 24, on record. The
scene is a tally: one row of state, `Tally { count }`, counting the
evaluations through a `<~` binding. `agent.jsonl` is what an agent sent
over the creation protocol to change what the tally carries while it
ran:

1. Three `evaluate`s, then `read`: the count is 3.
2. `publish tally.upgrade@1`, a map from the old `Tally` to a new one
   with `doubled`; `publish tally.step@1` again, carrying the new
   `Tally`. The Kernel's own `struct Tally` stands in for the scene's.
3. `compose` the scene with `previous <~ count.result from seed.result
   migrate tally.upgrade@1`.
4. `evaluate`: the report says the state of `count.previous` was
   migrated by `tally.upgrade@1`; `read`: the count is 4 and its double
   8. The next evaluation counts 5 and 10, migrating nothing.
5. `compose` the scene without `migrate`, `publish` the old
   `tally.step@1` again: its port wants the old layout and the state has
   the new one. `evaluate`: the report says the state reset, with why;
   `read`: the count is 1, from the seed.

Nothing here costs a frame that migrates nothing: the layout check is
the one every evaluation already made. The test in
`crates/cli/tests/protocol.rs` replays this record on a copy of the
example.

```sh
cargo build --release
cp -r examples/migrate /tmp/migrate
./target/release/orla serve /tmp/migrate/scene.osn < examples/migrate/agent.jsonl
```
