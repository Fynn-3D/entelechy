# Migrate

State migration (architecture plan, section 24) as a protocol record. The
scene is a tally: one row of state, `Tally { count }`, carried through a
`<~` binding. `agent.jsonl`:

1. Three `evaluate`s, then `read`: the count is 3.
2. `publish tally.upgrade@1`, a map from the old `Tally` to one with
   `doubled`, and `publish tally.step@1` carrying the new `Tally`.
3. `compose` with `previous <~ count.result from seed.result migrate
   tally.upgrade@1`.
4. `evaluate`: the report names the migration; `read`: count 4, double 8.
5. `compose` without `migrate` and publish the old `tally.step@1`: the
   layouts differ, the report says the state reset and why; `read`: 1.

`crates/interface/orla/tests/protocol.rs` replays this record on a copy
of the example.

```sh
cargo build --release
cp -r examples/migrate /tmp/migrate
./target/release/orla serve /tmp/migrate/scene.osn < examples/migrate/agent.jsonl
```
