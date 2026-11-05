//! File inputs: a Field file, an image, a sound or a data table, read again
//! when it changes.

use std::path::{Path, PathBuf};

use entelechy_kernel::{FieldData, Handle, Runtime};

use super::{
    Declared, Inputs, TableElement, head_within, kept_as, put_on_device, read_within, room,
};
use crate::store::{Stamp, Store};
use crate::{Diagnostic, Event};

/// A file an input is read from, and the element a table was read as and
/// the tables its rows named, as they were when it was read.
pub(super) struct Asset {
    pub(super) path: PathBuf,
    stamp: Option<Stamp>,
    table: Option<TableElement>,
    named: Vec<(PathBuf, Option<Stamp>)>,
}

/// A data table's rows as the element the Kernel reading input `name`
/// lays out; refused when no Kernel reads it.
fn read_table(
    bytes: &[u8],
    extension: &str,
    name: &str,
    table: Option<&TableElement>,
    room: usize,
    tables: &dyn Fn(&str) -> Result<Vec<String>, String>,
) -> Result<(FieldData, String), String> {
    let table = table.ok_or_else(|| {
        format!("a data table is read as the Kernel reading it lays out its element, and no published Kernel reads `{name}`")
    })?;
    let element = entelechy_formats::table::Element {
        names: &table.names,
        schema: &table.schema,
    };
    let data = entelechy_formats::table::decode(bytes, extension, &element, room, tables)?;
    let detail = format!("{} row(s) of {}", data.domain().count(), data.schema());
    Ok((data, detail))
}

/// The row names of the table `name` beside `dir` (`name.csv`, else
/// `name.json`), read as any input is, and the file and its stamp noted.
fn table_names(
    store: &dyn Store,
    readable: &dyn Fn(&Path) -> Result<(), String>,
    dir: &Path,
    name: &str,
    noted: &std::cell::RefCell<Vec<(PathBuf, Option<Stamp>)>>,
) -> Result<Vec<String>, String> {
    if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') {
        return Err(format!(
            "`{name}` is no table's name: a table is named by its file's stem beside this one"
        ));
    }
    for extension in ["csv", "json"] {
        let path = dir.join(format!("{name}.{extension}"));
        if !store.exists(&path) {
            continue;
        }
        readable(&path)?;
        let bytes = store
            .read(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        noted.borrow_mut().push((path.clone(), store.stamp(&path)));
        return entelechy_formats::table::row_names(&bytes, extension)
            .map_err(|why| format!("{}: {why}", path.display()));
    }
    Err(format!(
        "no table `{name}` beside it ({name}.csv or {name}.json)"
    ))
}

/// A Field file's Field and its line, its header read first and its
/// elements then as words, never held as bytes beside them.
fn read_field(store: &dyn Store, path: &Path, room: usize) -> Result<(FieldData, String), String> {
    let start = head_within(store, path, room)?;
    let (schema, domain, from) =
        entelechy_formats::file::layout(&start).map_err(|e| e.to_string())?;
    let (words, bytes) = store.read_words(path, from).map_err(|e| e.to_string())?;
    let data = entelechy_formats::file::from_words(schema, domain, words, bytes)
        .map_err(|e| e.to_string())?;
    let detail = entelechy_formats::describe(&data);
    Ok((data, detail))
}

impl Inputs {
    /// Reads new or changed input files; forgets undeclared ones.
    pub(super) fn sync_assets(
        &mut self,
        store: &dyn Store,
        declared: &Declared,
        readable: &dyn Fn(&Path) -> Result<(), String>,
        runtime: &mut Runtime,
        events: &mut Vec<Event>,
        on_device: &mut Vec<Handle>,
    ) {
        let Declared {
            graph,
            files,
            tables,
            unread,
        } = *declared;
        let wanted: Vec<entelechy_scene::Asset> = files.assets.clone();
        let stale: Vec<String> = self
            .assets
            .keys()
            .filter(|name| !wanted.iter().any(|asset| &asset.name == *name))
            .cloned()
            .collect();
        for name in stale {
            self.assets.remove(&name);
            self.forget(runtime.host_mut(), &name);
        }
        for entelechy_scene::Asset {
            name,
            path,
            ty,
            bytes,
        } in wanted
        {
            let current = store.stamp(&path);
            let table = entelechy_formats::table::is_table(&entelechy_formats::extension(&path))
                .then(|| tables.get(&name).cloned())
                .flatten();
            if self.assets.get(&name).is_some_and(|asset| {
                asset.path == path
                    && asset.stamp == current
                    && asset.table == table
                    && asset
                        .named
                        .iter()
                        .all(|(named, stamp)| store.stamp(named) == *stamp)
            }) {
                continue;
            }
            self.assets.insert(
                name.clone(),
                Asset {
                    path: path.clone(),
                    stamp: current,
                    table: table.clone(),
                    named: Vec::new(),
                },
            );
            // Checked at read time, so a refusal reports once per change.
            if let Err(why) = readable(&path) {
                self.forget(runtime.host_mut(), &name);
                events.push(Event::Rejected {
                    path,
                    error: Diagnostic::new(why),
                });
                continue;
            }
            // A save a replay does not start from reads as if never made.
            if unread.contains(&path) && graph.is_some_and(|g| g.fallback(&name).is_some()) {
                self.forget(runtime.host_mut(), &name);
                events.push(Event::Absent { name, path });
                continue;
            }
            let extension = entelechy_formats::extension(&path);
            let noted = std::cell::RefCell::new(Vec::new());
            let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
            let tables = |table: &str| table_names(store, readable, &dir, table, &noted);
            let read = match bytes {
                false if extension == "field" => read_field(store, &path, room(runtime.host())),
                _ => {
                    read_within(store, &path, room(runtime.host())).and_then(|found| match bytes {
                        true => entelechy_formats::decode_bytes(&found),
                        false if entelechy_formats::table::is_table(&extension) => read_table(
                            &found,
                            &extension,
                            &name,
                            table.as_ref(),
                            room(runtime.host()),
                            &tables,
                        ),
                        false => entelechy_formats::decode(
                            &found,
                            &path,
                            Some(&ty),
                            room(runtime.host()),
                        ),
                    })
                }
            };
            if let Some(asset) = self.assets.get_mut(&name) {
                asset.named = noted.into_inner();
            }
            let read = read.and_then(|(data, detail)| {
                let data = kept_as(data, &ty)?;
                // A file that stands in for its fallback must be what the
                // fallback gives: one of another layout is refused here,
                // by its name, as a state of another layout restarts.
                if let Some(element) = declared.tables.get(&name)
                    && graph.is_some_and(|g| g.fallback(&name).is_some())
                    && !element.takes(data.schema())
                {
                    return Err(format!(
                        "holds a Field of {}, which is not what `{name}` is read as; reading its fallback, and the next write replaces the file",
                        data.schema().describe()
                    ));
                }
                let detail = match data.schema().uniform_encoding() {
                    Some(encoding) if !data.schema().is_wide() => {
                        format!("{detail}, kept as {encoding}")
                    }
                    _ => detail,
                };
                Ok((data, detail))
            });
            let (data, detail) = match read {
                Ok(loaded) => loaded,
                // A missing file with a fallback is absent, not failed.
                Err(_)
                    if !store.exists(&path)
                        && graph.is_some_and(|g| g.fallback(&name).is_some()) =>
                {
                    self.forget(runtime.host_mut(), &name);
                    events.push(Event::Absent { name, path });
                    continue;
                }
                Err(error) => {
                    events.push(Event::Rejected {
                        path: path.clone(),
                        error: Diagnostic::new(error.to_string()),
                    });
                    continue;
                }
            };
            match runtime.host_mut().adopt(data) {
                Ok(handle) => {
                    self.hold(runtime.host_mut(), &name, handle);
                    if put_on_device(runtime, handle) {
                        on_device.push(handle);
                    }
                    events.push(Event::Loaded { name, path, detail });
                }
                Err(error) => events.push(Event::Rejected {
                    path,
                    error: Diagnostic::new(error.to_string()),
                }),
            }
        }
    }
}
