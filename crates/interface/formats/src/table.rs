//! Data tables: CSV and JSON read as rows of the element a Kernel reads
//! them as, and rows written as them. The caller gives the name of each
//! number an element holds (`pos.x`, `items.0`, `look.race`; one number
//! alone is unnamed) and the element's layout; a record gives each named
//! number its value.
//!
//! A CSV file's first line names its columns; each line after it is a
//! row, its cells separated by commas, a cell in double quotes holding
//! commas and doubled quotes. A JSON file is an array of objects, a row
//! each, nested objects and arrays naming their numbers by a dot (`{"pos":
//! [1, 2]}` gives `pos.0` and `pos.1`); an array of numbers, or of arrays,
//! is rows of one unnamed number, or of numbered ones. A vector's lane is
//! named `x` to `w` or `0` to `3`. Every number of the element must be
//! given, and every column or key must name one, but for a note: a
//! column or key whose name begins with `#` (`#name`) is for whoever
//! reads the table, and is left out.
//!
//! An `[Int; N]` may be given as text under its own name (`name`, not
//! `name.0`): a character's code an element, zeros after, the cell as
//! written (a CSV's quotes and a JSON string's escapes taken off); a
//! text of more than N characters is refused. Written, an array whose
//! every row holds characters, then zeros, goes out as that text.
//!
//! A row's `#name` names it to other tables: a whole-number cell written
//! `table:name` (`beasts:goblin`) is the place of the row so named in
//! the table of that name beside this one, counted from 0. A name no row
//! has, or two rows share, is refused.

use std::collections::BTreeMap;

use entelechy_host::{Domain, FieldData, Scalar, Schema};

/// The element a table's rows are read as: the name of each number it
/// holds, in layout order, and the layout.
pub struct Element<'a> {
    /// A name for each number, as [`entelechy_ir`]'s `number_names` gives them.
    pub names: &'a [String],
    /// The layout the numbers are kept in.
    pub schema: &'a Schema,
}

/// Whether `extension` names a data table.
pub fn is_table(extension: &str) -> bool {
    extension == "csv" || extension == "json"
}

/// The names of a table's rows, its `#name` cells in order (empty where
/// a row has none): what another table's `table:name` cells count by.
pub fn row_names(bytes: &[u8], extension: &str) -> Result<Vec<String>, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "the table is not UTF-8 text".to_owned())?;
    let records = match extension {
        "csv" => csv_records(text)?,
        "json" => json_records(text)?,
        other => return Err(format!("`.{other}` is not a data table; use .csv or .json")),
    };
    Ok(records
        .iter()
        .map(|r| {
            r.cells
                .iter()
                .find(|(name, _)| name == "#name")
                .map(|(_, value)| value.trim().to_owned())
                .unwrap_or_default()
        })
        .collect())
}

/// A table's rows as a `Rows` Field of `element`, from the file's bytes;
/// `extension` says which format. A table that would hold more than
/// `words` words is refused before it is laid out. `tables` gives the
/// row names of the table a `table:name` cell names.
pub fn decode(
    bytes: &[u8],
    extension: &str,
    element: &Element,
    words: usize,
    tables: &dyn Fn(&str) -> Result<Vec<String>, String>,
) -> Result<FieldData, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "the table is not UTF-8 text".to_owned())?;
    let records = match extension {
        "csv" => csv_records(text)?,
        "json" => json_records(text)?,
        other => return Err(format!("`.{other}` is not a data table; use .csv or .json")),
    };
    let per_row = element.schema.value_words();
    let needed = records.len().saturating_mul(per_row);
    if needed > words {
        return Err(entelechy_host::HostError::BudgetExceeded {
            requested: needed,
            available: words,
        }
        .to_string());
    }
    // One number alone is unnamed: a CSV's one column is it, whatever
    // its heading.
    let unnamed = element.names.len() == 1 && element.names[0].is_empty();
    let records: Vec<Record> = records
        .into_iter()
        .map(|r| {
            if unnamed && r.cells.len() == 1 {
                Record {
                    cells: vec![(String::new(), r.cells[0].1.clone())],
                    ..r
                }
            } else {
                r
            }
        })
        .collect();
    let layout = laid(element.names, element.schema)?;
    let runs = text_runs(element.names, &layout);
    let mut wanted = wanted_names(element.names);
    for run in &runs {
        wanted.insert(run.name.clone(), run.name.clone());
    }
    let mut values = Vec::with_capacity(needed);
    let records: Vec<Record> = records
        .into_iter()
        .map(|r| Record {
            cells: r
                .cells
                .into_iter()
                .filter(|(name, _)| !is_note(name))
                .collect(),
            ..r
        })
        .collect();
    for record in &records {
        // Every column or key names a number of the element.
        if let Some((name, _)) = record
            .cells
            .iter()
            .find(|(name, _)| !wanted.contains_key(name.as_str()))
        {
            return Err(format!(
                "{}: `{name}` names no number of the element; it holds {}",
                record.at,
                listed(element.names)
            ));
        }
        let mut place = 0;
        while place < element.names.len() {
            // An array given as text: a character's code an element, zeros
            // after.
            if let Some(run) = runs.iter().find(|run| run.first == place)
                && let Some((_, text)) = record.cells.iter().find(|(name, _)| *name == run.name)
            {
                let members = &element.names[place..place + run.count];
                if let Some(member) = members.iter().find(|m| given(record, m).is_some()) {
                    return Err(format!(
                        "{}: `{}` is given as text and `{member}` as a number; give one",
                        record.at, run.name
                    ));
                }
                let codes: Vec<u32> = text.chars().map(u32::from).collect();
                if codes.len() > run.count {
                    return Err(format!(
                        "{}: `{}` holds {} characters, and {text:?} is {}",
                        record.at,
                        run.name,
                        run.count,
                        codes.len()
                    ));
                }
                values.extend_from_slice(&codes);
                values.extend(std::iter::repeat_n(0, run.count - codes.len()));
                place += run.count;
                continue;
            }
            let name = &element.names[place];
            let scalar = layout[place].1;
            place += 1;
            let text = given(record, name).ok_or_else(|| {
                format!(
                    "{}: no {} given",
                    record.at,
                    if name.is_empty() {
                        "value".to_owned()
                    } else {
                        format!("`{name}`")
                    }
                )
            })?;
            let text = &match named_row(text.trim(), scalar, tables) {
                Some(Ok(row)) => row.to_string(),
                Some(Err(why)) => {
                    return Err(format!(
                        "{}: `{}` is {:?}: {why}",
                        record.at,
                        if name.is_empty() { "the value" } else { name },
                        text.trim()
                    ));
                }
                None => text.to_owned(),
            };
            let words_of = parsed(text.trim(), scalar).ok_or_else(|| {
                format!(
                    "{}: `{}` is {:?}, not {}",
                    record.at,
                    if name.is_empty() { "the value" } else { name },
                    text,
                    scalar_name(scalar)
                )
            })?;
            values.extend_from_slice(&words_of[..scalar.words()]);
        }
    }
    let count =
        u32::try_from(records.len()).map_err(|_| "the table has too many rows".to_owned())?;
    let domain = Domain::Rows(count);
    FieldData::from_values(element.schema.clone(), domain, &values).map_err(|e| e.to_string())
}

/// A `Rows` Field as the table `extension` names, each number under
/// `names` (the element's, as [`Element`] holds them): what [`decode`]
/// reads back word for word. A CSV's heading line names the columns, one
/// number alone `value`; a JSON array holds a row an object, the dots
/// nesting objects and the numbered parts arrays, one number alone a
/// number. An `[Int; N]` whose every row holds text (characters, then
/// zeros) is written as that text under the array's name. A JSON table
/// cannot hold a NaN or an infinity.
pub fn encode(data: &FieldData, names: &[String], extension: &str) -> Result<Vec<u8>, String> {
    if !matches!(data.domain(), Domain::Rows(_)) {
        return Err("not rows, which a table holds".to_owned());
    }
    let layout = laid(names, data.schema())?;
    let per_row = data.schema().value_words();
    let values = data.values();
    let rows: Vec<&[u32]> = values.chunks_exact(per_row.max(1)).collect();
    // Each column: a number by its place among the names, or an array's
    // text.
    let mut columns: Vec<(String, Option<&TextRun>, usize)> = Vec::new();
    let runs = text_runs(names, &layout);
    let mut place = 0;
    while place < names.len() {
        let texts = runs.iter().find(|run| {
            run.first == place
                && rows
                    .iter()
                    .all(|row| text_of(&row[layout[place].0..][..run.count]).is_some())
                && rows.iter().any(|row| row[layout[place].0] != 0)
        });
        match texts {
            Some(run) => {
                columns.push((run.name.clone(), Some(run), place));
                place += run.count;
            }
            None => {
                columns.push((names[place].clone(), None, place));
                place += 1;
            }
        }
    }
    let cells = |row: &[u32]| -> Vec<Cell> {
        columns
            .iter()
            .map(|&(_, run, place)| {
                let (at, scalar) = layout[place];
                match run {
                    Some(run) => Cell::Text(text_of(&row[at..][..run.count]).unwrap_or_default()),
                    None => Cell::Number(written(&row[at..at + scalar.words()], scalar)),
                }
            })
            .collect()
    };
    let unnamed = names.len() == 1 && names[0].is_empty();
    let mut out = String::new();
    match extension {
        "csv" => {
            let headings: Vec<&str> = columns.iter().map(|(name, ..)| name.as_str()).collect();
            out.push_str(&if unnamed {
                "value".to_owned()
            } else {
                headings.join(",")
            });
            out.push('\n');
            for row in &rows {
                let row: Vec<String> = cells(row).iter().map(Cell::csv).collect();
                out.push_str(&row.join(","));
                out.push('\n');
            }
        }
        "json" => {
            out.push('[');
            for (index, row) in rows.iter().enumerate() {
                let cells = cells(row);
                if let Some(Cell::Number(bad)) = cells.iter().find(|cell| {
                    matches!(cell, Cell::Number(n) if matches!(n.as_str(), "NaN" | "inf" | "-inf"))
                }) {
                    return Err(format!(
                        "row {index} holds {bad}, which JSON cannot hold; write it as .csv"
                    ));
                }
                out.push_str(if index == 0 { "" } else { "," });
                if unnamed {
                    out.push_str(if index == 0 { "" } else { " " });
                    out.push_str(&cells[0].json());
                    continue;
                }
                let mut object = Json::Object(Vec::new());
                for ((name, ..), cell) in columns.iter().zip(&cells) {
                    nest(
                        &mut object,
                        &name.split('.').collect::<Vec<_>>(),
                        cell.json(),
                    );
                }
                out.push_str("\n  ");
                written_json(&object, &mut out);
            }
            out.push_str(if unnamed || rows.is_empty() {
                "]\n"
            } else {
                "\n]\n"
            });
        }
        other => return Err(format!("`.{other}` is not a data table; use .csv or .json")),
    }
    Ok(out.into_bytes())
}

/// Each name's first word in the element and its scalar.
fn laid(names: &[String], schema: &Schema) -> Result<Vec<(usize, Scalar)>, String> {
    let scalars = schema.scalars();
    let mut at = 0;
    let mut layout = Vec::with_capacity(names.len());
    for _ in names {
        let scalar = *scalars
            .get(at)
            .ok_or("the names are more than the element's numbers")?;
        layout.push((at, scalar));
        at += scalar.words();
    }
    if at != schema.value_words() {
        return Err("the names are fewer than the element's numbers".to_owned());
    }
    Ok(layout)
}

/// An `[Int; N]` an element holds, which a cell may give as text, a
/// character's code an element and zeros after: its name, the place of
/// its first number among the element's names, and N. An array's numbers
/// are named `name.0` to `name.N-1` and no field of a struct is named by
/// a number, so a run of Ints so named is an array of them.
struct TextRun {
    name: String,
    first: usize,
    count: usize,
}

fn text_runs(names: &[String], layout: &[(usize, Scalar)]) -> Vec<TextRun> {
    let int = |place: usize| layout.get(place).is_some_and(|(_, s)| *s == Scalar::Int);
    let mut runs = Vec::new();
    let mut place = 0;
    while place < names.len() {
        let Some(name) = names[place].strip_suffix(".0").filter(|_| int(place)) else {
            place += 1;
            continue;
        };
        let count = (0..)
            .take_while(|&k| {
                int(place + k)
                    && names
                        .get(place + k)
                        .is_some_and(|n| *n == format!("{name}.{k}"))
            })
            .count();
        runs.push(TextRun {
            name: name.to_owned(),
            first: place,
            count,
        });
        place += count;
    }
    runs
}

/// Whether `code` is a character a text cell holds: a tab, a new line, or
/// a printable one.
fn is_character(code: u32) -> bool {
    matches!(code, 9 | 10) || (code >= 32 && code != 127 && char::from_u32(code).is_some())
}

/// The text an array's codes hold, characters then zeros; `None` for
/// numbers that are no text.
fn text_of(codes: &[u32]) -> Option<String> {
    let length = codes
        .iter()
        .position(|code| *code == 0)
        .unwrap_or(codes.len());
    if codes[length..].iter().any(|code| *code != 0) {
        return None;
    }
    codes[..length]
        .iter()
        .map(|&code| char::from_u32(code).filter(|_| is_character(code)))
        .collect()
}

/// A cell to write: a number as its text, or a text.
enum Cell {
    Number(String),
    Text(String),
}

impl Cell {
    /// As a CSV cell: a text in quotes when it is empty or holds what
    /// parts cells or lines.
    fn csv(&self) -> String {
        match self {
            Self::Number(number) => number.clone(),
            Self::Text(text) if text.is_empty() || text.contains([',', '"', '\n']) => {
                format!("\"{}\"", text.replace('"', "\"\""))
            }
            Self::Text(text) => text.clone(),
        }
    }

    /// As a JSON value.
    fn json(&self) -> String {
        match self {
            Self::Number(number) => number.clone(),
            Self::Text(text) => {
                let mut out = String::from("\"");
                for c in text.chars() {
                    match c {
                        '"' => out.push_str("\\\""),
                        '\\' => out.push_str("\\\\"),
                        '\n' => out.push_str("\\n"),
                        '\t' => out.push_str("\\t"),
                        c => out.push(c),
                    }
                }
                out.push('"');
                out
            }
        }
    }
}

/// A number's words as a cell: the shortest text that reads back as them.
fn written(words: &[u32], scalar: Scalar) -> String {
    let wide = || u64::from(words[0]) | (u64::from(words[1]) << 32);
    match scalar {
        Scalar::Float => f32::from_bits(words[0]).to_string(),
        Scalar::Int => (words[0] as i32).to_string(),
        Scalar::Bool => (words[0] != 0).to_string(),
        Scalar::Long => (wide() as i64).to_string(),
        Scalar::Double => f64::from_bits(wide()).to_string(),
    }
}

/// Puts `cell` into `object` at the path `parts` (a number's name, its
/// dots apart), an object for each part before the last.
fn nest(object: &mut Json, parts: &[&str], cell: String) {
    let (Json::Object(fields), [head, rest @ ..]) = (object, parts) else {
        return;
    };
    if rest.is_empty() {
        fields.push(((*head).to_owned(), Json::Scalar(cell)));
        return;
    }
    let at = match fields.iter().position(|(key, _)| key == head) {
        Some(at) => at,
        None => {
            fields.push(((*head).to_owned(), Json::Object(Vec::new())));
            fields.len() - 1
        }
    };
    nest(&mut fields[at].1, rest, cell);
}

/// `value` as JSON text; an object whose keys count from 0 in order is an
/// array.
fn written_json(value: &Json, out: &mut String) {
    match value {
        Json::Scalar(text) => out.push_str(text),
        Json::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                out.push_str(if index == 0 { "" } else { ", " });
                written_json(item, out);
            }
            out.push(']');
        }
        Json::Object(fields) => {
            let numbered = fields
                .iter()
                .enumerate()
                .all(|(index, (key, _))| *key == index.to_string());
            out.push(if numbered { '[' } else { '{' });
            for (index, (key, item)) in fields.iter().enumerate() {
                out.push_str(if index == 0 { "" } else { ", " });
                if !numbered {
                    out.push_str(&format!("\"{key}\": "));
                }
                written_json(item, out);
            }
            out.push(if numbered { ']' } else { '}' });
        }
    }
}

/// A whole-number cell written `table:name`: the place of the row so
/// named in that table; `None` for any other cell.
fn named_row(
    text: &str,
    scalar: Scalar,
    tables: &dyn Fn(&str) -> Result<Vec<String>, String>,
) -> Option<Result<usize, String>> {
    let (table, row) = text.split_once(':')?;
    if !matches!(scalar, Scalar::Int | Scalar::Long) {
        return Some(Err(format!(
            "only a whole number names a row, not {}",
            scalar_name(scalar)
        )));
    }
    let (table, row) = (table.trim(), row.trim());
    Some(tables(table).and_then(|names| {
        let places: Vec<usize> = names
            .iter()
            .enumerate()
            .filter(|(_, n)| *n == row)
            .map(|(i, _)| i)
            .collect();
        match places.as_slice() {
            [place] => Ok(*place),
            [] => Err(format!("no row of `{table}` is named `{row}`")),
            _ => Err(format!(
                "{} rows of `{table}` are named `{row}`",
                places.len()
            )),
        }
    }))
}

/// Whether a column or key is a note for the reader: its name, or a
/// nested key's, begins with `#`.
fn is_note(name: &str) -> bool {
    name.starts_with('#') || name.contains(".#")
}

/// A row as read: where it was written, for errors, and each cell by the
/// name it was given.
struct Record {
    at: String,
    cells: Vec<(String, String)>,
}

/// The names a record may give a number by: its own, and a vector lane's
/// letter as its number (`pos.x` as `pos.0`). Each maps to the element's
/// name.
fn wanted_names(names: &[String]) -> BTreeMap<String, String> {
    let mut wanted = BTreeMap::new();
    for name in names {
        wanted.insert(name.clone(), name.clone());
        if let Some(other) = lane_number(name) {
            wanted.insert(other, name.clone());
        }
    }
    wanted
}

/// A vector lane's name with its letter as its number: `pos.y` as
/// `pos.1`, a lone `y` as `1`.
fn lane_number(name: &str) -> Option<String> {
    ["x", "y", "z", "w"]
        .iter()
        .enumerate()
        .find_map(|(index, lane)| {
            if name == *lane {
                return Some(index.to_string());
            }
            name.strip_suffix(lane)
                .filter(|head| head.ends_with('.'))
                .map(|head| format!("{head}{index}"))
        })
}

/// The cell a record gives the element's number `name`, by its name or
/// its lane's number.
fn given<'a>(record: &'a Record, name: &str) -> Option<&'a str> {
    let other = lane_number(name);
    record
        .cells
        .iter()
        .find(|(given, _)| given == name || other.as_deref() == Some(given.as_str()))
        .map(|(_, value)| value.as_str())
}

fn listed(names: &[String]) -> String {
    if names.len() == 1 && names[0].is_empty() {
        return "one unnamed value".to_owned();
    }
    names
        .iter()
        .map(|n| format!("`{n}`"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn scalar_name(scalar: Scalar) -> &'static str {
    match scalar {
        Scalar::Float => "a Float",
        Scalar::Int => "an Int",
        Scalar::Bool => "a Bool (true, false, 1 or 0)",
        Scalar::Double => "a Double",
        Scalar::Long => "a Long",
    }
}

/// A cell's value as the words its scalar takes, low first.
fn parsed(text: &str, scalar: Scalar) -> Option<[u32; 2]> {
    Some(match scalar {
        Scalar::Float => [text.parse::<f32>().ok()?.to_bits(), 0],
        Scalar::Int => [text.parse::<i32>().ok()? as u32, 0],
        Scalar::Bool => [
            match text {
                "true" | "1" => 1,
                "false" | "0" => 0,
                _ => return None,
            },
            0,
        ],
        Scalar::Long => {
            let v = text.parse::<i64>().ok()? as u64;
            [v as u32, (v >> 32) as u32]
        }
        Scalar::Double => {
            let v = text.parse::<f64>().ok()?.to_bits();
            [v as u32, (v >> 32) as u32]
        }
    })
}

// ---------------------------------------------------------------- CSV

/// A CSV file's rows, each cell named by its column's heading. Lines
/// with nothing but spaces in them are left out.
fn csv_records(text: &str) -> Result<Vec<Record>, String> {
    let lines = csv_lines(text)?;
    let mut lines = lines.into_iter().filter(|(_, _, filled)| *filled);
    let Some((_, headings, _)) = lines.next() else {
        return Err("the table has no heading line".to_owned());
    };
    let headings: Vec<String> = headings.iter().map(|h| h.trim().to_owned()).collect();
    let mut records = Vec::new();
    for (line, cells, _) in lines {
        if cells.len() != headings.len() {
            return Err(format!(
                "line {line}: {} cell(s) under {} heading(s)",
                cells.len(),
                headings.len()
            ));
        }
        records.push(Record {
            at: format!("line {line}"),
            cells: headings.iter().cloned().zip(cells).collect(),
        });
    }
    Ok(records)
}

/// Each line's cells, with the line's number and whether it holds more
/// than spaces; a quoted cell may hold a comma, a doubled quote and a
/// line break.
fn csv_lines(text: &str) -> Result<Vec<(usize, Vec<String>, bool)>, String> {
    let mut out = Vec::new();
    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut quoted = false;
    let mut line = 1;
    let mut began = 1;
    // Whether the line holds more than spaces: a quoted empty cell is a
    // cell.
    let mut filled = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        filled |= !c.is_whitespace();
        if quoted {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    chars.next();
                    cell.push('"');
                }
                '"' => quoted = false,
                '\n' => {
                    line += 1;
                    cell.push(c);
                }
                _ => cell.push(c),
            }
            continue;
        }
        match c {
            '"' if cell.trim().is_empty() => {
                cell.clear();
                quoted = true;
            }
            ',' => cells.push(std::mem::take(&mut cell)),
            '\r' => {}
            '\n' => {
                cells.push(std::mem::take(&mut cell));
                out.push((
                    began,
                    std::mem::take(&mut cells),
                    std::mem::take(&mut filled),
                ));
                line += 1;
                began = line;
            }
            _ => cell.push(c),
        }
    }
    if quoted {
        return Err(format!("line {began}: a quoted cell is not closed"));
    }
    if !cell.is_empty() || !cells.is_empty() {
        cells.push(cell);
        out.push((began, cells, filled));
    }
    Ok(out)
}

// ---------------------------------------------------------------- JSON

/// A JSON value, as far as a table needs one: numbers and booleans kept
/// as written.
enum Json {
    Scalar(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

/// A JSON file's rows: the top array's items, each flattened to its
/// numbers' names.
fn json_records(text: &str) -> Result<Vec<Record>, String> {
    let mut reader = JsonReader {
        text: text.as_bytes(),
        at: 0,
    };
    let top = reader.value()?;
    reader.space();
    if reader.at < reader.text.len() {
        return Err(reader.error("more after the table's array"));
    }
    let Json::Array(items) = top else {
        return Err("the table is not a JSON array of rows".to_owned());
    };
    Ok(items
        .into_iter()
        .enumerate()
        .map(|(row, item)| {
            let mut cells = Vec::new();
            flatten(&item, "", &mut cells);
            Record {
                at: format!("row {row}"),
                cells,
            }
        })
        .collect())
}

fn flatten(value: &Json, name: &str, out: &mut Vec<(String, String)>) {
    let joined = |part: &str| {
        if name.is_empty() {
            part.to_owned()
        } else {
            format!("{name}.{part}")
        }
    };
    match value {
        Json::Scalar(text) => out.push((name.to_owned(), text.clone())),
        Json::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                flatten(item, &joined(&i.to_string()), out);
            }
        }
        Json::Object(fields) => {
            for (key, item) in fields {
                flatten(item, &joined(key), out);
            }
        }
    }
}

struct JsonReader<'a> {
    text: &'a [u8],
    at: usize,
}

impl JsonReader<'_> {
    fn error(&self, what: &str) -> String {
        let before = &self.text[..self.at.min(self.text.len())];
        let line = before.iter().filter(|b| **b == b'\n').count() + 1;
        let column = before.iter().rev().take_while(|b| **b != b'\n').count() + 1;
        format!("line {line}, column {column}: {what}")
    }

    fn space(&mut self) {
        while self.at < self.text.len() && self.text[self.at].is_ascii_whitespace() {
            self.at += 1;
        }
    }

    fn eat(&mut self, byte: u8) -> bool {
        self.space();
        if self.text.get(self.at) == Some(&byte) {
            self.at += 1;
            true
        } else {
            false
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.space();
        match self.text.get(self.at) {
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                if self.eat(b']') {
                    return Ok(Json::Array(items));
                }
                loop {
                    items.push(self.value()?);
                    if self.eat(b']') {
                        return Ok(Json::Array(items));
                    }
                    if !self.eat(b',') {
                        return Err(self.error("expected `,` or `]`"));
                    }
                }
            }
            Some(b'{') => {
                self.at += 1;
                let mut fields = Vec::new();
                if self.eat(b'}') {
                    return Ok(Json::Object(fields));
                }
                loop {
                    self.space();
                    let key = self.string()?;
                    if !self.eat(b':') {
                        return Err(self.error("expected `:`"));
                    }
                    fields.push((key, self.value()?));
                    if self.eat(b'}') {
                        return Ok(Json::Object(fields));
                    }
                    if !self.eat(b',') {
                        return Err(self.error("expected `,` or `}`"));
                    }
                }
            }
            Some(b'"') => Ok(Json::Scalar(self.string()?)),
            Some(_) => {
                let start = self.at;
                while self.at < self.text.len() && !b",]} \t\r\n".contains(&self.text[self.at]) {
                    self.at += 1;
                }
                let word = std::str::from_utf8(&self.text[start..self.at]).unwrap_or("");
                if word.is_empty() || word == "null" {
                    return Err(self.error("expected a number, true or false"));
                }
                Ok(Json::Scalar(word.to_owned()))
            }
            None => Err(self.error("the table ends early")),
        }
    }

    /// A quoted string, its escapes as JSON writes them: `\uXXXX` a UTF-16
    /// unit, two of them a character beyond the first 65536.
    fn string(&mut self) -> Result<String, String> {
        if self.text.get(self.at) != Some(&b'"') {
            return Err(self.error("expected a quoted name"));
        }
        self.at += 1;
        let mut out = Vec::new();
        while let Some(&b) = self.text.get(self.at) {
            self.at += 1;
            match b {
                b'"' => {
                    return String::from_utf8(out).map_err(|_| self.error("a string is not UTF-8"));
                }
                b'\\' => {
                    let escaped = self.text.get(self.at).copied();
                    self.at += 1;
                    let c = match escaped {
                        Some(b'n') => '\n',
                        Some(b't') => '\t',
                        Some(b'r') => '\r',
                        Some(b'b') => '\u{8}',
                        Some(b'f') => '\u{c}',
                        Some(b'u') => self.unit_char()?,
                        Some(c @ (b'"' | b'\\' | b'/')) => char::from(c),
                        Some(_) => return Err(self.error("an unknown escape")),
                        None => return Err(self.error("the table ends early")),
                    };
                    out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes());
                }
                _ => out.push(b),
            }
        }
        Err(self.error("a quoted string is not closed"))
    }

    /// The character after `\u`: one unit, or a surrogate pair.
    fn unit_char(&mut self) -> Result<char, String> {
        let high = self.unit()?;
        let code = if (0xD800..0xDC00).contains(&high) {
            let low = if self.text.get(self.at..self.at + 2) == Some(b"\\u") {
                self.at += 2;
                self.unit()?
            } else {
                0
            };
            if !(0xDC00..0xE000).contains(&low) {
                return Err(self.error("a surrogate without its pair"));
            }
            0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
        } else {
            high
        };
        char::from_u32(code).ok_or_else(|| self.error("a surrogate without its pair"))
    }

    /// Four hexadecimal digits.
    fn unit(&mut self) -> Result<u32, String> {
        let digits = self
            .text
            .get(self.at..self.at + 4)
            .and_then(|d| std::str::from_utf8(d).ok())
            .and_then(|d| u32::from_str_radix(d, 16).ok())
            .ok_or_else(|| self.error("`\\u` takes four hexadecimal digits"))?;
        self.at += 4;
        Ok(digits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn no_tables(table: &str) -> Result<Vec<String>, String> {
        Err(format!("no table `{table}`"))
    }

    #[test]
    fn a_cell_names_a_row_of_another_table() {
        let beasts = b"#name,health\ngoblin,900\nwolf,1200\ncave croc,1500\n";
        let names = row_names(beasts, "csv").unwrap();
        assert_eq!(names, ["goblin", "wolf", "cave croc"]);
        let tables = |table: &str| {
            if table == "beasts" {
                Ok(names.clone())
            } else {
                Err(format!("no table `{table}`"))
            }
        };
        let element_names = vec!["slot".to_owned(), "beast".to_owned()];
        let schema = Schema::new(vec![Scalar::Int, Scalar::Int]);
        let e = Element {
            names: &element_names,
            schema: &schema,
        };
        let rows = decode(
            b"slot,beast\n0,beasts:cave croc\n1,beasts:goblin\n2,1\n",
            "csv",
            &e,
            usize::MAX,
            &tables,
        )
        .unwrap();
        assert_eq!(rows.values()[..], [0, 2, 1, 0, 2, 1]);
        let json = decode(
            br#"[{"slot": 0, "beast": "beasts:wolf"}]"#,
            "json",
            &e,
            usize::MAX,
            &tables,
        )
        .unwrap();
        assert_eq!(json.values()[..], [0, 1]);
        let missing = decode(
            b"slot,beast\n0,beasts:gobln\n",
            "csv",
            &e,
            usize::MAX,
            &tables,
        )
        .unwrap_err();
        assert!(
            missing.contains(
                "line 2: `beast` is \"beasts:gobln\": no row of `beasts` is named `gobln`"
            ),
            "{missing}"
        );
        let unknown = decode(
            b"slot,beast\n0,foes:goblin\n",
            "csv",
            &e,
            usize::MAX,
            &tables,
        )
        .unwrap_err();
        assert!(unknown.contains("no table `foes`"), "{unknown}");
        let twice = |_: &str| Ok(vec!["goblin".to_owned(), "goblin".to_owned()]);
        let shared = decode(
            b"slot,beast\n0,beasts:goblin\n",
            "csv",
            &e,
            usize::MAX,
            &twice,
        )
        .unwrap_err();
        assert!(
            shared.contains("2 rows of `beasts` are named `goblin`"),
            "{shared}"
        );
        let floats = Schema::new(vec![Scalar::Int, Scalar::Float]);
        let f = Element {
            names: &element_names,
            schema: &floats,
        };
        let float = decode(
            b"slot,beast\n0,beasts:goblin\n",
            "csv",
            &f,
            usize::MAX,
            &tables,
        )
        .unwrap_err();
        assert!(float.contains("only a whole number names a row"), "{float}");
    }

    fn element(names: &[&str], scalars: Vec<Scalar>) -> (Vec<String>, Schema) {
        (
            names.iter().map(|n| (*n).to_owned()).collect(),
            Schema::new(scalars),
        )
    }

    /// `name: [Int; 4]` and `hp: Int`.
    fn named_beast() -> (Vec<String>, Schema) {
        element(
            &["name.0", "name.1", "name.2", "name.3", "hp"],
            vec![Scalar::Int; 5],
        )
    }

    #[test]
    fn an_int_array_holds_text_a_code_a_character() {
        let (names, schema) = named_beast();
        let e = Element {
            names: &names,
            schema: &schema,
        };
        let csv = "#name,name,hp\ngoblin,哥布林,30\nwolf,\"a, \"\"\",45\nnone,,1\n";
        let rows = decode(csv.as_bytes(), "csv", &e, usize::MAX, &no_tables).unwrap();
        let codes = |s: &str| s.chars().map(|c| c as u32).collect::<Vec<_>>();
        assert_eq!(
            rows.values()[..5],
            [codes("哥布林"), vec![0, 30]].concat()[..]
        );
        assert_eq!(
            rows.values()[5..10],
            [codes("a, \""), vec![45]].concat()[..]
        );
        assert_eq!(rows.values()[10..], [0, 0, 0, 0, 1]);
        let json = "[{\"name\": \"哥布林\", \"hp\": 30}, {\"hp\": 45, \"name\": \"a, \\\"\"}, {\"name\": \"\", \"hp\": 1}]";
        let from_json = decode(json.as_bytes(), "json", &e, usize::MAX, &no_tables).unwrap();
        assert_eq!(from_json.values(), rows.values());
        // Written back, the text stays text, and reads back word for word.
        for extension in ["csv", "json"] {
            let bytes = encode(&rows, &names, extension).unwrap();
            let back = decode(&bytes, extension, &e, usize::MAX, &no_tables).unwrap();
            assert_eq!(back.values(), rows.values(), "{extension}");
        }
        assert_eq!(
            encode(&rows, &names, "csv").unwrap(),
            "name,hp\n哥布林,30\n\"a, \"\"\",45\n\"\",1\n".as_bytes()
        );
        assert_eq!(
            String::from_utf8(encode(&rows, &names, "json").unwrap()).unwrap(),
            "[\n  {\"name\": \"哥布林\", \"hp\": 30},\n  {\"name\": \"a, \\\"\", \"hp\": 45},\n  {\"name\": \"\", \"hp\": 1}\n]\n"
        );
        // Numbers that are no text stay numbers; the numbers are read still.
        let numbers =
            FieldData::from_values(schema.clone(), Domain::Rows(1), &[7, 0, 3, 0, 2]).unwrap();
        let csv = String::from_utf8(encode(&numbers, &names, "csv").unwrap()).unwrap();
        assert_eq!(csv, "name.0,name.1,name.2,name.3,hp\n7,0,3,0,2\n");
        let back = decode(csv.as_bytes(), "csv", &e, usize::MAX, &no_tables).unwrap();
        assert_eq!(back.values(), numbers.values());
        // A text longer than its array, and a text given with its numbers, are said.
        let long = decode(b"name,hp\nabcde,1\n", "csv", &e, usize::MAX, &no_tables).unwrap_err();
        assert!(
            long.contains("line 2") && long.contains("4 characters"),
            "{long}"
        );
        let both = decode(
            b"name,name.0,hp\nab,1,1\n",
            "csv",
            &e,
            usize::MAX,
            &no_tables,
        )
        .unwrap_err();
        assert!(
            both.contains("`name`") && both.contains("`name.0`"),
            "{both}"
        );
        // A one-column table keeps a row whose text is empty.
        let names = vec!["name.0".to_owned(), "name.1".to_owned()];
        let schema = Schema::new(vec![Scalar::Int; 2]);
        let e = Element {
            names: &names,
            schema: &schema,
        };
        let rows = decode(b"name\nab\n\"\"\nc\n", "csv", &e, usize::MAX, &no_tables).unwrap();
        assert_eq!(rows.values()[..], [97, 98, 0, 0, 99, 0]);
    }

    #[test]
    fn json_strings_take_every_escape() {
        let (names, schema) = named_beast();
        let e = Element {
            names: &names,
            schema: &schema,
        };
        let json = r#"[{"name": "A\/\r😀", "hp": 1}]"#;
        let rows = decode(json.as_bytes(), "json", &e, usize::MAX, &no_tables).unwrap();
        assert_eq!(rows.values()[..], [65, 47, 13, 0x1F600, 1]);
        let bad = decode(
            br#"[{"name": "\ud83d", "hp": 1}]"#,
            "json",
            &e,
            usize::MAX,
            &no_tables,
        )
        .unwrap_err();
        assert!(bad.contains("surrogate"), "{bad}");
    }

    #[test]
    fn csv_and_json_give_the_same_rows() {
        let (names, schema) = element(
            &["id", "power", "pos.x", "pos.y", "on"],
            vec![
                Scalar::Int,
                Scalar::Float,
                Scalar::Float,
                Scalar::Float,
                Scalar::Bool,
            ],
        );
        let e = Element {
            names: &names,
            schema: &schema,
        };
        let csv = "id,power,pos.x,pos.y,on\n3,1.5,0,2,true\n\n4,-2,\"1\",3,0\n";
        let json = r#"[{"id": 3, "power": 1.5, "pos": [0, 2], "on": true}, {"on": false, "pos": {"x": 1, "y": 3}, "id": 4, "power": -2}]"#;
        let a = decode(csv.as_bytes(), "csv", &e, usize::MAX, &no_tables).unwrap();
        let b = decode(json.as_bytes(), "json", &e, usize::MAX, &no_tables).unwrap();
        assert_eq!(a.values(), b.values());
        assert_eq!(a.domain().count(), 2);
        assert_eq!(a.values()[0], 3);
        assert_eq!(f32::from_bits(a.values()[6]), -2.0);
        assert_eq!(a.values()[9], 0);
    }

    #[test]
    fn a_table_says_what_is_wrong_and_where() {
        let (names, schema) = element(&["id", "power"], vec![Scalar::Int, Scalar::Float]);
        let e = Element {
            names: &names,
            schema: &schema,
        };
        let missing = decode(b"id\n3\n", "csv", &e, usize::MAX, &no_tables).unwrap_err();
        assert!(missing.contains("line 2: no `power` given"), "{missing}");
        let wrong = decode(b"id,power\n3,1\nx,2\n", "csv", &e, usize::MAX, &no_tables).unwrap_err();
        assert!(
            wrong.contains("line 3: `id` is \"x\", not an Int"),
            "{wrong}"
        );
        let stray = decode(
            br#"[{"id": 1, "power": 2, "speed": 3}]"#,
            "json",
            &e,
            usize::MAX,
            &no_tables,
        )
        .unwrap_err();
        assert!(stray.contains("row 0: `speed` names no number"), "{stray}");
        let broken = decode(
            b"[{\"id\": 1,\n \"power\" 2}]",
            "json",
            &e,
            usize::MAX,
            &no_tables,
        )
        .unwrap_err();
        assert!(
            broken.contains("line 2, column 10: expected `:`"),
            "{broken}"
        );
        let big = decode(b"id,power\n1,2\n3,4\n", "csv", &e, 3, &no_tables).unwrap_err();
        assert!(big.contains('4') && big.contains('3'), "{big}");
    }

    #[test]
    fn a_note_is_left_out() {
        let (names, schema) = element(&["id", "power"], vec![Scalar::Int, Scalar::Float]);
        let e = Element {
            names: &names,
            schema: &schema,
        };
        let csv = decode(
            b"#name,id,power\ngoblin,0,1.5\n\"cave, croc\",11,2\n",
            "csv",
            &e,
            usize::MAX,
            &no_tables,
        )
        .unwrap();
        let json = decode(br##"[{"#name": "goblin", "id": 0, "power": 1.5}, {"#name": "cave croc", "id": 11, "power": 2}]"##, "json", &e, usize::MAX, &no_tables).unwrap();
        assert_eq!(csv.values(), json.values());
        assert_eq!(csv.values()[2], 11);
    }

    #[test]
    fn written_rows_read_back_word_for_word() {
        let (names, schema) = element(
            &[
                "id", "power", "pos.x", "pos.y", "on", "big", "fine", "items.0", "items.1",
            ],
            vec![
                Scalar::Int,
                Scalar::Float,
                Scalar::Float,
                Scalar::Float,
                Scalar::Bool,
                Scalar::Long,
                Scalar::Long,
                Scalar::Double,
                Scalar::Double,
                Scalar::Int,
                Scalar::Int,
            ],
        );
        let e = Element {
            names: &names,
            schema: &schema,
        };
        let row = |id: i32, power: f32, x: f32, big: i64, fine: f64| {
            let big = big as u64;
            let fine = fine.to_bits();
            vec![
                id as u32,
                power.to_bits(),
                x.to_bits(),
                (-x).to_bits(),
                u32::from(id % 2 == 0),
                big as u32,
                (big >> 32) as u32,
                fine as u32,
                (fine >> 32) as u32,
                7,
                id as u32,
            ]
        };
        let words: Vec<u32> = [
            row(3, 0.1, 1e-30, -5_000_000_000, std::f64::consts::PI),
            row(-4, 3.4e38, 0.0, i64::MAX, -1e300),
        ]
        .concat();
        let rows = FieldData::from_values(schema.clone(), Domain::Rows(2), &words).unwrap();
        for extension in ["csv", "json"] {
            let bytes = encode(&rows, &names, extension).unwrap();
            let back = decode(&bytes, extension, &e, usize::MAX, &no_tables).unwrap();
            assert_eq!(back.values(), rows.values(), "{extension}");
        }
        let csv = String::from_utf8(encode(&rows, &names, "csv").unwrap()).unwrap();
        assert!(
            csv.starts_with("id,power,pos.x,pos.y,on,big,fine,items.0,items.1\n3,0.1,"),
            "{csv}"
        );
        let json = String::from_utf8(encode(&rows, &names, "json").unwrap()).unwrap();
        assert!(
            json.contains(r#""pos": {"x": "#) && json.contains(r#""items": [7, 3]"#),
            "{json}"
        );
        // One number alone is a CSV's one column and a JSON array of numbers.
        let one = vec![String::new()];
        let schema = Schema::new(vec![Scalar::Int]);
        let rows = FieldData::from_values(schema, Domain::Rows(2), &[5, 6]).unwrap();
        assert_eq!(encode(&rows, &one, "csv").unwrap(), b"value\n5\n6\n");
        assert_eq!(encode(&rows, &one, "json").unwrap(), b"[5, 6]\n");
        // What JSON cannot hold, and what is no table, are said.
        let schema = Schema::new(vec![Scalar::Float]);
        let rows =
            FieldData::from_values(schema.clone(), Domain::Rows(1), &[f32::NAN.to_bits()]).unwrap();
        assert!(encode(&rows, &one, "json").unwrap_err().contains("NaN"));
        assert_eq!(encode(&rows, &one, "csv").unwrap(), b"value\nNaN\n");
        let grid = FieldData::from_values(schema, Domain::Grid(1, 1), &[0]).unwrap();
        assert!(encode(&grid, &one, "csv").unwrap_err().contains("not rows"));
    }

    #[test]
    fn one_unnamed_number_and_wide_numbers() {
        let names = vec![String::new()];
        let schema = Schema::new(vec![Scalar::Long, Scalar::Long]);
        let e = Element {
            names: &names,
            schema: &schema,
        };
        let rows = decode(b"[5000000000, -1]", "json", &e, usize::MAX, &no_tables).unwrap();
        assert_eq!(rows.values()[..2], [5_000_000_000u64 as u32, 1]);
        assert_eq!(rows.values()[2..], [u32::MAX, u32::MAX]);
        let schema = Schema::new(vec![Scalar::Float]);
        let e = Element {
            names: &names,
            schema: &schema,
        };
        let csv = decode(b"weight\n0.5\n0.25\n", "csv", &e, usize::MAX, &no_tables).unwrap();
        assert_eq!(f32::from_bits(csv.values()[1]), 0.25);
    }
}
