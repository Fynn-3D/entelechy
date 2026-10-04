//! Data tables: CSV and JSON read as rows of the element a Kernel reads
//! them as. The caller gives the name of each number an element holds
//! (`pos.x`, `items.0`, `look.race`; one number alone is unnamed) and
//! the element's layout; a record gives each named number its value.
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
//! A row's `#name` names it to other tables: a whole-number cell written
//! `table:name` (`beasts:goblin`) is the place of the row so named in
//! the table of that name beside this one, counted from 0. A name no row
//! has, or two rows share, is refused.

use std::collections::BTreeMap;

use orla_host::{Domain, FieldData, Scalar, Schema};

/// The element a table's rows are read as: the name of each number it
/// holds, in layout order, and the layout.
pub struct Element<'a> {
    /// A name for each number, as [`orla_ir`]'s `number_names` gives them.
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
        return Err(orla_host::HostError::BudgetExceeded {
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
    let wanted = wanted_names(element.names);
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
        let mut word = 0;
        for name in element.names {
            let scalar = element.schema.scalars()[word];
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
                Some(Ok(place)) => place.to_string(),
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
            word += scalar.words();
        }
    }
    let count =
        u32::try_from(records.len()).map_err(|_| "the table has too many rows".to_owned())?;
    let domain = Domain::Rows(count);
    FieldData::from_values(element.schema.clone(), domain, &values).map_err(|e| e.to_string())
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
/// with nothing in them are left out.
fn csv_records(text: &str) -> Result<Vec<Record>, String> {
    let lines = csv_lines(text)?;
    let mut lines = lines
        .into_iter()
        .filter(|(_, cells)| !(cells.len() == 1 && cells[0].trim().is_empty()));
    let Some((_, headings)) = lines.next() else {
        return Err("the table has no heading line".to_owned());
    };
    let headings: Vec<String> = headings.iter().map(|h| h.trim().to_owned()).collect();
    let mut records = Vec::new();
    for (line, cells) in lines {
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

/// Each line's cells, with the line's number; a quoted cell may hold a
/// comma, a doubled quote and a line break.
fn csv_lines(text: &str) -> Result<Vec<(usize, Vec<String>)>, String> {
    let mut out = Vec::new();
    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut quoted = false;
    let mut line = 1;
    let mut began = 1;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
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
                out.push((began, std::mem::take(&mut cells)));
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
        out.push((began, cells));
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
                    return String::from_utf8(out).map_err(|_| self.error("a name is not UTF-8"));
                }
                b'\\' => {
                    let escaped = self.text.get(self.at).copied();
                    self.at += 1;
                    out.push(match escaped {
                        Some(b'n') => b'\n',
                        Some(b't') => b'\t',
                        Some(other) => other,
                        None => return Err(self.error("the table ends early")),
                    });
                }
                _ => out.push(b),
            }
        }
        Err(self.error("a quoted name is not closed"))
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
