//! A request as its operation reads it, with one error form for a
//! missing or mistyped field.

use orla_kernel::KernelId;
use serde_json::Value as Json;

use crate::wire::failure;

/// One request, by its operation.
pub(crate) struct Request<'a> {
    op: &'a str,
    json: &'a Json,
}

impl<'a> Request<'a> {
    /// The request, or an error when it names no operation.
    pub(crate) fn new(json: &'a Json) -> Result<Self, Json> {
        let op = json
            .get("op")
            .and_then(Json::as_str)
            .ok_or_else(|| failure("request needs an `op`"))?;
        Ok(Self { op, json })
    }

    /// The operation it asks for.
    pub(crate) fn op(&self) -> &'a str {
        self.op
    }

    /// The request as it came.
    pub(crate) fn json(&self) -> &'a Json {
        self.json
    }

    /// A field, when it is there.
    pub(crate) fn get(&self, name: &str) -> Option<&'a Json> {
        self.json.get(name)
    }

    /// A field the operation needs.
    pub(crate) fn field(&self, name: &str) -> Result<&'a Json, Json> {
        self.json
            .get(name)
            .ok_or_else(|| failure(format!("`{}` needs `{name}`", self.op)))
    }

    /// A field the operation needs, which is a string.
    pub(crate) fn text(&self, name: &str) -> Result<String, Json> {
        self.field(name)?
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| failure(format!("`{name}` must be a string")))
    }

    /// The Kernel it names, in `kernel`, else in `id` (which is also the
    /// request's own id).
    pub(crate) fn kernel(&self) -> Result<KernelId, Json> {
        let named = self
            .json
            .get("kernel")
            .or_else(|| self.json.get("id"))
            .and_then(Json::as_str)
            .ok_or_else(|| failure("`kernel` names the Kernel, for example `image.blur@1`"))?;
        named
            .parse::<KernelId>()
            .map_err(|e| failure(e.to_string()))
    }
}
