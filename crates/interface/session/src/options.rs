//! The flags every session-opening command shares, read in one place:
//! scene, library, inputs, params, backend, fusion, limits, allowances
//! and grants. Commands read their own flags beside these.

use std::path::{Path, PathBuf};

use orla_kernel::{Runtime, Value};

use crate::{Allowance, Limits, Session, parse_allow};

/// What a command line says of the session it opens.
#[derive(Clone, Debug)]
pub struct SessionOptions {
    /// The scene document.
    pub scene: Option<PathBuf>,
    /// A directory or file of Kernels the scene uses beside its own.
    pub library: Option<PathBuf>,
    /// Inputs bound to files, by name.
    pub inputs: Vec<(String, PathBuf)>,
    /// Graph parameters set, by name.
    pub params: Vec<(String, Value)>,
    /// The backend asked for; the command's own default otherwise.
    pub backend: Option<String>,
    /// Whether pointwise producers fuse into their consumers.
    pub fusion: bool,
    /// What the Host may hold and spend.
    pub limits: Limits,
    /// Directories the scene may read from and write into.
    pub allow: Vec<Allowance>,
    /// The capabilities a Kernel may be given (`--grant`); `None` grants
    /// every capability.
    pub grants: Option<orla_kernel::Capabilities>,
}

impl Default for SessionOptions {
    fn default() -> Self {
        Self {
            scene: None,
            library: None,
            inputs: Vec::new(),
            params: Vec::new(),
            backend: None,
            fusion: true,
            limits: Limits::default(),
            allow: Vec::new(),
            grants: None,
        }
    }
}

impl SessionOptions {
    /// Reads `arg` if it is one of these flags or the scene, taking a
    /// value from `next`; returns whether it was. Other flags are the
    /// caller's.
    pub fn read(
        &mut self,
        arg: &str,
        next: &mut dyn FnMut() -> Option<String>,
    ) -> Result<bool, String> {
        let mut value = |flag: &str| next().ok_or_else(|| format!("{flag} needs a value"));
        match arg {
            "--library" | "--kernels" => self.library = Some(PathBuf::from(value(arg)?)),
            "--backend" => self.backend = Some(value(arg)?),
            "--no-fusion" => self.fusion = false,
            "--allow" => self.allow.push(parse_allow(&value(arg)?)?),
            // `--grant field.read,field.write`: the only capabilities a
            // Kernel may be given; repeats accumulate.
            "--grant" => {
                let grants = self.grants.get_or_insert_with(Default::default);
                for name in value(arg)?.split(',').filter(|name| !name.is_empty()) {
                    let capability: orla_kernel::Capability =
                        name.trim().parse().map_err(|_| {
                            format!(
                                "--grant takes capabilities ({}), not `{name}`",
                                orla_kernel::Capability::ALL.map(|c| c.name()).join(", ")
                            )
                        })?;
                    grants.insert(capability);
                }
            }
            flag if Limits::takes(flag) => self.limits.read(flag, &value(flag)?)?,
            "--param" => {
                let binding = value(arg)?;
                let (name, text) = binding
                    .split_once('=')
                    .ok_or_else(|| "--param needs <name>=<literal>".to_string())?;
                let expr = orla_language::parse_literal(text).map_err(|e| e.to_string())?;
                let literal = orla_compiler::literal(&expr).map_err(|e| e.to_string())?;
                self.params.push((name.to_owned(), literal));
            }
            "--input" => {
                let binding = value(arg)?;
                let (name, path) = binding
                    .split_once('=')
                    .ok_or_else(|| "--input needs <name>=<file.ppm>".to_string())?;
                self.inputs.push((name.to_owned(), PathBuf::from(path)));
            }
            flag if flag.starts_with("--") => return Ok(false),
            scene => {
                if self.scene.replace(PathBuf::from(scene)).is_some() {
                    return Err("exactly one scene file is expected".into());
                }
            }
        }
        Ok(true)
    }

    /// The scene, which every command needs.
    pub fn scene(&self) -> Result<&Path, String> {
        self.scene
            .as_deref()
            .ok_or_else(|| "a scene file is required".to_string())
    }

    /// Opens the described session on the named backend, else `default`.
    pub fn open(&self, default: &str) -> Result<Session, String> {
        let backend = self.backend.as_deref().unwrap_or(default);
        self.open_over(Session::runtime_for(backend, self.limits)?)
    }

    /// Opens the described session over `runtime`, unsynced.
    pub fn open_over(&self, runtime: Runtime) -> Result<Session, String> {
        let mut runtime = runtime;
        if let Some(grants) = &self.grants {
            runtime.set_grants(grants.clone());
        }
        let mut session = Session::open_with(self.scene()?, self.library.as_deref(), runtime);
        for (name, path) in &self.inputs {
            session
                .bind_external(name, path)
                .map_err(|e| e.to_string())?;
        }
        for (name, value) in &self.params {
            session.set_param(name, value.clone())?;
        }
        session.set_fusion(self.fusion);
        for allowance in &self.allow {
            session.allow(allowance)?;
        }
        Ok(session)
    }
}
