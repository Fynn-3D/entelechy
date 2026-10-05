//! What a package is to the session that reads it: the parts of a
//! package directory and how a scene reaches one by name, each kept by
//! the code that reads it: the manifest and the Kernels by the sources,
//! the presets by the scene's lowering, the directories a name is looked
//! for in by the session's options. `help`'s `packages` section
//! answers this table; `tests/package.rs` holds a package to it.

/// One part of a package, as the session reads it.
pub struct Part {
    /// Its name.
    pub name: &'static str,
    /// How it is written or where it lies.
    pub shape: &'static str,
    /// What the session makes of it, one or two sentences.
    pub doc: &'static str,
}

/// The parts of a package directory, and how one is found by name.
pub const LAYOUT: [Part; 6] = [
    Part {
        name: "manifest",
        shape: "name.orlapkg/package.osn: package name@1 { language = 1; ir = 1; }",
        doc: "Its name and edition, and the least language and IR it needs: one this Orla does not have refuses the package, naming both. The directory is named for the package, `name.orlapkg`; a scene reaches one package of a name, and two, as two editions of one, are refused.",
    },
    Part {
        name: "kernels",
        shape: "use \"kernels/\";   (in the manifest)",
        doc: "The documents the manifest uses, inside the directory, are the package's Kernels; a scene that uses the package publishes every one of them. A document outside the package uses it whole, never a file inside it, so the manifest's versions hold.",
    },
    Part {
        name: "graphs",
        shape: "name.orlapkg/graphs/*.osn",
        doc: "Preset graphs, placed with `node n: graph \"name.orlapkg/graphs/g.osn\" { ... }`.",
    },
    Part {
        name: "docs",
        shape: "//! at the top of a manifest or a preset;   /// above a Kernel, a port or a param",
        doc: "What each part is for, in its source; `kernel` and `kernels` answer a published Kernel's.",
    },
    Part {
        name: "assets",
        shape: "name.orlapkg/font.png",
        doc: "Any other file in the directory, which a scene's `input` reads by its path or through the package's name (`name.orlapkg/font.png`), as a document is; a package's files are readable wherever reads are held.",
    },
    Part {
        name: "by name",
        shape: "use \"name.orlapkg\";",
        doc: "A package not beside the document is found by its name: beside the package the document is in, so packages kept together name one another; then in each directory of `ORLA_PACKAGES`; then in `packages/` beside the program.",
    },
];
