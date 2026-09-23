//! The exporter's writer that needs more than `orla-formats`: a mesh
//! and a rig as GLB, from what the Host wrote, so what a Kernel made can
//! leave for other software. A sound, a score and a picture leave by
//! `orla_formats::encode`; a lossy copy of a sound is a job for ffmpeg
//! on the WAV.

pub mod gltf;
