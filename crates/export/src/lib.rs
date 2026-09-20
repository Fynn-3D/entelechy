//! The exporter's writers: a mesh as GLB, a sound as WAV, a picture as
//! PNG. Every writer starts from what the Host wrote, so what a Kernel
//! made can leave for other software; a lossy copy of a sound is a job
//! for ffmpeg on the WAV.

pub mod gltf;
pub mod sound;
