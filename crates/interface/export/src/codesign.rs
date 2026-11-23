//! Pure-Rust Mach-O bundle signature sealing (`CodeResources` + `CS_CodeDirectory`
//! special slots `CSSLOT_INFOSLOT` and `CSSLOT_RESOURCEDIR`).
//!
//! When `entelechy-export` packs a `.app` from a pre-signed Mach-O
//! `entelechy-runner` binary (on Linux, Windows, or inside WASI `exporter.wasm`
//! where `/usr/bin/codesign` is unavailable), the executable pages in
//! `Contents/MacOS/` never change — only `Contents/Info.plist` and
//! `Contents/Resources/app.entapp` vary per exported scene. Writing
//! `Contents/_CodeSignature/CodeResources` and updating special slots `-1` and
//! `-3` inside every `CS_CodeDirectory` seals a strict-valid ad-hoc `.app` bundle
//! without needing macOS host tools.

use entelechy_formats::base64::base64;

use crate::sha::{sha1, sha256};

const MH_MAGIC_64_LE: [u8; 4] = [0xCF, 0xFA, 0xED, 0xFE];
const LC_CODE_SIGNATURE: u32 = 0x1D;
const CSMAGIC_EMBEDDED_SIGNATURE: u32 = 0xFADE_0CC0;
const CSMAGIC_CODEDIRECTORY: u32 = 0xFADE_0C02;
const CS_HASHTYPE_SHA1: u8 = 1;
const CS_HASHTYPE_SHA256: u8 = 2;

fn read_u32_le(bytes: &[u8], at: usize) -> Option<u32> {
    let s = bytes.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}

fn read_u32_be(bytes: &[u8], at: usize) -> Option<u32> {
    let s = bytes.get(at..at.checked_add(4)?)?;
    Some(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

fn locate_superblob(macho: &[u8]) -> Option<(usize, usize)> {
    if macho.get(..4)? != MH_MAGIC_64_LE {
        return None;
    }
    let ncmds = read_u32_le(macho, 16)? as usize;
    let mut off = 32usize;
    for _ in 0..ncmds {
        let cmd = read_u32_le(macho, off)?;
        let cmdsize = read_u32_le(macho, off.checked_add(4)?)? as usize;
        if cmdsize < 8 || off.checked_add(cmdsize)? > macho.len() {
            return None;
        }
        if cmd == LC_CODE_SIGNATURE && cmdsize >= 16 {
            let dataoff = read_u32_le(macho, off + 8)? as usize;
            let datasize = read_u32_le(macho, off + 12)? as usize;
            if dataoff.checked_add(datasize)? <= macho.len() && datasize >= 12 {
                return Some((dataoff, datasize));
            }
            return None;
        }
        off += cmdsize;
    }
    None
}

/// If `macho` carries an embedded `CS_CodeDirectory` with bundle special slots
/// (`nSpecialSlots >= 3` covering `Info.plist` and `CodeResources`), returns its
/// signing identifier (e.g. `"run.entelechy.app"`).
pub fn bundle_signing_id(macho: &[u8]) -> Option<String> {
    let (sb_off, sb_len) = locate_superblob(macho)?;
    let sb = &macho[sb_off..sb_off + sb_len];
    if read_u32_be(sb, 0)? != CSMAGIC_EMBEDDED_SIGNATURE {
        return None;
    }
    let count = read_u32_be(sb, 8)? as usize;
    for i in 0..count {
        let idx_at = 12usize.checked_add(i.checked_mul(8)?)?;
        let rel_off = read_u32_be(sb, idx_at + 4)? as usize;
        if rel_off.checked_add(40)? > sb.len() {
            continue;
        }
        let cd = &sb[rel_off..];
        if read_u32_be(cd, 0)? != CSMAGIC_CODEDIRECTORY {
            continue;
        }
        let n_special = read_u32_be(cd, 24)?;
        if n_special < 3 {
            continue;
        }
        let ident_off = read_u32_be(cd, 20)? as usize;
        let tail = cd.get(ident_off..)?;
        let nul = tail.iter().position(|&b| b == 0)?;
        let id = std::str::from_utf8(&tail[..nul]).ok()?;
        if !id.is_empty() {
            return Some(id.to_owned());
        }
    }
    None
}

/// Generates `Contents/_CodeSignature/CodeResources` XML sealing
/// `Resources/app.entapp` with both SHA-1 (`files`) and SHA-256 (`files2`).
pub fn code_resources_xml(entapp_bytes: &[u8]) -> Vec<u8> {
    let h1 = base64(&sha1(entapp_bytes));
    let h2 = base64(&sha256(entapp_bytes));
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n\
<plist version=\"1.0\">\n\
<dict>\n\
\t<key>files</key>\n\
\t<dict>\n\
\t\t<key>Resources/app.entapp</key>\n\
\t\t<data>\n\
\t\t{h1}\n\
\t\t</data>\n\
\t</dict>\n\
\t<key>files2</key>\n\
\t<dict>\n\
\t\t<key>Resources/app.entapp</key>\n\
\t\t<dict>\n\
\t\t\t<key>hash2</key>\n\
\t\t\t<data>\n\
\t\t\t{h2}\n\
\t\t\t</data>\n\
\t\t</dict>\n\
\t</dict>\n\
\t<key>rules</key>\n\
\t<dict>\n\
\t\t<key>^Resources/</key>\n\
\t\t<true/>\n\
\t\t<key>^Resources/.*\\.lproj/</key>\n\
\t\t<dict>\n\
\t\t\t<key>optional</key>\n\
\t\t\t<true/>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>1000</real>\n\
\t\t</dict>\n\
\t\t<key>^Resources/.*\\.lproj/locversion.plist$</key>\n\
\t\t<dict>\n\
\t\t\t<key>omit</key>\n\
\t\t\t<true/>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>1100</real>\n\
\t\t</dict>\n\
\t\t<key>^Resources/Base\\.lproj/</key>\n\
\t\t<dict>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>1010</real>\n\
\t\t</dict>\n\
\t\t<key>^version.plist$</key>\n\
\t\t<true/>\n\
\t</dict>\n\
\t<key>rules2</key>\n\
\t<dict>\n\
\t\t<key>.*\\.dSYM($|/)</key>\n\
\t\t<dict>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>11</real>\n\
\t\t</dict>\n\
\t\t<key>^(.*/)?\\.DS_Store$</key>\n\
\t\t<dict>\n\
\t\t\t<key>omit</key>\n\
\t\t\t<true/>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>2000</real>\n\
\t\t</dict>\n\
\t\t<key>^(Frameworks|SharedFrameworks|PlugIns|Plug-ins|XPCServices|Helpers|MacOS|Library/(Automator|Spotlight|LoginItems))/</key>\n\
\t\t<dict>\n\
\t\t\t<key>nested</key>\n\
\t\t\t<true/>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>10</real>\n\
\t\t</dict>\n\
\t\t<key>^.*</key>\n\
\t\t<true/>\n\
\t\t<key>^Info\\.plist$</key>\n\
\t\t<dict>\n\
\t\t\t<key>omit</key>\n\
\t\t\t<true/>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>20</real>\n\
\t\t</dict>\n\
\t\t<key>^PkgInfo$</key>\n\
\t\t<dict>\n\
\t\t\t<key>omit</key>\n\
\t\t\t<true/>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>20</real>\n\
\t\t</dict>\n\
\t\t<key>^Resources/</key>\n\
\t\t<dict>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>20</real>\n\
\t\t</dict>\n\
\t\t<key>^Resources/.*\\.lproj/</key>\n\
\t\t<dict>\n\
\t\t\t<key>optional</key>\n\
\t\t\t<true/>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>1000</real>\n\
\t\t</dict>\n\
\t\t<key>^Resources/.*\\.lproj/locversion.plist$</key>\n\
\t\t<dict>\n\
\t\t\t<key>omit</key>\n\
\t\t\t<true/>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>1100</real>\n\
\t\t</dict>\n\
\t\t<key>^Resources/Base\\.lproj/</key>\n\
\t\t<dict>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>1010</real>\n\
\t\t</dict>\n\
\t\t<key>^[^/]+$</key>\n\
\t\t<dict>\n\
\t\t\t<key>nested</key>\n\
\t\t\t<true/>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>10</real>\n\
\t\t</dict>\n\
\t\t<key>^embedded\\.provisionprofile$</key>\n\
\t\t<dict>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>20</real>\n\
\t\t</dict>\n\
\t\t<key>^version\\.plist$</key>\n\
\t\t<dict>\n\
\t\t\t<key>weight</key>\n\
\t\t\t<real>20</real>\n\
\t\t</dict>\n\
\t</dict>\n\
</dict>\n\
</plist>\n"
    )
    .into_bytes()
}

/// Updates special slots `-1` (`CSSLOT_INFOSLOT` = `Info.plist`) and `-3`
/// (`CSSLOT_RESOURCEDIR` = `_CodeSignature/CodeResources`) in every embedded
/// `CS_CodeDirectory` inside `macho`. Returns `true` when at least one directory
/// was sealed.
pub fn seal_macho_bundle(macho: &mut [u8], plist: &[u8], code_resources: &[u8]) -> bool {
    let Some((sb_off, sb_len)) = locate_superblob(macho) else {
        return false;
    };
    let Some(magic) = read_u32_be(macho, sb_off) else {
        return false;
    };
    if magic != CSMAGIC_EMBEDDED_SIGNATURE {
        return false;
    }
    let Some(count) = read_u32_be(macho, sb_off + 8) else {
        return false;
    };
    let plist_sha1 = sha1(plist);
    let plist_sha256 = sha256(plist);
    let cr_sha1 = sha1(code_resources);
    let cr_sha256 = sha256(code_resources);

    let mut sealed = false;
    for i in 0..(count as usize) {
        let Some(idx_at) = sb_off.checked_add(12 + i * 8) else {
            break;
        };
        let Some(rel_off) = read_u32_be(macho, idx_at + 4) else {
            continue;
        };
        let rel_off = rel_off as usize;
        if rel_off.saturating_add(40) > sb_len {
            continue;
        }
        let cd_at = sb_off + rel_off;
        if read_u32_be(macho, cd_at) != Some(CSMAGIC_CODEDIRECTORY) {
            continue;
        }
        let Some(cd_len) = read_u32_be(macho, cd_at + 4) else {
            continue;
        };
        if rel_off.saturating_add(cd_len as usize) > sb_len {
            continue;
        }
        let Some(hash_off) = read_u32_be(macho, cd_at + 16) else {
            continue;
        };
        let Some(n_special) = read_u32_be(macho, cd_at + 24) else {
            continue;
        };
        if n_special < 3 {
            continue;
        }
        let hash_size = macho[cd_at + 36] as usize;
        let hash_type = macho[cd_at + 37];
        let hash_off = hash_off as usize;
        let Some(slot3_rel) = hash_off.checked_sub(3 * hash_size) else {
            continue;
        };
        let slot1_rel = hash_off - hash_size;
        if slot1_rel + hash_size > cd_len as usize {
            continue;
        }
        match (hash_type, hash_size) {
            (CS_HASHTYPE_SHA1, 20) => {
                macho[cd_at + slot1_rel..cd_at + slot1_rel + 20].copy_from_slice(&plist_sha1);
                macho[cd_at + slot3_rel..cd_at + slot3_rel + 20].copy_from_slice(&cr_sha1);
                sealed = true;
            }
            (CS_HASHTYPE_SHA256, 32) => {
                macho[cd_at + slot1_rel..cd_at + slot1_rel + 32].copy_from_slice(&plist_sha256);
                macho[cd_at + slot3_rel..cd_at + slot3_rel + 32].copy_from_slice(&cr_sha256);
                sealed = true;
            }
            _ => {}
        }
    }
    sealed
}
