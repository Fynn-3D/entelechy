//! L7 meta-runtime primitives: speculative shadow verification (`trial`)
//! and structured world diagnostics (`diagnose`).
//!
//! `trial` tests candidate document changes inside an isolated in-memory
//! shadow session (`Kept` store) against automated mechanical gates
//! (compile, runtime, numerical finiteness, frame budget, zero mid-frame
//! downloads) before optionally committing them atomically to the live
//! session. Failed trials leave both disk and the live world untouched.

use std::collections::BTreeSet;
use std::time::Instant;

use serde_json::{Value as Json, json};

use entelechy_formats::base64::base64;
use entelechy_graph::Source;
use entelechy_session::Session;

use crate::create::{document_version, resolve_changes};
use crate::fields::{diff_fields, field_stats};
use crate::request::Request;
use crate::wire::{
    context_given, context_of, diagnostic_json, evaluation_error, events_json, failure, place_json,
    trace_json,
};

/// Maximum frames a single `trial` evaluates.
const MAX_TRIAL_STEPS: u64 = 60;

/// Tests document changes in an isolated shadow session against verification
/// gates, optionally committing them atomically when every gate passes.
pub(crate) fn trial(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let changes = resolve_changes(session, request, true)?;
    let steps = match request.get("steps") {
        None => 1u64,
        Some(val) => {
            let n = val
                .as_i64()
                .ok_or_else(|| failure("`steps` must be an integer between 1 and 60"))?;
            if !(1..=MAX_TRIAL_STEPS as i64).contains(&n) {
                return Err(failure(format!(
                    "`steps` must be between 1 and {MAX_TRIAL_STEPS}, got {n}"
                )));
            }
            n as u64
        }
    };
    let dt = request
        .get("dt")
        .and_then(Json::as_f64)
        .unwrap_or(1.0 / 60.0);
    if !dt.is_finite() || dt < 0.0 {
        return Err(failure("`dt` must be a non-negative finite number"));
    }
    let fresh_state = request
        .get("fresh_state")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let measure = request
        .get("measure")
        .and_then(Json::as_bool)
        .unwrap_or(true);
    let forbid_nonfinite = request
        .get("forbid_nonfinite")
        .and_then(Json::as_bool)
        .unwrap_or(true);
    let max_frame_ms = request.get("max_frame_ms").and_then(Json::as_f64);
    if let Some(budget) = max_frame_ms
        && (!budget.is_finite() || budget <= 0.0)
    {
        return Err(failure("`max_frame_ms` must be a positive finite number"));
    }
    let forbid_download = request
        .get("forbid_download")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let include_png = request
        .get("include_png")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let commit = request
        .get("commit")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let wanted_outputs: Option<BTreeSet<String>> = request
        .get("read_outputs")
        .and_then(Json::as_array)
        .map(|list| {
            list.iter()
                .filter_map(Json::as_str)
                .map(str::to_owned)
                .collect()
        });

    // In `trial`, `text` carries document source (like `compose`), never
    // keyboard input, so strip it before reading evaluation context fields.
    let ctx_json = match request.json().as_object() {
        Some(obj) if obj.contains_key("text") => {
            let mut stripped = obj.clone();
            stripped.remove("text");
            Json::Object(stripped)
        }
        _ => request.json().clone(),
    };
    let mut base_ctx = if context_given(&ctx_json).is_empty() {
        session.last_context()
    } else {
        context_of(session, &ctx_json)?
    };
    if (base_ctx.window[0] <= 0.0 || base_ctx.window[1] <= 0.0)
        && let Some((w, s)) = session.window()
    {
        base_ctx.window = w;
        base_ctx.scale = s;
    }

    let texts: Vec<(std::path::PathBuf, String)> = changes
        .iter()
        .map(|c| (c.path.clone(), c.text.clone()))
        .collect();

    // Gate 1: Compile & load in isolated in-memory shadow session.
    let opened = session
        .shadow_for(&texts, fresh_state, &base_ctx)
        .map_err(failure)?;
    let entelechy_session::ShadowOpen {
        session: mut shadow,
        state_warning,
    } = match opened {
        Ok(shadow) => shadow,
        Err(refusal) => {
            let mut fault = diagnostic_json(&refusal.diagnostic);
            fault["path"] = json!(refusal.path.display().to_string());
            if let Some(change) = changes.iter().find(|c| c.path == refusal.path) {
                fault["version"] = json!(change.version);
                if change.edited {
                    fault["text"] = json!(change.text);
                }
            }
            return Ok(json!({
                "passed": false,
                "committed": false,
                "written": false,
                "steps": 0,
                "gates": [{ "name": "compile", "passed": false, "reason": refusal.diagnostic.message }],
                "fault": fault,
                "trace": null,
                "outputs": [],
            }));
        }
    };

    let mut gates = vec![json!({ "name": "compile", "passed": true })];

    // Gate 2: Evaluate `steps` frames in shadow.
    let mut last_report = None;
    let mut last_wall_ms = 0.0f64;
    let mut last_ctx = base_ctx;
    for step in 0..steps {
        let mut ctx = base_ctx;
        ctx.time = base_ctx.time + (step as f64) * dt;
        last_ctx = ctx;
        let measure_this = measure && (step + 1 == steps);
        shadow.set_measured(measure_this);
        let started = Instant::now();
        let evaluated = shadow.evaluate(&ctx);
        let took_ms = started.elapsed().as_secs_f64() * 1000.0;
        if measure_this {
            shadow.set_measured(false);
        }
        match evaluated {
            Ok(report) => {
                last_wall_ms = took_ms;
                last_report = Some(report);
            }
            Err(error) => {
                let mut fault = evaluation_error(&error, Some(step));
                if let Some(place) = shadow.failure_place(&error, &ctx) {
                    place_json(&mut fault, &place);
                }
                let traced = shadow.failure_trace(&error, &ctx).map(|(node, t)| {
                    trace_json(&t.module, &node, &t.kernel, &t.trace, t.place.as_ref())
                });
                gates.push(json!({
                    "name": "runtime",
                    "passed": false,
                    "step": step,
                    "reason": shadow.told(&error),
                }));
                return Ok(json!({
                    "passed": false,
                    "committed": false,
                    "written": false,
                    "steps": step,
                    "gates": gates,
                    "state_warning": state_warning,
                    "fault": fault,
                    "trace": traced,
                    "outputs": [],
                }));
            }
        }
    }
    gates.push(json!({ "name": "runtime", "passed": true, "steps": steps }));

    let report = last_report.expect("steps >= 1");
    let device_ms = shadow.device_times().map(|d| d.busy.as_secs_f64() * 1000.0);
    let timings_sum_ms: f64 = report
        .timings
        .iter()
        .map(|(_, d)| d.as_secs_f64() * 1000.0)
        .sum();
    let fallback_ms = if timings_sum_ms > 0.0 {
        timings_sum_ms
    } else {
        last_wall_ms
    };
    let frame_ms = device_ms.unwrap_or(fallback_ms);
    let uploaded_bytes: u64 = report.placements.iter().map(|p| p.uploaded).sum();
    let downloaded_bytes: u64 = report.placements.iter().map(|p| p.downloaded).sum();

    // Gate 3: Numerical finiteness + output summaries & baseline diff.
    let output_names: Vec<(String, Source)> = shadow
        .graph()
        .map(|g| g.outputs.clone())
        .unwrap_or_default();
    let mut total_nonfinite = 0u64;
    let mut bad_outputs = Vec::new();
    let mut first_nonfinite_trace: Option<Json> = None;
    let mut output_reports = Vec::new();

    for (name, source) in &output_names {
        let Ok(handle) = shadow.output(Some(name)) else {
            continue;
        };
        let stats = field_stats(&mut shadow, handle)?;
        if stats.nonfinite > 0 {
            total_nonfinite += stats.nonfinite;
            bad_outputs.push(json!({
                "output": name,
                "nonfinite": stats.nonfinite,
                "nan": stats.nan,
                "inf": stats.inf,
                "first_nonfinite": stats.first_nonfinite,
            }));
            if first_nonfinite_trace.is_none()
                && let (Source::Node { node, .. }, Some(at)) = (source, stats.first_nonfinite_at)
                && let Ok(t) = shadow.trace(node, at, &last_ctx)
            {
                first_nonfinite_trace = Some(trace_json(
                    &t.module,
                    &node.to_string(),
                    &t.kernel,
                    &t.trace,
                    t.place.as_ref(),
                ));
            }
        }
        let include_detail = wanted_outputs.as_ref().is_none_or(|set| set.contains(name));
        if include_detail {
            let diff = match (session.output(Some(name)), shadow.field(handle)) {
                (Ok(base_h), Ok(trial_data)) => {
                    let trial_clone = trial_data.clone();
                    session
                        .field(base_h)
                        .ok()
                        .and_then(|base_data| diff_fields(base_data, &trial_clone))
                }
                _ => None,
            };
            let mut entry = json!({
                "name": name,
                "nonfinite": stats.nonfinite,
                "nan": stats.nan,
                "inf": stats.inf,
                "summary": stats.summary,
                "diff": diff,
            });
            if include_png
                && shadow.is_picture(handle)
                && let Ok(image) = shadow.picture(handle)
            {
                let bytes =
                    entelechy_formats::png::encode(image.width(), image.height(), image.pixels());
                entry["width"] = json!(image.width());
                entry["height"] = json!(image.height());
                entry["png"] = json!(base64(&bytes));
            }
            output_reports.push(entry);
        }
    }

    if forbid_nonfinite {
        let mut gate = json!({
            "name": "finite",
            "passed": total_nonfinite == 0,
            "nonfinite": total_nonfinite,
        });
        if !bad_outputs.is_empty() {
            gate["bad_outputs"] = Json::Array(bad_outputs);
        }
        gates.push(gate);
    }

    // Gate 4a: Frame time budget gate.
    if let Some(limit_ms) = max_frame_ms {
        gates.push(json!({
            "name": "budget",
            "passed": frame_ms <= limit_ms,
            "frame_ms": frame_ms,
            "max_frame_ms": limit_ms,
        }));
    }

    // Gate 4b: Zero mid-frame GPU->CPU download stall gate.
    if forbid_download {
        let stalls: Vec<Json> = report
            .placements
            .iter()
            .filter(|p| p.downloaded > 0)
            .map(|p| json!({ "node": p.node.to_string(), "bytes": p.downloaded }))
            .collect();
        gates.push(json!({
            "name": "no_download",
            "passed": downloaded_bytes == 0,
            "downloaded_bytes": downloaded_bytes,
            "stalls": stalls,
        }));
    }

    let passed = gates
        .iter()
        .all(|g| g.get("passed").and_then(Json::as_bool) == Some(true));

    let timings: Vec<Json> = report
        .timings
        .iter()
        .map(|(node, d)| json!({ "node": node.to_string(), "millis": d.as_secs_f64() * 1000.0 }))
        .collect();
    let placements: Vec<Json> = report
        .placements
        .iter()
        .map(|p| {
            json!({
                "node": p.node.to_string(),
                "backend": p.backend,
                "routed": p.routed.as_deref(),
                "uploaded": p.uploaded,
                "downloaded": p.downloaded,
            })
        })
        .collect();

    // Release the shadow session before touching the live session.
    drop(shadow);

    let mut answer = json!({
        "passed": passed,
        "committed": false,
        "written": false,
        "steps": steps,
        "gates": gates,
        "frame_ms": frame_ms,
        "wall_ms": last_wall_ms,
        "device_ms": device_ms,
        "uploaded_bytes": uploaded_bytes,
        "downloaded_bytes": downloaded_bytes,
        "timings": timings,
        "placements": placements,
        "outputs": output_reports,
        "state_warning": state_warning,
        "trace": first_nonfinite_trace,
    });

    // Atomic verify-then-promote: only write & swap when every gate passed.
    if commit && passed && !texts.is_empty() {
        match session
            .write_documents(&texts)
            .map_err(|e| failure(e.to_string()))?
        {
            Ok(()) => {
                session.claim_scene();
                let events = session.sync_settled();
                let scene_ver = session
                    .scene_text()
                    .map(|t| document_version(&t))
                    .unwrap_or_default();
                answer["committed"] = json!(true);
                answer["written"] = json!(true);
                answer["version"] = json!(scene_ver);
                answer["events"] = json!(events_json(&events));
                answer["documents"] = json!(
                    changes
                        .iter()
                        .map(|c| json!({
                            "path": c.path.display().to_string(),
                            "version": document_version(&c.text),
                        }))
                        .collect::<Vec<_>>()
                );
            }
            Err(refusal) => {
                answer["passed"] = json!(false);
                answer["fault"] = diagnostic_json(&refusal.diagnostic);
            }
        }
    }

    Ok(answer)
}

/// Structured live world health, hotspot, transfer, fusion, and memory report.
pub(crate) fn diagnose(session: &mut Session, request: &Request) -> Result<Json, Json> {
    let ask_measure = request
        .get("measure")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let check_finite = request
        .get("check_finite")
        .and_then(Json::as_bool)
        .unwrap_or(true);
    let top = match request.get("top") {
        None => 10usize,
        Some(val) => {
            let n = val
                .as_i64()
                .ok_or_else(|| failure("`top` must be an integer between 1 and 100"))?;
            if !(1..=100).contains(&n) {
                return Err(failure(format!("`top` must be between 1 and 100, got {n}")));
            }
            n as usize
        }
    };

    let loaded = session.graph().is_some();
    if !loaded {
        return Ok(json!({
            "healthy": false,
            "backend": session.runtime().backend_name(),
            "loaded": false,
            "refused": session.why_no_scene(),
            "revision": session.position().0,
            "frame": session.position().1,
            "failing": session.failing(),
            "hotspots": [],
            "transfers": { "uploaded_bytes": 0, "downloaded_bytes": 0, "stalls": [] },
            "fusions": { "fused_groups": 0, "refused": [] },
            "outputs": [],
            "recommendations": [{
                "severity": "error",
                "kind": "no_scene",
                "message": session.why_no_scene().unwrap_or_else(|| "no scene is loaded".into()),
            }],
        }));
    }

    let mut eval_error: Option<Json> = None;
    if ask_measure || session.last_report().is_none() {
        let mut ctx = if context_given(request.json()).is_empty() {
            session.last_context()
        } else {
            context_of(session, request.json())?
        };
        if (ctx.window[0] <= 0.0 || ctx.window[1] <= 0.0)
            && let Some((w, s)) = session.window()
        {
            ctx.window = w;
            ctx.scale = s;
        }
        session.set_measured(true);
        let res = session.evaluate(&ctx);
        session.set_measured(false);
        if let Err(err) = res {
            let mut fault = evaluation_error(&err, None);
            if let Some(place) = session.failure_place(&err, &ctx) {
                place_json(&mut fault, &place);
            }
            eval_error = Some(fault);
        }
    }

    let report = session.last_report().cloned();
    let device_times = session.device_times();
    let device_ms = device_times.as_ref().map(|d| d.busy.as_secs_f64() * 1000.0);

    let mut recommendations = Vec::new();
    if let Some(why) = session.failing() {
        recommendations.push(json!({
            "severity": "error",
            "kind": "failing",
            "message": why,
        }));
    }

    // Build hotspots, transfers, and fusions from the latest Report.
    let mut hotspots = Vec::new();
    let mut uploaded_bytes = 0u64;
    let mut downloaded_bytes = 0u64;
    let mut stalls = Vec::new();
    let mut fused_groups = 0usize;
    let mut refused_fusions = Vec::new();
    let mut frame_ms = 0.0f64;

    if let Some(rep) = &report {
        fused_groups = rep.fused.len();
        for rf in &rep.refused {
            refused_fusions.push(json!({
                "node": rf.node.to_string(),
                "members": rf.members.iter().map(ToString::to_string).collect::<Vec<_>>(),
                "reason": rf.reason,
            }));
        }

        struct HotspotRow {
            name: String,
            ms: f64,
            backend: Option<&'static str>,
            routed: Option<String>,
            up: u64,
            down: u64,
        }
        let mut rows: Vec<HotspotRow> = rep
            .timings
            .iter()
            .map(|(node, dur)| {
                let name = node.to_string();
                let ms = dur.as_secs_f64() * 1000.0;
                let placed = rep.placements.iter().find(|p| p.node == *node);
                HotspotRow {
                    name,
                    ms,
                    backend: placed.map(|p| p.backend),
                    routed: placed.and_then(|p| p.routed.as_deref().map(str::to_owned)),
                    up: placed.map_or(0, |p| p.uploaded),
                    down: placed.map_or(0, |p| p.downloaded),
                }
            })
            .collect();
        frame_ms = rows.iter().map(|r| r.ms).sum();
        let denom = device_ms.unwrap_or(frame_ms).max(1e-6);
        rows.sort_by(|a, b| b.ms.total_cmp(&a.ms));

        for row in rows.into_iter().take(top) {
            hotspots.push(json!({
                "node": row.name,
                "millis": row.ms,
                "share": row.ms / denom,
                "backend": row.backend,
                "routed": row.routed,
                "uploaded": row.up,
                "downloaded": row.down,
            }));
        }

        for p in &rep.placements {
            uploaded_bytes += p.uploaded;
            downloaded_bytes += p.downloaded;
            if p.downloaded > 0 {
                stalls.push(json!({
                    "node": p.node.to_string(),
                    "bytes": p.downloaded,
                    "backend": p.backend,
                }));
                recommendations.push(json!({
                    "severity": "warning",
                    "kind": "download_stall",
                    "node": p.node.to_string(),
                    "bytes": p.downloaded,
                    "message": format!(
                        "node `{}` downloaded {} bytes from device mid-frame; keep its small producer on the ordered/host backend (`scan` or host-resident inputs) to avoid GPU stalls",
                        p.node, p.downloaded
                    ),
                }));
            } else if p.uploaded >= (1 << 20) {
                recommendations.push(json!({
                    "severity": "info",
                    "kind": "large_upload",
                    "node": p.node.to_string(),
                    "bytes": p.uploaded,
                    "message": format!(
                        "node `{}` uploaded {} bytes from host to device in one frame",
                        p.node, p.uploaded
                    ),
                }));
            }
        }
    }

    // Check numerical finiteness across outputs.
    let output_names: Vec<String> = session
        .graph()
        .map(|g| g.outputs.iter().map(|(n, _)| n.clone()).collect())
        .unwrap_or_default();
    let mut output_health = Vec::new();
    let mut total_nonfinite = 0u64;
    if check_finite {
        for name in &output_names {
            if let Ok(handle) = session.output(Some(name))
                && let Ok(stats) = field_stats(session, handle)
            {
                total_nonfinite += stats.nonfinite;
                if stats.nonfinite > 0 {
                    recommendations.push(json!({
                        "severity": "error",
                        "kind": "nonfinite_output",
                        "output": name,
                        "nonfinite": stats.nonfinite,
                        "first_nonfinite": stats.first_nonfinite,
                        "message": format!(
                            "output `{name}` has {} non-finite value(s) ({} NaN, {} Inf)",
                            stats.nonfinite, stats.nan, stats.inf
                        ),
                    }));
                }
                output_health.push(json!({
                    "name": name,
                    "nonfinite": stats.nonfinite,
                    "nan": stats.nan,
                    "inf": stats.inf,
                    "first_nonfinite": stats.first_nonfinite,
                }));
            }
        }
    }

    // Memory health.
    let used = session.runtime().host().used_words() as u64 * 4;
    let budget = session.runtime().host().budget_words() as u64 * 4;
    let usage_fraction = if budget > 0 {
        (used as f64) / (budget as f64)
    } else {
        0.0
    };
    let map = session.memory_map();
    let cold: Vec<String> = session
        .cold_nodes()
        .iter()
        .map(|n| n.0.to_string())
        .collect();
    if usage_fraction > 0.80 {
        recommendations.push(json!({
            "severity": "warning",
            "kind": "memory_pressure",
            "usage_fraction": usage_fraction,
            "message": format!(
                "Host memory is at {:.0}% of budget ({used} / {budget} bytes)",
                usage_fraction * 100.0
            ),
        }));
    }
    if !cold.is_empty() {
        recommendations.push(json!({
            "severity": "info",
            "kind": "cold_eviction",
            "nodes": cold,
            "message": format!("{} node output(s) were evicted cold to stay within memory budget", cold.len()),
        }));
    }

    let healthy = session.failing().is_none()
        && eval_error.is_none()
        && total_nonfinite == 0
        && downloaded_bytes == 0;

    Ok(json!({
        "healthy": healthy,
        "backend": session.runtime().backend_name(),
        "loaded": true,
        "revision": session.position().0,
        "frame": session.position().1,
        "failing": session.failing(),
        "fault": eval_error,
        "frame_ms": device_ms.unwrap_or(frame_ms),
        "wall_nodes_ms": frame_ms,
        "device_ms": device_ms,
        "hotspots": hotspots,
        "transfers": {
            "uploaded_bytes": uploaded_bytes,
            "downloaded_bytes": downloaded_bytes,
            "stalls": stalls,
        },
        "fusions": {
            "fused_groups": fused_groups,
            "refused": refused_fusions,
        },
        "outputs": output_health,
        "memory": {
            "used": used,
            "budget": budget,
            "usage_fraction": usage_fraction,
            "host": map.census.host,
            "device": map.census.device,
            "kept": map.census.kept,
            "cold": cold,
            "top": map.held.iter().take(top).map(|e| json!({
                "owner": e.owner,
                "kind": e.kind,
                "host": e.bytes.host,
                "device": e.bytes.device,
            })).collect::<Vec<_>>(),
        },
        "recommendations": recommendations,
    }))
}
