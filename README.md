# Spot

Spot finds **isolated misplaced vertices** in a triangle mesh and nudges them
back to where the surface says they should be. It targets one specific case:
*"I sculpted or generated something and one vertex got shoved out of place by
accident."* It works on any mesh regardless of origin (Meshy/Tripo export, hand
sculpt, scan), because it only ever looks at geometry, never at where the mesh
came from.

> Named after Keenan Crane's cow, the mascot mesh of discrete differential
> geometry, whose course notes teach the cotangent Laplacian this tool
> implements. The name is also the job description: Spot finds the bad vertex.
> (Spot finds; [Suzanne](#the-report-is-the-api), the judgment layer, rules.)

## What it is not

Hard scope boundary: Spot does **not**:

- do global retopology or quad remeshing (ZRemesher / Instant Meshes own that);
- reason about intent (is this sharp corner intentional or a mistake?), which
  needs a learned prior and is a research problem;
- repair holes, non-manifold edges or self-intersections (basic validation only).

A flagged spike is *assumed* to be a defect. Spot surfaces candidates; it does
not claim to know intent.

## How it works

A mesh is a graph, and a carelessly misplaced vertex is a **local high-frequency
spike** on it. The pipeline:

1. **Load & validate.** `trimesh` for IO; require a manifold triangle mesh and
   fail loudly otherwise.
2. **Three defect signals per vertex**, each made dimensionless by the local edge
   length so a single threshold works across meshes:
   - **mean curvature** `|H|` via the hand-built cotangent Laplacian
     (`Hvec = M⁻¹ L V`);
   - **Laplacian displacement** `|vᵢ − avg(neighbors)|`, the most direct
     "it's offset" cue;
   - **normal coherence**, the angle between a vertex's normal and its
     neighborhood's.
3. **Flag as _local_ outliers, never global.** Curvature is legitimately high in
   curved regions, so a global threshold would flag every real feature. Instead
   each vertex gets a robust z-score (median + MAD) **within its k-ring** (k=2);
   only local outliers flag. Boundary vertices are excluded, and non-maximum
   suppression keeps only the peak of an isolated spike.
4. **Fix locally.** Flagged vertices are pulled back with **Taubin smoothing**
   (the λ|μ scheme, so the mesh doesn't shrink), capped so we nudge rather than
   melt. Every non-flagged vertex stays **byte-for-byte unchanged**.
5. **Report & view.** Emit a versioned JSON report; optionally open a polyscope
   before/after view with flagged vertices highlighted.

The cotangent Laplacian and mass matrix are built by hand from the triangle
geometry (`spot/operators.py`) rather than pulled from a library. That is the
part worth understanding, and it sidesteps libigl's painful Windows build.

## Install

Python 3.11+. From a fresh virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install trimesh numpy scipy polyscope pytest
```

## Usage

```powershell
# Inspect a mesh (counts + polyscope viewer)
python -m spot view model.obj

# Detect misplaced vertices, correct them, write the fixed mesh + a JSON report
python -m spot fix model.obj --out fixed.obj --report report.json

# ...and open a before/after view of the correction
python -m spot fix model.obj --out fixed.obj --view

# Measure V1's false positives on genuine features (see "The hard edge")
python -m spot bench --out-dir benchmark/
```

`fix` knobs: `--k` (k-ring radius, default 2), `--threshold` (local z-score,
default 3.5), `--iterations` (Taubin pairs, default 20), `--max-displacement`
(cap in local-neighborhood units, default 3.0).

Output format follows the file extension. **OBJ / GLB / PLY** preserve shared
vertices; **STL** is a triangle soup and will not (a format limitation, not a
bug).

## The report is the API

The JSON report is Spot's public contract, the input for **Suzanne**, the VLM
judgment layer built on top of Spot. Its schema is versioned (`schema_version`,
semver); consumers should read it and refuse an unexpected major version. Each
flagged vertex carries its id, original and corrected positions, applied
displacement, combined `local_z`, and the three per-signal values and z-scores.
See `spot/report.py` for the full schema.

## The hard edge (measured, not solved)

Spot's `bench` command measures how often V1 false-positives on **legitimate**
features, the doorway to the research version, kept shut on purpose. On clean
meshes (every flag is therefore a false positive):

| mesh | kind | false-positive rate |
|------|------|--------------------:|
| cube | feature | 16.3% |
| cylinder | feature | 0.0% |
| cone | feature | 4.0% |
| sphere | smooth | 0.0% |
| torus | smooth | 0.0% |

Two findings fall out:

- **Uniform ridges** (a cylinder rim, most of a cube's edges) are *not* flagged:
  a whole ridge is "normal for around here." Only **isolated** sharp features
  (cube corners, a cone apex) are.
- Those isolated features score `local_z ≈ 40–78`, **the same range as a real
  injected defect (`≈ 76`)**. No threshold separates them. They are
  geometrically indistinguishable from a defect without a notion of *intent*.

That is precisely why the next step needs a learned prior (Suzanne), not a
parameter. The flagged-feature set becomes Suzanne's "feature" benchmark class
for free.

## Testing

Ground truth is free here: take a clean mesh, shove one vertex by a known
offset, and require that the detector finds it and the fix returns it. That
synthetic-defect harness (`spot/synthetic.py`) backs the whole suite.

```powershell
python -m pytest
```

## License

Apache 2.0. See [LICENSE](LICENSE).
