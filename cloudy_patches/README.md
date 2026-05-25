# Cloudy Patches for TODDLERS

## New command: `save diffuse continuum unattenuated`

These patches add a new save command to Cloudy that outputs the cloud-integrated
**gas-phase nebular continuum** (free-bound, free-free, two-photon) with the **dust
extinction removed but the gas extinction retained**, and with grain thermal emission
excluded. It is the diffuse continuum analogue of Cloudy's *intrinsic* line luminosities:
the emission a downstream dust radiative-transfer code (e.g. SKIRT) needs as input, so the
code can apply dust attenuation itself without double-counting.

"Unattenuated" here means **un-attenuated by dust**, not optically thin to everything — see
*Physics* below for why the gas opacity must stay.

### Usage

```
save last diffuse continuum unattenuated ".diffContUnatt" units micron
```

### Output columns

1. Wavelength (in the specified units)
2. `DiffContUnatt`: gas-only nebular continuum, **dust extinction removed, gas extinction kept**
3. `DiffContAtt`: standard attenuated diffuse continuum (gas + dust extinction, includes grain
   thermal emission), for comparison

### Base version and compatibility

These patches are against Cloudy C22, commit `01c4cfc6` (master branch, updated 2025).

Verified to apply (`git apply`) **and** compile cleanly on **C22.00**, the 2025 master
branch, and **C25.00** — the continuum internals the patch touches
(`ConEmitOut`/`ConEmitLocal`, `opac.ExpZone`/`opac.opacity_abs`/`opac.tmn`,
`radius.dVolOutwrd`, `gv.dstab`/`gv.GrainEmission`, the `save diffuse continuum` parser)
are unchanged across that range, so `apply_patch.sh` works on all of them. A future release
that refactors those regions may need the change re-ported by hand (one parallel
`ConEmitOutUnatt` accumulator in `rt_continuum.cpp` plus the save handler); `apply_patch.sh`
then errors with that guidance rather than silently overwriting your source (override with
`FORCE_COPY=1`).

### Modified files (7)

| File | Change |
|------|--------|
| `rfield.h` | New `ConEmitOutUnatt` Spectrum array |
| `cont_createmesh.cpp` | Resize new array |
| `cont_setintensity.cpp` | Zero at initialization |
| `iter_startend.cpp` | Zero at iteration start (2 locations) |
| `rt_continuum.cpp` | Accumulate the gas-only outward diffuse continuum: propagate with the **gas-only** inter-zone transmission (dust term removed from `ExpZone`), keep the within-zone escape `opac.tmn`, and subtract `gv.GrainEmission` |
| `parse_save.cpp` | Parse `UNAT` keyword within the `DIFF` block |
| `save_do.cpp` | New `CT_OUTW_DIFF_UNATT` enum + `CODU` save handler |

### How to apply

Use `apply_patch.sh` (tries `git apply`, falls back through `--3way` and `patch -p1`, and
errors clearly if none apply):

```bash
./apply_patch.sh /path/to/cloudy        # then it builds (MAKE_JOBS=N to parallelize)
```

Or manually:

```bash
cd /path/to/cloudy
git apply /path/to/cloudy_unattenuated_diffuse_continuum.patch
cd source && make -j4
```

(The seven full source files are also included for a last-resort verbatim copy; `apply_patch.sh
FORCE_COPY=1` uses them. They are the C22/`01c4cfc6` versions, so only use that on a close base.)

### Physics

Cloudy builds the outward diffuse continuum zone by zone. The standard (emergent) array is

```
ConEmitOut[i] *= AttenuationDilutionFactor;                          // = opac.ExpZone[i] * DilutionHere
ConEmitOut[i] += ConEmitLocal[nzone][i] * dVolOutwrd * opac.tmn[i];  // local emission, within-zone escape
```

with two distinct opacity effects, both using `opac.opacity_abs` (**gas + dust**):

* `opac.ExpZone[i] = exp(-opacity_abs * dr/2)` — the **inter-zone** transmission applied to the
  accumulated continuum at every outward step (this is the dominant, cumulative attenuation).
* `opac.tmn[i] = (1 - e^-dτ)/dτ` — the **within-zone** mean escape factor for emission generated
  inside the current zone (a small, ~1–5 %, energy-conservation term).

`ConEmitOutUnatt` keeps the same structure but removes **only the dust extinction**, by rescaling
`ExpZone` by the gas fraction of the absorption opacity, and subtracts the grain thermal emission:

```
dustOpacAbs = gv.dstab[i] * hden;                                    // dust term in opacity_abs (opacity_addtotal.cpp)
ExpZoneGas  = ExpZone[i] ^ ((opacity_abs - dustOpacAbs)/opacity_abs); // = exp(-(opacity_abs - dust)*dr/2)
ConEmitOutUnatt[i] *= ExpZoneGas * DilutionHere;
gasEmit = ConEmitLocal[nzone][i] - GrainEmission[i];                 // gas-only source
ConEmitOutUnatt[i] += gasEmit * dVolOutwrd * opac.tmn[i];            // within-zone escape kept
```

**Why keep the gas extinction.** The gas continuous opacity is large below the Lyman limit
(H photoionization), where the ground-state recombination continuum is reabsorbed on-the-spot.
Removing *all* continuous extinction (gas included) resurrects that Lyman continuum as a huge,
unphysical bump (`DiffContUnatt/DiffContAtt` ~ 10⁶ below 912 Å) that would double-count ionizing
photons downstream. Keeping the gas opacity leaves the LyC correctly reabsorbed (ratio ~1 there),
while removing the dust opacity lets a dust-RT code (SKIRT) apply extinction itself. This is
consistent with Cloudy's intrinsic *lines*, which carry no **dust** foreground attenuation (and
have no transitions blueward of ~970 Å, so they never probe the LyC regime).

A quick sanity check on a dust-rich, optically-thick model: `DiffContUnatt/DiffContAtt` is ≈1
below the Lyman limit (gas reabsorption retained), >1 through the UV/optical (dust extinction
removed), →1 in the transparent NIR, and ≪1 at the 10–100 µm dust peak (grain thermal emission
excluded).
