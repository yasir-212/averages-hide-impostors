# Data

No data files are included in this repository. All three datasets are public; obtain them from their authors and keep them outside version control (`data/` is git-ignored except this file).

## 1. SMoRFFI (main dataset, Sections 4–7)

- Paper: Z. Guo, Z. Jia, J. Zhu, W. Huang, Y. Chen, "SMoRFFI: A Large-Scale Same-Model 2.4 GHz Wi-Fi Dataset and Reproducible Framework for RF Fingerprinting," *Computer Networks* 282:112309, 2026. https://doi.org/10.1016/j.comnet.2026.112309 (preprint: https://arxiv.org/abs/2511.07770)
- Content used: 123 same-model 802.11g transmitters, 1,000 frames each, 15 per-frame physical-layer features
  (`CFO, short_freq, long_freq, frac_dimension_1, frac_dimension_2, iqi_1, iqi_2, mag_error_mean_1, mag_error_mean_2,
  mag_error_var_1, mag_error_var_2, phase_error_mean_1, phase_error_mean_2, phase_error_var_1, phase_error_var_2`).
  MAC address and device-number columns are **not** used as features.

The code expects one file, `features_v1.npz`, with arrays

| Array | Shape | Meaning |
|---|---|---|
| `x` | (122511, 15) | per-frame features, float |
| `y` | (122511,) | device index 0–122 |
| `row` | (122511,) | frame index within the device's capture (chronological) |
| `labels` | (123,) | device MAC address for each index (identifier only, never a feature) |

The file used in the paper has SHA-256 `4d3efa580e733493add9a920a457517bb9c17a5b91dc0611d77c8217aef29ab8`.
To build it from the per-device CSV files of SMoRFFI (`<MAC>_pre.csv`), run

```bash
python data/build_smorffi_features.py /path/to/SMoRFFI/csv_folder data/features_v1.npz
```

A rebuilt file holds the same arrays, but its SHA-256 can differ because of file metadata; the script prints the shapes
to compare with the table above. One device has fewer than 999 frames and is excluded by the code, leaving 122.

## 2. INRIA PLA BladeRF captures (Section 4.3, supplement S2)

- 12 commercial 2.4 GHz transmitters recorded with a BladeRF AX4 at 20 MS/s. Zenodo: https://doi.org/10.5281/zenodo.18268648
- Used by `experiments/inria_captures/eval/run_measured_inria.py` and by `rapid/run_on_your_data.py --iq-cache`.
- `run_measured_inria.py` reads the windowed I/Q cache (`meta.json`, `iq_f16.dat`, `y.npy`) from `$INRIA_CACHE`.

## 3. Emulated impersonation corpus (supplement S2)

- Four legitimate devices and four GNU Radio impersonators. Zenodo record 20186947.
- Expected files in `$EMULATED_DIR`: `legitimate_device_01.npy` … `_04.npy` and
  `attacker_device_05_impersonates_01.npy` … `attacker_device_08_impersonates_04.npy`.

```bash
export EMULATED_DIR=/path/to/RFFI_Emulated
```
