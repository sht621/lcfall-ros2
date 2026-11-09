# Pi build inputs (not bundled)

requires Pi verification. Supply an ARM64 HailoRT runtime `.deb` and its matching
CPython 3.12 / Linux ARM64 `hailort` / `hailo_platform` Python `.whl` here.
Both must match the Pi host Hailo-8 PCIe kernel driver. No automatic download,
version fallback, driver installation or ABI claim is provided.

Compose requires the exact basenames in HAILORT_DEB and HAILORT_WHEEL. Binary
artifacts are gitignored. This is a build-input directory, not a host installer.
