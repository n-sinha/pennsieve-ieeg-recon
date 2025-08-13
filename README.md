# pennsieve-ieeg-recon application

This application performs iEEG reconstruction using pre-implant MRI, post-implant CT, and electrode coordinates.

## Environment Variables

The application uses two key environment variables:
- `INPUT_DIR`: Directory containing input files (default: `/data`)
- `OUTPUT_DIR`: Directory for output files (default: `/output`)

## Usage Examples

### Basic Usage (Using Environment Variables)

Mount your data directories and run with default file names:

```bash
docker run -v "/path/to/your/data:/data" \
  -v "/path/to/your/output:/output" \
  nishant/ieeg_recon \
  --modules "1,2,3,4" \
  --qa-viewer niplot \
  --reg-type gc_noCTthereshold
```

**Expected file structure in INPUT_DIR (`/data`):**
```
/data/
├── T1.nii.gz                    # Pre-implant MRI
├── CT.nii.gz                    # Post-implant CT  
└── electrodes.txt               # Electrode coordinates
```

### Advanced Usage (Custom File Names)

Specify custom file paths while still using environment variables:

```bash
docker run -v "/Users/nishant/Dropbox/Sinha/Lab/Research/epi_t3_iEEG/data:/data" \
  -v "/Users/nishant/Dropbox/Sinha/Lab/Research/pennsieve-applications/pennsieve-ieeg-recon/data:/output" \
  nishant/ieeg_recon \
  --t1 "/data/BIDS/sub-RID0031/derivatives/freesurfer/mri/T1.nii.gz" \
  --ct "/data/BIDS/sub-RID0031/ses-clinical01/ct/sub-RID0031_ses-clinical01_acq-3D_space-T01ct_ct.nii.gz" \
  --elec "/data/BIDS/sub-RID0031/ses-clinical01/ieeg/sub-RID0031_ses-clinical01_space-T01ct_desc-vox_electrodes.txt" \
  --freesurfer-dir "/data/BIDS/sub-RID0031/derivatives/freesurfer" \
  --output-dir "/output/recon" \
  --skip-existing \
  --modules "1,2,3,4" \
  --qa-viewer niplot \
  --reg-type gc_noCTthereshold
```

### Minimal Usage (Just Required Arguments)

Run with minimal arguments, letting the app use environment variable defaults:

```bash
docker run -v "/your/data:/data" \
  -v "/your/output:/output" \
  nishant/ieeg_recon \
  --modules "1,2,3,4"
```

### Interactive Mode

Run the container interactively to explore and debug:

```bash
docker run -it --entrypoint bash \
  -v "/your/data:/data" \
  -v "/your/output:/output" \
  nishant/ieeg_recon
```

## File Naming Conventions

When using environment variables, the application expects these default file names in `INPUT_DIR`:

| File Type | Default Name | Description |
|-----------|--------------|-------------|
| T1 MRI | `T1.nii.gz` | Pre-implant structural MRI |
| CT | `CT.nii.gz` | Post-implant CT scan |
| Electrodes | `electrodes.txt` | Electrode coordinates (space-separated) |

## Output Structure

All outputs are written to `OUTPUT_DIR` with this structure:

```
OUTPUT_DIR/
└── ieeg_recon/
    ├── module1/          # Electrode coordinates in CT space
    ├── module2/          # Registration and electrode spheres
    ├── module3/          # Brain region mapping
    └── module4/          # MNI space transformation
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--t1` | T1 MRI file path | `INPUT_DIR/T1.nii.gz` |
| `--ct` | CT file path | `INPUT_DIR/CT.nii.gz` |
| `--elec` | Electrode file path | `INPUT_DIR/electrodes.txt` |
| `--output-dir` | Output directory | `OUTPUT_DIR` |
| `--modules` | Modules to run | `"1,2,3,4"` |
| `--qa-viewer` | Quality assurance viewer | `"niplot"` |
| `--reg-type` | Registration type | `"gc_noCTthereshold"` |
| `--skip-existing` | Skip if output exists | `False` |
| `--save-channels` | Save individual channels | `False` |