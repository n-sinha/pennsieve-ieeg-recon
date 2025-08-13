# pennsieve-ieeg-recon

This application performs iEEG reconstruction using pre-implant MRI, post-implant CT, and electrode coordinates.

## Environment Variables

The application uses two key environment variables:
- `INPUT_DIR`: Directory containing input files (default: `/data`)
- `OUTPUT_DIR`: Directory for output files (default: `/output`)

## Usage Examples

### Pull the Container

First, pull the container from Docker Hub:

```bash
docker pull nishantsinha89/ieeg_recon:latest
```

### Basic Usage (Using Environment Variables)

Mount your data directories and run with default file names:

```bash
docker run -v "/path/to/your/data:/data" \
  -v "/path/to/your/output:/output" \
  nishantsinha89/ieeg_recon:latest \
  --t1 "/data/T1.nii.gz" \
  --ct "/data/CT.nii.gz" \
  --elec "/data/electrodes.txt" \
  --output-dir "/output" \
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
  nishantsinha89/ieeg_recon:latest \
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

### Minimal Usage (Required Arguments Only)

Run with only the required arguments:

```bash
docker run -v "/your/data:/data" \
  -v "/your/output:/output" \
  nishantsinha89/ieeg_recon:latest \
  --t1 "/data/T1.nii.gz" \
  --ct "/data/CT.nii.gz" \
  --elec "/data/electrodes.txt" \
  --output-dir "/output" \
  --modules "1,2,3,4"
```

### Interactive Mode

Run the container interactively to explore and debug:

```bash
docker run -it --entrypoint bash \
  -v "/your/data:/data" \
  -v "/your/output:/output" \
  nishantsinha89/ieeg_recon:latest
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

| Option | Description | Required | Default |
|--------|-------------|----------|---------|
| `--t1` | T1 MRI file path | **Yes** | None |
| `--ct` | CT file path | **Yes** | None |
| `--elec` | Electrode file path | **Yes** | None |
| `--output-dir` | Output directory | **Yes** | None |
| `--freesurfer-dir` | FreeSurfer directory path | No | None |
| `--env-path` | Environment file path | No | `.env` |
| `--modules` | Modules to run | No | `"1,2,3,4"` |
| `--qa-viewer` | Quality assurance viewer | No | `"niplot"` |
| `--reg-type` | Registration type | No | `"gc_noCTthereshold"` |
| `--skip-existing` | Skip if output exists | No | `False` |
| `--save-channels` | Save individual channels | No | `False` |

**Note:** The `--t1`, `--ct`, `--elec`, and `--output-dir` arguments are **required** and must be specified for the application to run.