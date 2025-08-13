# pennsieve-ieeg-recon

This application performs iEEG reconstruction using pre-implant MRI, post-implant CT, and electrode coordinates.

## Environment Variables

The application uses two key environment variables:
- `INPUT_DIR`: Directory containing input files (default: `/data/input`)
- `OUTPUT_DIR`: Directory for output files (default: `/data/output`)

## Usage Examples

### Pull the Container

First, pull the container from Docker Hub:

```bash
docker pull nishantsinha89/ieeg_recon:latest
```

### Basic Usage (Using Environment Variables)

Mount your data directories and run with default file names:

```bash
docker run -v "/path/to/your/data:/data/input" \
  -v "/path/to/your/output:/data/output" \
  nishantsinha89/ieeg_recon:latest \
  --t1 "/data/input/T1.nii.gz" \
  --ct "/data/input/CT.nii.gz" \
  --elec "/data/input/electrodes.txt" \
  --output-dir "/data/output" \
  --modules "1,2,3,4" \
  --qa-viewer niplot \
  --reg-type gc_noCTthereshold
```

**Expected file structure in INPUT_DIR (`/data/input`):**
```
/data/input/
├── T1.nii.gz                    # Pre-implant MRI
├── CT.nii.gz                    # Post-implant CT  
└── electrodes.txt               # Electrode coordinates
```

### Advanced Usage (Using Demo Data)

Run with the included demo data for testing and learning:

```bash
docker run -v "$(pwd)/demodata/input:/data/input" \
  -v "$(pwd)/demodata/output:/data/output" \
  nishantsinha89/ieeg_recon:latest \
  --t1 "/data/input/T1.nii.gz" \
  --ct "/data/input/CT.nii.gz" \
  --elec "/data/input/electrodes.txt" \
  --freesurfer-dir "/data/input/freesurfer" \
  --output-dir "/data/output" \
  --skip-existing \
  --modules "1,2,3,4" \
  --qa-viewer niplot \
  --reg-type gc_noCTthereshold
```

**Demo data structure:**
```
demodata/input/
├── T1.nii.gz                    # Demo pre-implant MRI
├── CT.nii.gz                    # Demo post-implant CT  
├── electrodes.txt               # Demo electrode coordinates
└── freesurfer/                  # Demo FreeSurfer directory
```

### Minimal Usage (Required Arguments Only)

Run with only the required arguments:

```bash
docker run -v "/your/data:/data/input" \
  -v "/your/output:/data/output" \
  nishantsinha89/ieeg_recon:latest \
  --t1 "/data/input/T1.nii.gz" \
  --ct "/data/input/CT.nii.gz" \
  --elec "/data/input/electrodes.txt" \
  --output-dir "/data/output" \
  --modules "1,2,3,4"
```

### Interactive Mode

Run the container interactively to explore and debug:

```bash
docker run -it --entrypoint bash \
  -v "/your/data:/data/input" \
  -v "/your/output:/data/output" \
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