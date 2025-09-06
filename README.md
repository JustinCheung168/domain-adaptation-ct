# domain-generalization-ct

## Usage

### Setting up Docker

This project uses Docker to control requirements and aid in cross-platform compatibility.

#### Prerequisites

1. **Install Docker:**
   - For Linux: Follow the instructions at https://docs.docker.com/engine/install/.
   - For Mac & Windows: Download Docker Desktop from https://www.docker.com/products/docker-desktop/.

2. **Windows Only - Install Git Bash:**
   - Download and install Git Bash from https://git-scm.com/downloads.

#### Building the Docker Image (one-time setup)

1. Open a terminal (if you are using Windows, this terminal should be a Git Bash terminal).
2. Navigate to the project directory:
   ```bash
   cd /path/to/domain-generalization-ct
   ```
3. Run the `build.sh` script to build the Docker image:
   ```bash
   ./docker/build.sh
   ```

#### Setting your data path

Open `./docker/config.env` and set the path to where your data will live.

For example, on a Windows system:
``

#### Running the Project

1. Start the Docker container by running the `run.sh` script:
   ```bash
   ./docker/run.sh
   ```

### Troubleshooting

- **Docker not running:** Ensure Docker Desktop is running on your system.
- **Permission issues:** If you encounter permission errors, try running the scripts with `sudo` (Linux/Mac) or ensure your user is added to the `docker` group.
- **Incorrect `DATA_PATH`:** Verify that the `DATA_PATH` in `docker/config.env` points to the correct directory.

## Contents

- `geirhos/`: Preprocessing & training code for reproducing results from Geirhos et al. (2018).
- `metric_csvs/`: Raw training/validation/test result outputs.
- `src/`: Source code files. Primarily contains sinogram manipulation code right now.
- `*_pipeline*.ipynb`: Jupyter Notebooks used for model training for each experiment on OrganAMNIST data. In our convention expanding on notation used by Geirhos et al., "A" models are trained on single distortions, "C" models are trained on all-but-one distortion, and "D" models are based on Ganin & Lempitsky (2015)'s domain adaptation architecture.
- `GaninDALoss.ipynb`: Quick demonstration that the loss function component used for the label predictor successfully excludes influence of target domain instances.
- `Image_Manipulation*.ipynb`: Jupyter Notebooks for producing distorted data.
- `*resnet*.py`: Classes and script for our custom ResNet-50 configuration based on Ganin & Lempitsky (2015), and for a comparable unmodified ResNet-50.
- `ct_projection_proto.ipynb`: Exploration of sinogram manipulation.
- `evaluate_experiment.ipynb`: Model evaluation code.
- `medmnist_eda.ipynb`: Exploratory data analysis of MedMNIST datasets.
- `view_test_results.ipynb`: Model training/validation curve and test matrix visualization code. 

## References

- Ganin, Y., & Lempitsky, V. (2015, June). Unsupervised domain adaptation by backpropagation. In International conference on machine learning (pp. 1180-1189). PMLR.
- Geirhos, R., Temme, C. R., Rauber, J., Schütt, H. H., Bethge, M., & Wichmann, F. A. (2018). Generalisation in humans and deep neural networks. Advances in neural information processing systems, 31.

## TODO

Check docker process for various platforms
