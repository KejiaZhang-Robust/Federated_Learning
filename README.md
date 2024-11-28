**Title:** Federated Learning with DNNs)([https://github.com/KejiaZhang-Robust/](https://www.google.com/url?sa=E&source=gmail&q=https://github.com/KejiaZhang-Robust/)[Federated_Learning])

**Description:**

This repository implements federated learning (FL) using a ResNet architecture for image classification tasks. The configuration file (`configs_train.yml`) provides various parameters to customize the training process.

**Key Features:**

  - **Federated Learning:** Enables training on distributed datasets residing on multiple devices.
  - **WRN-34 Model:** Utilizes a deep convolutional neural network architecture for efficient learning.
  - **Configurable Training:** Offers flexibility through various settings in `configs_train.yml`.

**Installation:**

1.  Clone this repository:

    ```bash
    git clone https://github.com/KejiaZhang-Robust/[your-repository-name].git
    ```

2.  Install dependencies :

```
pytorch
numpy
tqdm
```

**Usage:**

1.  **Configure `configs_train.yml`:**

      - Adjust parameters as needed based on your dataset and training goals. Refer to the detailed explanation below for guidance.

2.  **Run Training:**

      - Execute the main training script (replace `train_FL.py` if different):

    <!-- end list -->

    ```bash
    python train_FL.py
    ```

**Explanation of `configs_train.yml`:**

**Operation:**

  - **Prefix:** Appends this string to model filenames (default: "WRN34\_10").
  - **record\_words:** Optional string to append to filenames for record-keeping purposes.
  - **Resume:** Set to `True` to resume training from a checkpoint (default: `False`).
  - **GPU:** Specify a GPU ID for training using CUDA (default: `None` for CPU).

**Train:**

  - **Train\_Method:** Set to `"FL"` for federated learning.
  - **Optimizer:** Choose an optimizer like `"SGD"` or `"Adam"` (default: `"SGD"`).
  - **Epoch:** Number of training rounds (default: `10`).
  - **Num\_users:** Number of participating devices/clients (default: `100`).
  - **Local\_epoch:** Number of local training epochs on each device (default: `10`).
  - **C:** Fraction of clients selected in each round (default: `0.1`).
  - **Local\_bs:** Local batch size on each device (default: `10`).
  - **Lr:** Learning rate (default: `0.01`).
  - **Momentum:** SGD momentum (default: `0.5`).
  - **lr\_change\_iter:** List of iterations at which to reduce the learning rate (default: `[100, 105]`).

**DATA:**

  - **Dataset:** Supported datasets (replace based on your implementation, e.g., `"CIFAR10"`, `"MNIST"`) (default: `"MNIST"`).
  - **Split:** Data distribution strategy (e.g., `"IID"` for Independent and Identically Distributed, `"NonIID"` for non-IID) (default: `"IID"`).
  - **num\_class:** Number of classes in the dataset (default: `10`).
  - **mean, std:** Dataset mean and standard deviation for normalization (replace with values relevant to your dataset).




  - Replace bracketed placeholders with your actual information.
  - Consider adding screenshots or GIFs to showcase the project's functionality (optional).
  - Include links to relevant resources or tutorials on federated learning and the WRN-34 architecture (optional).

**Remember to replace the bracketed placeholders with your specific details and files.** This comprehensive README file will provide a clear understanding of your federated learning code on GitHub, making it easier for others to contribute and use it effectively.
