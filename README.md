# THOP by Micron MDLA: Onnx-OpCounter from Micron DLA Team

THOP is Python package that allows you calculate the FLOPs for the [ONNX](https://github.com/onnx/onnx) file.

## Requirements

- Python 3.6+
- torch
- torchvision
- onnx
- xlsxwriter
- rich
- pandas
- tqdm
- onnx-simplifier
- PyYAML
- dohq-artifactory
- python-dotenv

## Install

```bash
$ pip install --upgrade git+https://bitbucket.micron.com/bbdc/scm/acemsml/onnx-opcounter.git
```

## How to use

#### To Run Local

```bash
$ python onnx_profiling_local.py /path/to/onnx-files-directory
```

- If you would like to run all the models in ***model_list.yaml*** we can use `all` argument.

  Example:

  ```bash
  $ python onnx_profiling_local.py /path/to/onnx-files-directory --all
  ```
- If you would like to export the summary of all the models in ***model_list.yaml*** we can use `summary` argument.

  Example:

  ```bash
  $ python onnx_profiling_local.py /path/to/onnx-files-directory --summary
  ```

  This will generate two files -

  1. ***regression_flops_output.xlsx*** which provides the aggregated summary of FLOPs for all the layers thats is present in each model.
  2. ***regression_output_local.xlsx*** which captures all the details that is displayed on the console. The information that is displayed for each model would be the `Total Parameters` and `Total FLOPs`. It also displays a table with other details for each layer.
- If you would like to simplify before profiling the onnx model we can use `simplify` argument.

  Example:

  ```bash
  $ python onnx_profiling_local.py /path/to/onnx-files-directory --simplify
  ```
- If you would like to round of the calculation we can use `decimal_roundoff` argument.

  Example:

  ```bash
  $ python onnx_profiling_local.py /path/to/onnx-files-directory --decimal_roundoff 3
  ```
- If you would like to display different FLOPs unit we can use `flops` argument.

  Example:

  ```bash
  $ python onnx_profiling_local.py /path/to/onnx-files-directory --flops MEGA
  ```
- We support calculation in three different units ***KILO***, ***MEGA*** and ***GIGA***.

Also, you can use all the `flags` together.

  Example:

```bash
  $ python onnx_profiling_local.py /path/to/onnx-files-directory --all --summary --simplify --decimal_roundoff 3 --flops KILO
```

#### To Run Cloud

We need to create a `.env` file using the following command from the base directory.

```bash
$ touch .env
```

Create an ***`APIKEY`*** by logging into [boartifactory](https://boartifactory.micron.com/ui/login/) using your _Micron_ credentials.

Copy the ***`APIKEY`*** to the `.env` file.

```
export API_KEY="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

Once the `.env` file is updated we can now run the cloud version of the script. This script has four different arguments you can pass `summary`, `simplify`, `decimal_roundoff` and `flops`. Each of this arguments perform the above mentioned task.

Example:

```bash
$ python onnx_profiling_cloud.py --summary --simplify --decimal_roundoff 1 --flops KILO
```

> **_NOTE:_**  For now the cloud version of the script runs on simplified version of the onnx directory. If you need to run on the non-simplified version of the onnx directory, then the `URL` variable in  ***onnx_profiling_cloud.py*** needs to be updated to
> `https://boartifactory.micron.com:443/artifactory/ndcg-generic-dev-local/MDLA-ModelZoo/models`

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

* [Ligeng Zhu -> GitHub Page](https://github.com/Lyken17/pytorch-OpCounter/)
