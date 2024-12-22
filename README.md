# ABC EqF: Attitude Biase and Calibration states Equivariant Filter

[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-informational)](./LICENSE)

Maintainer: [Alessandro Fornasier](mailto:alessandro.fornasier@aau.at)

- [Credit](#credit)
  * [Usage for academic purposes](#usage-for-academic-purposes)
- [Description](#description)
- [Requirements](#requirements)

## Credit
This code was written by the [Control of Networked System (CNS)](https://www.aau.at/en/smart-systems-technologies/control-of-networked-systems/), University of Klagenfurt.

### Usage for academic purposes
If you use this software in an academic research setting, please cite the
corresponding paper and consult the `LICENSE` file for a detailed explanation.

```latex
@ARTICLE{9905914,
  author={Fornasier, Alessandro and Ng, Yonhon and Brommer, Christian and Böhm, Christoph and Mahony, Robert and Weiss, Stephan},
  journal={IEEE Robotics and Automation Letters}, 
  title={Overcoming Bias: Equivariant Filter Design for Biased Attitude Estimation With Online Calibration}, 
  year={2022},
  volume={7},
  number={4},
  pages={12118-12125},
  doi={10.1109/LRA.2022.3210867}}
```

## Description
**ABC EqF** contains an educational implementation of the Equivariant filter presented in the Robotics and Automation Letter "Overcoming Bias: Equivariant Filter Design for Biased Attitude Estimation with Online Calibration".

## Requirements

* [pylie](https://github.com/pvangoor/pylie) (latest commit tested: 65922fc)

## Run the example

`python3 examples/simulation.py examples/data.csv --show`

