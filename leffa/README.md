# VTON

Based on `genads/imagenet`.


## Methods

- IDM-VTON
- CatVTON
- SimpleVTON
- Learning Flow Fields in Attention


## Prerequisites

- **Genie CLI**. Jobs are launched with `genie`. Install the CLI on your devserver with `feature install genie --persist`. For more information, visit https://www.internalfb.com/intern/wiki/Genie/Genie_101/Genie_CLI/

- **ACL permissions**. To launch jobs, you need to be added to [`oncall_ai_genads`](https://www.internalfb.com/omh/view/ai_genads/oncall_management_settings/members).

- **Hive permissions**. Once you have your dataset you will need it

### Starting a New project:
Create a fbpkg:
`fbpkg create genads.train.idm_vton --oncall-team=ai_genads --pkg-desc='idm_vton for genads usage' --acl-name ai_genads_fbpkg`
Create a new model type:
https://www.internalfb.com/mlhub/models/model_type


## Train

### Local training

You need a local machine with at least 1 GPUs of 80G.

```
CUDA_VISIBLE_DEVICES=7 genie launch genads/idm_vton --config pkg://genads/idm_vton/launcher/launch_train.yaml launcher=local launcher.torchx_options.num_processes=1 conf@app.run_fn.cfg=train_local
```

### Cluster training (MAST, Distributed)
If you are granted access to the pool

```
genie launch genads/idm_vton --config pkg://genads/idm_vton/launcher/launch_train.yaml launcher=mast launcher.torchx_options.num_hosts=4 launcher.torchx_options.num_processes=8 conf@app.run_fn.cfg=train.yaml ++launcher.torchx_options.job_name_suffix=vton_v0_0

genie launch genads/idm_vton --config pkg://genads/idm_vton/launcher/launch_train.yaml launcher=mast launcher.torchx_options.host_type=grandteton launcher.torchx_options.num_hosts=4 launcher.torchx_options.num_processes=8 conf@app.run_fn.cfg=train.yaml ++launcher.torchx_options.job_name_suffix=vton_v0_0
```


## Predict

First:
```
cd /path/to/genads/idm_vton/
```

### Local predicting
```
CUDA_VISIBLE_DEVICES=7 torchx run --scheduler local_penv fb.dist.hpc -m leffa.predict -j 1x1 -- -cn predict.yaml max_steps_per_epoch=2
```

### Cluster predicting
More information, please see this [doc](https://docs.google.com/document/d/1PW-ABvpjtUiwghXz6ZqL_sobjFCMinRfLN6QICS37-E/edit).

```
torchx run --scheduler mast fb.dist.hpc -m leffa.predict -j 2x8 -- -cn predict.yaml
```
