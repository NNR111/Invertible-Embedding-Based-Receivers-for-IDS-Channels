This project develops invertible embedding-based receivers for insertion-deletion-substitution (IDS) channels. The main goal is to learn a robust embedding representation that enables recovery of transmitted messages despite synchronization errors such as insertions, deletions, and substitutions. The framework includes several decoding setups, including a direct embedding-based decoder, embedding-assisted neural decoding, and hybrid classical decoding with BCJR and soft-Viterbi. This allows us to compare baseline, neural, and hybrid receivers under the same IDS channel setting.


In our experiments, we use the following IDS setting:

- insertion probability: `p_ins = 0.01, 0.02, 0.03`
- deletion probability: `p_del = 0.01, 0.02, 0.03`
- substitution probability for evaluation: `p_sub in {0.01, 0.02, 0.03, 0.04, 0.05}`

For training, the current codebase supports:
- fixed `p_ins` per run
- fixed `p_del` per run
- curriculum over substitution through:
  - `--p_sub_train_min`
  - `--p_sub_train_max`

Therefore, the recommended setup is:
- use **stage-wise curriculum** for insertion/deletion:
  - stage 1: `p_ins=0.01`, `p_del=0.01`
  - stage 2: `p_ins=0.02`, `p_del=0.02`
  - stage 3: `p_ins=0.03`, `p_del=0.03`
- use **range-based curriculum** for substitution during training:
  - `p_sub_train_min=0.01`
  - `p_sub_train_max=0.05`

This means the model is gradually exposed to harder insertion/deletion conditions across stages, while already learning over the full substitution range during each stage.


### Stage 1
- `p_ins = 0.01`
- `p_del = 0.01`
- `p_sub_train_min = 0.01`
- `p_sub_train_max = 0.05`

### Stage 2
- `p_ins = 0.02`
- `p_del = 0.02`
- `p_sub_train_min = 0.01`
- `p_sub_train_max = 0.05`

### Stage 3
- `p_ins = 0.03`
- `p_del = 0.03`
- `p_sub_train_min = 0.01`
- `p_sub_train_max = 0.05`

At each stage, training continues from the checkpoint of the previous stage.

---

## 1. Train the Embedding Model

### Stage 1
```bash
python -m ids_receiver.train.train_embedding \
  --epochs 100 \
  --batch_size 128 \
  --lr 8e-4 \
  --save_dir runs_embed/embed_stage1 \
  --lambda_local 0.35 \
  --train_samples 120000 \
  --val_samples 12000 \
  --workers 0 \
  --p_ins 0.01 \
  --p_del 0.01 \
  --p_sub_train_min 0.01 \
  --p_sub_train_max 0.05 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --seed 0
````

### Stage 2

```bash
python -m ids_receiver.train.train_embedding \
  --epochs 100 \
  --batch_size 128 \
  --lr 8e-4 \
  --save_dir runs_embed/embed_stage2 \
  --lambda_local 0.35 \
  --train_samples 120000 \
  --val_samples 12000 \
  --workers 0 \
  --p_ins 0.02 \
  --p_del 0.02 \
  --p_sub_train_min 0.01 \
  --p_sub_train_max 0.05 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --seed 0
```

### Stage 3

```bash
python -m ids_receiver.train.train_embedding \
  --epochs 100 \
  --batch_size 128 \
  --lr 8e-4 \
  --save_dir runs_embed/embed_stage3 \
  --lambda_local 0.35 \
  --train_samples 120000 \
  --val_samples 12000 \
  --workers 0 \
  --p_ins 0.03 \
  --p_del 0.03 \
  --p_sub_train_min 0.01 \
  --p_sub_train_max 0.05 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --seed 0
```

Use the final embedding checkpoint from:

```
runs_embed/embed_stage3/best.pt
```

---

## 2. Train the Proposed Direct Decoder

This is the proposed pipeline:

* embedding encoder
* direct neural decoder

### Stage 1

```bash
python -m ids_receiver.train.train_decoder_direct \
  --embed_ckpt runs_embed/embed_stage3/best.pt \
  --freeze_embed 1 \
  --epochs 100 \
  --batch_size 128 \
  --lr 8e-4 \
  --save_dir runs_embed/direct_decoder_stage1 \
  --train_samples 120000 \
  --val_samples 12000 \
  --workers 0 \
  --p_ins 0.01 \
  --p_del 0.01 \
  --p_sub_train_min 0.01 \
  --p_sub_train_max 0.05 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --seed 0
```

### Stage 2

```bash
python -m ids_receiver.train.train_decoder_direct \
  --embed_ckpt runs_embed/embed_stage3/best.pt \
  --freeze_embed 1 \
  --epochs 100 \
  --batch_size 128 \
  --lr 7e-4 \
  --save_dir runs_embed/direct_decoder_stage2 \
  --train_samples 120000 \
  --val_samples 12000 \
  --workers 0 \
  --p_ins 0.02 \
  --p_del 0.02 \
  --p_sub_train_min 0.01 \
  --p_sub_train_max 0.05 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --seed 0
```

### Stage 3

```bash
python -m ids_receiver.train.train_decoder_direct \
  --embed_ckpt runs_embed/embed_stage3/best.pt \
  --freeze_embed 1 \
  --epochs 100 \
  --batch_size 128 \
  --lr 7e-4 \
  --save_dir runs_embed/direct_decoder_stage3 \
  --train_samples 120000 \
  --val_samples 12000 \
  --workers 0 \
  --p_ins 0.03 \
  --p_del 0.03 \
  --p_sub_train_min 0.01 \
  --p_sub_train_max 0.05 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --seed 0
```

Final checkpoint:

```text
runs_embed/direct_decoder_stage3/best.pt
```

---

## 3. Train the Embedding-Based NBM

### Stage 1

```bash
python -m ids_receiver.train.train_nbm_embed \
  --embed_ckpt runs_embed/embed_stage3/best.pt \
  --freeze_embed 1 \
  --epochs 100 \
  --batch_size 128 \
  --lr 8e-4 \
  --save_dir runs_embed/nbm_stage1 \
  --train_samples 120000 \
  --val_samples 12000 \
  --workers 0 \
  --p_ins 0.01 \
  --p_del 0.01 \
  --p_sub_train_min 0.01 \
  --p_sub_train_max 0.05 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --seed 0
```

### Stage 2

```bash
python -m ids_receiver.train.train_nbm_embed \
  --embed_ckpt runs_embed/embed_stage3/best.pt \
  --freeze_embed 1 \
  --epochs 100 \
  --batch_size 128 \
  --lr 8e-4 \
  --save_dir runs_embed/nbm_stage2 \
  --train_samples 120000 \
  --val_samples 12000 \
  --workers 0 \
  --p_ins 0.02 \
  --p_del 0.02 \
  --p_sub_train_min 0.01 \
  --p_sub_train_max 0.05 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --seed 0
```

### Stage 3

```bash
python -m ids_receiver.train.train_nbm_embed \
  --embed_ckpt runs_embed/embed_stage3/best.pt \
  --freeze_embed 1 \
  --epochs 100 \
  --batch_size 128 \
  --lr 8e-4 \
  --save_dir runs_embed/nbm_stage3 \
  --train_samples 120000 \
  --val_samples 12000 \
  --workers 0 \
  --p_ins 0.03 \
  --p_del 0.03 \
  --p_sub_train_min 0.01 \
  --p_sub_train_max 0.05 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --seed 0
```

Final checkpoint:

```text
runs_embed/nbm_stage3/best.pt
```

---

## 4. Train the Embedding-Based Decoder

```bash
python -m ids_receiver.train.train_decoder_embed \
  --embed_ckpt runs_embed/embed_stage3/best.pt \
  --nbm_ckpt runs_embed/nbm_stage3/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 7e-4 \
  --save_dir runs_embed/decoder_stage3
```

---

## 5. Train the No-Embedding NBM Baseline

### Stage 1

```bash
python -m ids_receiver.train.train_nbm_noembed \
  --epochs 100 \
  --batch_size 128 \
  --lr 9e-4 \
  --save_dir runs_noembed/nbm_stage1 \
  --train_samples 120000 \
  --val_samples 12000 \
  --workers 0 \
  --p_ins 0.01 \
  --p_del 0.01 \
  --p_sub_train_min 0.01 \
  --p_sub_train_max 0.05 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --seed 0
```

### Stage 2

```bash
python -m ids_receiver.train.train_nbm_noembed \
  --epochs 100 \
  --batch_size 128 \
  --lr 9e-4 \
  --save_dir runs_noembed/nbm_stage2 \
  --train_samples 120000 \
  --val_samples 12000 \
  --workers 0 \
  --p_ins 0.02 \
  --p_del 0.02 \
  --p_sub_train_min 0.01 \
  --p_sub_train_max 0.05 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --seed 0
```

### Stage 3

```bash
python -m ids_receiver.train.train_nbm_noembed \
  --epochs 100 \
  --batch_size 128 \
  --lr 9e-4 \
  --save_dir runs_noembed/nbm_stage3 \
  --train_samples 120000 \
  --val_samples 12000 \
  --workers 0 \
  --p_ins 0.03 \
  --p_del 0.03 \
  --p_sub_train_min 0.01 \
  --p_sub_train_max 0.05 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --seed 0
```

---

## 6. Train the No-Embedding Decoder Baseline

```bash
python -m ids_receiver.train.train_decoder_noembed \
  --nbm_ckpt runs_noembed/nbm_stage3/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 8e-4 \
  --save_dir runs_noembed/decoder_stage3
```

---

## Evaluation

For evaluation, use:

* `p_ins = 0.01-0.03`
* `p_del = 0.01-0.03`
* `p_sub_list = 0.01,0.02,0.03,0.04,0.05`

### Proposed direct decoder

```bash
python -m ids_receiver.eval.evaluate_direct \
  --ckpt runs_embed/direct_decoder_stage3/best.pt \
  --p_ins 0.03 \
  --p_del 0.03 \
  --p_sub_list 0.01,0.02,0.03,0.04,0.05 \
  --n_trials 3000 \
  --batch_size 256 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --out_csv runs_embed/eval_direct_curriculum.csv
```

### Embedding-based neural pipeline

```bash
python -m ids_receiver.eval.evaluate_embed \
  --ckpt runs_embed/decoder_stage3/best.pt \
  --p_ins 0.03 \
  --p_del 0.03 \
  --p_sub_list 0.01,0.02,0.03,0.04,0.05 \
  --n_trials 3000 \
  --batch_size 256 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --out_csv runs_embed/eval_embed_curriculum.csv
```

### No-embedding baseline

```bash
python -m ids_receiver.eval.evaluate_noembed \
  --ckpt runs_noembed/decoder_stage3/best.pt \
  --p_ins 0.03 \
  --p_del 0.03 \
  --p_sub_list 0.01,0.02,0.03,0.04,0.05 \
  --n_trials 3000 \
  --batch_size 256 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --out_csv runs_noembed/eval_noembed_curriculum.csv
```

### Classical BCJR + soft-Viterbi

```bash
python -m ids_receiver.eval.evaluate_conv_bcjr_softviterbi \
  --p_ins 0.03 \
  --p_del 0.03 \
  --p_sub_list 0.01,0.02,0.03,0.04,0.05 \
  --n_trials 3000 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --out_csv runs_bcjr_viterbi/eval_conv_bcjr_softviterbi_curriculum.csv
```

### Hybrid embedding-assisted BCJR + soft-Viterbi

```bash
python -m ids_receiver.eval.evaluate_conv_embed_bcjr_softviterbi \
  --nbm_ckpt runs_embed/nbm_stage3/best.pt \
  --prior_scale 1.0 \
  --p_ins 0.03 \
  --p_del 0.03 \
  --p_sub_list 0.01,0.02,0.03,0.04,0.05 \
  --n_trials 3000 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --out_csv runs_embed_bcjr_viterbi/eval_conv_embed_bcjr_softviterbi_curriculum.csv
```


git config --global user.name "NNR111"
git config --global user.email "naufalrafi781@gmail.com"
git add -A
git commit -m "Invertible-Embedding-for-IDS-Channels"
git push --set-upstream origin main

