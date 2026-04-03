# Invertible Embedding-Based Receivers for IDS Channels

This project develops invertible embedding-based receivers for insertion-deletion-substitution (IDS) channels.  
The main goal is to learn a robust embedding representation that enables recovery of transmitted messages despite synchronization errors such as insertions, deletions, and substitutions.

The framework supports several decoding setups:

- direct embedding-based decoder
- embedding-assisted neural decoder
- hybrid classical decoding with BCJR + soft-Viterbi
- no-embedding neural baseline

This allows fair comparison between classical, neural, and hybrid receivers under the same IDS setting.

---

## IDS Setting

In our experiments, we use the following IDS setting:

- insertion probability: `p_ins ∈ {0.01, 0.02, 0.03}`
- deletion probability: `p_del ∈ {0.01, 0.02, 0.03}`
- substitution probability for evaluation: `p_sub ∈ {0.01, 0.02, 0.03, 0.04, 0.05}`

For training:

- `p_ins` and `p_del` are trained stage-by-stage
- `p_sub` is trained using a range:
  - `--p_sub_train_min`
  - `--p_sub_train_max`

So the recommended curriculum is:

- **Stage 1**: `p_ins=0.01`, `p_del=0.01`
- **Stage 2**: `p_ins=0.02`, `p_del=0.02`
- **Stage 3**: `p_ins=0.03`, `p_del=0.03`

with

- `p_sub_train_min=0.01`
- `p_sub_train_max=0.05`

for all stages.

---

## Curriculum / Resume

### 1. Curriculum warm-start
Use:

```bash
--init_ckpt <previous_stage_checkpoint>
```

This loads model weights from the previous stage, but starts a new optimizer/scheduler state for the new stage.

This is the recommended mode for curriculum training:

- stage 2 loads from stage 1
- stage 3 loads from stage 2

### 2. Full resume
Use:

```bash
--resume_ckpt <checkpoint>
```

This resumes full training state, including:

- model weights
- optimizer state
- scheduler state
- epoch
- best validation metric

Use this only if a training run was interrupted and you want to continue the same stage.

---

## Recommended Curriculum

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

---

# 1. Train the Embedding Model

## Stage 1

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
```

## Stage 2

```bash
python -m ids_receiver.train.train_embedding \
  --init_ckpt runs_embed/embed_stage1/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 6e-4 \
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

## Stage 3

```bash
python -m ids_receiver.train.train_embedding \
  --init_ckpt runs_embed/embed_stage2/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 5e-4 \
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

Final embedding checkpoint:

```text
runs_embed/embed_stage3/best.pt
```

---

# 2. Train the Proposed Direct Decoder

Pipeline:

- embedding encoder
- direct neural decoder

## Stage 1

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

## Stage 2

```bash
python -m ids_receiver.train.train_decoder_direct \
  --embed_ckpt runs_embed/embed_stage3/best.pt \
  --freeze_embed 1 \
  --init_ckpt runs_embed/direct_decoder_stage1/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 6e-4 \
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

## Stage 3

```bash
python -m ids_receiver.train.train_decoder_direct \
  --embed_ckpt runs_embed/embed_stage3/best.pt \
  --freeze_embed 1 \
  --init_ckpt runs_embed/direct_decoder_stage2/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 5e-4 \
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

# 3. Train the Embedding-Based NBM

## Stage 1

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

## Stage 2

```bash
python -m ids_receiver.train.train_nbm_embed \
  --embed_ckpt runs_embed/embed_stage3/best.pt \
  --freeze_embed 1 \
  --init_ckpt runs_embed/nbm_stage1/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 7e-4 \
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

## Stage 3

```bash
python -m ids_receiver.train.train_nbm_embed \
  --embed_ckpt runs_embed/embed_stage3/best.pt \
  --freeze_embed 1 \
  --init_ckpt runs_embed/nbm_stage2/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 6e-4 \
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

# 4. Train the Embedding-Based Decoder

This decoder uses the final embedding checkpoint and the final NBM checkpoint.

## Stage 1

```bash
python -m ids_receiver.train.train_decoder_embed \
  --embed_ckpt runs_embed/embed_stage3/best.pt \
  --nbm_ckpt runs_embed/nbm_stage3/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 7e-4 \
  --save_dir runs_embed/decoder_stage1 \
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

## Stage 2

```bash
python -m ids_receiver.train.train_decoder_embed \
  --embed_ckpt runs_embed/embed_stage3/best.pt \
  --nbm_ckpt runs_embed/nbm_stage3/best.pt \
  --init_ckpt runs_embed/decoder_stage1/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 6e-4 \
  --save_dir runs_embed/decoder_stage2 \
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

## Stage 3

```bash
python -m ids_receiver.train.train_decoder_embed \
  --embed_ckpt runs_embed/embed_stage3/best.pt \
  --nbm_ckpt runs_embed/nbm_stage3/best.pt \
  --init_ckpt runs_embed/decoder_stage2/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 5e-4 \
  --save_dir runs_embed/decoder_stage3 \
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
runs_embed/decoder_stage3/best.pt
```

---

# 5. Train the No-Embedding NBM Baseline

## Stage 1

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

## Stage 2

```bash
python -m ids_receiver.train.train_nbm_noembed \
  --init_ckpt runs_noembed/nbm_stage1/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 8e-4 \
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

## Stage 3

```bash
python -m ids_receiver.train.train_nbm_noembed \
  --init_ckpt runs_noembed/nbm_stage2/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 7e-4 \
  --save_dir runs_noembed/nbm_stage3 \
  --train_samples 100000 \
  --val_samples 10000 \
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
runs_noembed/nbm_stage3/best.pt
```

---

# 6. Train the No-Embedding Decoder Baseline

## Stage 1

```bash
python -m ids_receiver.train.train_decoder_noembed \
  --nbm_ckpt runs_noembed/nbm_stage3/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 8e-4 \
  --save_dir runs_noembed/decoder_stage1 \
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

## Stage 2

```bash
python -m ids_receiver.train.train_decoder_noembed \
  --nbm_ckpt runs_noembed/nbm_stage3/best.pt \
  --init_ckpt runs_noembed/decoder_stage1/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 7e-4 \
  --save_dir runs_noembed/decoder_stage2 \
  --train_samples 120000 \
  --val_samples 10000 \
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

## Stage 3

```bash
python -m ids_receiver.train.train_decoder_noembed \
  --nbm_ckpt runs_noembed/nbm_stage3/best.pt \
  --init_ckpt runs_noembed/decoder_stage2/best.pt \
  --epochs 100 \
  --batch_size 128 \
  --lr 6e-4 \
  --save_dir runs_noembed/decoder_stage3 \
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
runs_noembed/decoder_stage3/best.pt
```

---



# Evaluation

For evaluation, use:

- `p_ins ∈ {0.01, 0.02, 0.03}`
- `p_del ∈ {0.01, 0.02, 0.03}`
- `p_sub_list = 0.01,0.02,0.03,0.04,0.05`

## A. Proposed direct decoder

```bash
python -m ids_receiver.eval.evaluate_direct \
  --ckpt runs_embed/direct_decoder_stage3/best.pt \
  --p_ins 0.03 \
  --p_del 0.03 \
  --p_sub_list 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10 \
  --n_trials 3000 \
  --batch_size 256 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --out_csv runs_embed/eval_direct_original_id=3.csv
```

---

## B. Embedding-based neural pipeline

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
  --out_csv runs_embed/eval_embed_original_id=3.csv
```

---

## C. No-embedding baseline

```bash
python -m ids_receiver.eval.evaluate_noembed \
  --ckpt runs_noembed/decoder_stage3/best.pt \
  --p_ins 0.03 \
  --p_del 0.03 \
  --p_sub_list 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10 \
  --n_trials 3000 \
  --batch_size 256 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --out_csv runs_noembed/eval_noembed_nbm_id=1.csv
```

---

## D. Classical BCJR + soft-Viterbi

```bash
python -m ids_receiver.eval.evaluate_conv_bcjr_softviterbi \
  --p_ins 0.01 \
  --p_del 0.01 \
  --p_sub_list 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10 \
  --n_trials 3000 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --out_csv runs_bcjr_viterbi/eval_conv_bcjr_softviterbi_original_id=1.csv
```

---

## E. Hybrid embedding-assisted BCJR + soft-Viterbi

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
  --out_csv runs_embed_bcjr_viterbi/eval_conv_embed_bcjr_softviterbi_original.csv
```
