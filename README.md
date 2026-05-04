# Invertible Embedding-Based Receivers for IDS Channels

This project develops invertible embedding-based receivers for insertion-deletion-substitution (IDS) channels.  
The main goal is to learn a robust embedding representation that enables recovery of transmitted messages despite synchronization errors such as insertions, deletions, and substitutions.

The framework supports several decoding setups:

- direct embedding-based decoder
- embedding-assisted neural decoder
- hybrid classical decoding with BCJR + soft-Viterbi


This allows fair comparison between classical, neural, and hybrid receivers under the same IDS setting.

---

## IDS Setting

In our experiments, we use the following IDS setting:

- insertion probability: `p_ins ∈ {0.01, 0.02, 0.03}`
- deletion probability: `p_del ∈ {0.01, 0.02, 0.03}`
- substitution probability for evaluation: `p_sub ∈ {0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10}`




# Evaluation

For evaluation, use:

- `p_ins ∈ {0.01, 0.02, 0.03}`
- `p_del ∈ {0.01, 0.02, 0.03}`
- `p_sub_list = 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10`

## A. Direct decoder

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
  --p_sub_list 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10 \
  --n_trials 3000 \
  --batch_size 256 \
  --use_marker 1 \
  --marker 0,3 \
  --num_blocks 20 \
  --out_csv runs_embed/eval_embed_original_id=3.csv
```

---

---

## C. Classical BCJR + soft-Viterbi

```bash
python -m ids_receiver.eval.evaluate_conv_bcjr_softviterbi \
  --p_ins 0.03 \
  --p_del 0.03 \
  --p_sub_list 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10 \
  --n_trials 1000 \
  --out_csv runs_bcjr_viterbi/eval_conv_bcjr_softviterbi_original_id=3.csv
```

---

## D. Hybrid embedding-assisted BCJR + soft-Viterbi

```bash
python -m ids_receiver.eval.evaluate_conv_embed_bcjr_softviterbi \
  --nbm_ckpt runs_embed/nbm_stage3/best.pt \
  --prior_scale 1.0 \
  --p_ins 0.03 \
  --p_del 0.03 \
  --p_sub_list 0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10 \
  --n_trials 1000 \
  --out_csv runs_embed_bcjr_viterbi/eval_conv_embed_bcjr_softviterbi_id=3.csv
```

