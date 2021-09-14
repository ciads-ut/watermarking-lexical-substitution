# Results of provided lexical substitution models:

#### Sbert_sub (Model 1, 1 Generator + Discriminator loss + SBERT):
Training method:
- 3000 steps (1 ep) discriminator only
  - Done by setting the discriminator frequency to 1 and generator frequency to 0
  - Saved as `sbert_model_disc.ckpt`
- 12000 steps (4 ep) both discriminator and generator
  - Saved as `sbert_model_sota.ckpt`

Key hyperparameters (see: hparams.py):
- ALPHA = 16
- DISC_LEARNING_RATE = 2e-6
- GEN_LEARNING_RATE = 2e-6
- BATCH_SIZE = 32

single score (sum of k=10 and k=50 single scores): 0.4676

|type| k | L F10 | L F10C | S F10 | S F10C|
|---|---|---|---|---|---|
|dev| k=10|39.84|41.59|22.90|29.77|
|dev| k=50|37.68|60.67|21.44|30.42|
|test| k=10|35.95|39.18|19.61|27.67|
|test| k=50|33.97|59.61|18.55|28.62|


#### Lex_sub (Model 2, 2x Generators + Discriminator):
Training method:
- 200 steps discriminator only with the generators outputting random values
  - Done to give discriminator a concept of location for classifier heads. Future work could attach the discriminator linear layers to the word output logits, like in the SBERT substitution model
  - Done by multiplying padded_sent_ids by 0 when calculating out_a and out_b and by setting the discriminator frequency to 1 and generator frequency to 0
- 3000 steps (1 ep) discriminator only
  - Saved as `lex_sub_disc.ckpt`
  - Done by setting the discriminator frequency to 1 and generator frequency to 0. padded_sent_ids multiplied by 1 (kept the same)
- 4500 steps (1.5 ep) both discriminator and generator
  - Saved as `lex_sub_best.ckpt`

Key hyperparameters (see: hparams.py):
- DISC_LEARNING_RATE = 1e-5
- GEN_LEARNING_RATE = 2e-6
- BATCH_SIZE = 32

single score (sum of k=10 and k=50 single scores): 0.4385

|type| k | L F10 | L F10C | S F10 | S F10C|
|---|---|---|---|---|---|
|dev| k=10|38.05|40.88|20.20|27.71|
|dev| k=50|36.10|60.06|19.41|27.79|
|test| k=10|34.63|38.79|17.78|26.25|
|test| k=50|32.18|58.28|17.40|26.57|

