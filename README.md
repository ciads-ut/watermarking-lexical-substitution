## Watermarking and Lexical Substitution

This repository contains code for text-based watermarking and lexical substitution.
This project is still in the research phase, so reproducibility may be limited.

Each of the three main folders corresponds to a different model:

- BART_Watermarker: A watermarker similar to AWT [1], but using pretrained model BART [3].
- Sbert_sub: Model 1 for lexical substitution (generator, discriminator, and sbert)
- Lex_sub: Model 2 for lexical substitution (two generators and one discriminator)

These folders contain code for both training and loading models.

Hyperparameters and checkpointing can be navigated through the train.py files and hparams.py files.

### Training

To train the models, run the train.py script in the respective folder.
At the end of every epoch, the SWORDS [2] dev set will be run.

Required format for the training data:

- The format for the BART Pretrained Watermarker is an untokenized sentence on each line.
- The format of the data files for the substitution models is the following on each line:
``[Target Word no Space] [Space] [Sentence with [MASK] in the position of target word]``

### Evaluation

For evaluation, follow the same format as the SWORDS generator function,
described on their github repository,
https://github.com/p-lambda/swords#evaluating-new-lexical-substitution-methods-on-swords.

To see the BART watermarker in action, run the playground.py file. To train the model, run train.py.

References
----------

[1] Abdelnabi, Sahar, and Mario Fritz. "Adversarial watermarking transformer: Towards tracing text provenance with data hiding." 2021 IEEE Symposium on Security and Privacy (SP). IEEE, 2021.

[2] Lee, Mina, et al. "Swords: A Benchmark for Lexical Substitution with Improved Data Coverage and Quality." NAACL. 2021.

[3] Lewis, Mike, et al. "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." ACL 2020.
