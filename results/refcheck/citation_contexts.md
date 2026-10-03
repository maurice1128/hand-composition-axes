# Every citing sentence per reference key (regenerated 2026-10-03 after the reference check)

## yang2026lamp

```bibtex
@misc{yang2026lamp,
  author        = {Yang, Xinye and Ma, Zhiyuan and Yu, Hongze and Chen, Yuanpei and Yang, Yaodong and Chai, Xiaojie and Chen, Xinlei and Yu, Chao},
  title         = {{LAMP}: Latent Motion Prior-Guided Real-World Learning for Dexterous Hand Manipulation},
  year          = {2026},
  eprint        = {2607.06323},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  howpublished  = {arXiv preprint arXiv:2607.06323}
}
```

Cited in 2 sentence(s):
- Priors with a continuous latent space [@yang2026lamp] and priors built from a bank of primitives are both used on the assumption that what is learned separately recombines, so that an action recorded with one tool and a tool recorded in another action remain usable together.
- On TACO the fine label is the tool-action-object triplet. ### 3.2 The measuring prior The prior is a latent motion prior in the sense of LAMP [@yang2026lamp], a Gaussian latent over a window of hand motion trained with a reconstruction term and a KL penalty; we do not propose this idea.

## yang2022oakink

```bibtex
@inproceedings{yang2022oakink,
  author    = {Yang, Lixin and Li, Kailin and Zhan, Xinyu and Wu, Fei and Xu, Anran and Liu, Liu and Lu, Cewu},
  title     = {{OakInk}: A Large-Scale Knowledge Repository for Understanding Hand-Object Interaction},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {20953--20962},
  year      = {2022},
  eprint    = {2203.15709},
  archivePrefix = {arXiv}
}
```

Cited in 2 sentence(s):
- Method ### 3.1 Datasets and hand representation We use four public datasets: OakInk-Image [@yang2022oakink] (770 trajectories at 30 fps, median 72 frames), GRAB [@taheri2020grab] (1,048 trajectories at 30 fps, median 254 frames, from subject archives s1 to s7 and s10), OakInk2 [@zhan2024oakink2] (609 trajectories, motion capture subsampled by 4 to 30 fps, median 613 frames) and TACO [@liu2024taco] (2,317 trajectories at 30 fps, median 148 frames).
- On OakInk-Image a trajectory's fine label is its object identity and intent; the object's category, functional class and affordance come from the dataset's own taxonomy [@yang2022oakink], and for one axis the capture subject joins the fine label.

## zhan2024oakink2

```bibtex
@inproceedings{zhan2024oakink2,
  author    = {Zhan, Xinyu and Yang, Lixin and Zhao, Yifei and Mao, Kangrui and Xu, Hanlin and Lin, Zenan and Li, Kailin and Lu, Cewu},
  title     = {{OAKINK2}: A Dataset of Bimanual Hands-Object Manipulation in Complex Task Completion},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {445--456},
  year      = {2024},
  eprint    = {2403.19417},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- Method ### 3.1 Datasets and hand representation We use four public datasets: OakInk-Image [@yang2022oakink] (770 trajectories at 30 fps, median 72 frames), GRAB [@taheri2020grab] (1,048 trajectories at 30 fps, median 254 frames, from subject archives s1 to s7 and s10), OakInk2 [@zhan2024oakink2] (609 trajectories, motion capture subsampled by 4 to 30 fps, median 613 frames) and TACO [@liu2024taco] (2,317 trajectories at 30 fps, median 148 frames).

## taheri2020grab

```bibtex
@inproceedings{taheri2020grab,
  author    = {Taheri, Omid and Ghorbani, Nima and Black, Michael J. and Tzionas, Dimitrios},
  title     = {{GRAB}: A Dataset of Whole-Body Human Grasping of Objects},
  booktitle = {Computer Vision -- ECCV 2020},
  series    = {Lecture Notes in Computer Science},
  volume    = {12349},
  publisher = {Springer International Publishing},
  pages     = {581--600},
  year      = {2020},
  doi       = {10.1007/978-3-030-58548-8_34},
  eprint    = {2008.11200},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- Method ### 3.1 Datasets and hand representation We use four public datasets: OakInk-Image [@yang2022oakink] (770 trajectories at 30 fps, median 72 frames), GRAB [@taheri2020grab] (1,048 trajectories at 30 fps, median 254 frames, from subject archives s1 to s7 and s10), OakInk2 [@zhan2024oakink2] (609 trajectories, motion capture subsampled by 4 to 30 fps, median 613 frames) and TACO [@liu2024taco] (2,317 trajectories at 30 fps, median 148 frames).

## liu2024taco

```bibtex
@inproceedings{liu2024taco,
  author    = {Liu, Yun and Yang, Haolin and Si, Xu and Liu, Ling and Li, Zipeng and Zhang, Yuxiang and Liu, Yebin and Yi, Li},
  title     = {{TACO}: Benchmarking Generalizable Bimanual Tool-{ACtion}-Object Understanding},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {21740--21751},
  year      = {2024},
  eprint    = {2401.08399},
  archivePrefix = {arXiv}
}
```

Cited in 5 sentence(s):
- The test is established in language [@lake2018generalization; @keysers2020measuring], vision [@materzynska2020something; @atzmon2020causal], robot manipulation [@gao2024efficient; @chen2025robohiman; @wang2026diagnosing] and hand-object interaction, where TACO holds out the tool-action-object triplet while keeping every tool in training [@liu2024taco] and verb-noun splits with held-out objects exist for H2O and FPHA [@tse2025collaborative].
- Held-out-triplet results for hand-object motion forecasting come from one partition and one run per split [@liu2024taco], so neither the size of this floor nor the spread over splits is known.
- In hand-object interaction, TACO evaluates motion forecasting on held-out triplets [@liu2024taco], and Tse et al.
- Method ### 3.1 Datasets and hand representation We use four public datasets: OakInk-Image [@yang2022oakink] (770 trajectories at 30 fps, median 72 frames), GRAB [@taheri2020grab] (1,048 trajectories at 30 fps, median 254 frames, from subject archives s1 to s7 and s10), OakInk2 [@zhan2024oakink2] (609 trajectories, motion capture subsampled by 4 to 30 fps, median 613 frames) and TACO [@liu2024taco] (2,317 trajectories at 30 fps, median 148 frames).
- On TACO the right hand usually holds the tool [@liu2024taco], so the label describes it.

## romero2017embodied

```bibtex
@article{romero2017embodied,
  author  = {Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
  title   = {Embodied Hands: Modeling and Capturing Hands and Bodies Together},
  journal = {ACM Transactions on Graphics},
  volume  = {36},
  number  = {6},
  pages   = {245:1--245:17},
  year    = {2017},
  doi     = {10.1145/3130800.3130883},
  note    = {Proc. SIGGRAPH Asia 2017},
  eprint  = {2201.02610},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- Each includes MANO hand parameters [@romero2017embodied].

## higgins2017beta

```bibtex
@inproceedings{higgins2017beta,
  author    = {Higgins, Irina and Matthey, Loic and Pal, Arka and Burgess, Christopher and Glorot, Xavier and Botvinick, Matthew and Mohamed, Shakir and Lerchner, Alexander},
  title     = {{β-VAE}: Learning Basic Visual Concepts With a Constrained Variational Framework},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2017},
  url       = {https://openreview.net/forum?id=Sy2fzU9gl}
}
```

Cited in 1 sentence(s):
- The prior trains as a VAE [@kingma2014auto] whose KL weight beta [@higgins2017beta] rises linearly to 1.0 over the first 12 epochs, for 120 epochs on windows cut every 4 frames, with mean-squared error on pose, a first-difference term weighted 0.1 and a free-bits floor [@kingma2016improved] of 0.02 nats per dimension.

## kingma2014auto

```bibtex
@inproceedings{kingma2014auto,
  author    = {Kingma, Diederik P. and Welling, Max},
  title     = {Auto-Encoding Variational {Bayes}},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2014},
  eprint    = {1312.6114},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- The prior trains as a VAE [@kingma2014auto] whose KL weight beta [@higgins2017beta] rises linearly to 1.0 over the first 12 epochs, for 120 epochs on windows cut every 4 frames, with mean-squared error on pose, a first-difference term weighted 0.1 and a free-bits floor [@kingma2016improved] of 0.02 nats per dimension.

## kingma2016improved

```bibtex
@inproceedings{kingma2016improved,
  author    = {Kingma, Diederik P. and Salimans, Tim and Jozefowicz, Rafal and Chen, Xi and Sutskever, Ilya and Welling, Max},
  title     = {Improved Variational Inference with Inverse Autoregressive Flow},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {29},
  year      = {2016},
  url       = {https://papers.nips.cc/paper_files/paper/2016/hash/ddeebdeefdb7e7e7a697e1c3e3d8ef54-Abstract.html},
  eprint    = {1606.04934},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- The prior trains as a VAE [@kingma2014auto] whose KL weight beta [@higgins2017beta] rises linearly to 1.0 over the first 12 epochs, for 120 epochs on windows cut every 4 frames, with mean-squared error on pose, a first-difference term weighted 0.1 and a free-bits floor [@kingma2016improved] of 0.02 nats per dimension.

## cho2014learning

```bibtex
@inproceedings{cho2014learning,
  author    = {Cho, Kyunghyun and van Merri{\"e}nboer, Bart and Gulcehre, Caglar and Bahdanau, Dzmitry and Bougares, Fethi and Schwenk, Holger and Bengio, Yoshua},
  title     = {Learning Phrase Representations Using {RNN} Encoder--Decoder for Statistical Machine Translation},
  booktitle = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages     = {1724--1734},
  address   = {Doha, Qatar},
  publisher = {Association for Computational Linguistics},
  year      = {2014},
  doi       = {10.3115/v1/D14-1179},
  eprint    = {1406.1078},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- Two-layer bidirectional gated recurrent units [@cho2014learning] of 256 units encode a 32-frame window of 27 degrees of freedom into a 12-dimensional Gaussian latent per frame, and unidirectional ones decode it under a bounded output (2,283,059 parameters).

## finegan2018improving

```bibtex
@inproceedings{finegan2018improving,
  author    = {Finegan-Dollak, Catherine and Kummerfeld, Jonathan K. and Zhang, Li and Ramanathan, Karthik and Sadasivam, Sesh and Zhang, Rui and Radev, Dragomir},
  title     = {Improving Text-to-{SQL} Evaluation Methodology},
  booktitle = {Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL, Volume 1: Long Papers)},
  pages     = {351--360},
  address   = {Melbourne, Australia},
  publisher = {Association for Computational Linguistics},
  year      = {2018},
  doi       = {10.18653/v1/P18-1033},
  eprint    = {1806.09029},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- [@finegan2018improving] show that splitting text-to-SQL data by question leaves the SQL queries being predicted in both training and test, and propose splitting by query instead.

## lake2018generalization

```bibtex
@inproceedings{lake2018generalization,
  author    = {Lake, Brenden and Baroni, Marco},
  title     = {Generalization without Systematicity: On the Compositional Skills of Sequence-to-Sequence Recurrent Networks},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {80},
  pages     = {2873--2882},
  publisher = {PMLR},
  year      = {2018},
  eprint    = {1711.00350},
  archivePrefix = {arXiv}
}
```

Cited in 3 sentence(s):
- The test is established in language [@lake2018generalization; @keysers2020measuring], vision [@materzynska2020something; @atzmon2020causal], robot manipulation [@gao2024efficient; @chen2025robohiman; @wang2026diagnosing] and hand-object interaction, where TACO holds out the tool-action-object triplet while keeping every tool in training [@liu2024taco] and verb-noun splits with held-out objects exist for H2O and FPHA [@tse2025collaborative].
- SCAN [@lake2018generalization] and CFQ [@keysers2020measuring] do so for language, and CFQ varies how far the test distribution diverges from training across several splits; Hupkes et al.
- SCAN contains a related case: models trained with "turn left" seen only as a bare command still generalise to its compositions, which the authors attribute to its output action appearing in training inside other commands, whereas they fail on "jump", whose action appears nowhere else [@lake2018generalization].

## keysers2020measuring

```bibtex
@inproceedings{keysers2020measuring,
  author    = {Keysers, Daniel and Sch{\"a}rli, Nathanael and Scales, Nathan and Buisman, Hylke and Furrer, Daniel and Kashubin, Sergii and Momchev, Nikola and Sinopalnikov, Danila and Stafiniak, Lukasz and Tihon, Tibor and Tsarkov, Dmitry and Wang, Xiao and van Zee, Marc and Bousquet, Olivier},
  title     = {Measuring Compositional Generalization: A Comprehensive Method on Realistic Data},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2020},
  url       = {https://openreview.net/forum?id=SygcCnNKwr},
  eprint    = {1912.09713},
  archivePrefix = {arXiv}
}
```

Cited in 2 sentence(s):
- The test is established in language [@lake2018generalization; @keysers2020measuring], vision [@materzynska2020something; @atzmon2020causal], robot manipulation [@gao2024efficient; @chen2025robohiman; @wang2026diagnosing] and hand-object interaction, where TACO holds out the tool-action-object triplet while keeping every tool in training [@liu2024taco] and verb-noun splits with held-out objects exist for H2O and FPHA [@tse2025collaborative].
- SCAN [@lake2018generalization] and CFQ [@keysers2020measuring] do so for language, and CFQ varies how far the test distribution diverges from training across several splits; Hupkes et al.

## hupkes2023taxonomy

```bibtex
@article{hupkes2023taxonomy,
  author  = {Hupkes, Dieuwke and Giulianelli, Mario and Dankers, Verna and Artetxe, Mikel and Elazar, Yanai and Pimentel, Tiago and Christodoulopoulos, Christos and Lasri, Karim and Saphra, Naomi and Sinclair, Arabella and Ulmer, Dennis and Schottmann, Florian and Batsuren, Khuyagbaatar and Sun, Kaiser and Sinha, Koustuv and Khalatbari, Leila and Ryskina, Maria and Frieske, Rita and Cotterell, Ryan and Jin, Zhijing},
  title   = {A Taxonomy and Review of Generalization Research in {NLP}},
  journal = {Nature Machine Intelligence},
  volume  = {5},
  number  = {10},
  pages   = {1161--1174},
  year    = {2023},
  doi     = {10.1038/s42256-023-00729-y},
  eprint  = {2210.03050},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- [@hupkes2023taxonomy] place such tests in a taxonomy of generalisation research.

## hupkes2020compositionality

```bibtex
@article{hupkes2020compositionality,
  author  = {Hupkes, Dieuwke and Dankers, Verna and Mul, Mathijs and Bruni, Elia},
  title   = {Compositionality Decomposed: How do Neural Networks Generalise?},
  journal = {Journal of Artificial Intelligence Research},
  volume  = {67},
  pages   = {757--795},
  year    = {2020},
  doi     = {10.1613/jair.1.11674},
  eprint  = {1908.08351},
  archivePrefix = {arXiv}
}
```

Cited in 0 sentence(s):
- (NOT CITED in the manuscript)

## sun2023validity

```bibtex
@inproceedings{sun2023validity,
  author    = {Sun, Kaiser and Williams, Adina and Hupkes, Dieuwke},
  title     = {The Validity of Evaluation Results: Assessing Concurrence Across Compositionality Benchmarks},
  booktitle = {Proceedings of the 27th Conference on Computational Natural Language Learning (CoNLL)},
  pages     = {274--293},
  address   = {Singapore},
  publisher = {Association for Computational Linguistics},
  year      = {2023},
  doi       = {10.18653/v1/2023.conll-1.19},
  eprint    = {2310.17514},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- [@sun2023validity] show that compositional benchmarks rank the same models inconsistently, even when they share a data source.

## atzmon2020causal

```bibtex
@inproceedings{atzmon2020causal,
  author    = {Atzmon, Yuval and Kreuk, Felix and Shalit, Uri and Chechik, Gal},
  title     = {A Causal View of Compositional Zero-Shot Recognition},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  volume    = {33},
  pages     = {1462--1473},
  year      = {2020},
  eprint    = {2006.14610},
  archivePrefix = {arXiv}
}
```

Cited in 2 sentence(s):
- The test is established in language [@lake2018generalization; @keysers2020measuring], vision [@materzynska2020something; @atzmon2020causal], robot manipulation [@gao2024efficient; @chen2025robohiman; @wang2026diagnosing] and hand-object interaction, where TACO holds out the tool-action-object triplet while keeping every tool in training [@liu2024taco] and verb-noun splits with held-out objects exist for H2O and FPHA [@tse2025collaborative].
- [@atzmon2020causal] audit label quality on an observational compositional benchmark, and Sun et al.

## nikolaus2019compositional

```bibtex
@inproceedings{nikolaus2019compositional,
  author    = {Nikolaus, Mitja and Abdou, Mostafa and Lamm, Matthew and Aralikatte, Rahul and Elliott, Desmond},
  title     = {Compositional Generalization in Image Captioning},
  booktitle = {Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL)},
  pages     = {87--98},
  address   = {Hong Kong, China},
  publisher = {Association for Computational Linguistics},
  year      = {2019},
  doi       = {10.18653/v1/K19-1009}
}
```

Cited in 0 sentence(s):
- (NOT CITED in the manuscript)

## materzynska2020something

```bibtex
@inproceedings{materzynska2020something,
  author    = {Materzynska, Joanna and Xiao, Tete and Herzig, Roei and Xu, Huijuan and Wang, Xiaolong and Darrell, Trevor},
  title     = {Something-Else: Compositional Action Recognition With Spatial-Temporal Interaction Networks},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1049--1059},
  year      = {2020},
  eprint    = {1912.09930},
  archivePrefix = {arXiv}
}
```

Cited in 2 sentence(s):
- The test is established in language [@lake2018generalization; @keysers2020measuring], vision [@materzynska2020something; @atzmon2020causal], robot manipulation [@gao2024efficient; @chen2025robohiman; @wang2026diagnosing] and hand-object interaction, where TACO holds out the tool-action-object triplet while keeping every tool in training [@liu2024taco] and verb-noun splits with held-out objects exist for H2O and FPHA [@tse2025collaborative].
- In vision, Something-Else holds out verb-object combinations in video [@materzynska2020something].

## montero2021role

```bibtex
@inproceedings{montero2021role,
  author    = {Montero, Milton Llera and Ludwig, Casimir J. H. and Costa, Rui Ponte and Malhotra, Gaurav and Bowers, Jeffrey},
  title     = {The Role of Disentanglement in Generalisation},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2021},
  url       = {https://openreview.net/forum?id=qbH974jKUVy}
}
```

Cited in 0 sentence(s):
- (NOT CITED in the manuscript)

## montero2022lost

```bibtex
@inproceedings{montero2022lost,
  author    = {Montero, Milton and Bowers, Jeffrey and Ponte Costa, Rui and Ludwig, Casimir and Malhotra, Gaurav},
  title     = {Lost in Latent Space: Examining Failures of Disentangled Models at Combinatorial Generalisation},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {10136--10149},
  year      = {2022},
  doi       = {10.52202/068431-0736},
  eprint    = {2204.02283},
  archivePrefix = {arXiv}
}
```

Cited in 0 sentence(s):
- (NOT CITED in the manuscript)

## schott2022visual

```bibtex
@inproceedings{schott2022visual,
  author    = {Schott, Lukas and von K{\"u}gelgen, Julius and Tr{\"a}uble, Frederik and Gehler, Peter and Russell, Chris and Bethge, Matthias and Sch{\"o}lkopf, Bernhard and Locatello, Francesco and Brendel, Wieland},
  title     = {Visual Representation Learning Does Not Generalize Strongly Within the Same Domain},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  url       = {https://openreview.net/forum?id=9RUHPlladgh},
  eprint    = {2107.08221},
  archivePrefix = {arXiv}
}
```

Cited in 0 sentence(s):
- (NOT CITED in the manuscript)

## tse2025collaborative

```bibtex
@inproceedings{tse2025collaborative,
  author    = {Tse, Tze Ho Elden and Feng, Runyang and Zheng, Linfang and Park, Jiho and Gao, Yixing and Kim, Jihie and Leonardis, Ale\v{s} and Chang, Hyung Jin},
  title     = {Collaborative Learning for {3D} Hand-Object Reconstruction and Compositional Action Recognition from Egocentric {RGB} Videos Using Superquadrics},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {39},
  number    = {7},
  pages     = {7437--7445},
  year      = {2025},
  doi       = {10.1609/aaai.v39i7.32800},
  eprint    = {2501.07100},
  archivePrefix = {arXiv}
}
```

Cited in 2 sentence(s):
- The test is established in language [@lake2018generalization; @keysers2020measuring], vision [@materzynska2020something; @atzmon2020causal], robot manipulation [@gao2024efficient; @chen2025robohiman; @wang2026diagnosing] and hand-object interaction, where TACO holds out the tool-action-object triplet while keeping every tool in training [@liu2024taco] and verb-noun splits with held-out objects exist for H2O and FPHA [@tse2025collaborative].
- [@tse2025collaborative] build verb-noun splits on H2O and FPHA for action recognition, removing every training sequence that contains the held-out object.

## gao2024efficient

```bibtex
@inproceedings{gao2024efficient,
  author    = {Gao, Jensen and Xie, Annie and Xiao, Ted and Finn, Chelsea and Sadigh, Dorsa},
  title     = {Efficient Data Collection for Robotic Manipulation via Compositional Generalization},
  booktitle = {Robotics: Science and Systems (RSS) XX},
  year      = {2024},
  doi       = {10.15607/RSS.2024.XX.013},
  eprint    = {2403.05110},
  archivePrefix = {arXiv}
}
```

Cited in 2 sentence(s):
- The test is established in language [@lake2018generalization; @keysers2020measuring], vision [@materzynska2020something; @atzmon2020causal], robot manipulation [@gao2024efficient; @chen2025robohiman; @wang2026diagnosing] and hand-object interaction, where TACO holds out the tool-action-object triplet while keeping every tool in training [@liu2024taco] and verb-noun splits with held-out objects exist for H2O and FPHA [@tse2025collaborative].
- In robot learning, unseen combinations of environmental factors [@gao2024efficient], of atomic skills [@chen2025robohiman] and of instruction components [@wang2026diagnosing] evaluate policies.

## wang2026diagnosing

```bibtex
@misc{wang2026diagnosing,
  author        = {Wang, Yixiao and Wu, Cheng-En and Sun, Lingfeng and Wang, Pengcheng and Ji, Xiang and Liang, Boyuan and Zhan, Guojian and Tomizuka, Masayoshi},
  title         = {Diagnosing Compositional Generalization in Sequential Robot Tasks},
  year          = {2026},
  eprint        = {2607.29687},
  archivePrefix = {arXiv},
  howpublished  = {arXiv preprint arXiv:2607.29687}
}
```

Cited in 2 sentence(s):
- The test is established in language [@lake2018generalization; @keysers2020measuring], vision [@materzynska2020something; @atzmon2020causal], robot manipulation [@gao2024efficient; @chen2025robohiman; @wang2026diagnosing] and hand-object interaction, where TACO holds out the tool-action-object triplet while keeping every tool in training [@liu2024taco] and verb-noun splits with held-out objects exist for H2O and FPHA [@tse2025collaborative].
- In robot learning, unseen combinations of environmental factors [@gao2024efficient], of atomic skills [@chen2025robohiman] and of instruction components [@wang2026diagnosing] evaluate policies.

## chen2025robohiman

```bibtex
@misc{chen2025robohiman,
  author        = {Chen, Yangtao and Chen, Zixuan and Chan, Nga Teng and Chen, Junting and Yin, Junhui and Shi, Jieqi and Gao, Yang and Li, Yong-Lu and Huo, Jing},
  title         = {{RoboHiMan}: A Hierarchical Evaluation Paradigm for Compositional Generalization in Long-Horizon Manipulation},
  year          = {2025},
  eprint        = {2510.13149},
  archivePrefix = {arXiv},
  howpublished  = {arXiv preprint arXiv:2510.13149}
}
```

Cited in 2 sentence(s):
- The test is established in language [@lake2018generalization; @keysers2020measuring], vision [@materzynska2020something; @atzmon2020causal], robot manipulation [@gao2024efficient; @chen2025robohiman; @wang2026diagnosing] and hand-object interaction, where TACO holds out the tool-action-object triplet while keeping every tool in training [@liu2024taco] and verb-noun splits with held-out objects exist for H2O and FPHA [@tse2025collaborative].
- In robot learning, unseen combinations of environmental factors [@gao2024efficient], of atomic skills [@chen2025robohiman] and of instruction components [@wang2026diagnosing] evaluate policies.

## punnakkal2021babel

```bibtex
@inproceedings{punnakkal2021babel,
  author    = {Punnakkal, Abhinanda R. and Chandrasekaran, Arjun and Athanasiou, Nikos and Quiros-Ramirez, Alejandra and Black, Michael J.},
  title     = {{BABEL}: Bodies, Action and Behavior With {English} Labels},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {722--731},
  year      = {2021},
  doi       = {10.1109/CVPR46437.2021.00078},
  eprint    = {2106.09696},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- For motion data, a single sequence label can leave most of a motion-capture sequence undescribed: in BABEL's worked example the labelled action covers 20 % of the sequence [@punnakkal2021babel], and the temporal bounds of object interactions in egocentric video change recognition accuracy [@moltisanti2017trespassing].

## moltisanti2017trespassing

```bibtex
@inproceedings{moltisanti2017trespassing,
  author    = {Moltisanti, Davide and Wray, Michael and Mayol-Cuevas, Walterio and Damen, Dima},
  title     = {Trespassing the Boundaries: Labeling Temporal Bounds for Object Interactions in Egocentric Video},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages     = {2886--2894},
  year      = {2017},
  doi       = {10.1109/ICCV.2017.314},
  eprint    = {1703.09026},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- For motion data, a single sequence label can leave most of a motion-capture sequence undescribed: in BABEL's worked example the labelled action covers 20 % of the sequence [@punnakkal2021babel], and the temporal bounds of object interactions in egocentric video change recognition accuracy [@moltisanti2017trespassing].

## gowda2021new

```bibtex
@inproceedings{gowda2021new,
  author    = {Gowda, Shreyank N. and Sevilla-Lara, Laura and Kim, Kiyoon and Keller, Frank and Rohrbach, Marcus},
  title     = {A New Split for Evaluating True Zero-Shot Action Recognition},
  booktitle = {Pattern Recognition -- DAGM GCPR 2021},
  series    = {Lecture Notes in Computer Science},
  volume    = {13024},
  publisher = {Springer International Publishing},
  pages     = {191--205},
  year      = {2021},
  doi       = {10.1007/978-3-030-92659-5_12},
  eprint    = {2107.13029},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- In zero-shot action recognition, classes treated as unseen reach the training or pre-training data under other names [@roitberg2018towards; @brattoli2020rethinking; @gowda2021new], an overlap decided class by class, by name or by visual similarity.

## brattoli2020rethinking

```bibtex
@inproceedings{brattoli2020rethinking,
  author    = {Brattoli, Biagio and Tighe, Joseph and Zhdanov, Fedor and Perona, Pietro and Chalupka, Krzysztof},
  title     = {Rethinking Zero-Shot Video Classification: End-to-End Training for Realistic Applications},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {4613--4623},
  year      = {2020},
  doi       = {10.1109/CVPR42600.2020.00467},
  eprint    = {2003.01455},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- In zero-shot action recognition, classes treated as unseen reach the training or pre-training data under other names [@roitberg2018towards; @brattoli2020rethinking; @gowda2021new], an overlap decided class by class, by name or by visual similarity.

## roitberg2018towards

```bibtex
@inproceedings{roitberg2018towards,
  author    = {Roitberg, Alina and Martinez, Manuel and Haurilet, Monica and Stiefelhagen, Rainer},
  title     = {Towards a Fair Evaluation of Zero-Shot Action Recognition Using External Data},
  booktitle = {Computer Vision -- ECCV 2018 Workshops},
  series    = {Lecture Notes in Computer Science},
  volume    = {11132},
  publisher = {Springer International Publishing},
  pages     = {97--105},
  year      = {2019},
  doi       = {10.1007/978-3-030-11018-5_8}
}
```

Cited in 1 sentence(s):
- In zero-shot action recognition, classes treated as unseen reach the training or pre-training data under other names [@roitberg2018towards; @brattoli2020rethinking; @gowda2021new], an overlap decided class by class, by name or by visual similarity.

## bansal2018zero

```bibtex
@inproceedings{bansal2018zero,
  author    = {Bansal, Ankan and Sikka, Karan and Sharma, Gaurav and Chellappa, Rama and Divakaran, Ajay},
  title     = {Zero-Shot Object Detection},
  booktitle = {Computer Vision -- ECCV 2018},
  series    = {Lecture Notes in Computer Science},
  volume    = {11205},
  publisher = {Springer International Publishing},
  pages     = {397--414},
  year      = {2018},
  doi       = {10.1007/978-3-030-01246-5_24},
  eprint    = {1804.04340},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- Zero-shot detection removes training images that contain unseen objects where the annotation allows it, on MS-COCO but not on Visual Genome [@bansal2018zero], and zero-shot temporal activity detection removes training clips that contain an unseen activity [@zhang2020zstad]; both state this as protocol rather than measure what happens without it.

## zhang2020zstad

```bibtex
@inproceedings{zhang2020zstad,
  author    = {Zhang, Lingling and Chang, Xiaojun and Liu, Jun and Luo, Minnan and Wang, Sen and Ge, Zongyuan and Hauptmann, Alexander},
  title     = {{ZSTAD}: Zero-Shot Temporal Activity Detection},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {876--885},
  year      = {2020},
  doi       = {10.1109/CVPR42600.2020.00096},
  eprint    = {2003.05583},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- Zero-shot detection removes training images that contain unseen objects where the annotation allows it, on MS-COCO but not on Visual Genome [@bansal2018zero], and zero-shot temporal activity detection removes training clips that contain an unseen activity [@zhang2020zstad]; both state this as protocol rather than measure what happens without it.

## fu2025gigahands

```bibtex
@inproceedings{fu2025gigahands,
  author    = {Fu, Rao and Zhang, Dingxi and Jiang, Alex and Fu, Wanjia and Funk, Austin and Ritchie, Daniel and Sridhar, Srinath},
  title     = {{GigaHands}: A Massive Annotated Dataset of Bimanual Hand Activities},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {17461--17474},
  year      = {2025},
  doi       = {10.1109/CVPR52734.2025.01627},
  eprint    = {2412.04244},
  archivePrefix = {arXiv}
}
```

Cited in 0 sentence(s):
- (NOT CITED in the manuscript)

## kapoor2023leakage

```bibtex
@article{kapoor2023leakage,
  author  = {Kapoor, Sayash and Narayanan, Arvind},
  title   = {Leakage and the Reproducibility Crisis in Machine-Learning-Based Science},
  journal = {Patterns},
  volume  = {4},
  number  = {9},
  pages   = {100804},
  year    = {2023},
  doi     = {10.1016/j.patter.2023.100804},
  eprint  = {2207.07048},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- A survey of leakage in machine-learning-based science does not single out partial label coverage among its sources [@kapoor2023leakage].

## lipsitch2010negative

```bibtex
@article{lipsitch2010negative,
  author  = {Lipsitch, Marc and Tchetgen Tchetgen, Eric and Cohen, Ted},
  title   = {Negative Controls: A Tool for Detecting Confounding and Bias in Observational Studies},
  journal = {Epidemiology},
  volume  = {21},
  number  = {3},
  pages   = {383--388},
  year    = {2010},
  doi     = {10.1097/EDE.0b013e3181d61eeb}
}
```

Cited in 1 sentence(s):
- A condition whose true effect is zero is the negative control of epidemiology [@lipsitch2010negative] and the basis of empirical calibration of estimates [@schuemie2014interpreting; @schuemie2018empirical]; in machine learning, Engstrom et al.

## schuemie2014interpreting

```bibtex
@article{schuemie2014interpreting,
  author  = {Schuemie, Martijn J. and Ryan, Patrick B. and DuMouchel, William and Suchard, Marc A. and Madigan, David},
  title   = {Interpreting Observational Studies: Why Empirical Calibration Is Needed to Correct {p}-Values},
  journal = {Statistics in Medicine},
  volume  = {33},
  number  = {2},
  pages   = {209--218},
  year    = {2014},
  doi     = {10.1002/sim.5925}
}
```

Cited in 1 sentence(s):
- A condition whose true effect is zero is the negative control of epidemiology [@lipsitch2010negative] and the basis of empirical calibration of estimates [@schuemie2014interpreting; @schuemie2018empirical]; in machine learning, Engstrom et al.

## schuemie2018empirical

```bibtex
@article{schuemie2018empirical,
  author  = {Schuemie, Martijn J. and Hripcsak, George and Ryan, Patrick B. and Madigan, David and Suchard, Marc A.},
  title   = {Empirical Confidence Interval Calibration for Population-Level Effect Estimation Studies in Observational Healthcare Data},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {115},
  number  = {11},
  pages   = {2571--2577},
  year    = {2018},
  doi     = {10.1073/pnas.1708282114}
}
```

Cited in 1 sentence(s):
- A condition whose true effect is zero is the negative control of epidemiology [@lipsitch2010negative] and the basis of empirical calibration of estimates [@schuemie2014interpreting; @schuemie2018empirical]; in machine learning, Engstrom et al.

## ojala2010permutation

```bibtex
@article{ojala2010permutation,
  author  = {Ojala, Markus and Garriga, Gemma C.},
  title   = {Permutation Tests for Studying Classifier Performance},
  journal = {Journal of Machine Learning Research},
  volume  = {11},
  number  = {62},
  pages   = {1833--1863},
  year    = {2010},
  url     = {https://jmlr.org/papers/v11/ojala10a.html}
}
```

Cited in 1 sentence(s):
- Permutation nulls that preserve structure are standard for classifier evaluation [@ojala2010permutation], and shuffle and random-geometry nulls calibrate cross-condition generalisation in neuroscience [@bernardi2020geometry].

## bernardi2020geometry

```bibtex
@article{bernardi2020geometry,
  author  = {Bernardi, Silvia and Benna, Marcus K. and Rigotti, Mattia and Munuera, J{\'e}r{\^o}me and Fusi, Stefano and Salzman, C. Daniel},
  title   = {The Geometry of Abstraction in the Hippocampus and Prefrontal Cortex},
  journal = {Cell},
  volume  = {183},
  number  = {4},
  pages   = {954--967.e21},
  year    = {2020},
  doi     = {10.1016/j.cell.2020.09.031}
}
```

Cited in 1 sentence(s):
- Permutation nulls that preserve structure are standard for classifier evaluation [@ojala2010permutation], and shuffle and random-geometry nulls calibrate cross-condition generalisation in neuroscience [@bernardi2020geometry].

## petigura2013prevalence

```bibtex
@article{petigura2013prevalence,
  author  = {Petigura, Erik A. and Howard, Andrew W. and Marcy, Geoffrey W.},
  title   = {Prevalence of {Earth}-Size Planets Orbiting {Sun}-Like Stars},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {110},
  number  = {48},
  pages   = {19273--19278},
  year    = {2013},
  doi     = {10.1073/pnas.1319909110}
}
```

Cited in 1 sentence(s):
- Planting an effect of known size and measuring its recovery is the injection-recovery test of astronomy [@petigura2013prevalence], the spike-in of genomics [@jiang2011synthetic] and, in machine learning, the planted artefact [@adebayo2022post].

## christiansen2013measuring

```bibtex
@article{christiansen2013measuring,
  author  = {Christiansen, Jessie L. and Clarke, Bruce D. and Burke, Christopher J. and Jenkins, Jon M. and Barclay, Thomas S. and Ford, Eric B. and Haas, Michael R. and Sabale, Anima and Seader, Shawn and Smith, Jeffrey Claiborne and Tenenbaum, Peter and Twicken, Joseph D. and Uddin, Akm Kamal and Thompson, Susan E.},
  title   = {Measuring Transit Signal Recovery in the {Kepler} Pipeline. {I}. {Individual} Events},
  journal = {The Astrophysical Journal Supplement Series},
  volume  = {207},
  number  = {2},
  pages   = {35},
  year    = {2013},
  doi     = {10.1088/0067-0049/207/2/35},
  eprint  = {1303.0255},
  archivePrefix = {arXiv}
}
```

Cited in 0 sentence(s):
- (NOT CITED in the manuscript)

## jiang2011synthetic

```bibtex
@article{jiang2011synthetic,
  author  = {Jiang, Lichun and Schlesinger, Felix and Davis, Carrie A. and Zhang, Yu and Li, Renhua and Salit, Marc and Gingeras, Thomas R. and Oliver, Brian},
  title   = {Synthetic Spike-in Standards for {RNA}-seq Experiments},
  journal = {Genome Research},
  volume  = {21},
  number  = {9},
  pages   = {1543--1551},
  year    = {2011},
  doi     = {10.1101/gr.121095.111}
}
```

Cited in 1 sentence(s):
- Planting an effect of known size and measuring its recovery is the injection-recovery test of astronomy [@petigura2013prevalence], the spike-in of genomics [@jiang2011synthetic] and, in machine learning, the planted artefact [@adebayo2022post].

## engstrom2020identifying

```bibtex
@inproceedings{engstrom2020identifying,
  author    = {Engstrom, Logan and Ilyas, Andrew and Santurkar, Shibani and Tsipras, Dimitris and Steinhardt, Jacob and Madry, Aleksander},
  title     = {Identifying Statistical Bias in Dataset Replication},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning (ICML)},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {2922--2932},
  publisher = {PMLR},
  year      = {2020},
  eprint    = {2005.09619},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- [@engstrom2020identifying] measure the bias of a replicated test set in the same spirit.

## adebayo2022post

```bibtex
@inproceedings{adebayo2022post,
  author    = {Adebayo, Julius and Muelly, Michael and Abelson, Hal and Kim, Been},
  title     = {Post Hoc Explanations May Be Ineffective for Detecting Unknown Spurious Correlation},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2022},
  url       = {https://openreview.net/forum?id=xNOVfCCvDpM},
  eprint    = {2212.04629},
  archivePrefix = {arXiv}
}
```

Cited in 1 sentence(s):
- Planting an effect of known size and measuring its recovery is the injection-recovery test of astronomy [@petigura2013prevalence], the spike-in of genomics [@jiang2011synthetic] and, in machine learning, the planted artefact [@adebayo2022post].

## welch1947generalization

```bibtex
@article{welch1947generalization,
  author  = {Welch, B. L.},
  title   = {The Generalization of ‘{Student's}’ Problem When Several Different Population Variances Are Involved},
  journal = {Biometrika},
  volume  = {34},
  number  = {1--2},
  pages   = {28--35},
  year    = {1947},
  doi     = {10.1093/biomet/34.1-2.28}
}
```

Cited in 1 sentence(s):
- Two-sided Welch tests [@welch1947generalization] on per-seed values compare penalties, with 95 % confidence intervals on the difference of means, against the zero-truth control at the same budget wherever one exists; the two stride-32 GRAB sweeps are compared with their unmodified sweep, and the budget-896 OakInk2 sweep with whole recordings.

## demsar2006statistical

```bibtex
@article{demsar2006statistical,
  author  = {Dem{\v{s}}ar, Janez},
  title   = {Statistical Comparisons of Classifiers over Multiple Data Sets},
  journal = {Journal of Machine Learning Research},
  volume  = {7},
  number  = {1},
  pages   = {1--30},
  year    = {2006},
  url     = {https://jmlr.org/papers/v7/demsar06a.html}
}
```

Cited in 0 sentence(s):
- (NOT CITED in the manuscript)

## bouthillier2021accounting

```bibtex
@inproceedings{bouthillier2021accounting,
  author    = {Bouthillier, Xavier and Delaunay, Pierre and Bronzi, Mirko and Trofimov, Assya and Nichyporuk, Brennan and Szeto, Justin and Mohammadi Sepahvand, Nazanin and Raff, Edward and Madan, Kanika and Voleti, Vikram and Ebrahimi Kahou, Samira and Michalski, Vincent and Arbel, Tal and Pal, Chris and Varoquaux, Ga{\"e}l and Vincent, Pascal},
  title     = {Accounting for Variance in Machine Learning Benchmarks},
  booktitle = {Proceedings of Machine Learning and Systems (MLSys)},
  volume    = {3},
  pages     = {747--769},
  year      = {2021},
  eprint    = {2103.03098},
  archivePrefix = {arXiv}
}
```

Cited in 0 sentence(s):
- (NOT CITED in the manuscript)

## dietterich1998approximate

```bibtex
@article{dietterich1998approximate,
  author  = {Dietterich, Thomas G.},
  title   = {Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms},
  journal = {Neural Computation},
  volume  = {10},
  number  = {7},
  pages   = {1895--1923},
  year    = {1998},
  doi     = {10.1162/089976698300017197}
}
```

Cited in 0 sentence(s):
- (NOT CITED in the manuscript)
