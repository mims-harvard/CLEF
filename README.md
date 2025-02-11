# CLEF: Controllable Sequence Editing for Counterfactual Generation

**Authors**:
- [Michelle M. Li](http://michellemli.com)
- [Kevin Li](https://www.linkedin.com/in/kevinli5941/)
- [Yasha Ektefaie](https://www.yashaektefaie.com)
- [Shvat Messica](https://www.linkedin.com/in/shvatmessica/)
- [Marinka Zitnik](http://zitniklab.hms.harvard.edu)

## Overview of CLEF

Counterfactual thinking is a fundamental objective in biology and medicine. "What if" scenarios are critical for reasoning about the underlying mechanisms of cells, patients, diseases, and drugs: "What if we treat the cells with the drug every one or 24 hours?" and "What if we perform the surgery on the patient today or next year?" We should reason about both the choice and timing of the counterfactual condition. Thus, counterfactual generation requires precise and context-specific edits that adhere to temporal and structural constraints.

Sequence models generate counterfactuals by modifying parts of a sequence based on a given condition, enabling reasoning about "what if" scenarios. While these models excel at conditional generation, they lack fine-grained control over when and where edits occur. Existing approaches either focus on univariate sequences or assume that interventions affect the entire sequence globally. However, many applications require precise, localized modifications, where interventions take effect only after a specified time and impact only a subset of co-occurring variables.

We develop CLEF, a controllable sequence editing approach for instance-wise counterfactual generation. CLEF learns temporal concepts that represent the trajectories of the sequences to enable accurate counterfactual generation guided by a given condition. We show that the learned temporal concepts help preserve temporal and structural constraints in the generated outputs. By design, CLEF is flexible with any type of sequential data encoder. We demonstrate through comprehensive experiments on four novel benchmark datasets in cellular reprogramming and patient immune dynamics that CLEF outperforms state-of-the-art models by up to 36.01% and 65.71% (MAE) on immediate and delayed sequence editing, respectively. We also show that any pretrained sequence encoder can gain controllable sequence editing capabilities when finetuned with CLEF. Moreover, CLEF outperforms baselines in zero-shot counterfactual generation of cellular trajectories by up to 14.45% and 63.19% (MAE) on immediate and delayed sequence editing, respectively. Further, precise edits via user interaction can be performed directly on CLEF's learned concepts. We demonstrate through real-world case studies that CLEF, given precise edits on specific temporal concepts, can generate realistic "healthy" counterfactual trajectories for patients originally with type 1 diabetes mellitus.

<p align="center">
<img src="img/clef_overview.png?raw=true" width="700" >
</p>



## Additional Resources

- [Paper](https://arxiv.org/abs/2502.03569)
- [Project Website (Coming Soon)]()

```
@article{li2025clef,
  title={Controllable Sequence Editing for Counterfactual Generation},
  author={Li, Michelle M and Li, Kevin and Ektefaie, Yasha and Messica, Shvat and Zitnik, Marinka},
  journal={arXiv:2502.03569},
  year={2025}
}
```


## Questions

Please leave a Github issue or contact Michelle Li at michelleli@g.harvard.edu.
