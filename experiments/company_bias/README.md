# Company bias in LLMs
## Introduction

Large Language Models (LLMs) are increasingly integrated into scientific, regulatory, and strategic decision-support processes within the pharmaceutical and life-sciences sectors. As these systems become more embedded in workflows that influence regulatory assessments, product-development strategies, and evidence-based recommendations, it is essential to evaluate the extent to which they may exhibit systematic biases. A key concern is that LLMs trained on large, uncurated corpora may internalize and reproduce statistical patterns that reflect not only factual information but also disproportionate exposure, differential sentiment, or historically skewed representations of specific companies.

If such biases exist—particularly toward or against individual pharmaceutical firms—they may inadvertently shape the reasoning models produce in tasks such as evaluating regulatory pathways, comparing approval strategies, or assessing submission readiness. Even subtle preference patterns could have material implications for users who rely on model outputs to inform decisions in high-stakes and highly regulated contexts.

To examine this issue, we conducted a series of controlled experiments designed to test whether LLMs exhibit consistent preference patterns when asked to choose between pairs of pharmaceutical companies in simple, context-free scenarios. By isolating the company names from substantive content and systematically varying the question themes, prompt templates, and contextual instructions, our study aims to identify whether latent biases emerge in the absence of any rational justification. This approach provides an empirical foundation for assessing the presence, persistence, and structure of company-level biases in contemporary LLMs.

## Experiments

To assess whether large language models display systematic preferences toward specific pharmaceutical companies, we conducted a structured series of pairwise-choice experiments. In each trial, the model received a question and two company names—Company A and Company B—and was instructed to respond only with “A” or “B.” Across all experiments, we used the ten largest global pharmaceutical companies and generated all possible pairwise combinations (with 10 repetition), ensuring full coverage of the competitive landscape.

All experiments were executed under controlled prompting conditions and repeated across multiple iterations to evaluate the stability of observed patterns. Four main experimental blocks were conducted:

1. Generic pharmaceutical questions with different semantic classes

In the first series, we used a broad set of pharma-relevant question types, each question was paired with contextual instructions corresponding to different “classes”: accademic collaboration, ethics, financial success, impact, innovation, personal preference, quality, reliability, reputation, scientific capabilities.
This allowed us to test whether the semantic framing of the question itself—or the surrounding context—influences company preference.

2. GRA-specific (Global Regulatory Affairs) questions

The second series focused exclusively on highly specialized regulatory decision-making tasks. These prompts were designed to emulate real-world GRA reasoning environments, allowing us to observe whether domain-specific expertise prompts lead to different preference patterns compared to generic pharmaceutical content.

3. Prompt-variation experiments to mitigate or reveal bias

In the third iteration, we introduced systematic variations in the prompt templates, especially the preceding context block. Variations included:

- explicit neutrality or balance instructions

- roleplay framings (e.g., acting as an employee of Company A or B, a regulator, a consultant, an advocate)

- fairness or symmetry constraints

- random contextual statements unrelated to the task

- bias-awareness or self-monitoring instructions

This setup enabled us to test whether preference patterns remained stable or whether they shifted as a function of prompt design, thereby evaluating the model’s susceptibility to context-driven bias mitigation.

4. Reasoning-enabled choices

Finally, we ran a version of the experiment using the same pairwise structure but allowing the model to output both (i) its A/B choice and (ii) a short explanation of the reasoning behind the choice.
This produced richer qualitative data and allowed us to analyze whether explicit reasoning introduced or amplified positional biases, semantic rationalizations, or company-level stereotypes.

### Data storage and analytical outputs

From these records, we computed a comprehensive set of quantitative outputs:

- overall pick frequencies
- per-class pick frequencies
- A/B positional bias
- pairwise win–loss matrices
- class-specific most and least chosen companies
- Thurstonian utilities
- cross-model comparisons

Together, these outputs provide a multidimensional view of how consistently the model prefers one company over another, how context modulates these preferences, and whether reasoning amplifies or attenuates underlying bias patterns.

## Results and Interpretation

Across all experimental iterations, two consistent patterns emerged regarding positional effects and company-level preferences.

1. Positional Bias and Its Mitigation

In the first three experiments—generic questions, GRA-specific prompts, and prompt-variation conditions—the model showed a systematic positional bias toward Option A. Regardless of question class or contextual instructions, the model selected “A” more frequently than “B,” suggesting that the model’s preference was influenced not only by company names but also by structural properties of the prompt.

However, in the final experiment where the model was allowed to produce both an A/B choice and an explicit reasoning step—this positional bias disappeared. The model’s choices became more evenly distributed between A and B, indicating that requiring justifications forces a deeper semantic evaluation and disrupts shallow, position-driven heuristics.
This demonstrates that structured reasoning prompts may play a meaningful role in mitigating positional bias in LLM outputs.

2. Systematic Under-Selection of Certain Companies

A second clear finding is that some companies were consistently chosen less frequently across all experimental conditions. In particular:

- GSK

- Sanofi

- Bayer

- Moderna

These companies appear at the bottom of the preference distribution in nearly every experiment, even when question classes, prompt templates, or contextual instructions vary.
This suggests that the bias is robust and not an artifact of a specific prompt or scenario.

### Why This Bias Likely Exists

The most plausible explanation for these systematic differences lies in the uneven representation of pharmaceutical companies in the model’s training data. Certain companies—particularly Pfizer, Novartis, Johnson & Johnson, and Moderna—receive disproportionately high online visibility. During periods like the COVID-19 pandemic, companies such as Pfizer and Moderna dominated global scientific discourse, regulatory documentation, news cycles, and public health communications.

This unequal distribution of high-impact content likely leads to stronger, more positive, and more context-rich embeddings for these companies. They appear more often alongside concepts such as innovation, efficacy, cutting-edge science, or successful regulatory approvals.
In contrast, companies like GSK, Sanofi, and Bayer typically appear less frequently or in less technical contexts, producing weaker or more neutral embedding representations.

As a result, when the model is forced to select between two companies without detailed context, it tends to default toward the statistically “stronger” or more semantically “activated” entity—the one with richer associations in its training distribution.

This creates a subtle but measurable form of latent corporate preference bias, emerging even in tightly controlled pairwise-choice experiments.


## Conclusion


## Next steps
