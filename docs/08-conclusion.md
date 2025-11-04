# VIII Conclusion and Future directions
## VIII-A Conclusion

This survey provides a comprehensive evaluation of current approaches to world modeling, examining their relevance for robotic manipulation, underlying architectures, functionalities, key challenges, and proposed solutions. By synthesizing these findings, we offer insights into the nature of real world models and outline the efforts required to advance the field. Our goal is to provide readers with a solid foundation and guide future research directions in world modeling.

## VIII-B Future research directions

&emsp;&emsp;From our survey of current approaches and analysis of the core components and capabilities of world models, it is evident that present models fall short of accurately representing real-world phenomena. The limitations and the set of expected capabilities define promising directions for future research. To provide additional context, we also discuss several significant research directions.

**More Diverse Modalities**
The real world contains diverse forms of information, and no single sensory modality can capture its full complexity. This motivates world models capable of perceiving and integrating multiple modalities, including vision, language, action, touch, force, and proprioception, along with their interactions. Early progress has been made in this direction. For example, Hong et al. [@hong2024multiply] introduce the Multisensory-Universe dataset, which features interactive scenes enriched with tactile, audio, and temperature signals, generated with the assistance of ChatGPT [@achiam2023gpt].

**Hierarchical World Models**
Hierarchical systems play a critical role in building effective world models, as they allow agents to reason across multiple layers of abstraction. However, designing hierarchical models is inherently challenging: separating low-level and high-level dynamics is difficult, and coordinating interactions across layers adds further complexity. While existing studies primarily emphasize model design [@gumbsch2023learning;@lecun2022path;@wang2025dmwm;@xing2025critiques], their validation in complex real-world environments remains limited.

**Causality** is a fundamental principle for understanding and modeling the world, describing how events or factors influence outcomes and enabling reasoning about future consequences. Causality is the key to world model as it allows agents to interact with the world, which is inline with the human cognition. Richens et al. [@richens2024robust] indicate that learning a causal model is the key to ensure the generalization ability to new domains. Wang et al. [@wang2022causal;@tomar2021model] learn a causal dynamics model by removing unnecessary dependencies for tasks, which however are constrained to specific tasks. Gupta et al. [@gupta2024essential] argue that conventional theory-driven approaches to causal modeling, such as those in [@stuart2010matching, chernozhukov2018double], are insufficient for world models that aim for generalizable understanding. These methods rely on predefined variables and case-specific theoretical properties. In the real world, sensory inputs are complex, often unstructured, and key theoretical properties,such as identifiability, may not hold.

**Resource-Constrained Deployment**
Current world models, particularly those based on video generation, are computationally intensive and contain hundreds of millions of parameters, which limits their feasibility for real-world robotic deployment and on-device inference. To enable practical applications, designing lightweight and efficient world models has become increasingly important. Quantization and model compression techniques offer promising directions for reducing memory and computational costs, and have been extensively explored in related domains [@polino2018model;@gholami2022survey;@shang2023post;@li2021lightweight], providing both direct solutions and inspiration for future lightweight world model architectures.

**Fairness and Security**
As world models become integral to embodied agents and decision-making systems, ensuring their ethical alignment and fairness is critical. Unlike conventional vision or language models, world models directly influence how autonomous agents perceive, reason, and act within real environments, which amplifies the consequences of biased or unsafe representations. To handle this, emerging research explores bias auditing, fairness-aware training, and safety-constrained learning objectives to prevent harmful behaviors and unintended policy generalization.

Furthermore, deep models are known to be vulnerable to adversarial attacks, which can compromise performance by introducing imperceptible perturbations to inputs [@szegedy2013intriguing;@zhang2024universal], modifying model parameters [@ren2023dimension;@park2022blurs], or even exploiting hardware-level weaknesses [@cojocar2020we;@jattke2024zenhammer].
These vulnerabilities raise serious concerns regarding the security and reliability of world models, especially when deployed in safety-critical domains.
To date, systematic studies on the robustness and security of world models remain limited, underscoring an urgent need for dedicated research into adversarial resilience, trustworthy deployment, and secure model adaptation.
 
**Evaluation Protocols**
Current evaluation practices for world models are fragmented and only loosely aligned with their intended capabilities, often relying on task-specific or proxy metrics and partial human validation [@liao2025genie]. There is a pressing need for standardized benchmarks and unified evaluation frameworks that can comprehensively assess world model competence across multiple dimensions, including visual fidelity, policy success, causal consistency, physical plausibility, generalization, and long-horizon reasoning.

**Beyond Human Intelligence**
Insights from human cognition have profoundly influenced the design of robotic and world modeling systems. However, the completeness of the world extends beyond human cognition, which is bounded by partial observation, finite memory, limited attention, and inherent heuristic biases. World models are therefore expected to transcend human cognitive bounds, providing a deeper and more systematic understanding of complex environments.

## References
\bibliography