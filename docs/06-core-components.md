# **VI Towards Defining Core Components and Capabilities of World Models**

&emsp;&emsp;From our survey of current approaches, we summarize some potential key components and capabilities that a world model should possess. Future research may identify additional dimensions necessary for comprehensive world modeling.  

&emsp;&emsp;**1) Comprehensive Multimodal Perception.**World models should be capable of perceiving and integrating information across all available modalities, such as vision, language, action, touch, force, and proprioception, along with the spatial and temporal structures. By jointly modeling these modalities and dimensions, they can construct a unified and dynamic understanding of the environment that facilitate decision-making and support robot training. 

&emsp;&emsp;**2) Interactivity.** World models should engage dynamically with their environments, not merely by passively observing or predicting changes, but by modeling how actions influence future states. Such action-conditioned dynamics enable agents to simulate interactions, evaluate potential outcomes, and plan behaviors grounded in causal understanding of the world. 

&emsp;&emsp;**3) Imagination.**Imagination enables world models to simulate and evaluate possible futures, allowing agents to learn, plan, and reason without external interaction. 

&emsp;&emsp;**4) Long-horizon Reasoning.** It enables world models to anticipate distant consequences of actions, plan multi-step behaviors, and optimize long-term outcomes rather than short-term rewards. 

&emsp;&emsp;**5) Spatiotemporal Reasoning.**World models should reason about spatial and temporal relationships among entities to understand and predict dynamic changes in the environment.  

&emsp;&emsp;**6) Counterfactual Reasoning.**This enables world models to imagine alternative futures under different actions, allowing agents to evaluate possible outcomes and select the most effective course of action. 

&emsp;&emsp;**7) Abstract Reasoning.**The world is immensely complex, and world models cannot capture every detail. Therefore, they must extract and represent the underlying principles and basic mechanisms that govern the world’s dynamics. 

&emsp;&emsp;**8) High-fidelity Prediction.** World models should generate accurate and detailed predictions of future states or observations, maintaining spatial, temporal, and physical consistency to ensure reliable simulation and planning.  

&emsp;&emsp;**9) Physics Awareness.** World models should maintain consistency with physical principles, enabling them to generate dynamically plausible predictions that support safe and reliable robotic interaction. 

&emsp;&emsp;**10) Generalization Ability.** To operate effectively in complex real-world settings, world models must generalize beyond their training distributions, adapting to new tasks, objects, and domains. 

&emsp;&emsp;**11) Causality.** World models should understand relationship between actions (causes) and their effects (outcomes) in the world. This causal understanding enables agents to predict how interventions will change future states, distinguish correlation from true influence, and generalize their behavior to unseen situations by reasoning about cause–effect mechanisms rather than memorized patterns. 

&emsp;&emsp;**12) Memory.**It enables world models to store and recall past experiences, ensuring temporal consistency and coherent predictions. In addition, world models should be able to access and integrate external information, thereby supporting richer reasoning, long-term planning, and adaptability—analogous to the role of retrieval-augmented generation (RAG) in language models. 

&emsp;&emsp;**13) Collaboration Ability.** World models should support both inter-agent and intra-agent coordination by reasoning about the behaviors, goals, and intentions of others and managing cooperation among multiple effectors (e.g., multi-arm systems).

