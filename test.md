&emsp;&emsp;Given the limited availability of training data, many approaches leverage existing pre-trained models. For example, Xiang *et al.* [@xiang2024pandora] bypass the need for training from scratch by integrating a pre-trained LLM and a pre-trained video model, requiring only lightweight fine-tuning. Zhu *et al.* [@zhu2025irasim] initialize IRASim with the pre-trained weights of OpenSora [@zheng2024open] to expedite training. Similarly, Sudhakar *et al.* [@sudhakar2024controlling] leverage a pre-trained diffusion model, while Wang *et al.* [@wang2025language] utilize Stable Video Diffusion, fine-tuned with robotic videos to adapt to the robotics domain. Song *et al.* [@song2025physical] further exploit the world knowledge embedded in pre-trained autoregressive video generation models such as NOVA [@deng2025autoregressive].

&emsp;&emsp;\paragraph{Incorporating Auxiliary Data Sources}
&emsp;&emsp;Some works tackle the shortage of robot data by using other available sources, such as human manipulation datasets. For instance, Zhi *et al.* [@zhi20253dflowaction] use both human and robot manipulation videos for training. However, these datasets often contain cluttered backgrounds and similar-looking objects. To address this, they apply optical flow constraints to make the learned representation embodiment-agnostic. Sudhakar *et al.* [@sudhakar2024controlling] leverage an automatic hand segmentation method to obtain agent-agnostic data for robot learning. Others resort to more diverse data. For example, Yang *et al.* [@yang2023learning] leverage diverse kinds of data, including objects, scenes, actions, motions, language, and motor control, and convert all actions into a common format.

&emsp;&emsp;\paragraph{Synthetic Data Generation}
&emsp;&emsp;Instead of relying on real-world data, Deng *et al.* [@deng2025graspvla] synthesize large-scale action data to train their model. To address the scarcity of 4D data, the Aether team [@team2025aether] generate RGB-D synthetic videos and develop a robust camera-pose annotation pipeline to reconstruct full 4D dynamics. Similarly, Zhen *et al.* [@zhen2025tesseract] build a 4D embodied video dataset that combines synthetic data with ground-truth depth, normal information and real-world data with estimated depth and normal maps obtained from off-the-shelf estimators.

&emsp;&emsp;**) Heterogeneous Action Data**


&emsp;&emsp;World models should be able to understand different forms of actions and embodiments to ensure their real-world applications. A basic strategy is to utilize diverse datasets for training. However, the inherent cross-domain and cross-embodiment nature of datasets lead to heterogeneous actions data, including action spaces, action frequencies, and action horizon. For example, diverse embodiment (e.g., different degrees of freedom across robotic arms) and control interface (end effector (EEF) position for arms) would lead to actions of different forms. To handle this, Zheng *et al.* [@zheng2025universal] learn to capture their shared structural features to obtain the generic atomic behaviors by means of vision language models. Similarly, Zheng *et al.* [@wang2025learning] lean a share latent space for actions by decoupling observation and actions. More strategies can borrow from relevant fields [@doshi2025scaling;@team2024octo;@wang2024scaling].

&emsp;&emsp;**) Action Label Missing**

&emsp;&emsp;Action-labeled data, which are essential for learning action-conditioned future predictions [@yang2023learning], are particularly scarce in real-world settings.

&emsp;&emsp;\paragraph{Self-supervised Learning}
&emsp;&emsp;Finn *et al.* [@finn2016unsupervised;@finn2017deep] propose to learn pixel-level motion in a self-supervised manner, while Ebert *et al.* [@ebert2018robustness;@ebert2018visual] leverage image-to-image registration between consecutive video frames to capture dynamics without explicit action labels. However, goal image-based learning presents several drawbacks: such goals are inconvenient for humans to specify, may over-constrain the desired behavior (leading to sparse rewards), or under-specify task-relevant information for non-goal-reaching tasks.

&emsp;&emsp;\paragraph{Action Label Extraction}
&emsp;&emsp;Another approach to handling missing action labels is to infer them directly from unlabeled videos. More specifically,Bruce *et al.* [@bruce2024genie;@gao2025adaworld] employ latent action autoencoders to extract latent actions in a self-supervised manner. In their studies, Bruce *et al.* [@bruce2024genie] sample actions uniformly, while Gao *et al.* [@gao2025adaworld] introduce biased action sampling to encourage broader exploration and enable action reuse across contexts.Jiang *et al.* [@jang2025dreamgen] extract pseudo-actions using either a latent action model [@ye2025latent] or an inverse dynamics model (IDM) [@baker2022video]. Du *et al.* [@du2023learning;@ren2025videoworld;@villar2025playslot;@ko2024learning] learn from unlabeled videos by training inverse dynamics models to infer actions or their embeddings. Ren *et al.* [@ren2025videoworld] further integrate an inverse dynamics module into a latent dynamics model to leverage rich temporal representations, improving the temporal consistency of predicted actions. Villar *et al.* [@villar2025playslot] predict latent actions from object-centric representations.

&emsp;&emsp;\paragraph{Other Strategies}
&emsp;&emsp;Some works aim to leverage **pre-trained video models**. For instance, Rigter *et al.* [@rigter2025avid] adapt a pre-trained video diffusion model for action-conditioned world modeling by training a lightweight adapter, which is then fine-tuned on a small set of domain-specific, action-labeled videos. Black *et al.* [@black2024zero] similarly employ a pre-trained image-editing diffusion model to support video-based world modeling. In addition, Zhu *et al.* [@zhu2025unified] design a **unified world model** that integrates the action and video diffusion processes within a unified transformer architecture using separate diffusion timesteps. This can enable learning from action-free video data. Ko *et al.* [@ko2024learning] utilize **optical flow** extracted from videos, thereby circumventing the need for explicit action labels.

&emsp;&emsp;\begin{tcolorbox}[colback=blue!5!white, colframe=blue!70!white, title=Implications for Data Limitations]
&emsp;&emsp;The scarcity of data can be alleviated by leveraging external datasets or pre-trained base models from other sources. The key challenge lies in bridging the gap between the source and target domains. Furthermore, it is desirable to uncover the underlying knowledge through supervised learning.
&emsp;&emsp;\end{tcolorbox}

## **Perception and Representation**


&emsp;&emsp;Perception lies at the heart of robotic world models, enabling systems to interpret task instructions and transform raw sensory inputs into meaningful representations. These representations allow robots to understand structured environments and, in turn, predict, react, and plan effectively.

&emsp;&emsp;**) Inputs**

&emsp;&emsp;**Language.** Task instructions are usually given in language. Many methods use pretrained models such as CLIP [@bu2024closed;@ko2024learning;@radford2021learning;@tian2025predictive], Phi [@javaheripi2023phi;@song2025physical], or conditional VAEs [@song2025physical] to extract semantic representations from the instructions.

&emsp;&emsp;**Visual data.** Similarly, visual inputs are often processed using pre-trained visual encoders. For example, Tian *et al.* [@tian2025predictive] leverage pre-trained Vision Transformers (ViTs) [@he2022masked] to process image observations.
&emsp;&emsp;Wu *et al.* [@wu2024ivideogpt] employ a conditional VQGAN that encodes only task-relevant dynamic information, such as the position and pose of moving objects, to reduce temporal redundancy across frames. An autoregressive, GPT-like transformer is then used to generate the next tokens, which are decoded into future frames.

&emsp;&emsp;**Action data.** Actions are sometimes represented as integer values, which lack the contextual richness. This limitation can prevent world models from accurately capturing the intended meaning behind actions. To address this, He *et al.* [@he2025pre] propose representing actions through language templates that explicitly encode their semantic meaning. In many cases, actions are instead expressed in natural language, as noted above. While this enables richer semantic representations, it also introduces challenges, such as instruction-following ambiguity, which are discussed in Section **@@@**.

&emsp;&emsp;**Diverse data inputs.** Robots need to gain a structured understanding of the world by jointly considering diverse sensory inputs. To achieve this, Song *et al.* [@song2025physical] embed images and robot actions into a unified physical space, enabling the model to capture the sequential evolution of both the robot and its environment. Hong *et al.* [@hong2024multiply] incorporate visual, auditory, tactile, and thermal modalities, projecting them into a shared feature space where a language model generates subsequent states and action tokens.


&emsp;&emsp;**) Challenges**


&emsp;&emsp;\paragraph{Instruction Understanding and Following} 
&emsp;&emsp;Instructions convey task goals and can take various forms, including linguistic directives (natural language or structured text), visual cues (sketches, images, or demonstration videos), and others. Compared to image-based goals, textual descriptions provide a more abstract, compositional, and flexible way of specifying objectives, enabling better generalization, clearer intent communication, and more efficient human–robot interaction. Many recent works express target goals through text descriptions [@du2023learning]. Ideally, language instructions should clearly describe the task and remain easily interpretable by the model. However, real-world scenarios often involve ambiguous or novel instructions, making effective interpretation and grounding critical for successful task execution.

&emsp;&emsp;**Ambiguous Instructions.**
&emsp;&emsp;In real-world scenarios, language instructions are often ambiguous (e.g., ``put this near here'' [@wang2025language]).To resolve such ambiguity, Wang *et al.* [@wang2025language] use pointing gestures, interpreted through 2D gripper and object tracking, as an additional instruction modality.


&emsp;&emsp;**New Instructions.**
&emsp;&emsp;World models are constrained to make predictions based on language instructions similar to those encountered during training, limiting their ability to generalize to novel commands. To solve this problem, Xiang *et al.* [@xiang2024pandora] curate a large and diverse set of action-state sequences from re-captioned videos and simulations, and fine-tune world models on this data to improve instruction interpretation and generalize to novel commands and tasks. Li *et al.* [@zhou2024robodreamer] employ a text parser to decompose language instructions into primitives, separating actions and spatial relationships. This decomposition allows the model to flexibly recombine these components and generalize to previously unseen combinations of instructions. However, decomposing instructions into primitives can ignore their interrelationships. To address this, Li *et al.* [@li2025manipdreamer] represent each instruction as an action tree, capturing the hierarchical structure among primitives to better model task organization.

&emsp;&emsp;Some studies suggest that humans make predictions based on abstract concepts rather than raw pixels [@chen2025egoagent].Instead of converting images into discrete tokens [@yang2023learning;@wu2024ivideogpt], Chen *et al.* [@chen2025egoagent] use learnable convolutional layers to project images into continuous semantic embeddings.Song *et al.* [@song2025physical] adopt an open-source 3D variational autoencoder (Open-Sora [@zheng2024open]) to obtain video representations. In contrast, another line of work operates directly in pixel space. For instance, Ko *et al.* [@ko2024learning] adapt a U-Net-based image diffusion model with factorized spatial–temporal convolutions [@dhariwal2021diffusion] to jointly capture spatial and temporal information.

&emsp;&emsp;\paragraph{Task-irrelevant Issues}
&emsp;&emsp;Visual data often contain information irrelevant to the task, and models such as Vision Transformers (ViTs) may produce hundreds of features per image, affecting both efficiency and effectiveness. To address this, Tian *et al.* [@tian2025predictive] extract task-relevant features using a perceiver resampler [@alayrac2022flamingo]. Ren *et al.* [@ren2025videoworld] learn compact visual representations that preserve fine-grained temporal dynamics through a causal encoder–decoder structure and quantization with a discrete codebook [@mentzer2024finite].

&emsp;&emsp;\paragraph{Spatiotemporal Awareness}
&emsp;&emsp;Understanding the world requires modeling how spatial structures evolve over time. To this end, several works design architectures that explicitly capture spatial and temporal dependencies. Tian *et al.* [@tian2025predictive] enhance token representations with learnable positional embeddings at each timestep to capture temporal information. Bruce *et al.* [@bruce2024genie] develop a spatiotemporal transformer composed of multiple spatiotemporal blocks to model spatial–temporal relationships in dynamic scenes. Ko *et al.* [@ko2024learning] adopt factorized spatiotemporal convolutions following the design of [@ho2022video]. Zhang *et al.* [@zhang2025dreamvla] extract spatiotemporal patch representations using a masked autoencoder [@he2022masked]. Other studies incorporate additional cues to better understand the three-dimensional structure of the environment. For example, Zhang *et al.* [@zhang2025dreamvla] estimate depth information using depth estimation techniques [@yang2024depth] to enhance 3D spatial understanding. When encoding multi-view inputs, Liao *et al.* [@liao2025genie] augment each token with 2D rotary positional embeddings, view-specific learnable embeddings, and timestep encodings to promote spatiotemporal alignment while preserving viewpoint-specific distinctions.

&emsp;&emsp;\begin{tcolorbox}[colback=blue!5!white, colframe=blue!70!white, title=Implications for Perception]
&emsp;&emsp;World models should process and integrate diverse sensory inputs to build a coherent understanding of real-world dynamics.
&emsp;&emsp;While current models primarily rely on vision and language, incorporating additional modalities such as tactile and proprioceptive sensing is crucial for achieving comprehensive perception in complex environments. It is also important to consider which information to perceive and how to model its spatial and temporal structure.
&emsp;&emsp;\end{tcolorbox}

## **Long-horizon Reasoning**


&emsp;&emsp;Many robotic tasks require coherent long-horizon reasoning, where achieving the final objective depends on executing a temporally consistent sequence of actions over extended time scales. Existing methods are limited in long-horizon predictions [@nair2022learning;@ha2018world;@hafner2019learning;@hafner2021mastering;@hafner2023mastering]. For example, Ha *et al.* [@ha2018world;@hafner2019learning;@hafner2021mastering;@hafner2023mastering] predefine temporal horizons to guide planning in their world models. In terms of video generation, existing methods still suffer from limited length (short-horizon future video) [@gao2024flip]. For example, Ko *et al.* [@ko2024learning] predicts a fixed number (eight) of future frames with U-Net based diffusion model [@dhariwal2021diffusion]. Bruce *et al.* [@bruce2024genie] can only memorize 16 frames and cannot produce consistent predictions. For autoregressive models, small prediction errors compound sequentially, leading to substantial inaccuracies in long-horizon forecasts.

&emsp;&emsp;**) Closed-loop Learning**

&emsp;&emsp;A line of work enabling long-term planning/predictions by learning through interaction with feedback and adjusting their behaviour accordingly [@du2023video;@bu2024closed] . For example, Ebert *et al.* [@ebert2018robustness;@ebert2018visual] utilize image-to-image registration between predicted video frames and both the start and the goal images with the average length of the warping vectors as a cost function. The model would continue to retry until the task is completed. Du *et al.* [@du2023video] proposes a recursive planning framework comprising action proposal, video rollout generation, and evaluation. Vision–language models (VLMs) are used to propose potential next actions, while video generation models simulate multiple possible future rollouts. The resulting trajectories are then evaluated by the VLMs to select the optimal action. Du *et al.* [@liao2025genie] design a neural simulator that predicts future visuals, enabling policy models to interact within a consistent environment. A sparse memory mechanism is leveraged to further enhance the consistency over the time.

&emsp;&emsp;**) Subgoals**

&emsp;&emsp;Pre-trained models possess a vast repository of commonsense and procedural knowledge that can be leveraged to decompose a high-level goal, often specified in natural language (e.g., "make a cup of coffee"), into a logical sequence of concrete sub-goals or skills. Bu *et al.* [@bu2024closed] propose to promote long-horizon manipulation tasks by decomposing the goal into sub-goals and handling error accumulations by designing a real-time feedback mechanism. Yang *et al.* [@yang2025roboenvision] leverage VLM to produce sub-goals and utilize coarse and fine video diffusion models to generate long-horizon videos. Chen *et al.* [@chen2025robohorizon] utilizes an LLM to generate a multi-stage plan and design a LLM-based dense reward generator for sub-tasks, providing crucial guidance for long-horizon planning.

&emsp;&emsp;**) Hierarchical Structures**

&emsp;&emsp;Bu *et al.* [@gumbsch2023learning]  propose hierarchical world models with Adaptive Temporal Abstractions that separate the modeling of dynamics into high-level and low-level latent states. The low-level model captures fine-grained, short-term dynamics for immediate reactions, while the high-level model abstracts over longer temporal horizons to represent extended dependencies and long-term goals. By dynamically adapting the temporal granularity of the high-level latent states, the model can efficiently plan and predict over long horizons while maintaining accurate short-term predictions through the low-level module.

&emsp;&emsp;**) More Strategies.**


## **Spatiotemporal Consistency**

&emsp;&emsp;Spatiotemporal consistency plays a vital role in ensuring coherent and physically plausible predictions of future states. It guarantees that the model preserves object continuity, motion smoothness, and causal relationships across time, enabling stable video simulation and reliable dynamics forecasting.

&emsp;&emsp;**) Data Perspective**


&emsp;&emsp;**) Model Perspective**


&emsp;&emsp;**) Memory Mechanism**

&emsp;&emsp;Memory mechanisms preserve historical information, enabling the coherent evolution of spatial and temporal patterns over time. For example, Liao *et al.* [@liao2025genie] design a sparse memory mechanism to provide long-term historical context, improving spatiotemporal consistency and task relevance. More information can refer to Section **@@@**.

## **Generalization**

&emsp;&emsp;Robots are expected to operate robustly in complex and novel environments, interacting with unfamiliar objects and performing tasks beyond their training distribution.

&emsp;&emsp;**) Data Scaling**

&emsp;&emsp;An intuitive and effective strategy to enhance generalization is to scale the diversity and volume of training data. For example, Cheang *et al.* [@cheang2024gr] increase the number of pre-training videos from 0.8 million in [@wu2024unleashing] to 38 million. Assran *et al.* [@assran2025v] expand the dataset from 2 million used by [@bardes2024revisiting] to 22 million videos. Wang *et al.* [@wang2025learning] expand each of the 40 datasets by increasing trajectories from 10 up to $10^{6}$. Cheang *et al.* [@cheang2025gr] train the model with web-scale vision-language data,  human trajectory data and robot trajectory data. Kevin *et al.* [@intelligence2025pi_] leverage diverse mobile manipulator data, diverse multi-environment non-mobile robot data, cross-embodiment laboratory data, high-level subtask prediction, and multi-modal web data. Cheang *et al.* [@barcellona2025dream;@cheang2024gr] investigate **data augmentation** strategies to enhance generalization. In [@barcellona2025dream], object rotation and roto-translation are applied. Cheang *et al.* [@cheang2024gr] generate novel scenes by injecting objects using a diffusion model [@ho2020denoising] and/or altering backgrounds with the Segment Anything Model (SAM)  [@kirillov2023segment]. A video generation model [@kirillov2023segment] is subsequently employed to synthesize videos that preserve the original robot motions from the inpainted frames. Liao *et al.*   [@liao2025genie] augment the dataset with a diverse set of failure cases, including erroneous executions, incomplete behaviors, and suboptimal control trajectories—collected from both human teleoperation and real-world robotic deployments. One problem of data scaling is that it is unlikely to collect all data for each tasks. At the same time, how to balance different data tasks is also challenging. Moreover, performance gains by scaling data is also limited for consistent performance improvements.

&emsp;&emsp;**) Use of Pretrained Models**

&emsp;&emsp;Many methods aim to enhance generalization by leveraging the generative capabilities of video models. For example, Zhu *et al.* [@team2025aether] combine video generation with geometric-aware learning to improve synthetic-to-real generalization across unseen viewpoints and support multiple downstream tasks. Zhen *et al.* [@zhen2025tesseract] fine-tune a video generation model on RGB, depth, and normal videos to encode detailed shape, configuration, and temporal dynamics, enabling generalization to unseen scenes, objects, and cross-domain scenarios. The generalization capabilities of large language models, such as video-language models [@wang2025founder] and vision-language models [@mazzaglia2024genrl], can be leveraged to enhance world models. By extracting high-level knowledge about the environment, these models facilitate more effective low-level dynamics modeling.

&emsp;&emsp;**) Instructions Decomposing**

&emsp;&emsp;Another generation issue comes from unseen instructions. To handle this, Zhou *et al.* [@zhou2024robodreamer] enhance the ability to unseen instructions by decomposing each spatial relation phrase into a set of compositional components with the pre-trained parser [@kitaev2019multilingual] and the rule-based approach. Detailed information can refer to Section **@@@**.

&emsp;&emsp;**) Invariant Representations**

&emsp;&emsp;Generalization can be significantly improved by learning invariant representations to superficial or task-irrelevant changes in the environment. For example, Pang *et al.* [@pang2025reviwo] model learns to explicitly decompose visual observations into a view-invariant representation, which is used for the control policy, and a view-dependent representation. This decoupling makes the resulting policy robust to changes in camera viewpoint, a common source of failure in visuomotor control. Similarly, the Martinez *et al.* [@martinez2025coral] framework learns a transferable communicative context between two agents, which enables zero-shot adaptation to entirely unseen sparse-reward environments by decoupling the representation learning from the control problem. Wu *et al.* [@wu2023pre] disentangle the modeling of context and dynamics by introducing a context encoder, enabling the model to capture shared knowledge for predictions.

&emsp;&emsp;**) Task-relevant Information Focused**

&emsp;&emsp;Video data often contain irrelevant data to the actions such as background and robot arm, which would limited the generalization ability of the learned world models. To handle this, [@zhi20253dflowaction] propose to object-centric world models, which concentrated on object movements via the optical flow predictions that is independent of embodiment. Finn *et al.* [@finn2016unsupervised] propose to explicitly model and predict motion that are relatively invariant to the object appearance, enabling long-range predictions and generalize to unseen objects.

&emsp;&emsp;**) Other Strategies**

&emsp;&emsp;Black *et al.* [@black2024zero] use a pretrained image-editing model to generate subgoals from language commands and current observations, enabling low-level controllers to act and generalize to novel objects and scenarios. Self-supervised learning without task-specific rewards that can enhancing generalization abilities into different tasks [@sekar2020planning].

## **Physics-informed Learning**

&emsp;&emsp;Existing world models struggle to generate physically consistent videos because they lack an inherent understanding of physics, often producing unrealistic dynamics and implausible event sequences. Simply scaling up training data or model size is insufficient to capture the underlying physical laws [@kang2025far]. To address this challenge, several approaches have been proposed.
&emsp;&emsp;For example, Yang *et al.* [@yang2025vlipp] introduce a two-stage image-to-video generation framework that explicitly incorporates physics through vision- and language-informed physical priors. Team *et al.* [@team2025aether] estimate depth and camera pose directly from videos, facilitating physics-informed learning and enabling world models to infer and predict physically consistent dynamics. Peper *et al.* [@peper2025four] argue that advancing from physics-informed to physics-interpretable world models requires rethinking model design, and propose four guiding principles: organizing latent spaces by physical intent, encoding invariant and equivariant environmental representations, integrating multiple supervision signals, and partitioning generative outputs to improve both scalability and verifiability.

&emsp;&emsp;\begin{tcolorbox}[colback=blue!5!white, colframe=blue!70!white, title=Implications for Generalization and Physics-informed World Modeling]
&emsp;&emsp;While large-scale training improves the predictive, and generative abilities of world models, handling complex environments requires going beyond simple replication of observations. World models must capture the underlying physical and causal mechanisms of the world, enabling them to generate and predict consistent dynamics across diverse and unseen scenarios.
&emsp;&emsp;\end{tcolorbox}

## **Memory**

&emsp;&emsp;Memory mechanisms enable world models to store and retrieve relevant past information, supporting hidden-state disambiguation and long-horizon reasoning. For example, LeCun *et al.* [@lecun2022path] incorporate a memory module that maintains past, current, and predicted world states along with intrinsic costs, allowing retrieval of contextual information for reasoning and training. Huang *et al.* [@huang2025enerverse] propose a sparse contextual memory mechanism that preserves essential prior information throughout the generation process in a non-redundant manner, theoretically enabling the generation of sequences of arbitrary length. Zhou *et al.* [@zhou2025learning] employ a 3D feature-map memory to maintain temporal consistency during sequence generation.

&emsp;&emsp;**Memory Efficiency.**
&emsp;&emsp;Standard transformer blocks apply multi-head self-attention to all tokens in the input token sequence, resulting in quadratic computation cost. Zhu *et al.* [@zhu2025irasim] leverage the memory-efficient spatial-temporal attention mechanism to reduce the computation cost. Liao *et al.* [@liao2025genie] randomly sampled parse memory frames from prior video history to augment temporal diversity to improve representational invariance, and use low-frame-rate video sequence for fine-tuning frames.

## **Other Techniques and Challenges**

&emsp;&emsp;**) Video Fidelity**

&emsp;&emsp;To achieve high-fidelity video generation, several methods leverage powerful generative models. For instance, Ko *et al.* [@ko2024learning] employ an image diffusion model based on a U-Net with factorized spatiotemporal convolutions as the fundamental building block. Guo *et al.* [@guo2025flowdreamer] utilize the pre-trained variational autoencoder from Stable Diffusion [@rombach2022high]. Souvcek *et al.* [@souvcek2024genhowto] propose to make use of a variety of action and final state prompts.

&emsp;&emsp;**) Closed-loop Learning**

&emsp;&emsp;Closed-loop learning enables agents to actively refine their internal world models by observing and responding to real-time feedback from the environment. This continuous perception–action cycle grounds learning in physical reality, enhances generalization, and allows adaptive correction. Driess *et al.* [@driess2023palm] update observations based on the actions executed, which are then fed into VLMs to enable the robot to correct or reorganize its plan in response to environmental changes and task progress. Bu *et al.* [@bu2024closed] design a feedback mechanism that is based on the element-wise discrepancy measure between current and goal state embeddings. Zhi *et al.* [@zhi20253dflowaction], estimate the location of the moving objects, depth prediction, 3D optical flow by input into GPT-4o to verify alignment with given instructions, enabling closed-loop planning.

&emsp;&emsp;**) Reasoning**

&emsp;&emsp;Reasoning enables a robot to interpret sensory input and translate it into purposeful actions rather than reactive responses. Zhou *et al.* [@zhang2025dreamvla]  enhance the reasoning and genrealization ability by incorporating context information and predicting dynamic regions, depth map, semantic knowledge by means of foundation models, e.g., DINOv2 [@oquab2024dinov2] and SAM [@kirillov2023segment]. Ye *et al.* [@Ye2025GigaBrain] introduce an Embodied Chain-of-Thought as an intermediate reasoning representation, enabling more structured and interpretable decision-making in embodied tasks. Ye *et al.* [@Ye2025GigaBrain] [@zhao2025cot] generates a sub-goal image that represents the robot’s planned state in pixel space, and then conditions its action on both the current observation and the generated subgoal image.

&emsp;&emsp;**) Sim-to-real Gap**

&emsp;&emsp;The substantial gap between synthetic and real-world data limits the transferability and real-world performance of robotic systems. To handle this, Huang *et al.* [@huang2025enerverse] propose combining the generative model with 4D Gaussian Splatting, forming a self-reinforcing data loop to reduce the sim-to-real gap.

&emsp;&emsp;**) 3D Robotics World Prediction**

&emsp;&emsp;General-purpose video generation models neglect the substantial gap between their representation space and the three-dimensional, temporally interconnected robotics environment, thereby hindering accurate action policy prediction. For example, Wen *et al.* [@wen2024vidman] focuses on 2D image prediction before action generation. To handle this, Huang *et al.* [@huang2025enerverse]  propose Free Anchor Views, a multi-view video representation offering flexible, task-adaptive perspectives to address challenges like motion ambiguity and environmental constraints.

&emsp;&emsp;**) Fine-grained Robot-object Interaction**

&emsp;&emsp;Robots are expected to perform precise manipulation, which requires world models to support fine-grained robot-object interactions. To achieve this, Zhu *et al.* [@zhu2025irasim] design a novel frame-level action-conditioning module to achieve precise action-frame alignment. He *et al.* [@he2025pre] adopt two different pre-trained video generative models as the base models, and introduce a minimalist yet powerful add-on action-conditioned module that improves frame-level action awareness while maintaining architectural flexibility.

&emsp;&emsp;**) Multi-agent Operation**

&emsp;&emsp;Certain tasks necessitate coordinated operation among multiple robots to achieve successful completion. To this end, Zhang *et al.* [@zhang2025combo] factorize the joint actions of different agents as a set of text prompt and leverage composable video diffusion models to learn world dynamics and make predictions. An agent-dependent loss is imposed to let the model focus on the related pixel, where the loss coefficient matrix is based on each agent’s reachable region.

&emsp;&emsp;% \subsubsection{Multi-view.}
&emsp;&emsp;% Zhang *et al.* [@liao2025genie] concentrate on the egocentric nature of perception in dual-arm robotic systems.  Zhang *et al.* [@liao2025genie] extend the world model into a multi-view, language-and-image-conditioned generation framework that leverages temporally synchronised inputs from three on-board cameras: a head-mounted view and two wrist-mounted views. It concatenates all views and leverages cross-view attentions.

&emsp;&emsp;**) Error Propagation**

&emsp;&emsp;In autoregressive models, subsequent actions are generated based on previous predictions, leading to error propagation over time. To handle this, Cen *et al.* [@cen2025worldvla] propose an attention mask strategy that selectively masks prior actions during the generation of the current action. It enables both future imagination and action generation.