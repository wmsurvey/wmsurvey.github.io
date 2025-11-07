# **III Overview of the World Model**
“What we observe is not nature itself, but nature exposed to our method of questioning.” 
<p style="text-align:right">— Werner Heisenberg</p>  

## **A. Paradigms**
&emsp;&emsp;Building on the previous review of current models, contemporary architectures for capturing world dynamics can be broadly stratified along a methodological spectrum: implicit world modeling (e.g., LLMs, VLMs, and VLAs) [@chen2025egoagent;@team2023internlm;@zhen20243d;@hong20233d], latent dynamics modeling [@wu2023daydreamer;@hafnerdream;@hafner2021mastering;@hafner2023mastering], and video generation paradigms [@wu2024ivideogpt;@du2023video;@zhang2025combo;@rigter2025avid], each targeting distinct representational granularities and predictive mechanisms.
<figure markdown>
  ![Perspectives on world models](assets/img/03-01.png){ width="100%" }
  <figcaption markdown="span">Fig. 2. A visualization of LLM-based world models [@driess2023palm].</figcaption>
</figure>
&emsp;&emsp;**1) Implicit World Modeling**  

&emsp;&emsp;Representative models include LLMs, VLMs, and VLAs, which offer distinct advantages in semantic grounding, generalization, and interpretability [@ahn2022can;@xiang2024pandora;@driess2023palm;@zhang2025dreamvla;@zhao2024vlmpc]. An illustration of these models is shown in Figure&nbsp;1 At the same time, these models can be integrated into broader world-modeling architectures to capture temporal dependencies and enable long-horizon prediction [@zitkovich2023rt;@hong20233d;@bjorck2025gr00t]. Detailed discussions of these models are provided in Sections II-D1 and IV-A1.

&emsp;&emsp;**2) Latent Dynamics Modeling**  

&emsp;&emsp;Latent dynamics models typically encode high-dimensional observations into compact latent states through a variational autoencoder (VAE) or encoder network, and employ recurrent or transformer modules (e.g., RNNs or Transformers) to predict the temporal evolution of these latent representations [@hafnerdream;@hafner2021mastering;@wu2023daydreamer;@hafner2023mastering]. This architecture is characterized by latent-space imagination and task-oriented optimization over visual granularity, facilitating long-horizon learning by forecasting future states without the need for pixel-level reconstruction.

&emsp;&emsp;Recurrent State-Space Model (RSSM) [@hafner2019learning] resembles the structure of a partially observable Markov decision process.  Its learning framework consists of three main components: an encoder, a decoder, and a dynamics model. The encoder network fuses sensory inputs (observations) *o* together into the stochastic representations *z*. The dynamics model learns to predict the sequence of stochastic representations by using its recurrent state *s*. The decoder reconstructs sensory inputs to provide a rich signal for learning representations and enables human inspection of model predictions, but is not needed while learning behaviors from latent rollouts. Specifically, at time step *t*, let the image observation be *oₜ*, the action vectors at and the reward *rₜ*. RSSM can be formulated as the generative process of the images and rewards conditioned a hidden state sequence *sₜ*:  

&emsp;&emsp;Encoder/representation model:&emsp;&emsp;&emsp;  $s_t \sim p_\theta\left(s_t \mid s_{t-1}, a_{t-1}, o_t\right)$

&emsp;&emsp;Decoder/observation model:&emsp;&emsp;&emsp;&emsp; $o_t \sim p_\theta\left(o_t \mid s_t\right)$

&emsp;&emsp;Dynamics/Transition model:&emsp;&emsp;&emsp;&emsp;  $s_t \sim p_\theta\left(s_t \mid s_{t-1}, a_{t-1}\right)$

&emsp;&emsp;Reward model:&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;  $r_t \sim p_\theta\left(r_t \mid s_t\right)$


<p style="text-align:right">(1)</p>


<figure markdown>
  ![Perspectives on world models](assets/img/03-02.png){ width="80%" }
  <figcaption markdown="span">Fig. 3. A visualization of Dreamer architecture [@hafnerdream], which encodes visual representations into latent states through recurrent estimation.</figcaption>
</figure>

&emsp;&emsp;PlaNet [@hafner2019learning] first demonstrates the effectiveness of learning dynamics in a latent space. The Dreamer family of models (a visualization is shown in Fig. 3) [@hafnerdream;@hafner2021mastering;@wu2023daydreamer;@hafner2023mastering] further verify this paradigm and establish a representative framework that reduces reliance on real-world data by performing imagination directly in latent space. Dreamer enables policy learning through imagined trajectories without continuous interaction with the environment, allowing agents to simulate multi-step consequences of actions and generalize to new states, objects, and environments.
<figure markdown>
  ![Perspectives on world models](assets/img/03-03.png){ width="60%" }
  <figcaption markdown="span">Fig. 4. A visualization of Joint Embedding-Action-Prediction (JEPA) Archi-
tecture [@lecun2022path], where self-supervised learning is used to learn the future world
state representations.</figcaption>
</figure>
&emsp;&emsp;While sharing the objective of learning predictive worldstate representations, Joint-Embedding Predictive Architecture (JEPA) [@lecun2022path;@chen2025egoagent] and RSSM diverge fundamentally in their learning mechanisms. RSSM relies on generative reconstruction of observations to model latent dynamics, whereas JEPA (a visualization is shown in Fig. 4) employs selfsupervised predictive coding in embedding spaces—directly forecasting future state representations without decoding to raw sensory inputs. This paradigm eliminates the computational cost of pixel-level reconstruction but necessitates powerful hierarchical encoders to compress sufficient environmental information into abstract embeddings, creating an implicit information bottleneck that demands careful architectural balancing to preserve task-relevant features. Under the JEPA framework, Assran *et al.* [@assran2025v] combine pre-trained video models with an action-conditioned predictor to autoregressively predict future states and actions. 
<figure markdown>
  ![Perspectives on world models](assets/img/Genie_envisioner.png){ width="120%" }
  <figcaption markdown="span">Fig. 5. An illustration of video-geneation based world models [@liao2025genie]. World
model serve as the core component, modelling the world dynamics and
enabling action planning and generation.</figcaption>
</figure>

&emsp;&emsp;The **MuZero** series [@schrittwieser2020mastering;@ye2021mastering;@wang2024efficientzero] represent another form of latent-dynamics-based world modeling. Instead of modeling the complete environment dynamics, MuZero predicts only future quantities directly relevant to planning, such as rewards, values, and policies, given the complexity of real-world environments, and employs a tree-based search algorithm [@silver2018general] to select optimal actions.

&emsp;&emsp;**3) Video Generation.**

&emsp;&emsp;Video-based generative models are powerful tools for capturing environmental dynamics and predicting future scenes. These models operate directly on high-dimensional raw observations, such as RGB images, depth maps, or force fields [@wu2024ivideogpt;@team2025aether;@yang2023learning;@bruce2024genie;@brooks2024video;@xiang2024pandora;@zheng2024open;@zhou2024robodreamer;@ali2025humanoid], treating the environment as a sequence of frames. By generating future scenes, they can support a wide range of applications, including visual planning, simulation, and action generation [@finn2017deep;@ebert2018visual;@wu2024ivideogpt;@zhang2025combo;@rigter2025avid]. Moreover, they can leverage large-scale pre-training to enhance generalization and improve sample efficiency [@rigter2025avid;@team2025aether;@wang2025language;@jang2025dreamgen]. Depending on the input modality, world models can be constructed using action-conditioned video prediction models [@wu2024ivideogpt], text-to-video models [@du2023video;@yang2023learning;@jang2025dreamgen;@zhou2024robodreamer], or trajectory-to-video models [@zhu2025irasim;@cheang2024gr].

&emsp;&emsp;There are several architectural families of video-based world models. Diffusion-based world models generate videos by progressively denoising random noise through multiple iterative steps. Representative examples include U-Net-based models [@ho2022video;@ko2024learning] and diffusion transformer (DiT)-based architectures [@ferraro2025focus;@agarwal2025cosmos;@zhu2025irasim;@wan2025wan;@yang2025roboenvision]. Autoregressive world models, in contrast, predict the next token or frame conditioned on previously generated ones, effectively modeling temporal dependencies in the sequence [@liao2025genie;@wu2024ivideogpt;@xiang2024pandora;@huang2025enerverse;@villar2025playslot;@bruce2024genie;@cheang2024gr]. Other architectures include variational autoencoder (VAE)-based models [@bruce2024genie] and convolutional LSTMs [@finn2017deep;@ebert2018visual]. 

&emsp;&emsp;Autoregressive-based world models generate each step conditioned on previous outputs, allowing them to predict sequences of arbitrary length and making them well-suited for long-horizon predictions. However, they often suffer from error accumulation over extended sequences [@yang2025roboenvision] and may struggle to represent highly multi-modal distributions. In contrast, diffusion-based models generate samples through an iterative denoising process, enabling them to model complex, multi-modal distributions and produce globally coherent sequences. This iterative refinement also makes diffusion models more robust to individual prediction errors, resulting in better performance on tasks requiring long-horizon consistency or high-quality generative outputs. On the downside, diffusion models are computationally intensive and slower during inference, and adapting them to sequential prediction requires careful conditioning. Overall, autoregressive world models tend to excel in scenarios demanding speed and accurate short-term predictions, whereas diffusion models are more suitable for tasks involving long-horizon, multi-modal, or high-dimensional outputs where maintaining global coherence is crucial. 

&emsp;&emsp;Compared with implicit world models and latent-space world models, video generation models provide more detailed visual predictions but at a higher computational cost, lower generation speed and sample efficiency. In addition, action predictions are only proved to be align with visual future generation [@wang2025learning], as visual data contain relevant information to actions.

## **B. Architectural Design**

&emsp;&emsp;**1) Flat architecture**  

&emsp;&emsp;Most current methods adopt flat architectures [@guo2025flowdreamer;@ferraro2025focus;@villar2025playslot;@bruce2024genie;@brooks2024video;@xiang2024pandora;@zheng2024open;@zhou2024robodreamer], which face critical limitations. They lack structured representations of the environment, resulting in poor handling of multi-scale dynamics, limited longhorizon prediction, error accumulation, and reduced generalization. Specifically, in robotic manipulation, placing fragile objects requires the robot to react instantly to unexpected slips while simultaneously planning the sequence of pickand-place actions to achieve the overall goal. Many tasks further involve long-term objectives that must be completed through sequential subgoals and temporally extended actions. For example, assembling a piece of furniture requires picking up components, aligning and attaching them correctly, and tightening screws for each part. Moreover, operating at a single level of abstraction causes small prediction errors to compound over time, degrading performance in long-horizon tasks. Finally, flat architectures fail to extract high-level abstractions, limiting transferability across tasks and environments. 

&emsp;&emsp;**2) Hierarchical architecture** 

&emsp;&emsp;Several studies have begun to explore and design hierarchical world models, in which lower-level modules handle intermediate reactions and short-term predictions, while higherlevel components are responsible for long-term planning and abstraction. Lecun *et al.* [@lecun2022path] hypothesize a hierarchical JEPA architecture, where low-level and high-level representations are learned for short- and long-term predictions, respectively.Gumbsch *et al.* [@gumbsch2023learning] propose an RSSM-based hierarchical world model, where the low-level module captures immediate dynamics for reactive control, and the high-level module models abstract temporal patterns for strategic planning. Bjo ̈rck *et al.* [@bjorck2025gr00t] introduce a dual-system architecture in which System 2 interprets the environment and task goals, while System 1 generates continuous motor commands in real time. Similarly, Wang *et al.* [@wang2025dmwm] design a dual-level world model consisting of an RSSM-based System 1 (RSSM-S1) and a logic-integrated neural network System 2 (LINN-S2). The inter-system feedback mechanism ensures that predicted sequences comply with domain-specific logical rules: LINN-S2 constrains RSSM-S1’s predictions, while RSSM-S1 updates LINN-S2 based on new observations, enabling dynamic adaptation. Wang *et al.* [@song2025hume] further employ System 2 for value-guided high-level planning by estimating state-action values and selecting optimal actions, while System 1 executes real-time motions via cascaded action denoising. 

&emsp;&emsp;Despite their advantages, hierarchical architectures introduce greater model complexity, higher computational cost, and increased training difficulty. Determining which goals or subtasks should be handled by high-level versus low-level modules remains challenging, as does designing appropriate architectures and preparing suitable training datasets. Moreover, maintaining effective information flow and coordination between layers is essential for stable and coherent performance. Consequently, developing hierarchical world models requires substantial effort in architecture design, goal decomposition, dataset construction, and inter-layer coordination.

<figure markdown>
  ![Perspectives on world models](assets/img/03-04.png){ width="50%" }
</figure>

## **C. World Observation and Representation**

&emsp;&emsp;**1) Dimensionality of the World**  

&emsp;&emsp;In designing world models, the dimensionality of the environment plays a critical role, shaping how effectively a model captures spatial structures, temporal evolution, and causal dynamics.  

&emsp;&emsp;Some works operate purely in 2D pixel space [@yang2023learning;@bruce2024genie;@brooks2024video;@xiang2024pandora;@zheng2024open;@zhou2024robodreamer], capturing visual appearance and short-term dynamics but ignoring the real-world geometry. While 2D pixel-space models [@yang2023learning] can model appearance and short-term dynamics, they lack geometric awareness of real-world structure. This limitation motivates the development of 3D-aware architectures.  
To incorporate geometric understanding of the 3D world, Bu *et al.* [@bu2024closed;@ko2024learning;@zhang2025combo] construct world models based on RGB-D data, while others extract richer 3D cues such as scene flow [@guo2025flowdreamer], motion fields [@zhi20253dflowaction], and 3D point clouds with associated language descriptions [@zhen20243d], enabling more comprehensive modeling of 3D world dynamics. Additionally, Lu *et al.* [@lu2025gwm] leverage 3D Gaussian Splatting, Diffusion Transformers, and a 3D Gaussian Variational Autoencoder to extract 3D representations from RGB observations. Zhang *et al.* [@zhang2025dreamvla] incorporate depth estimation to enhance understanding of 3D worlds.  

&emsp;&emsp;In addition to geometric structure, temporal dynamics are incorporated to construct 4D world models that jointly capture spatial and temporal evolution. For example, Zhu *et al.* [@team2025aether] synthesize 4D data from RGB-D videos by estimating depth and camera pose. Zhen *et al.* [@zhen2025tesseract] leverage a pre-trained 3D VAE [@kingma2013auto] to encode RGB, depth, and normal videos and combine them, while Huang *et al.* [@huang2025enerverse] employ 4D Gaussian splatting to model spatiotemporal dynamics in robotic environments.


&emsp;&emsp;**2) Observation Viewpoint of the World** 
 
&emsp;&emsp;Robots acquire skills by observing and imitating humans or other robots in their environment. Depending on the observation viewpoint, world models for robot learning can be categorized into **third-person (exocentric)** [@guo2025flowdreamer;@ferraro2025focus;@villar2025playslot] and **first-person (egocentric)** [@chen2025egoagent;@grauman2024ego] perspectives. Many existing methods learn from exocentric perspectives, capturing skills from an external viewpoint [@guo2025flowdreamer;@ferraro2025focus;@villar2025playslot]. However, exocentric observations do not fully align with how humans perceive the world, motivating the development of **egocentric world models**. For example, Chen *et al.* [@chen2025egoagent] observe a continuous loop of human interactions, in which humans perceive egocentric observations and take 3D actions repeatedly. They model these interactions as sequences of “state-action-state-action” tokens processed using a causal attention mechanism. Zhang *et al.* [@zhang2025combo] focus on multi-agent planning, inferring other agents’ actions from world states estimated via partial egocentric observations.


&emsp;&emsp;Grauman *et al.* [@grauman2024ego] argue that egocentric and exocentric viewpoints are complementary: egocentric viewpoints provide fine-grained cues about hand-object interactions and the camera wearer’s attention, while exocentric viewpoints supply broader context about the surrounding environment and whole-body poses.

&emsp;&emsp;**3) Representation of the World**

&emsp;&emsp;A central aspect of world models lies in how the environment is represented, which directly influences their ability to reason about dynamics, predict future states, and generalize across tasks. World representations can be broadly categorized into scene-centric, object-centric, and flow-centric approaches. In scene-centric representations, the environment is encoded as a single holistic latent, typically learned directly from pixels or raw sensory inputs [@hafner2019learning;@hafnerdream;@hafner2021mastering;@hafner2023mastering;@hafner2025mastering]. While video generation tasks aim to maximize the visual fidelity of predicted sequences, robotic manipulation often does not require the full visual detail. Irrelevant elements such as the background or parts of the robot arm can be ignored. This motivates the use of object-centric representations, which focus on task-relevant entities and their interactions [@ferraro2025focus;@barcellona2025dream;@villar2025playslot;@zhi20253dflowaction;@hong2024multiply]. Flow-centric representations, in contrast, are designed to capture the motion dynamics of the environment, emphasizing temporal change and spatial displacement [@gao2024flip].


<figure markdown>
  ![Perspectives on world models](assets/img/03-05.png){ width="50%" }
</figure>

## **D. Task Scope**
&emsp;&emsp;World models can also be categorized based on their task coverage. Some studies focus on single-task objectives, such as future-scene prediction [@sudhakar2024controlling;@finn2016unsupervised;@barcellona2025dream;@jang2025dreamgen;@ebert2018robustness], or planning and action prediction [@sekar2020planning].


&emsp;&emsp;In contrast, an increasing number of studies aim to support multiple tasks simultaneously, thereby enhancing the generality and applicability of world models.For instance, Cheang *et al.* [@cheang2024gr;@zhou2024robodreamer;@du2023video;@gao2024flip] generate videos for future-scene prediction and accordingly infer corresponding actions. Other works pursue simultaneous action prediction and world-scene forecasting [@cen2025worldvla;@chen2025egoagent;@zhen20243d;@song2025physical]. Beyond dual-task integration, several approaches extend world models to even broader capabilities. For instance, Bruce *et al.* [@bruce2024genie] propose interactive video generation that supports environment prediction and imitation learning, and utilize a latent action model to infer policies from unseen, action-free videos. Liao *et al.* [@liao2025genie] introduce a unified framework for embodied video generation, policy learning, and simulation. Lu *et al.* [@lu2025gwm] learn 3D world representations for future-state prediction, imitation learning, and simulator through video generation. Zhu *et al.* [@zhu2025irasim] develop an action-conditioned world model supporting trajectory-conditioned video generation, policy evaluation, and planning. Similarly, Huang *et al.* [@huang2025enerverse] achieve multi-view video generation, robotic action prediction, and a data flywheel mechanism for sim-to-real adaptation.

&emsp;&emsp;**Would foundation models.** When discussing task scope, the notion of “foundation world models” becomes essential. These approaches aim to generalize across diverse tasks through large-scale training, paving the way for world models that act as universal backbones for robotics. One line of research achieves this through large-scale pretraining followed by taskspecific fine-tuning [@lu2025gwm;@wu2024ivideogpt;@agarwal2025cosmos;@cheang2024gr;@mazzaglia2024genrl]. In particular, Mazzaglia *et al.* [@mazzaglia2024genrl] integrate a foundation VLM with a generative world model to enhance multimodal generalization. Other works directly pursue large-scale end-to-end training to build general-purpose world models [@bruce2024genie;@cen2025worldvla].

<figure markdown>
  ![Perspectives on world models](assets/img/03-06.png){ width="50%" }
</figure>

## **References**
\bibliography