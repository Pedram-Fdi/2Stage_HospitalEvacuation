# Deep-Reinforcement-Learning-Based Matheuristic

To solve our complex stochastic optimization problem, we propose a hybrid approach that integrates deep reinforcement learning (DRL) with the adaptive large neighborhood search (ALNS) algorithm, referred to as *DRL-ALNS*. This learning-enhanced algorithm is designed to improve the flexibility and intelligence of the search process, addressing key limitations of classical ALNS implementations...

\section{Deep-Reinforcement-Learning-Based
Matheuristic}\label{sec:SolutionApproach}
To solve our complex stochastic optimization problem, we propose a hybrid approach that integrates deep reinforcement learning (DRL) with the adaptive large neighborhood search (ALNS) algorithm, referred to as \textit{DRL-ALNS}. This learning-enhanced algorithm is designed to improve the flexibility and intelligence of the search process, addressing key limitations of classical ALNS implementations. Originally introduced by \citet{Shaw_1998}, ALNS is a widely used heuristic for solving NP-hard problems efficiently. It relies on two types of operators, \textit{destroy} and \textit{repair}, which iteratively modify and reconstruct solutions. One of the strengths of ALNS lies in its ability to explore a diverse set of neighborhoods adaptively. This adaptiveness is achieved by tracking the performance of each operator and updating their selection probabilities accordingly, promoting both intensification and diversification in the search process.

However, the effectiveness of ALNS heavily depends on the design of its adaptive operator selection (AOS) mechanism and the configuration of its acceptance parameters. Traditional acceptance and AOS techniques, such as the Metropolis acceptance criterion and the roulette wheel selection, are commonly employed to guide the search and select promising operators. These mechanisms form the core of our baseline ALNS implementation, which we use as a benchmark for evaluating the performance of our proposed learning-based approach. Despite its widespread use, standard ALNS has several limitations. As highlighted by \citet{Turkes_2021}, simple weight-based AOS strategies often fail to exploit meaningful patterns in the search trajectory and can struggle to prioritize truly effective operators. Moreover, traditional operator selection is typically based solely on historical performance, without accounting for the current context or state of the search. This reactive, rather than proactive, strategy may lead to suboptimal decisions—especially in highly dynamic and stochastic environments \citep{Reijnen_2024}. To overcome these limitations, we incorporate a deep reinforcement learning agent into the ALNS framework. DRL enables the algorithm to learn from the evolving state of the search process and proactively configure operator selection and acceptance strategies in each iteration. This integration allows for more intelligent decision-making, as the agent dynamically adapts based on the structure of the current solution rather than relying purely on past performance. As suggested in  \cite{Li_2024}, the ability of DRL to capture complex patterns and respond quickly to environmental changes makes it a natural fit for enhancing metaheuristic frameworks like ALNS. 

In addition to developing an efficient learning-based algorithm, generating high-quality scenarios is essential for capturing real-world uncertainty—particularly during testing after model training. To enhance the out-of-sample performance of our algorithm, we introduce two scenario pre-processing techniques: (i) an advanced scenario sampling method, and (ii) a decision-based scenario clustering approach. These steps improve the robustness of the DRL-ALNS solution and ensure that it performs well across diverse scenarios. The remainder of this section presents the full solution methodology. We first describe the scenario pre-processing steps, then outline the structure of the baseline ALNS algorithm, and finally explain how the DRL agent is integrated into the search process.

\subsection{Advanced Scenario Sampling}\label{subsec:ScenarioSampling}
In many stochastic optimization problems, the complete set of scenarios representing uncertain parameters is too large to be handled computationally. To address this, a representative subset of scenarios is often selected. However, enhancing the quality of this subset is crucial for improving the accuracy of the model’s results. One promising direction is to use more sophisticated sampling strategies that reflect the stochastic behavior of the system more effectively. Unlike standard random sampling methods, Quasi-Monte Carlo (QMC) techniques employ low-discrepancy sequences that aim to fill the sample space more evenly. This property helps reduce the variance of estimators and leads to more stable evaluations of expectations. However, QMC methods do not inherently include randomness, which limits their ability to generate unbiased estimators or confidence intervals. To overcome this limitation, Randomized Quasi-Monte Carlo (RQMC) methods have been introduced \citep{Cranley_1976}. RQMC introduces controlled randomness into low-discrepancy sequences, maintaining their uniform coverage while enabling unbiased statistical estimates. This hybrid approach offers the dual benefit of reduced estimator variance and the ability to assess reliability via confidence intervals, making it well-suited for generating efficient scenario sets with fewer samples.

\subsection{Decision-Based Scenario Clustering}\label{subsec:DBClustering}
This section introduces the proposed decision-based scenario clustering method, which is applied in two stages: (i) computing an opportunity cost distance between scenarios, and (ii) selecting representative scenarios. Before delving into these steps, we present the general formulation of the optimization model introduced in Section~\ref{sec:Model}. Suppose a decision-maker is faced with an optimization problem under uncertainty—such as selecting locations for ACFs given uncertain casualty levels. The decision variables are represented by the vector \(\mathbf{x}\), and the uncertainty is described by a random vector \(\xi\). Let \(\phi(\mathbf{x}, \xi)\) denote the objective function that evaluates the quality of decision \(\mathbf{x}\) under a specific realization \(\xi\). Assuming the decision-maker aims to minimize this objective, the problem can be expressed as:
\vspace{-0.75cm}
\begin{align}
    \min_{\mathbf{x} \in A} \mathbb{E}_{\xi}[\phi(\mathbf{x}, \xi)] \label{GeneralFormulation_1}
\end{align}

\vspace{-1 cm}
\noindent where \(A\) is the feasible region and \(\phi(\mathbf{x},\xi)\) denotes the cost when \(\mathbf{x}\) is implemented under realization \(\xi\).  In practice, a finite scenario set \(\mathcal{S}=\{1,\dots,S\}\) with probabilities \(\{p_s\}\) is drawn from a probability measure \(\mathbb{P}\), and one solves the sample‐average approximation:
\vspace{-0.75cm}
\begin{align}
    \min_{\mathbf{x} \in A} \sum_{s \in \mathcal{S}} p_s \phi(\mathbf{x}, \xi_s) \label{GeneralFormulation_2}
\end{align}

\vspace{-0.75 cm}
While this scenario-based formulation is widely used in stochastic optimization \citep{Birge_2011}, a key practical challenge is selecting which scenarios to retain. The number of potential realizations across all data sources is typically very large, making it computationally infeasible to include all of them. To address this, we propose a clustering approach that preserves scenario diversity based on decision relevance.

\paragraph{Step 1: Opportunity Cost Distance}

The goal is to interpret the information contained in each scenario \(\xi_s\) through the lens of its associated optimal decision. For each data source \(k \in \mathcal{K}\), such as synthetic aperture radar, drones, or mobile phone and GPS signals, let \(\mathcal{S}^k = \{s_1^k, \dots, s_{N_k}^k\}\) denote the set of scenarios derived from that source. We solve the deterministic version of the problem for each scenario \(s_i^k \in \mathcal{S}^k\), obtaining:
\vspace{-0.75cm}
\begin{align}
    \mathbf{x}(s_i^k) = \arg \min_{\mathbf{x} \in A} \phi(\mathbf{x}, \xi_{s_i^k}) \quad i = 1, \dots, N_k \label{GeneralFormulation_3}
\end{align}

\vspace{-0.75 cm}
Solution \(\mathbf{x}(s_i^k)\) represents the optimal decision assuming scenario \(s_i^k\) will occur with certainty. To quantify how “similar” two scenarios are from a decision-making standpoint, we use the \textit{opportunity cost distance} proposed by \citet{Hewitt_2022}. For any pair of scenarios \(s_1, s_2 \in \mathcal{S}\), this distance measures the cost incurred when using the decision optimized for one scenario in the context of the other:
\vspace{-0.75cm}
\begin{align}
    d(s_1, s_2) = \phi(\mathbf{x}(s_1), \xi_{s_2}) -\phi(\mathbf{x}(s_2), \xi_{s_2}) + \phi(\mathbf{x}(s_2), \xi_{s_1}) - \phi(\mathbf{x}(s_1), \xi_{s_1}) \label{GeneralFormulation_4}
\end{align}

\vspace{-0.75 cm}
A small value of \(d(s_1, s_2)\) implies that the two scenarios yield similar decisions, in the sense that their corresponding solutions perform well under both realizations. Using Equation~\eqref{GeneralFormulation_4}, we compute the full pairwise distance matrix across all scenarios in \(\mathcal{S}\).

\paragraph{Step 2: Scenario Selection}
Once the opportunity cost distance matrix has been constructed, the next step is to identify a representative subset of scenarios that reflects the diversity of the entire scenario space. Traditionally, this task is framed as a clustering problem over the scenario set \(\mathcal{S}\), aiming to group scenarios that are similar in terms of their decision impact. This grouping is performed using the distance matrix constructed in the previous step. We employ a greedy Max--Min diversity selection algorithm (see Algorithm~\ref{alg:MaxMinClustering}) inspired by farthest-point sampling. The algorithm begins by identifying the pair of scenarios with the greatest opportunity cost distance, and then iteratively adds the scenario that is farthest from its closest counterpart in the set of already selected scenarios. This ensures that each additional scenario brings in the most novel decision information relative to the current set, making the approach computationally efficient, scalable, and well-suited for scenario reduction in stochastic optimization.

\vspace{0.3cm}
\begin{algorithm}
\caption{Greedy Max--Min Scenario Selection} \label{alg:MaxMinClustering}
\begingroup
\resizebox{0.9\textwidth}{!}{ % Shrink the algorithm box
\begin{minipage}{1.0\textwidth}
\begin{algorithmic}[1]
\State \textbf{Input:} Distance matrix (\(D \in \mathbb{R}^{S \times S}\)), number of scenarios to select (\(k\))
\State \textbf{Output:} Set \(S'\) of \(k\) diverse scenario indices
\State Initialize \(S' \leftarrow \{i,j\}\), where \((i,j) = \arg\max_{i,j} D[i,j]\) \Comment{Select most distant initial pair $\ \ \ $}
\While{\(|S'| < k\)}
    \State Let \(R \leftarrow \mathcal{S} \setminus S'\)\Comment{Remaining unselected scenarios}
    \For{each \(r \in R\)}
        \State Compute \(d_{\min}(r) = \min_{s \in S'} D[r,s]\)
    \EndFor
    \State Select \(r^* = \arg\max_{r \in R} d_{\min}(r)\)
    \State Add \(r^*\) to \(S'\)
\EndWhile
\State \Return \(S'\)
\end{algorithmic}
\end{minipage}
}
\endgroup
\end{algorithm}

\subsection{Deep Reinforcement Learning-Based ALNS (DRL-ALNS)}\label{subsec:DRLALNS}
The two-stage stochastic MIP presented in Section \ref{sec:Model} is solved through a sequential decision-making process, where a reinforcement learning (RL) agent iteratively guides the ALNS destroy and repair operators. In each iteration, the agent treats the current first-stage solution as the state, selects a destroy/repair action to perturb this solution, and receives a reward based on the improvement in the objective after re-optimizing the MIP. A simulated annealing acceptance criterion then decides whether the new solution is accepted into the search. By embedding an RL agent within the ALNS framework, the method learns to adaptively select operators and escape local optima more effectively than a classical ALNS with fixed weight adaptation. We first formulate the DRL-ALNS as a Markov Decision Process (MDP), which is represented as a tuple \((S, \mathcal{A}, R)\), wherein $S$ represents the set of states, $\mathcal{A}$ represents the set of actions, and $R$ represents the reward function. The components of our formulation are:

\paragraph{State ($s$):} 
A representation of the current first-stage solution (decision vector) for the two-stage MIP. This includes all binary and integer first-stage decision variables (i.e. \(x_i, \vartheta_{mi}\), and, \(w_{hh'}\)). At the start ($t=0$), an initial state $s_0$ is obtained by solving a linear relaxation version of the model to generate a high-quality starting solution for the first-stage decisions. This initial optimization also provides an initial estimate of the value function.

\paragraph{Action ($a$):} 
A composite destroy-and-repair operator applied to the current solution. We define a discrete action space of heuristic operators for first-stage decisions. For binary variables, actions include either removing some currently selected ACFs or taking no action (NoDestroy), followed by either adding new ACFs or swapping between opened and closed ones. For integer variables, actions include either decreasing some values or taking no action (NoChange), followed by either increasing others or again applying NoChange. Each action $a$ thus specifies a pair of destroy/ repair heuristics for binary decisions and another pair for integer decisions. For example, an action might be $a=(\text{Remove}, \text{Add}, \text{Decrease}, \text{Increase})$, meaning we remove a subset of currently open ACFs, then add new ACFs, decrease some integer allocations, and then increase others. The magnitude of destruction is adaptive: at each iteration we randomly choose the number of elements to remove or change (e.g. between 10\% and 50\% of ACFs) . This ensures a variable neighborhood size (“destroy severity”). The RL agent’s policy will learn which operator combination to apply based on the current state.

\paragraph{Reward (\(r\)):}  
The reward reflects the improvement in the objective value. Let \(f(\mathbf{x})\) denote the total cost of the current first-stage solution \(\mathbf{x}\), calculated after solving all second-stage scenarios:

\begin{minipage}{\textwidth}
\footnotesize
\vspace{-0.4cm}
\begin{align}
f(\mathbf{x}) = \min \Bigg(& \varrho_{hh'} w_{hh'} + \sum_{s \in \mathcal{S}} p_s \Big(\sum_{t \in \mathcal{T}} \sum_{j \in \mathcal{J}} \Big[\sum_{l \in \mathcal{L}} \sum_{u \in \mathcal{U}} \sum_{m \in \mathcal{M}} \pi_{lu} q_{tjlums} + \sum_{h \in \mathcal{H}} \sum_{m \in \mathcal{M}} \Big(\sum_{u \in \mathcal{U}} \nu_j R^{L}_{tjhum} u^{L}_{tjhums} \nonumber \\
& + \sum_{i \in \mathcal{I}^A} \sum_{h' \in \text{BC}_h} \nu_j R^{A}_{tjhih'm} u^{A}_{tjhih'ms} \Big) + \sum_{l \in \mathcal{L}} \rho_j \mu_{tjls} + \sum_{\substack{h \in \mathcal{H}: \\ g_{sh} = 1}} \nu_j \Lambda_{Tjh} \varpi_{tjhs} \Big] \Big) \Bigg) \label{DRL_Obj}
\end{align}
\end{minipage}

At each iteration, the reward is computed as the difference between the cost of the current solution \(f(\mathbf{x})\) and the best cost found so far, i.e., \(f(\mathbf{x}^*)\):
\vspace{-0.75cm}
\begin{align}
r = f(\mathbf{x}) - f(\mathbf{x}^*) \label{DRL_reward}
\end{align}

\vspace{-0.75cm}
This reward encourages the agent to take actions that reduce the objective function. Since the goal of ALNS is cost minimization, using the cost difference as the reward aligns with the overall search objective. The RL agent learns a value function \(Q(s, a)\), estimating the long-term improvement in objective when action \(a\) is applied in state \(s\). The search proceeds iteratively. Each iteration \(t\) generates a new state \(s_{t+1}\) and reward \(r_t\), and continues either for a fixed number of iterations or until a stopping criterion is met (e.g., no improvement over a given number of steps).

\paragraph{Deep Q-Learning for Operator Selection}
We employ a Deep Q-Learning (DQL) agent to learn the value of state-action pairs $(s,a)$, i.e. the quality of applying each operator in a given solution state. Classic Q-learning uses a lookup table to store $Q(s,a)$ values, but in our setting the state $s$ is high-dimensional (hundreds of binary/integer variables) and the number of possible states is enormous. We therefore approximate the Q-value function with a deep neural network, which takes as input the state (encoded, for example, as a binary/continuous feature vector of the first-stage decisions) and returns Q-values for all possible actions . The network is trained online during the ALNS iterations via incremental Q-learning updates.

At each iteration \(t\), given state \(s_t\) (i.e., the current first-stage solution), the agent selects an action \(a_t\) using an \(\epsilon\)-greedy policy: with probability \(\epsilon\), it chooses a random operator (exploration), and with probability \(1 - \epsilon\), it chooses the action with the highest estimated \(Q(s_t, a)\) (exploitation). We initialize \(\epsilon = 1.0\) and gradually decay it to a small final value (e.g., 0.1) over the first few thousand iterations. This allows the search to start with broad exploration and become increasingly greedy as the agent gains confidence in its learned Q-values.

To stabilize learning, we maintain a replay buffer of past experiences \((s, a, r)\), from which mini-batches are sampled to update the Q-network. A separate target network is updated periodically (every 500 steps) to improve the stability of Q-value estimates.

The DQL agent is trained to satisfy the Bellman optimality condition. After observing reward \(r_t\) and next state \(s_{t+1}\), the network minimizes the temporal difference (TD) error:

\vspace{-0.75cm}
\begin{align}
Q(s_t,a_t) \;\leftarrow\; Q(s_t,a_t) \;+\; \alpha\Big[\,r_t \;+\; \gamma \max_{a'}Q(s_{t+1},a') \;-\; Q(s_t,a_t)\Big] \label{Bellman}
\end{align}

\vspace{-0.75cm}

\noindent where \(\alpha\) is the learning rate, and \(\gamma\) is a discount factor (we use \(\gamma = 0.9\) to mildly prioritize immediate rewards). The network parameters are updated via stochastic gradient descent on the mean-squared TD error, enabling the agent to learn which destroy/repair operators tend to produce better long-term improvements in the objective function.

In summary, the RL agent automates the operator selection scheme of ALNS by learning from experience rather than relying on hand-tuned weights. A traditional ALNS would adjust operator weights based only on recent performance (e.g. increasing weights for successful operators) . In contrast, our DQL agent can condition its choices on the current state of the solution and predict which operator is most effective in that context. This addresses a known limitation of classical ALNS weight updating, which “cannot take advantage of short-term dependencies between the current state of the search and the selection of operators”. By using deep Q-learning, the algorithm can capture complex patterns (for example, recognizing that certain destroy-repair moves work well when specific structures are present in the solution) and thus adapt more intelligently as the search progresses. A summary of the algorithm is provided in Algorithm \ref{alg:DRL-ALNS}.

\vspace{0.3cm}
\begin{algorithm}
\caption{DRL-ALNS Matheuristic Algorithm} \label{alg:DRL-ALNS}
\begingroup
\resizebox{0.9\textwidth}{!}{
\begin{minipage}{1.0\textwidth}
\begin{algorithmic}[1]
\State \textbf{Input:} Two-stage stochastic MIP model; operator set $\mathcal{A}$; maximum number of iterations $N$.
\State Solve the linear relaxation of the model (Section~\ref{sec:Model}) to obtain initial first-stage solution $\mathbf{x}^0$.
\State Set current solution $\mathbf{x} \gets \mathbf{x}^0$, best solution $\mathbf{x}^* \gets \mathbf{x}^0$, and best cost $f^* \gets f(\mathbf{x}^0)$.
\State Initialize the DQL agent with Q-network $Q_\theta(s, a)$ (state dimension = $|\mathbf{x}|$, action dimension = $|\mathcal{A}|$).
\State Initialize simulated annealing temperature $T \gets T_0$.
\For{$t = 1$ \textbf{to} $N$}
    \State Observe current state $s$ based on features of $\mathbf{x}$.
    \State Select action $a \in A$ using $\epsilon$-greedy policy based on $Q_\theta(s, a)$.
    \State Apply the destroy operator in $a$ to $\mathbf{x}$, then apply the repair operator to construct a new solution $\mathbf{x}'$.
    \State Fix $\mathbf{x}'$ as the first-stage decision, and solve the second-stage model (\ref{2_Obj})--(\ref{2_NonNegative}) to compute $f(\mathbf{x}')$.
    \State Compute reward $r \gets f(\mathbf{x}) - f(\mathbf{x}')$.
    \State Update the Q-network using the tuple $(s, a, r)$.
    \If{$f(\mathbf{x}') < f^*$}
        \State $\mathbf{x}^* \gets \mathbf{x}'$ \Comment{Update best solution $\ \ \ \ \ \ \ \ \ \ \ \ $}
        \State $f^* \gets f(\mathbf{x}')$
    \Else
        \State Compute acceptance probability $p \gets \min\{1,\; \exp[-(f(\mathbf{x}') - f(\mathbf{x}))/T]\}$
        \If{$p > \text{Uniform}(0,1)$}
            \State $\mathbf{x} \gets \mathbf{x}'$ \Comment{Accept $\mathbf{x}'$ as current solution}
            \State $f(\mathbf{x}) \gets f(\mathbf{x}')$
        \EndIf
    \EndIf
    \State $T \gets \rho \cdot T$ \Comment{Cool down temperature$\ \ \ \ \ \ \ \ $}
    \If{stopping criterion is met}
        \State \textbf{break}
    \EndIf
\EndFor
\State \textbf{Output:} Best solution $\mathbf{x}^*$ with objective value $f^*$.
\end{algorithmic}
\end{minipage}
}
\endgroup
\end{algorithm}

