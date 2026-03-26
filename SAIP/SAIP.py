import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv
from openai import OpenAI
import time
import re
import datetime
import json
from collections import defaultdict

# Load environment variables
load_dotenv()


# ====================== Enum Definitions ======================

class ClassificationStep(Enum):
    """Classification step enumeration"""
    INITIAL = "initial"  # Initial state
    ATOMICITY_CHECK = "atomicity_check"  # Atomicity check
    CONDITIONAL_CHECK = "conditional_check"  # Conditional check (new)
    USER_ACTION_CHECK = "user_action_check"  # User action check
    SYSTEM_CONDITION_CHECK = "system_condition_check"  # System condition check
    FINAL_CLASSIFICATION = "final_classification"  # Final classification


class RequirementType(Enum):
    """Requirement type enumeration"""
    COMPOSITE = "composite"  # Composite type
    CONDITIONAL = "conditional"  # Conditional type (new)
    INTERACTIVE = "interactive"  # Interactive type
    SEQUENTIAL = "sequential"  # Sequential type
    STRUCTURAL = "structural"  # Structural type
    UNKNOWN = "unknown"  # Unknown


class ActionType(Enum):
    """Step action type"""
    CHECK_ATOMICITY = "check_atomicity"
    CHECK_CONDITIONAL = "check_conditional"  # New
    CHECK_USER_ACTION = "check_user_action"
    CHECK_SYSTEM_CONDITION = "check_system_condition"
    MAKE_FINAL_CLASSIFICATION = "make_final_classification"
    REQUEST_CLARIFICATION = "request_clarification"


# ====================== Data Classes ======================

@dataclass
class ClassificationState:
    """Classification process state"""
    step: ClassificationStep  # Current step
    requirement_text: str  # Original requirement text
    intermediate_results: Dict[str, Any]  # Intermediate results
    history: List[Dict]  # History of steps
    is_complete: bool = False  # Whether complete
    final_classification: Optional[RequirementType] = None  # Final classification

    def to_dict(self):
        return {
            "step": self.step.value,
            "requirement": self.requirement_text[:100] + "..." if len(
                self.requirement_text) > 100 else self.requirement_text,
            "intermediate_results": self.intermediate_results,
            "history_length": len(self.history),
            "is_complete": self.is_complete,
            "final_classification": self.final_classification.value if self.final_classification else None
        }


@dataclass
class StepDecision:
    """Step decision result"""
    step: ClassificationStep
    action: ActionType
    decision: bool  # True/False decision result
    confidence: float  # Confidence 0.0-1.0
    reasoning: str  # Reasoning process
    evidence: List[str]  # Supporting evidence (extracted from requirement text)
    timestamp: str

    def to_dict(self):
        return {
            "step": self.step.value,
            "action": self.action.value,
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning": self.reasoning[:200] + "..." if len(self.reasoning) > 200 else self.reasoning,
            "evidence": self.evidence,
            "timestamp": self.timestamp
        }


@dataclass
class ClassificationEpisode:
    """Complete classification process record"""
    requirement_id: str
    requirement_text: str
    steps: List[StepDecision]
    final_classification: RequirementType
    ground_truth: Optional[RequirementType] = None
    total_reward: float = 0.0
    step_rewards: List[float] = None
    efficiency_metrics: Dict[str, float] = None

    def __post_init__(self):
        if self.step_rewards is None:
            self.step_rewards = []
        if self.efficiency_metrics is None:
            self.efficiency_metrics = {}

    def add_step(self, step_decision: StepDecision, step_reward: float):
        self.steps.append(step_decision)
        self.step_rewards.append(step_reward)

    def to_dict(self):
        return {
            "requirement_id": self.requirement_id,
            "requirement_text": self.requirement_text[:150] + "..." if len(
                self.requirement_text) > 150 else self.requirement_text,
            "steps_count": len(self.steps),
            "final_classification": self.final_classification.value,
            "ground_truth": self.ground_truth.value if self.ground_truth else None,
            "total_reward": self.total_reward,
            "step_rewards": self.step_rewards,
            "is_correct": self.ground_truth == self.final_classification if self.ground_truth else None,
            "efficiency": self.efficiency_metrics
        }


# ====================== Reinforcement Learning Agent ======================

class ClassificationAgent:
    """Requirement classification agent - executes multi-step decisions"""

    def __init__(self, llm_client, reward_calculator=None):
        self.llm_client = llm_client
        self.reward_calculator = reward_calculator or RewardCalculator()
        self.state: Optional[ClassificationState] = None
        self.current_episode: Optional[ClassificationEpisode] = None

        # Define decision tree state transitions - updated flow
        self.decision_tree = {
            ClassificationStep.INITIAL: {
                "next_step": ClassificationStep.ATOMICITY_CHECK,
                "action": ActionType.CHECK_ATOMICITY
            },
            ClassificationStep.ATOMICITY_CHECK: {
                "decision_false": {
                    "classification": RequirementType.COMPOSITE,
                    "is_terminal": True
                },
                "decision_true": {
                    "next_step": ClassificationStep.USER_ACTION_CHECK,
                    "action": ActionType.CHECK_USER_ACTION
                }
            },
            ClassificationStep.USER_ACTION_CHECK: {
                "decision_true": {
                    "classification": RequirementType.INTERACTIVE,
                    "is_terminal": True
                },
                "decision_false": {
                    "next_step": ClassificationStep.SYSTEM_CONDITION_CHECK,
                    "action": ActionType.CHECK_SYSTEM_CONDITION
                }
            },
            ClassificationStep.SYSTEM_CONDITION_CHECK: {
                "decision_true": {
                    "classification": RequirementType.SEQUENTIAL,
                    "is_terminal": True
                },
                "decision_false": {
                    "next_step": ClassificationStep.CONDITIONAL_CHECK,  # Place at the end
                    "action": ActionType.CHECK_CONDITIONAL
                }
            },
            ClassificationStep.CONDITIONAL_CHECK: {  # Check conditional last
                "decision_true": {
                    "classification": RequirementType.CONDITIONAL,  # Conditional requirement
                    "is_terminal": True
                },
                "decision_false": {
                    "classification": RequirementType.STRUCTURAL,  # If none match, it's structural
                    "is_terminal": True
                }
            }
        }

    def reset(self, requirement_text: str, requirement_id: str = None):
        """Reset agent state"""
        if requirement_id is None:
            requirement_id = f"req_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}_{hash(requirement_text) % 10000}"

        self.state = ClassificationState(
            step=ClassificationStep.INITIAL,
            requirement_text=requirement_text,
            intermediate_results={},
            history=[]
        )

        self.current_episode = ClassificationEpisode(
            requirement_id=requirement_id,
            requirement_text=requirement_text,
            steps=[],
            final_classification=RequirementType.UNKNOWN
        )

        return self.state

    def step(self) -> Tuple[StepDecision, ClassificationState, bool]:
        """Execute one step decision"""
        if self.state.is_complete:
            return None, self.state, True

        # Get current step configuration
        current_config = self.decision_tree.get(self.state.step, {})

        if not current_config:
            raise ValueError(f"Unknown step: {self.state.step}")

        # Execute decision for current step
        step_decision = self._execute_step_decision(self.state.step)

        # Record step
        self.state.history.append(step_decision.to_dict())
        self.current_episode.add_step(step_decision, 0.0)

        # Process decision result, update state
        is_terminal = self._process_decision_result(step_decision)

        # If terminal state, calculate final reward
        if is_terminal:
            self.state.is_complete = True
            self.current_episode.final_classification = self.state.final_classification

            # Calculate final reward
            if self.reward_calculator:
                total_reward, step_rewards = self.reward_calculator.calculate_episode_reward(
                    self.current_episode
                )
                self.current_episode.total_reward = total_reward
                self.current_episode.step_rewards = step_rewards

        return step_decision, self.state, is_terminal

    def _execute_step_decision(self, step: ClassificationStep) -> StepDecision:
        """Execute specific step decision"""

        # Build prompt for current step
        prompt = self._build_step_prompt(step, self.state.requirement_text)

        # Call LLM for decision
        response = self.llm_client.query_decision(prompt, step)

        # Parse response
        decision, confidence, reasoning, evidence = self._parse_llm_response(
            response, step
        )

        # Create decision record
        current_config = self.decision_tree[step]
        action = current_config.get("action",
                                    ActionType.MAKE_FINAL_CLASSIFICATION if step == ClassificationStep.FINAL_CLASSIFICATION
                                    else ActionType.CHECK_ATOMICITY
                                    )

        return StepDecision(
            step=step,
            action=action,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            timestamp=datetime.datetime.now().strftime("%H:%M:%S")
        )

    def _process_decision_result(self, step_decision: StepDecision) -> bool:
        """Process decision result, update state and return whether terminal"""
        current_config = self.decision_tree.get(self.state.step, {})

        # Determine next step based on decision result
        decision_key = "decision_true" if step_decision.decision else "decision_false"

        if decision_key in current_config:
            next_config = current_config[decision_key]

            if next_config.get("is_terminal", False):
                # Terminal state, set final classification
                self.state.final_classification = next_config["classification"]
                self.state.step = ClassificationStep.FINAL_CLASSIFICATION
                return True
            else:
                # Non-terminal state, transition to next step
                self.state.step = next_config["next_step"]
                return False
        else:
            # Default: continue to next step or terminate
            if "next_step" in current_config:
                self.state.step = current_config["next_step"]
                return False
            else:
                # No next step, set as terminal
                self.state.is_complete = True
                return True

    def _build_step_prompt(self, step: ClassificationStep, requirement_text: str) -> str:
        """Build step-specific prompt"""

        base_prompt = f"""You are a software requirements analysis expert. Please analyze the following requirement description:

Requirement Description: "{requirement_text}"

"""

        if step == ClassificationStep.ATOMICITY_CHECK:
            return base_prompt + """Please determine whether this requirement description satisfies "atomicity" (Atomic)?

Characteristics of atomic requirements:
1. Describes a single, complete functional point
2. Cannot be further decomposed into smaller independent requirements
3. Has clear completion criteria
4. Can deliver value independently once implemented

Please analyze:
1. Can this requirement be decomposed into multiple sub-requirements?
2. Does it describe a complete functional unit?
3. Are there clear completion criteria for implementing this requirement?

Please respond in the following format:
【Analysis】Your analysis process
【Decision】true/false (true indicates atomic, false indicates not atomic)
【Confidence】A number between 0.0-1.0
【Evidence】Keywords or phrases extracted from the requirement text supporting your decision, separated by commas"""

        elif step == ClassificationStep.USER_ACTION_CHECK:
            return base_prompt + """Please determine whether the execution conditions of this requirement include "actions that users must personally perform"?

User-performed actions include:
1. User clicks, inputs, selections and other interactive operations
2. User initiates requests or commands
3. User provides input data
4. User confirms or approves
5. User triggers a process

System-automated actions do not count, such as:
1. System scheduled tasks
2. System automatic responses
3. System background processing

Please analyze:
1. What actions are mentioned in the requirement description?
2. Do these actions require user execution?
3. Is there a clear user interaction component?

Please respond in the following format:
【Analysis】Your analysis process
【Decision】true/false (true indicates includes actions users must personally perform, false indicates does not include)
【Confidence】A number between 0.0-1.0
【Evidence】Keywords or phrases extracted from the requirement text supporting your decision, separated by commas"""

        elif step == ClassificationStep.SYSTEM_CONDITION_CHECK:
            return base_prompt + """Please determine whether the execution conditions of this requirement include "conditions the system must satisfy"?

System conditions include:
1. System state conditions (e.g., when the system is in X state)
2. Data conditions (e.g., when data meets condition Y)
3. Time conditions (e.g., specific time each day)
4. Event conditions (e.g., when an event occurs)
5. Preconditions (e.g., must complete A before executing B)

Note: This is different from conditional requirements. This emphasizes system's own conditions, not external conditions

Please analyze:
1. Are there prerequisites for executing the requirement?
2. Do these conditions require the system to actively check or satisfy?
3. Is there a clear dependency on system state?

Please respond in the following format:
【Analysis】Your analysis process
【Decision】true/false (true indicates includes conditions system must satisfy, false indicates does not include)
【Confidence】A number between 0.0-1.0
【Evidence】Keywords or phrases extracted from the requirement text supporting your decision, separated by commas"""

        elif step == ClassificationStep.CONDITIONAL_CHECK:  # New conditional check
            return base_prompt + """Please determine whether this requirement belongs to the "Conditional Requirement" type?

Core Characteristics of Conditional Requirements:
Used to describe mandatory prerequisites for system functionality to take effect or user operations to be executable, focusing on defining "necessary conditions" (i.e., "only if condition A is satisfied, can B be executed").

Typical expressions include:
- "only if... then...", "must... to...", "requires... to..."
- Modal verbs such as "must", "shall", "only", "cannot", etc.
- Expressions emphasizing necessary conditions like "if and only if", "unless... otherwise not"

Key Criteria:
1. Must be mandatory prerequisites; if conditions are not met, functionality/operation cannot be executed
2. Emphasizes logical relationship of "only if X, then Y"
3. Conditions must be necessary constraints for function execution, not optional conditions

【Special Distinction】:
- ❌ Most sentences with "if..." describing system reactions triggered by events (i.e., "sufficient conditions") belong to interactive requirements, not conditional requirements
- ✅ Key difference is whether it emphasizes mandatory prerequisites for function execution
- ❌ System default functions, general capabilities are not conditional requirements
- ✅ Must be specific prerequisites for specific functions

Typical Examples:
✅ Conditional requirements:
- "Only users with administrator privileges can delete system configurations"
- "Must complete real-name authentication to perform transaction operations"
- "If and only if inventory quantity is greater than 0, users can place orders for purchase"
- "User age must not be less than 18, otherwise cannot register account"

❌ Non-conditional requirements (may be interactive):
- "If user clicks save button, the system will save the currently edited content"
- "When user logs in successfully, the system displays a welcome page"
- "Users can choose to upload a profile picture"

Please analyze:
1. Does the requirement describe a mandatory prerequisite?
2. Does it emphasize the "only if... then..." necessary relationship?
3. If conditions are not met, is the related function completely non-executable?
4. Are expressions emphasizing necessity used, such as "must", "cannot", "only if... then..."?

Please respond in the following format:
【Analysis】Your analysis process
【Decision】true/false (true indicates conditional requirement, false indicates not conditional)
【Confidence】A number between 0.0-1.0
【Evidence】Keywords or phrases extracted from the requirement text supporting your decision, separated by commas"""

        else:
            return base_prompt + """Please classify this requirement.

Please respond in the following format:
【Analysis】Your analysis process
【Classification】composite/interactive/sequential/structural/conditional
【Confidence】A number between 0.0-1.0
【Evidence】Keywords or phrases extracted from the requirement text supporting your classification, separated by commas"""

    def _parse_llm_response(self, response: str, step: ClassificationStep) -> Tuple[bool, float, str, List[str]]:
        """Parse LLM response"""

        try:
            # Extract analysis section
            analysis_match = re.search(r'【Analysis】\s*([^【】]+)', response, re.DOTALL)
            reasoning = analysis_match.group(1).strip() if analysis_match else ""

            # Extract decision/classification section
            if step == ClassificationStep.FINAL_CLASSIFICATION:
                # Final classification
                classification_match = re.search(r'【Classification】\s*(\w+)', response)
                if classification_match:
                    class_str = classification_match.group(1).lower()
                    # Map classification string to boolean decision (simplified handling for final step)
                    decision = True  # Placeholder
                else:
                    decision = True
            else:
                # Boolean decision
                decision_match = re.search(r'【Decision】\s*(true|false)', response, re.IGNORECASE)
                if decision_match:
                    decision_str = decision_match.group(1).lower()
                    decision = decision_str == "true"
                else:
                    decision = False

            # Extract confidence
            confidence_match = re.search(r'【Confidence】\s*([0-9]*\.?[0-9]+)', response)
            if confidence_match:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = 0.5

            # Extract evidence
            evidence_match = re.search(r'【Evidence】\s*([^【】]+)', response)
            if evidence_match:
                evidence_text = evidence_match.group(1).strip()
                evidence = [e.strip() for e in evidence_text.split(',') if e.strip()]
            else:
                evidence = []

            return decision, confidence, reasoning, evidence

        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
            return False, 0.5, f"Parsing failed: {str(e)}", []


# ====================== LLM Client ======================

class DoubaoLLMClient:
    """Doubao LLM client"""

    def __init__(self, model_name: str = "doubao-seed-1-6-250615"):
        self.model_name = model_name

        # Initialize Doubao client
        api_key = os.getenv('DOUBAO_API_KEY')
        if not api_key:
            raise ValueError("DOUBAO_API_KEY environment variable not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3",
        )

        # System prompt
        self.system_prompt = """You are a professional software requirements analyst, skilled in analyzing requirement characteristics and performing classification.
Your task is to strictly follow the decision tree rules for analysis and provide detailed reasoning process."""

    def query_decision(self, prompt: str, step: ClassificationStep) -> str:
        """Query LLM for decision"""
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                stream=False
            )

            return completion.choices[0].message.content

        except Exception as e:
            print(f"LLM query failed: {e}")
            return f"""【Analysis】Due to technical issues, analysis cannot be completed.
【Decision】false
【Confidence】0.5
【Evidence】None"""


# ====================== Reward Calculator ======================

class RewardCalculator:
    """Reward calculator - based on multi-step decision process"""

    def __init__(self, config=None):
        self.config = config or self._default_config()

    def _default_config(self):
        return {
            "weights": {
                "accuracy": 0.40,
                "efficiency": 0.25,
                "confidence_quality": 0.20,
                "reasoning_quality": 0.15,
            },
            "step_rewards": {
                "correct_decision": 0.3,
                "incorrect_decision": -0.4,
                "high_confidence_correct": 0.1,
                "high_confidence_wrong": -0.2,
                "evidence_found": 0.05,
                "step_efficiency": 0.1,
            },
            "final_rewards": {
                "correct_classification": 2.0,
                "incorrect_classification": -1.5,
                "early_correct": 0.5,
                "optimal_path": 0.3,
            },
            "thresholds": {
                "high_confidence": 0.8,
                "low_confidence": 0.3,
                "max_steps": 5,  # Updated max steps
                "optimal_step_counts": {  # Optimal step count for each category
                    "composite": 1,  # Terminates at step 1
                    "interactive": 2,  # Terminates at step 2
                    "sequential": 3,  # Terminates at step 3
                    "conditional": 4,  # Terminates at step 4 (checked last)
                    "structural": 4  # Terminates at step 4 (after conditional)
                }
            }
        }

    def calculate_episode_reward(self, episode: ClassificationEpisode) -> Tuple[float, List[float]]:
        """Calculate complete episode reward"""
        step_rewards = []

        # 1. Calculate step rewards
        for i, step in enumerate(episode.steps):
            step_reward = self._calculate_step_reward(step, i, episode)
            step_rewards.append(step_reward)

        # 2. Calculate final reward
        final_reward = self._calculate_final_reward(episode)

        # 3. Combine total reward
        total_reward = sum(step_rewards) + final_reward

        # 4. Normalize (optional)
        total_reward = self._normalize_reward(total_reward, episode)

        return total_reward, step_rewards

    def _calculate_step_reward(self, step: StepDecision, step_index: int, episode: ClassificationEpisode) -> float:
        """Calculate single step reward"""
        reward = 0.0

        # Confidence quality reward
        reward += self._confidence_quality_reward(step.confidence)

        # Evidence extraction reward
        if step.evidence and len(step.evidence) > 0:
            reward += self.config["step_rewards"]["evidence_found"]

        # Reasoning quality assessment
        if len(step.reasoning) > 50:
            reward += 0.05

        return reward

    def _calculate_final_reward(self, episode: ClassificationEpisode) -> float:
        """Calculate final reward"""
        if episode.ground_truth is None:
            return 0.0

        reward = 0.0

        # 1. Final classification correctness
        is_correct = episode.final_classification == episode.ground_truth

        if is_correct:
            reward += self.config["final_rewards"]["correct_classification"]

            # 2. Check if optimal path
            optimal_steps = self.config["thresholds"]["optimal_step_counts"].get(
                episode.ground_truth.value, 3
            )

            if len(episode.steps) <= optimal_steps:
                reward += self.config["final_rewards"]["optimal_path"]

            # 3. Early correct termination reward
            if len(episode.steps) < optimal_steps:
                reward += self.config["final_rewards"]["early_correct"]
        else:
            reward += self.config["final_rewards"]["incorrect_classification"]

        # 4. Step efficiency penalty/reward
        if len(episode.steps) > self.config["thresholds"]["max_steps"]:
            reward -= 0.5

        return reward

    def _confidence_quality_reward(self, confidence: float) -> float:
        """Confidence quality reward"""
        if 0.4 <= confidence <= 0.7:
            return 0.1
        elif 0.7 < confidence <= 0.9:
            return 0.05
        elif confidence > 0.9:
            return -0.1
        else:
            return -0.05

    def _normalize_reward(self, reward: float, episode: ClassificationEpisode) -> float:
        """Normalize reward to reasonable range"""
        max_possible = (
                len(episode.steps) * 0.5 +
                self.config["final_rewards"]["correct_classification"] +
                self.config["final_rewards"]["optimal_path"] +
                self.config["final_rewards"]["early_correct"]
        )

        min_possible = (
                len(episode.steps) * -0.5 +
                self.config["final_rewards"]["incorrect_classification"] -
                0.5
        )

        if max_possible > min_possible:
            normalized = -1 + 3 * (reward - min_possible) / (max_possible - min_possible)
            return max(-1.0, min(2.0, normalized))
        else:
            return max(-1.0, min(2.0, reward))


# ====================== Processor and Main Program ======================

class SequentialClassificationProcessor:
    """Sequential classification processor"""

    def __init__(self, llm_client=None, agent=None):
        self.llm_client = llm_client or DoubaoLLMClient()
        self.agent = agent or ClassificationAgent(self.llm_client)
        self.results = []
        self.statistics = defaultdict(list)

    def classify_requirement(self, requirement_text: str,
                             requirement_id: str = None,
                             ground_truth: RequirementType = None) -> ClassificationEpisode:
        """Classify a single requirement"""

        print(f"\n{'=' * 60}")
        print(f"🔍 Starting requirement classification: {requirement_text[:80]}...")
        if ground_truth:
            print(f"📝 Ground truth: {ground_truth.value}")
        print(f"{'=' * 60}")

        # Reset agent
        state = self.agent.reset(requirement_text, requirement_id)

        step_count = 0
        is_complete = False

        while not is_complete and step_count < 10:
            step_count += 1

            print(f"\n📋 Step {step_count}: {state.step.value}")

            # Execute one step decision
            step_decision, new_state, is_complete = self.agent.step()

            if step_decision:
                print(f"  Decision: {'Pass' if step_decision.decision else 'Fail'}")
                print(f"  Confidence: {step_decision.confidence:.3f}")
                print(f"  Evidence: {', '.join(step_decision.evidence[:3])}" +
                      ("..." if len(step_decision.evidence) > 3 else ""))

            state = new_state

            if is_complete:
                print(f"\n✅ Classification complete!")
                print(f"  Final classification: {state.final_classification.value}")
                print(f"  Total steps: {step_count}")
                break

        # Get complete episode
        episode = self.agent.current_episode
        episode.ground_truth = ground_truth

        # Calculate reward
        if self.agent.reward_calculator:
            total_reward, step_rewards = self.agent.reward_calculator.calculate_episode_reward(episode)
            episode.total_reward = total_reward
            episode.step_rewards = step_rewards
            print(f"  Total reward: {total_reward:.3f}")

        # Record statistics
        self._record_statistics(episode)

        return episode

    def _record_statistics(self, episode: ClassificationEpisode):
        """Record statistical information"""
        self.results.append(episode)

        if episode.ground_truth:
            is_correct = episode.final_classification == episode.ground_truth
            self.statistics["accuracy"].append(1 if is_correct else 0)

        self.statistics["steps"].append(len(episode.steps))
        self.statistics["rewards"].append(episode.total_reward)
        self.statistics["confidences"].append(
            np.mean([s.confidence for s in episode.steps]) if episode.steps else 0
        )

    def process_dataset(self, dataset: List[Dict]) -> List[ClassificationEpisode]:
        """Process dataset"""
        episodes = []

        print(f"\n🚀 Starting batch processing of {len(dataset)} requirements")
        print(f"{'=' * 60}")

        for i, item in enumerate(dataset):
            requirement_text = item.get('requirement', item.get('content', ''))

            # Extract ground truth label (if exists)
            ground_truth = None
            if 'label' in item and item['label']:
                label_str = item['label'].strip().lower()
                for req_type in RequirementType:
                    if req_type.value == label_str:
                        ground_truth = req_type
                        break

            # Classify requirement
            episode = self.classify_requirement(
                requirement_text=requirement_text,
                requirement_id=f"req_{i + 1:04d}",
                ground_truth=ground_truth
            )

            episodes.append(episode)

            # Progress display
            if (i + 1) % 5 == 0:
                print(f"\n📊 Progress: {i + 1}/{len(dataset)}")
                if self.statistics.get("accuracy"):
                    accuracy = np.mean(self.statistics["accuracy"]) * 100
                    print(f"  Current accuracy: {accuracy:.1f}%")

            # Avoid API rate limits
            if i < len(dataset) - 1:
                time.sleep(1)

        return episodes

    def save_results(self, episodes: List[ClassificationEpisode], output_file: str):
        """Save results to Excel"""
        results_data = []

        for i, episode in enumerate(episodes):
            # Build step history string
            steps_summary = []
            for j, step in enumerate(episode.steps):
                steps_summary.append(
                    f"{step.step.value}: {'✓' if step.decision else '✗'}({step.confidence:.2f})"
                )

            row_data = {
                'Serial No.': i + 1,
                'Requirement ID': episode.requirement_id,
                'Requirement Content': episode.requirement_text,
                'Step Count': len(episode.steps),
                'Step Details': ' → '.join(steps_summary),
                'Final Classification': episode.final_classification.value,
                'Ground Truth': episode.ground_truth.value if episode.ground_truth else "N/A",
                'Consistent': '✅ Yes' if (
                        episode.ground_truth and
                        episode.final_classification == episode.ground_truth
                ) else '❌ No' if episode.ground_truth else 'N/A',
                'Total Reward': round(episode.total_reward, 3),
                'Average Confidence': round(np.mean([s.confidence for s in episode.steps]) if episode.steps else 0, 3),
                'Evidence Count': sum(len(s.evidence) for s in episode.steps),
                'Processing Time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Add detailed information for each step
            for j, step in enumerate(episode.steps):
                row_data[f'Step{j + 1}_Decision'] = step.step.value
                row_data[f'Step{j + 1}_Result'] = 'Pass' if step.decision else 'Fail'
                row_data[f'Step{j + 1}_Confidence'] = step.confidence
                row_data[f'Step{j + 1}_Evidence'] = ', '.join(step.evidence[:3]) + (
                    '...' if len(step.evidence) > 3 else ''
                )

            results_data.append(row_data)

        df = pd.DataFrame(results_data)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.',
                    exist_ok=True)

        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ Results saved to: {output_file}")

        # Save detailed report
        self._save_detailed_report(episodes, output_file)

    def _save_detailed_report(self, episodes: List[ClassificationEpisode], output_file: str):
        """Save detailed analysis report"""
        report_file = output_file.replace('.xlsx', '_detailed_report.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Software Requirements Sequential Classification System - Detailed Analysis Report\n")
            f.write("Classification Process: Atomicity → User Action → System Condition → Conditional → Final Classification\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generation Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Requirements: {len(episodes)}\n\n")

            # Statistics
            f.write("📊 Overall Statistics\n")
            f.write("-" * 80 + "\n")

            # Accuracy
            episodes_with_truth = [e for e in episodes if e.ground_truth]
            if episodes_with_truth:
                correct_count = sum(1 for e in episodes_with_truth
                                    if e.final_classification == e.ground_truth)
                accuracy = correct_count / len(episodes_with_truth) * 100
                f.write(f"Accuracy: {accuracy:.1f}% ({correct_count}/{len(episodes_with_truth)})\n")

            # Step statistics
            step_counts = [len(e.steps) for e in episodes]
            f.write(f"Average Steps: {np.mean(step_counts):.2f}\n")
            f.write(f"Minimum Steps: {np.min(step_counts)}\n")
            f.write(f"Maximum Steps: {np.max(step_counts)}\n")

            # Reward statistics
            rewards = [e.total_reward for e in episodes]
            f.write(f"Average Reward: {np.mean(rewards):.3f}\n")
            f.write(f"Maximum Reward: {np.max(rewards):.3f}\n")
            f.write(f"Minimum Reward: {np.min(rewards):.3f}\n\n")

            # Classification distribution
            f.write("📈 Classification Distribution\n")
            f.write("-" * 80 + "\n")

            from collections import Counter
            classification_counts = Counter([e.final_classification.value for e in episodes])
            for class_type, count in classification_counts.items():
                percentage = count / len(episodes) * 100
                f.write(f"{class_type}: {count} ({percentage:.1f}%)\n")

            f.write("\n")

            # Step efficiency analysis
            f.write("⚡ Step Efficiency Analysis\n")
            f.write("-" * 80 + "\n")

            # Optimal step counts per classification
            optimal_steps = {
                "composite": 1,
                "conditional": 2,
                "interactive": 3,
                "sequential": 4,
                "structural": 4
            }

            for class_type in ["composite", "conditional", "interactive", "sequential", "structural"]:
                class_episodes = [e for e in episodes
                                  if e.final_classification.value == class_type]
                if class_episodes:
                    avg_steps = np.mean([len(e.steps) for e in class_episodes])
                    optimal = optimal_steps.get(class_type, 3)
                    f.write(f"{class_type}: Average {avg_steps:.1f} steps (Optimal: {optimal} steps)\n")

            f.write("\n")

            # Inconsistency case analysis
            f.write("🔍 Inconsistency Case Analysis\n")
            f.write("-" * 80 + "\n")

            inconsistencies = []
            for e in episodes_with_truth:
                if e.final_classification != e.ground_truth:
                    inconsistencies.append(e)

            if inconsistencies:
                f.write(f"Found {len(inconsistencies)} inconsistent cases:\n\n")

                for i, e in enumerate(inconsistencies[:10]):
                    f.write(f"{i + 1}. Requirement ID: {e.requirement_id}\n")
                    f.write(f"   Model Classification: {e.final_classification.value}\n")
                    f.write(f"   Ground Truth: {e.ground_truth.value}\n")
                    f.write(f"   Steps: {len(e.steps)}\n")

                    # Show where error occurred
                    for j, step in enumerate(e.steps):
                        f.write(f"   Step{j + 1}({step.step.value}): {'✓' if step.decision else '✗'}\n")

                    f.write(f"   Requirement Content: {e.requirement_text[:100]}...\n\n")
            else:
                f.write("No inconsistent cases found.\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"✅ Detailed report saved to: {report_file}")

    def print_summary_statistics(self):
        """Print summary statistics"""
        print(f"\n{'=' * 60}")
        print("📊 Sequential Classification System - Summary Statistics")
        print(f"{'=' * 60}")

        if not self.statistics:
            print("No statistics data available")
            return

        # Accuracy
        if self.statistics.get("accuracy"):
            accuracy = np.mean(self.statistics["accuracy"]) * 100
            print(f"🎯 Accuracy: {accuracy:.1f}%")

        # Step statistics
        if self.statistics.get("steps"):
            avg_steps = np.mean(self.statistics["steps"])
            min_steps = np.min(self.statistics["steps"])
            max_steps = np.max(self.statistics["steps"])
            print(f"📈 Step Statistics: Average {avg_steps:.1f} steps (Range: {min_steps}-{max_steps})")

        # Reward statistics
        if self.statistics.get("rewards"):
            avg_reward = np.mean(self.statistics["rewards"])
            min_reward = np.min(self.statistics["rewards"])
            max_reward = np.max(self.statistics["rewards"])
            print(f"🏆 Reward Statistics: Average {avg_reward:.3f} (Range: {min_reward:.3f}-{max_reward:.3f})")

        # Confidence statistics
        if self.statistics.get("confidences"):
            avg_conf = np.mean(self.statistics["confidences"])
            print(f"📊 Average Confidence: {avg_conf:.3f}")

        print(f"\nTotal Requirements Processed: {len(self.results)}")


# ====================== Main Function ======================

def main():
    """Main function"""
    # File path configuration
    dataset_file = "dataset.xlsx"
    output_file = "sequential_classification_results.xlsx"

    print("=" * 80)
    print("🔧 Software Requirements Sequential Classification System")
    print("Classification Process: Atomicity → User Action → System Condition → Conditional → Final Classification")
    print("=" * 80)

    # Check environment variables
    print(f"\n🔧 Environment Check:")
    api_key = os.getenv('DOUBAO_API_KEY')

    if api_key:
        print(f"✅ DOUBAO_API_KEY is set")
    else:
        print(f"❌ DOUBAO_API_KEY is not set")
        return

    # Check data file
    if not os.path.exists(dataset_file):
        print(f"❌ Data file does not exist: {dataset_file}")
        return

    try:
        # Load data
        print(f"\n📁 Loading dataset...")
        df = pd.read_excel(dataset_file)

        # Convert to required format
        dataset = []
        for _, row in df.iterrows():
            if 'requirement' in df.columns:
                item = {'requirement': str(row['requirement'])}
                if 'label' in df.columns and pd.notna(row.get('label')):
                    item['label'] = str(row['label'])
                dataset.append(item)
            elif len(df.columns) > 0:
                item = {'requirement': str(row.iloc[0])}
                dataset.append(item)

        print(f"✅ Successfully loaded {len(dataset)} requirements")

        # Create processor and run
        print(f"\n🚀 Starting sequential classification processing...")

        processor = SequentialClassificationProcessor()
        episodes = processor.process_dataset(dataset)

        # Save results
        processor.save_results(episodes, output_file)

        # Print statistics
        processor.print_summary_statistics()

        # Display first few cases
        print(f"\n🔍 First 3 classification cases:")
        for i, episode in enumerate(episodes[:3]):
            print(f"\n{'=' * 60}")
            print(f"Case {i + 1}:")
            print(f"Requirement: {episode.requirement_text[:80]}...")
            print(f"Steps: {len(episode.steps)}")

            for j, step in enumerate(episode.steps):
                print(f"  Step{j + 1}({step.step.value}): {'Pass' if step.decision else 'Fail'} "
                      f"(Confidence: {step.confidence:.2f})")

            print(f"Final Classification: {episode.final_classification.value}")
            if episode.ground_truth:
                print(f"Ground Truth: {episode.ground_truth.value}")
                print(f"Consistent: {'✅ Yes' if episode.final_classification == episode.ground_truth else '❌ No'}")
            print(f"Total Reward: {episode.total_reward:.3f}")

        print(f"\n{'=' * 80}")
        print("🎉 Sequential classification processing complete!")
        print(f"📊 Results file: {output_file}")
        print(f"📄 Detailed report: {output_file.replace('.xlsx', '_detailed_report.txt')}")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
