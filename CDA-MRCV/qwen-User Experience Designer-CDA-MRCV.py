import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
import time
import re
import datetime
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import pearsonr

# Load environment variables
load_dotenv()


@dataclass
class ModelPrediction:
    """Stores model prediction results"""
    label: str
    confidence: float
    model_name: str
    attempts: List[Dict] = None
    human_label: str = None
    match_result: bool = False
    total_attempts: int = 0
    final_reason: str = ""
    calibration_error: float = None
    pearson_correlation: float = None
    agreement_status: str = ""  # Records LLM internal consistency status


@dataclass
class DataPoint:
    """Data point class"""
    content: str
    prediction: ModelPrediction = None
    final_label: str = None
    human_label: str = None


class SingleRoleQwenClient:
    """Single-role Qwen model client - User Experience Designer only"""

    def __init__(self, name: str = "qwen-plus-2025-07-28", model_name: str = "qwen-plus-2025-07-28"):
        self.name = name
        self.model_name = model_name

        api_key = os.getenv('DASHSCOPE_API_KEY')
        base_url = os.getenv('DASHSCOPE_BASE_URL')

        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is not set")
        if not base_url:
            raise ValueError("DASHSCOPE_BASE_URL environment variable is not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        self.role = {
            "name": "User Experience Designer",
            "system_prompt": """You are a User Experience Designer specializing in user interaction, interface design, and user experience optimization. You analyze requirements from the perspectives of user journey, interaction design, usability, accessibility, and emotional design. You focus on how requirements affect user satisfaction, ease of use, learning curve, and overall user experience. Your analysis focuses on how requirements are "perceived and used by users".

Please analyze according to the following requirements:
1. Conduct in-depth analysis of requirements from a user experience perspective
2. Select the most appropriate classification label
3. Evaluate your judgment confidence (0.0-1.0)
4. Provide clear analysis reasoning"""
        }

        print(f"✅ Successfully created single-role Qwen model client: {self.name}")
        print(f"  Professional role used: {self.role['name']}")
        print(f"  🔄 New attempt mechanism: At least 2 independent annotations, continue only if inconsistent")
        print(f"  📝 Subsequent annotations prioritize new categories")
        print(f"  📊 Includes confidence analysis: ECE and Pearson test")

    def analyze_with_independent_voting(self, text: str, labels: List[str], human_label: str = None) -> ModelPrediction:
        """Independent voting mechanism analysis: Execute at least 2 times, continue only if inconsistent, prioritize new categories for subsequent attempts"""
        print(f"\n🔍 Starting independent voting analysis: '{text[:50]}...'")
        print(f"  🏷️  Available labels: {', '.join(labels)}")
        print("-" * 60)

        attempts = []
        max_attempts = min(6, len(labels) + 1)  # Maximum 6 attempts or number of labels + 1
        tried_labels = set()  # Records labels that have been tried
        final_label = ""
        final_confidence = 0.0
        match_result = False
        final_reason = ""
        agreement_status = ""

        # Step 1: Perform at least 2 independent annotations
        print(f"    📌 Step 1: Perform at least 2 independent annotations...")

        # First independent annotation
        print(f"      1st independent annotation...")
        attempt1 = self._independent_vote(text, labels, 1, tried_labels, None)
        attempts.append(attempt1)
        tried_labels.add(attempt1["predicted_label"])

        # Second independent annotation (doesn't know first result, but can see tried labels)
        print(f"      2nd independent annotation (doesn't know 1st result)...")
        attempt2 = self._independent_vote(text, labels, 2, tried_labels, None)
        attempts.append(attempt2)
        tried_labels.add(attempt2["predicted_label"])

        # Check if first two are consistent
        label1 = attempt1["predicted_label"]
        label2 = attempt2["predicted_label"]
        confidence1 = attempt1["confidence"]
        confidence2 = attempt2["confidence"]

        if label1 == label2:
            # First two are consistent, end annotation
            agreement_status = "Consistent in first 2 attempts"
            final_label = label1
            final_confidence = (confidence1 + confidence2) / 2  # Average confidence
            print(f"    ✅ First 2 annotations consistent: '{final_label}' (confidence: {final_confidence:.3f})")
            print(f"    📌 Annotation completed, no more attempts needed")
        else:
            # First two are inconsistent, need to continue
            agreement_status = f"Inconsistent in first 2 attempts({label1} vs {label2})"
            print(f"    ⚠️ First 2 annotations inconsistent: '{label1}' vs '{label2}'")
            print(f"    🔄 Triggering 3rd annotation (prioritizing new categories)...")

            # Third annotation: prioritize new categories
            attempt3 = self._independent_vote(text, labels, 3, tried_labels, attempts[:2])
            attempts.append(attempt3)
            tried_labels.add(attempt3["predicted_label"])

            # Check if majority consistency is achieved
            labels_so_far = [a["predicted_label"] for a in attempts]
            label_counts = {}
            for label in labels_so_far:
                label_counts[label] = label_counts.get(label, 0) + 1

            # Find label with maximum votes
            max_count = max(label_counts.values())
            majority_labels = [l for l, c in label_counts.items() if c == max_count]

            if len(majority_labels) == 1 and max_count >= 2:
                # Majority consistency achieved (at least 2 votes)
                agreement_status += f" → Consistent after 3rd attempt"
                final_label = majority_labels[0]
                # Average confidence of all votes for this label
                relevant_confidences = [a["confidence"] for a in attempts if a["predicted_label"] == final_label]
                final_confidence = np.mean(relevant_confidences) if relevant_confidences else 0.5
                print(f"    ✅ Majority consistency achieved after 3rd attempt: '{final_label}' (confidence: {final_confidence:.3f})")
                print(f"    📌 Annotation completed")
            else:
                # Still no consistency, continue attempts (prioritize new categories)
                print(f"    ⚠️ Still no majority consistency after 3rd attempt, continuing (prioritizing new categories)...")

                # Continue until consistent or max attempts reached
                for attempt_num in range(4, max_attempts + 1):
                    # Stop if all labels have been tried
                    if len(tried_labels) >= len(labels):
                        print(f"    ℹ️ All {len(tried_labels)} labels tried, stopping attempts")
                        break

                    print(f"      {attempt_num}th annotation (prioritizing new categories)...")
                    attempt = self._independent_vote(text, labels, attempt_num, tried_labels, attempts)
                    attempts.append(attempt)
                    tried_labels.add(attempt["predicted_label"])

                    # Recheck for consistency
                    labels_so_far = [a["predicted_label"] for a in attempts]
                    label_counts = {}
                    for label in labels_so_far:
                        label_counts[label] = label_counts.get(label, 0) + 1

                    max_count = max(label_counts.values())
                    majority_labels = [l for l, c in label_counts.items() if c == max_count]

                    if len(majority_labels) == 1 and max_count >= 2:
                        agreement_status += f" → Consistent after {attempt_num}th attempt"
                        final_label = majority_labels[0]
                        relevant_confidences = [a["confidence"] for a in attempts if
                                                a["predicted_label"] == final_label]
                        final_confidence = np.mean(relevant_confidences) if relevant_confidences else 0.5
                        print(f"    ✅ Majority consistency achieved after {attempt_num}th attempt: '{final_label}'")
                        print(f"    📌 Annotation completed")
                        break
                    elif attempt_num == max_attempts or len(tried_labels) >= len(labels):
                        # Max attempts reached or all labels tried with no consistency
                        agreement_status += f" → Still inconsistent after max {len(attempts)} attempts"
                        # Select label with most votes, or highest confidence if tied
                        max_votes = max(label_counts.values())
                        candidate_labels = [l for l, c in label_counts.items() if c == max_votes]

                        if len(candidate_labels) == 1:
                            final_label = candidate_labels[0]
                        else:
                            # Tie: select label with highest average confidence
                            best_label = None
                            best_confidence = -1
                            for label in candidate_labels:
                                # Calculate average confidence for this label
                                label_confidences = [a["confidence"] for a in attempts if a["predicted_label"] == label]
                                avg_confidence = np.mean(label_confidences) if label_confidences else 0
                                if avg_confidence > best_confidence:
                                    best_confidence = avg_confidence
                                    best_label = label
                            final_label = best_label

                        # Calculate final confidence
                        relevant_confidences = [a["confidence"] for a in attempts if
                                                a["predicted_label"] == final_label]
                        final_confidence = np.mean(relevant_confidences) if relevant_confidences else 0.5

                        tried_count = len(tried_labels)
                        total_labels = len(labels)
                        print(f"    ⚠️ Still no consistency after max {len(attempts)} attempts (tried {tried_count}/{total_labels} labels)")
                        print(f"    📍 Final selection: '{final_label}' (confidence: {final_confidence:.3f})")

                        # Generate detailed reasoning
                        vote_summary = []
                        for label, count in label_counts.items():
                            # Get all confidences for this label
                            label_confs = [a["confidence"] for a in attempts if a["predicted_label"] == label]
                            avg_conf = np.mean(label_confs) if label_confs else 0
                            vote_summary.append(f"'{label}': {count} votes(avg confidence {avg_conf:.3f})")

                        final_reason = f"After {len(attempts)} voting rounds, distribution: {', '.join(vote_summary)}, final selection '{final_label}'"
                        break

        # Step 2: Compare with human annotation (for final validation only)
        if human_label:
            match_result = (final_label == human_label)
            if match_result:
                print(f"    🎯 Comparison with human annotation: ✅ Consistent (human label: {human_label})")
            else:
                print(f"    🎯 Comparison with human annotation: ❌ Inconsistent (model: {final_label}, human: {human_label})")
                if not final_reason:  # Add reason if none exists
                    final_reason = f"Model selected '{final_label}' after {len(attempts)} votes, but human annotation is '{human_label}'"
        else:
            print(f"    ℹ️ No human annotation available for comparison")
            match_result = True  # Default to consistent

        print(f"\n📊 Final Results:")
        print(f"  LLM Internal Status: {agreement_status}")
        print(f"  Model Final Prediction: {final_label}")
        print(f"  Final Confidence: {final_confidence:.3f}")
        print(f"  Total Voting Rounds: {len(attempts)}")
        print(f"  Tried Labels: {', '.join(sorted(tried_labels))}")
        if human_label:
            print(f"  Human Label: {human_label}")
            print(f"  Consistency: {'✅ Yes' if match_result else '❌ No'}")
        if final_reason:
            print(f"  Reasoning: {final_reason}")

        return ModelPrediction(
            label=final_label,
            confidence=final_confidence,
            model_name=f"{self.name}-independent-voting",
            attempts=attempts,
            human_label=human_label,
            match_result=match_result,
            total_attempts=len(attempts),
            final_reason=final_reason,
            agreement_status=agreement_status
        )

    def _independent_vote(self, text: str, labels: List[str], attempt_num: int,
                          tried_labels: set, previous_attempts: Optional[List[Dict]] = None) -> Dict:
        """Independent voting, prioritize new categories"""
        print(f"        🔄 Performing {attempt_num}th vote...")

        try:
            # Build prompt, adjust based on attempt number
            prompt = self._build_vote_prompt_with_priority(text, labels, attempt_num, tried_labels, previous_attempts)

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.role["system_prompt"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=self._get_temperature_for_attempt(attempt_num),
                max_tokens=300,
                stream=False
            )

            response_content = completion.choices[0].message.content
            predicted_label = self._extract_label_from_response(response_content, labels)
            confidence = self._extract_confidence_from_response(response_content)

            if predicted_label:
                print(f"          ✅ {attempt_num}th vote: {predicted_label} (confidence: {confidence:.3f})")

                # Check if it's a new label
                if predicted_label in tried_labels:
                    print(f"          ℹ️ Note: Selected already tried label '{predicted_label}'")
                else:
                    print(f"          🌟 Selected new label '{predicted_label}'")

                return {
                    "attempt_number": attempt_num,
                    "response": response_content,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "is_new_label": predicted_label not in tried_labels,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                }
            else:
                # If label extraction fails, select an untried label if possible
                untried_labels = [l for l in labels if l not in tried_labels]
                if untried_labels:
                    # Prioritize untried labels
                    fallback_label = untried_labels[0]
                    is_new = True
                else:
                    # Random selection if all labels tried
                    fallback_label = np.random.choice(labels)
                    is_new = False

                print(
                    f"          ⚠️ Label extraction failed for {attempt_num}th vote, selected: {fallback_label} {'(new label)' if is_new else '(tried label)'}")
                return {
                    "attempt_number": attempt_num,
                    "response": response_content,
                    "predicted_label": fallback_label,
                    "confidence": 0.5,
                    "is_new_label": is_new,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                }
        except Exception as e:
            print(f"          ❌ Exception in {attempt_num}th vote: {str(e)[:100]}")
            # Prioritize new labels even on exception
            untried_labels = [l for l in labels if l not in tried_labels]
            if untried_labels:
                fallback_label = untried_labels[0]
                is_new = True
            else:
                fallback_label = labels[0]
                is_new = False

            return {
                "attempt_number": attempt_num,
                "response": str(e),
                "predicted_label": fallback_label,
                "confidence": 0.5,
                "is_new_label": is_new,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }

    def _get_temperature_for_attempt(self, attempt_num: int) -> float:
        """Adjust temperature based on attempt number - minimal practical version"""
        temperature_map = {
            1: 0.3,  # First: deterministic
            2: 0.6,  # Start divergence
            3: 0.8,  # Maximum divergence
            4: 0.8,  # Maintain divergence
            5: 0.6,  # Start convergence
            6: 0.3,  # Final determination
        }
        return temperature_map.get(attempt_num, 0.5)

    def _build_vote_prompt_with_priority(self, text: str, labels: List[str], attempt_num: int,
                                         tried_labels: set, previous_attempts: Optional[List[Dict]] = None) -> str:
        """Build voting prompt, prioritize new categories"""
        labels_text = "\n".join([f"- {label}" for label in labels])

        if attempt_num == 1:
            # First vote: standard prompt
            return f"""As a User Experience Designer, independently analyze the following requirement:

Requirement Description:
"{text}"

Available Classification Labels:
{labels_text}

Please conduct an independent analysis from a user experience perspective and provide your judgment:

【User Experience Analysis】[Detailed analysis reasoning]
【Classification Label】[Must select one from the labels above]
【Confidence】[Decimal between 0.0-1.0]"""

        elif attempt_num == 2:
            # Second vote: independent analysis with tried labels notice
            tried_text = f"Note: The following labels have been tried: {', '.join(sorted(tried_labels))}" if tried_labels else ""

            return f"""As a User Experience Designer, please independently analyze the following requirement again:

Requirement Description:
"{text}"

{tried_text}

Available Classification Labels:
{labels_text}

Please provide independent analysis and judgment:

【Re-analysis】[Independent analysis]
【Classification Label】[Must select one from the labels above]
【Confidence】[Decimal between 0.0-1.0]"""

        else:
            # Third and subsequent votes: prioritize new categories
            untried_labels = [l for l in labels if l not in tried_labels]

            # Build history information
            history_text = ""
            if previous_attempts:
                history_text = "Previous voting history:\n"
                for attempt in previous_attempts:
                    history_text += f"{attempt['attempt_number']}th: {attempt['predicted_label']} "
                    history_text += f"(confidence: {attempt['confidence']:.3f})\n"

            priority_text = ""
            if untried_labels:
                if len(untried_labels) == 1:
                    priority_text = f"\nImportant Note: Only one untried label remains '{untried_labels[0]}', please prioritize it."
                else:
                    priority_text = f"\nImportant Note: Please prioritize the following untried labels: {', '.join(untried_labels)}"
            else:
                priority_text = "\nNote: All labels have been tried, please comprehensively consider all historical votes."

            return f"""As a User Experience Designer, please synthesize previous analyses and provide final judgment:

Requirement Description:
"{text}"

{history_text}

{priority_text}

Available Classification Labels:
{labels_text}

Please consider all analyses, with special focus on untried label possibilities:

【Comprehensive Analysis】[Synthesize history, specifically analyze untried label possibilities]
【Classification Label】[Must select one from the labels above]
【Confidence】[Decimal between 0.0-1.0]"""

    def _extract_label_from_response(self, response: str, labels: List[str]) -> str:
        """Extract label from response"""
        if not response or not labels:
            return labels[0] if labels else "Unknown"

        response_clean = response.strip()

        # 1. Try extracting from 【Classification Label】 format
        label_pattern = r'【Classification Label】\s*[:：]?\s*(.+)'
        label_match = re.search(label_pattern, response_clean, re.IGNORECASE | re.MULTILINE)
        if label_match:
            extracted = label_match.group(1).strip().strip('.,!?;:"\'')
            for label in labels:
                if label.lower() == extracted.lower():
                    return label

        # 2. Search for labels in entire response
        for label in labels:
            if label.lower() in response_clean.lower():
                return label

        # 3. Try other possible formats
        alt_patterns = [
            r'Classification[:：]\s*(.+)',
            r'Label[:：]\s*(.+)',
            r'Selection[:：]\s*(.+)',
            r'Result[:：]\s*(.+)',
            r'Final Classification[:：]\s*(.+)'
        ]

        for pattern in alt_patterns:
            match = re.search(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if match:
                extracted = match.group(1).strip().strip('.,!?;:"\'')
                for label in labels:
                    if label.lower() in extracted.lower():
                        return label

        # 4. Try finding bold or quoted labels
        bold_pattern = r'\*\*(.+?)\*\*|「(.+?)」|『(.+?)』|"(.+?)"|\'(.+?)\''
        bold_matches = re.findall(bold_pattern, response_clean)
        for match in bold_matches:
            for group in match:
                if group:
                    for label in labels:
                        if label.lower() == group.lower():
                            return label

        return ""  # Return empty string for caller to handle

    def _extract_confidence_from_response(self, response: str) -> float:
        """Extract confidence from response"""
        if not response:
            return 0.5

        response_clean = response.strip()

        confidence_patterns = [
            r'【Confidence】\s*[:：]?\s*([0-9]*\.?[0-9]+)',
            r'Confidence[:：]\s*([0-9]*\.?[0-9]+)',
            r'confidence[:：]\s*([0-9]*\.?[0-9]+)',
            r'Confidence\s*=\s*([0-9]*\.?[0-9]+)',
            r'Confidence Level[:：]\s*([0-9]*\.?[0-9]+)'
        ]

        for pattern in confidence_patterns:
            confidence_match = re.search(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    confidence = max(0.0, min(1.0, confidence))
                    return confidence
                except ValueError:
                    continue

        # Try finding numbers between 0-1 (looser matching)
        number_pattern = r'(0?\.\d{1,3}|1\.0{1,3}|0\.\d+|1\.0+)'
        number_matches = re.findall(number_pattern, response_clean)

        for num_str in number_matches:
            try:
                num = float(num_str)
                if 0 <= num <= 1:
                    return num
            except ValueError:
                continue

        # Try finding percentages
        percent_pattern = r'(\d{1,3})%'
        percent_matches = re.findall(percent_pattern, response_clean)
        for percent_str in percent_matches:
            try:
                percent = float(percent_str)
                if 0 <= percent <= 100:
                    return percent / 100.0
            except ValueError:
                continue

        return 0.5


class SingleRoleProcessor:
    """Single-role model processor - User Experience Designer only"""

    def __init__(self, single_role_client: SingleRoleQwenClient):
        self.single_role_client = single_role_client
        self.start_time = None
        self.end_time = None
        print(f"Single-role model processor initialized, using {single_role_client.role['name']} role")
        print(f"🔄 Independent voting mechanism: At least 2 attempts, continue only if inconsistent")
        print(f"🎯 Subsequent annotations: Prioritize new categories")

    def process_dataset(self, dataset_with_labels: List[Dict], labels: List[str]) -> List[DataPoint]:
        """Process labeled dataset"""
        data_points = []

        self.start_time = datetime.datetime.now()
        print(f"🚀 Independent voting analysis start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Starting to process {len(dataset_with_labels)} labeled software requirements...")
        print(f"Classification labels count: {len(labels)}")
        print(f"🎭 Role Configuration: User Experience Designer")
        print(f"🔄 Voting Mechanism: At least 2 independent annotations → if inconsistent → prioritize new labels → max {min(6, len(labels) + 1)} attempts")

        for i, item in enumerate(dataset_with_labels):
            text = item.get('requirement', item.get('content', ''))
            human_label = item.get('label', None)

            if i % 5 == 0 and i > 0:
                elapsed_time = datetime.datetime.now() - self.start_time
                processed = i
                remaining = len(dataset_with_labels) - i
                avg_time_per_req = elapsed_time / processed if processed > 0 else elapsed_time
                est_remaining = avg_time_per_req * remaining

                print(f"  📈 Progress: {i}/{len(dataset_with_labels)} - Time Elapsed: {elapsed_time} - Estimated Remaining: {est_remaining}")

            data_point = DataPoint(content=text, human_label=human_label)
            prediction = self.single_role_client.analyze_with_independent_voting(text, labels, human_label)

            data_point.prediction = prediction
            data_point.final_label = prediction.label
            data_points.append(data_point)

            if i < len(dataset_with_labels) - 1:
                time.sleep(1)  # Avoid API rate limits

        self.end_time = datetime.datetime.now()
        return data_points

    def calculate_calibration_metrics(self, results: List[DataPoint]):
        """Calculate calibration metrics: ECE and Pearson test"""
        confidences = []
        accuracies = []

        for dp in results:
            if dp.human_label and dp.prediction:
                confidences.append(dp.prediction.confidence)
                accuracies.append(1 if dp.prediction.match_result else 0)

        if len(confidences) < 10:
            print("⚠️ Insufficient sample size for reliable calibration analysis")
            return None, None

        # Calculate ECE (Expected Calibration Error)
        ece = self._calculate_ece(confidences, accuracies)

        # Calculate Pearson correlation coefficient
        if len(set(confidences)) > 1 and len(set(accuracies)) > 1:
            pearson_corr, p_value = pearsonr(confidences, accuracies)
        else:
            pearson_corr, p_value = 0.0, 1.0

        print(f"\n📊 Confidence Analysis Results:")
        print(f"  ECE (Expected Calibration Error): {ece:.4f}")
        print(f"  Pearson Correlation Coefficient: {pearson_corr:.4f} (p-value: {p_value:.4f})")
        print(f"  Sample Count: {len(confidences)}")

        # Interpret results
        if ece < 0.05:
            print(f"  ✅ ECE < 0.05, model calibration is good")
        elif ece < 0.1:
            print(f"  ⚠️ 0.05 ≤ ECE < 0.1, model calibration is average")
        else:
            print(f"  ❌ ECE ≥ 0.1, model calibration is poor")

        if abs(pearson_corr) > 0.3:
            direction = "positive" if pearson_corr > 0 else "negative"
            print(f"  ✅ Pearson correlation |r| > 0.3, confidence and accuracy have {direction} correlation")
        elif abs(pearson_corr) > 0.1:
            print(f"  ⚠️ 0.1 ≤ |correlation| ≤ 0.3, weak correlation")
        else:
            print(f"  ℹ️ |correlation| < 0.1, no significant linear relationship between confidence and accuracy")

        return ece, pearson_corr

    def _calculate_ece(self, confidences: List[float], accuracies: List[int], n_bins: int = 10) -> float:
        """Calculate Expected Calibration Error"""
        confidences = np.array(confidences)
        accuracies = np.array(accuracies)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        return ece

    def save_results(self, results: List[DataPoint], output_file: str):
        """Save final results to Excel file"""
        results_data = []

        for i, data_point in enumerate(results):
            attempts_info = ""
            attempts_summary = []
            tried_labels_set = set()
            new_label_count = 0

            if data_point.prediction and data_point.prediction.attempts:
                for a in data_point.prediction.attempts:
                    label = a["predicted_label"]
                    is_new = a.get("is_new_label", False)
                    star = "🌟" if is_new else ""
                    attempts_summary.append(f"Attempt {a['attempt_number']}: {label}{star}({a['confidence']:.3f})")

                    # Count tried labels and new label count
                    tried_labels_set.add(label)
                    if is_new:
                        new_label_count += 1

                attempts_info = "; ".join(attempts_summary)

            row_data = {
                'ID': i + 1,
                'Requirement Content': data_point.content,
                'LLM Internal Status': data_point.prediction.agreement_status if data_point.prediction else "",
                'Human Label': data_point.human_label or "N/A",
                'Model Final Prediction': data_point.final_label,
                'Consistency': '✅ Consistent' if data_point.prediction and data_point.prediction.match_result else '❌ Inconsistent',
                'Total Voting Attempts': data_point.prediction.total_attempts if data_point.prediction else 0,
                'Tried Labels Count': len(tried_labels_set),
                'New Labels Count': new_label_count,
                'Inconsistency Reason': data_point.prediction.final_reason if data_point.prediction else "",
                'Voting History': attempts_info,
                'Final Confidence': round(data_point.prediction.confidence, 3) if data_point.prediction else 0.0,
                'Notes': data_point.prediction.agreement_status if data_point.prediction else ""
            }
            results_data.append(row_data)

        df = pd.DataFrame(results_data)
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

        # Save to Excel
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Prediction Results', index=False)

            # Add statistics sheet
            stats_df = self._create_statistics_sheet(results)
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)

            # Add label exploration analysis sheet
            exploration_df = self._create_exploration_analysis_sheet(results)
            exploration_df.to_excel(writer, sheet_name='Label Exploration Analysis', index=False)

        print(f"\n✅ Independent voting analysis results saved to: {output_file}")

        # Calculate and save confidence analysis
        ece, pearson_corr = self.calculate_calibration_metrics(results)

        # Save detailed report
        self._save_detailed_report(results, output_file, ece, pearson_corr)

    def _create_statistics_sheet(self, results: List[DataPoint]) -> pd.DataFrame:
        """Create statistics sheet"""
        total = len(results)
        labeled = sum(1 for dp in results if dp.human_label)
        matched = sum(1 for dp in results if dp.prediction and dp.prediction.match_result)

        # LLM internal consistency statistics
        agreement_stats = {}
        vote_distribution = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

        # Label exploration statistics
        total_labels_explored = 0
        avg_labels_per_req = 0

        for dp in results:
            if dp.prediction:
                # Record consistency status
                status = dp.prediction.agreement_status
                agreement_stats[status] = agreement_stats.get(status, 0) + 1

                # Record voting attempt distribution
                attempts = dp.prediction.total_attempts
                if attempts in vote_distribution:
                    vote_distribution[attempts] += 1

                # Count tried labels
                tried_labels = set()
                for attempt in dp.prediction.attempts:
                    tried_labels.add(attempt["predicted_label"])
                total_labels_explored += len(tried_labels)

        avg_labels_per_req = total_labels_explored / total if total > 0 else 0

        stats_data = []
        stats_data.append(["Total Requirements", total])
        stats_data.append(["With Human Labels", labeled])
        if labeled > 0:
            stats_data.append(["Consistent with Human Labels", matched])
            stats_data.append(["Consistency Rate", f"{(matched / labeled * 100):.1f}%"])

        stats_data.append(["", ""])  # Empty line

        # Label exploration statistics
        stats_data.append(["Label Exploration Statistics", ""])
        stats_data.append(["Total Unique Tried Labels", total_labels_explored])
        stats_data.append(["Avg Labels Tried Per Requirement", f"{avg_labels_per_req:.2f}"])

        stats_data.append(["", ""])  # Empty line

        # LLM internal consistency status
        stats_data.append(["LLM Internal Consistency Statistics", ""])
        for status, count in sorted(agreement_stats.items()):
            stats_data.append([status, f"{count} items ({(count / total * 100):.1f}%)"])

        stats_data.append(["", ""])  # Empty line

        # Voting attempt distribution
        stats_data.append(["Voting Attempt Distribution", ""])
        for attempts, count in sorted(vote_distribution.items()):
            if count > 0:
                stats_data.append([f"{attempts} voting attempts", f"{count} items ({(count / total * 100):.1f}%)"])

        return pd.DataFrame(stats_data, columns=["Metric", "Value"])

    def _create_exploration_analysis_sheet(self, results: List[DataPoint]) -> pd.DataFrame:
        """Create label exploration analysis sheet"""
        exploration_data = []

        # Count label exploration for each requirement
        for i, dp in enumerate(results):
            if dp.prediction and dp.prediction.attempts:
                tried_labels = set()
                new_label_count = 0
                exploration_path = []

                for attempt in dp.prediction.attempts:
                    label = attempt["predicted_label"]
                    is_new = label not in tried_labels
                    tried_labels.add(label)

                    if is_new:
                        new_label_count += 1
                        exploration_path.append(f"{label}🌟")
                    else:
                        exploration_path.append(f"{label}")

                exploration_data.append({
                    'ID': i + 1,
                    'Final Label': dp.final_label,
                    'Total Voting Attempts': dp.prediction.total_attempts,
                    'Tried Labels Count': len(tried_labels),
                    'New Labels Count': new_label_count,
                    'Exploration Path': ' → '.join(exploration_path),
                    'LLM Status': dp.prediction.agreement_status
                })

        return pd.DataFrame(exploration_data)

    def _save_detailed_report(self, results: List[DataPoint], output_file: str, ece: float, pearson_corr: float):
        """Save detailed analysis report"""
        report_file = output_file.replace('.xlsx', '_Detailed_Report.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Qwen Model - Independent Voting Classification System Analysis Report\n")
            f.write("Special Configuration: At least 2 independent annotations → continue only if inconsistent → prioritize new labels → max 6 attempts\n")
            f.write("Human Labels: For final validation only\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Analysis Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Requirements: {len(results)}\n")

            # Consistency rate statistics
            labeled = sum(1 for dp in results if dp.human_label)
            matched = sum(1 for dp in results if dp.prediction and dp.prediction.match_result)

            f.write("\n📊 Human Label Consistency Statistics:\n")
            f.write("-" * 80 + "\n")
            if labeled > 0:
                f.write(f"Items with human labels: {labeled}\n")
                f.write(f"Consistent items: {matched}\n")
                f.write(f"Consistency rate: {(matched / labeled * 100):.1f}%\n")

            # Label exploration statistics
            total_labels_explored = 0
            for dp in results:
                if dp.prediction:
                    tried_labels = set()
                    for attempt in dp.prediction.attempts:
                        tried_labels.add(attempt["predicted_label"])
                    total_labels_explored += len(tried_labels)

            avg_labels_per_req = total_labels_explored / len(results) if results else 0

            f.write("\n🔍 Label Exploration Statistics:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total unique tried labels: {total_labels_explored}\n")
            f.write(f"Average labels tried per requirement: {avg_labels_per_req:.2f}\n")

            # LLM internal consistency statistics
            agreement_stats = {}
            for dp in results:
                if dp.prediction and dp.prediction.agreement_status:
                    status = dp.prediction.agreement_status
                    agreement_stats[status] = agreement_stats.get(status, 0) + 1

            f.write("\n🤖 LLM Internal Consistency Status Statistics:\n")
            f.write("-" * 80 + "\n")
            for status, count in sorted(agreement_stats.items()):
                f.write(f"{status}: {count} items ({(count / len(results) * 100):.1f}%)\n")

            # Confidence analysis
            if ece is not None and pearson_corr is not None:
                f.write("\n📈 Confidence Analysis:\n")
                f.write("-" * 80 + "\n")
                f.write(f"ECE (Expected Calibration Error): {ece:.4f}\n")
                f.write(f"Pearson Correlation Coefficient: {pearson_corr:.4f}\n")

                # Interpret ECE
                if ece < 0.05:
                    f.write("ECE Interpretation: < 0.05, good model calibration\n")
                elif ece < 0.1:
                    f.write("ECE Interpretation: 0.05-0.1, average model calibration\n")
                else:
                    f.write("ECE Interpretation: ≥ 0.1, poor model calibration\n")

            # Voting efficiency analysis
            total_votes = sum(dp.prediction.total_attempts for dp in results if dp.prediction)
            avg_votes = total_votes / len(results) if results else 0

            f.write("\n🔄 Voting Efficiency Analysis:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total voting attempts: {total_votes}\n")
            f.write(f"Average voting attempts per requirement: {avg_votes:.2f}\n")

            # New label exploration efficiency
            total_new_labels = 0
            for dp in results:
                if dp.prediction:
                    tried_labels = set()
                    for attempt in dp.prediction.attempts:
                        if attempt.get("is_new_label", False):
                            total_new_labels += 1

            avg_new_per_vote = total_new_labels / total_votes if total_votes > 0 else 0

            f.write(f"Total new label explorations: {total_new_labels}\n")
            f.write(f"Average new labels explored per vote: {avg_new_per_vote:.2f}\n")

            # Inconsistency case analysis
            inconsistencies = []
            for i, dp in enumerate(results):
                if dp.human_label and dp.prediction and not dp.prediction.match_result:
                    tried_labels = set()
                    for attempt in dp.prediction.attempts:
                        tried_labels.add(attempt["predicted_label"])

                    inconsistencies.append({
                        'index': i + 1,
                        'model': dp.final_label,
                        'human': dp.human_label,
                        'attempts': dp.prediction.total_attempts,
                        'tried_labels': len(tried_labels),
                        'status': dp.prediction.agreement_status,
                        'reason': dp.prediction.final_reason[:100] if dp.prediction.final_reason else ""
                    })

            if inconsistencies:
                f.write(f"\n💭 Human Label Inconsistency Case Analysis ({len(inconsistencies)} total):\n")
                f.write("-" * 80 + "\n")
                for inc in inconsistencies[:10]:  # Show first 10 only
                    f.write(f"Item {inc['index']}: \n")
                    f.write(
                        f"  Model Prediction: '{inc['model']}' ({inc['attempts']} votes, {inc['tried_labels']} labels tried)\n")
                    f.write(f"  Human Label: '{inc['human']}'\n")
                    f.write(f"  LLM Status: {inc['status']}\n")
                    if inc['reason']:
                        f.write(f"  Reason: {inc['reason']}\n")
                if len(inconsistencies) > 10:
                    f.write(f"... {len(inconsistencies) - 10} more inconsistency cases\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"✅ Detailed analysis report saved to: {report_file}")

    def print_statistics(self, results: List[DataPoint]):
        """Print statistics information"""
        print(f"\n{'=' * 60}")
        print("📊 Qwen Model - Independent Voting Classification Statistics")
        print(f"🔍 Label Exploration Priority Mechanism")
        print(f"{'=' * 60}")

        print(f"Total Data Processed: {len(results)} requirements")
        print(f"Role Used: User Experience Designer")

        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time
            print(f"\n⏰ Time Statistics:")
            print(f"  Total Runtime: {total_duration}")
            if len(results) > 0:
                avg_time_per_req = total_duration / len(results)
                print(f"  Average Processing Time Per Requirement: {avg_time_per_req}")

        # Human label consistency rate
        labeled = sum(1 for dp in results if dp.human_label)
        matched = sum(1 for dp in results if dp.prediction and dp.prediction.match_result)

        print(f"\n🎯 Human Label Consistency Rate:")
        if labeled > 0:
            match_rate = matched / labeled * 100
            print(f"  Items with human labels: {labeled}")
            print(f"  Consistent items: {matched}")
            print(f"  Consistency rate: {match_rate:.1f}%")
        else:
            print(f"  No human label data available")

        # Label exploration statistics
        total_labels_explored = 0
        total_new_labels = 0
        for dp in results:
            if dp.prediction:
                tried_labels = set()
                for attempt in dp.prediction.attempts:
                    tried_labels.add(attempt["predicted_label"])
                    if attempt.get("is_new_label", False):
                        total_new_labels += 1
                total_labels_explored += len(tried_labels)

        avg_labels_per_req = total_labels_explored / len(results) if results else 0

        print(f"\n🔍 Label Exploration Statistics:")
        print(f"  Total unique tried labels: {total_labels_explored}")
        print(f"  Average labels tried per requirement: {avg_labels_per_req:.2f}")
        print(f"  Total new label explorations: {total_new_labels}")

        # LLM internal consistency status
        agreement_stats = {}
        for dp in results:
            if dp.prediction and dp.prediction.agreement_status:
                status = dp.prediction.agreement_status
                agreement_stats[status] = agreement_stats.get(status, 0) + 1

        print(f"\n🤖 LLM Internal Consistency Status:")
        for status, count in sorted(agreement_stats.items()):
            percentage = count / len(results) * 100
            print(f"  {status}: {count} items ({percentage:.1f}%)")

        # Voting attempt distribution
        vote_dist = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
        total_votes = 0
        for dp in results:
            if dp.prediction:
                attempts = dp.prediction.total_attempts
                total_votes += attempts
                if attempts in vote_dist:
                    vote_dist[attempts] += 1

        print(f"\n🔄 Voting Attempt Distribution:")
        for attempts, count in sorted(vote_dist.items()):
            if count > 0:
                percentage = count / len(results) * 100
                print(f"  {attempts} voting attempts: {count} items ({percentage:.1f}%)")

        print(f"  Average voting attempts: {total_votes / len(results):.2f}")


# Original DataLoader class remains unchanged
class DataLoader:
    """Data loader class"""

    @staticmethod
    def load_dataset_with_labels(file_path: str) -> List[Dict]:
        """Load requirements and human annotations from dataset file"""
        try:
            df = pd.read_excel(file_path)
            dataset_with_labels = []

            if 'requirement' not in df.columns:
                print(f"⚠️ No 'requirement' column in file, using first column as requirement content")
                requirement_col = df.columns[0]
            else:
                requirement_col = 'requirement'

            if 'label' not in df.columns:
                print(f"⚠️ No 'label' column in file, annotation comparison unavailable")
                label_col = None
            else:
                label_col = 'label'

            for _, row in df.iterrows():
                if pd.notna(row[requirement_col]):
                    item = {'requirement': str(row[requirement_col]).strip()}
                    if label_col and pd.notna(row.get(label_col)):
                        item['label'] = str(row[label_col]).strip()
                    dataset_with_labels.append(item)

            print(f"✅ Successfully loaded {len(dataset_with_labels)} labeled test requirements")
            return dataset_with_labels

        except Exception as e:
            print(f"❌ Error loading dataset file: {e}")
            return []

    @staticmethod
    def load_categories_and_explanations(file_path: str) -> Dict[str, str]:
        """Load categories and explanations from concept file"""
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            categories = {}
            for _, row in df.iterrows():
                if pd.notna(row['category']):
                    category = str(row['category']).strip()
                    explanation = str(row['explanation']).strip() if pd.notna(row.get('explanation')) else ""
                    categories[category] = explanation
            print(f"✅ Successfully loaded {len(categories)} classification labels")
            return categories
        except Exception as e:
            print(f"❌ Error loading category file: {e}")
            return {}


def main():
    """Main function"""
    # File path configuration
    dataset_file = "dataset.xlsx"
    concept_file = "1123Concept.xlsx"
    output_file = "qwen_UXDesigner_priority_new_labels.xlsx"

    data_loader = DataLoader()

    print("=" * 80)
    print("🎨 Qwen Model - Independent Voting Classification System (Prioritize New Labels)")
    print("🔄 Voting Mechanism: At least 2 independent annotations → continue only if inconsistent")
    print("🌟 Key Feature: Subsequent annotations prioritize new categories")
    print("🎯 Human Labels: For final validation only")
    print("📊 Includes: Accuracy + LLM Internal Consistency Rate + Label Exploration Statistics + ECE + Pearson Test")
    print("=" * 80)

    # Check environment variables
    api_key = os.getenv('DASHSCOPE_API_KEY')
    base_url = os.getenv('DASHSCOPE_BASE_URL')

    if not api_key:
        print("❌ DASHSCOPE_API_KEY is not set")
        return

    # Check data files
    if not all(os.path.exists(f) for f in [dataset_file, concept_file]):
        print("❌ Data files do not exist")
        return

    try:
        # Create single-role client
        print(f"\n🚀 Creating single-role model client...")
        single_role_client = SingleRoleQwenClient(
            name="Qwen-Plus-UXDesigner-PriorityNew",
            model_name="qwen-plus-2025-07-28"
        )
    except Exception as e:
        print(f"❌ Failed to create model client: {e}")
        return

    # Load data
    print(f"\n📚 Loading data files...")
    category_explanations = data_loader.load_categories_and_explanations(concept_file)
    labels = list(category_explanations.keys())
    dataset_with_labels = data_loader.load_dataset_with_labels(dataset_file)

    if not dataset_with_labels or not labels:
        print("❌ Data loading failed")
        return

    print(f"\n✅ Data loading completed:")
    print(f"  Classification labels: {len(labels)}")
    print(f"  Test requirements: {len(dataset_with_labels)}")

    # Execute processing
    print(f"\n🚀 Starting independent voting classification processing (prioritize new labels)...")
    processor = SingleRoleProcessor(single_role_client)
    results = processor.process_dataset(dataset_with_labels, labels)

    # Save and display results
    processor.save_results(results, output_file)
    processor.print_statistics(results)

    # Display sample results
    print(f"\n🔍 Detailed information of first 3 results:")
    for i, data_point in enumerate(results[:3]):
        print(f"\n{'=' * 60}")
        print(f"{i + 1}. Requirement: {data_point.content[:80]}...")
        print(f"   LLM Internal Status: {data_point.prediction.agreement_status if data_point.prediction else 'N/A'}")
        print(f"   Model Final Prediction: {data_point.final_label}")
        print(f"   Final Confidence: {data_point.prediction.confidence:.3f if data_point.prediction else 0.0}")
        print(f"   Total Voting Attempts: {data_point.prediction.total_attempts if data_point.prediction else 0}")

        if data_point.prediction and data_point.prediction.attempts:
            print(f"   Voting History (🌟 = new label):")
            for attempt in data_point.prediction.attempts:
                is_new = attempt.get("is_new_label", False)
                star = "🌟" if is_new else ""
                print(f"     {attempt['attempt_number']}th: {attempt['predicted_label']}{star} "
                      f"(confidence: {attempt['confidence']:.3f})")

        if data_point.human_label:
            print(f"   Human Label: {data_point.human_label}")
            print(f"   Consistency: {'✅ Yes' if data_point.prediction.match_result else '❌ No'}")

    print(f"\n{'=' * 80}")
    print("🎉 Independent voting classification processing completed!")
    print(f"📊 Results file: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
