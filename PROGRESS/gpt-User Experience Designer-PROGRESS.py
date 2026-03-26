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

# Load environment variables
load_dotenv()


@dataclass
class ModelPrediction:
    """Store model prediction results"""
    label: str
    confidence: float
    model_name: str
    attempts: List[Dict] = None  # Record of each attempt
    human_label: str = None  # Human annotation
    match_result: bool = False  # Whether it matches human annotation
    total_attempts: int = 0  # Total number of attempts
    final_reason: str = ""  # Final reason for inconsistency


@dataclass
class DataPoint:
    """Data point class"""
    content: str
    prediction: ModelPrediction = None
    final_label: str = None
    human_label: str = None  # Human annotation label


class SingleRoleGPTClient:
    """Single-role GPT model client class - User Experience Designer only"""

    def __init__(self, name: str = "gpt-4.1-nano", model_name: str = "gpt-4.1-nano"):
        self.name = name
        self.model_name = model_name

        # GPT API configuration - using OpenAI proxy
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(
            base_url='https://api.openai-proxy.org/v1',
            api_key=api_key,
        )

        # Define User Experience Designer role only
        self.role = {
            "name": "User Experience Designer",
            "system_prompt": """You are a User Experience Designer focused on user interaction, interface design, and user experience optimization. You analyze requirements from the perspectives of user journey, interaction design, usability, accessibility, and emotional design. You focus on how requirements affect user satisfaction, ease of use, learning curve, and overall user experience. Your analysis focuses on how requirements are "perceived and used by users".""",
            "focus_areas": ["User Experience", "Interaction Design", "Usability", "Accessibility", "User Satisfaction"]
        }

        print(f"✅ Successfully created single-role GPT model client: {self.name}")
        print(f"  Used professional role: {self.role['name']}")
        print(f"  🔄 Retry mechanism: Vote once, compare with label, retry if inconsistent, up to 3 times")
        print(f"  📝 Inconsistency reason recording: Record reason if still inconsistent after 3 attempts")
        print(f"  API Configuration: OpenAI Proxy (https://api.openai-proxy.org/v1)")

    def single_vote(self, text: str, labels: List[str], attempt_num: int = 1) -> Dict:
        """Single vote"""
        role_name = self.role["name"]

        print(f"    🔄 Vote attempt {attempt_num}...")

        try:
            prompt = self._build_single_vote_prompt(text, labels, self.role, attempt_num)

            # Call GPT API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.role["system_prompt"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )

            response_content = completion.choices[0].message.content.strip()
            predicted_label = self._extract_label_from_response(response_content, labels)

            # Extract confidence
            confidence = self._extract_confidence_from_response(response_content)

            if predicted_label:
                print(f"      ✅ Vote attempt {attempt_num} result: {predicted_label} (confidence: {confidence:.3f})")
                return {
                    "attempt_number": attempt_num,
                    "response": response_content,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                }
            else:
                fallback_label = labels[0] if labels else "Unknown"
                print(f"      ⚠️ Vote attempt {attempt_num}: {fallback_label} (extraction failed, using default)")
                return {
                    "attempt_number": attempt_num,
                    "response": response_content,
                    "predicted_label": fallback_label,
                    "confidence": 0.5,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                }
        except Exception as e:
            print(f"      ❌ Vote attempt {attempt_num} exception: {str(e)[:100]}")
            fallback_label = labels[0] if labels else "Unknown"
            return {
                "attempt_number": attempt_num,
                "response": str(e),
                "predicted_label": fallback_label,
                "confidence": 0.5,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }

    def analyze_with_retry(self, text: str, labels: List[str], human_label: str = None,
                           category_explanations: Dict[str, str] = None,
                           requirements_examples: Dict[str, List[str]] = None) -> ModelPrediction:
        """Analysis with retry mechanism: Vote once, compare with label, retry if inconsistent, up to 3 times"""

        role_name = self.role["name"]

        print(f"\n🔍 Starting {role_name} analysis: '{text[:50]}...'")
        print(f"  📝 Human annotation: {human_label if human_label else 'No annotation'}")
        print("-" * 60)

        attempts = []
        max_attempts = 3
        final_label = ""
        final_confidence = 0.0
        match_result = False
        final_reason = ""

        # First vote
        attempt1 = self.single_vote(text, labels, 1)
        attempts.append(attempt1)

        # Check if it matches human annotation
        if human_label:
            match_result = (attempt1["predicted_label"] == human_label)

            if match_result:
                print(f"    ✅ First vote matches human annotation!")
                final_label = attempt1["predicted_label"]
                final_confidence = attempt1["confidence"]
            else:
                print(f"    ⚠️ First vote does not match human annotation, proceeding to second vote...")
                time.sleep(1)

                # Second vote (with rethinking)
                attempt2 = self._rethinking_vote(text, labels, human_label, attempt1["predicted_label"], 2)
                attempts.append(attempt2)

                # Check again
                match_result = (attempt2["predicted_label"] == human_label)

                if match_result:
                    print(f"    ✅ Second vote matches human annotation!")
                    final_label = attempt2["predicted_label"]
                    final_confidence = attempt2["confidence"]
                else:
                    print(f"    ⚠️ Second vote still does not match human annotation, proceeding to third vote...")
                    time.sleep(1)

                    # Third vote (final attempt)
                    attempt3 = self._rethinking_vote(text, labels, human_label, attempt2["predicted_label"], 3,
                                                     is_final=True)
                    attempts.append(attempt3)

                    # Final check
                    match_result = (attempt3["predicted_label"] == human_label)

                    if match_result:
                        print(f"    ✅ Third vote matches human annotation!")
                        final_label = attempt3["predicted_label"]
                        final_confidence = attempt3["confidence"]
                    else:
                        print(f"    ❌ Still does not match human annotation after 3 votes, extracting reason...")
                        final_label = attempt3["predicted_label"]
                        final_confidence = attempt3["confidence"]

                        # Extract reason for inconsistency
                        final_reason = self._extract_disagreement_reason(text, final_label, human_label, attempts)
                        print(f"    📝 Reason for inconsistency: {final_reason[:100]}...")
        else:
            # If no human annotation, directly use first vote result
            print(f"    ℹ️ No human annotation for comparison, using first vote result")
            final_label = attempt1["predicted_label"]
            final_confidence = attempt1["confidence"]
            match_result = True  # Default to consistent since no annotation

        # If no human annotation, calculate average confidence
        if not human_label and attempts:
            final_confidence = np.mean([a["confidence"] for a in attempts])

        # Print final results
        print(f"\n📊 Final results:")
        print(f"  Model prediction: {final_label}")
        if human_label:
            print(f"  Human annotation: {human_label}")
            print(f"  Consistent: {'✅ Yes' if match_result else '❌ No'}")
        print(f"  Confidence: {final_confidence:.3f}")
        print(f"  Total attempts: {len(attempts)}")
        if final_reason and not match_result:
            print(f"  Reason for inconsistency: {final_reason}")

        return ModelPrediction(
            label=final_label,
            confidence=final_confidence,
            model_name=f"{self.name}-single-role-retry",
            attempts=attempts,
            human_label=human_label,
            match_result=match_result,
            total_attempts=len(attempts),
            final_reason=final_reason
        )

    def _rethinking_vote(self, text: str, labels: List[str], human_label: str,
                         previous_label: str, attempt_num: int, is_final: bool = False) -> Dict:
        """Rethinking vote"""
        role_name = self.role["name"]

        print(f"    🔄 Vote attempt {attempt_num} (rethinking)...")

        try:
            if is_final:
                prompt = self._build_final_rethinking_prompt(text, labels, human_label, previous_label, self.role)
            else:
                prompt = self._build_rethinking_prompt(text, labels, human_label, previous_label, self.role)

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.role["system_prompt"]},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=250
            )

            response_content = completion.choices[0].message.content.strip()
            predicted_label = self._extract_label_from_response(response_content, labels)
            confidence = self._extract_confidence_from_response(response_content)

            if predicted_label:
                print(f"      ✅ Rethinking vote attempt {attempt_num} result: {predicted_label} (confidence: {confidence:.3f})")
                return {
                    "attempt_number": attempt_num,
                    "response": response_content,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "is_rethinking": True,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                }
            else:
                fallback_label = labels[0] if labels else "Unknown"
                print(f"      ⚠️ Rethinking vote attempt {attempt_num}: {fallback_label} (extraction failed)")
                return {
                    "attempt_number": attempt_num,
                    "response": response_content,
                    "predicted_label": fallback_label,
                    "confidence": 0.5,
                    "is_rethinking": True,
                    "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
                }
        except Exception as e:
            print(f"      ❌ Rethinking vote attempt {attempt_num} exception: {str(e)[:100]}")
            fallback_label = labels[0] if labels else "Unknown"
            return {
                "attempt_number": attempt_num,
                "response": str(e),
                "predicted_label": fallback_label,
                "confidence": 0.5,
                "is_rethinking": True,
                "timestamp": datetime.datetime.now().strftime("%H:%M:%S")
            }

    def _extract_disagreement_reason(self, text: str, model_label: str, human_label: str, attempts: List[Dict]) -> str:
        """Extract reason for disagreement"""
        try:
            # Use the last attempt's response to extract the reason
            last_attempt = attempts[-1]
            response = last_attempt["response"]

            # Try to extract reason from response
            reason_patterns = [
                r'【Reason Analysis】\s*[:：]?\s*(.+)',
                r'Reason Analysis[:：]\s*(.+)',
                r'Reason for Inconsistency[:：]\s*(.+)',
                r'Perspective Difference[:：]\s*(.+)'
            ]

            for pattern in reason_patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
                if match:
                    reason = match.group(1).strip()
                    if reason and len(reason) > 10:
                        return reason

            # If no explicit reason found, generate a summary reason
            return f"The UX Designer analyzed from a user experience perspective as '{model_label}', while the human annotation is '{human_label}'. After {len(attempts)} attempts, there remains a perspective difference."

        except Exception as e:
            return f"Unable to extract specific reason, possibly due to: {str(e)[:50]}"

    def _build_single_vote_prompt(self, text: str, labels: List[str],
                                  role_info: Dict, attempt_num: int = 1) -> str:
        """Build single vote prompt"""

        role_name = role_info["name"]

        labels_section = f"""Available classification labels (choose one strictly from the following labels):

{chr(10).join([f'• {label}' for label in labels])}

Important: Ensure you return exactly the label name as shown, without modification, abbreviation, or any additional content."""

        role_guidance = f"""
As a {role_name}, analyze this requirement from your professional perspective:

Your professional focus areas: {', '.join(role_info['focus_areas'])}
Please think from the {role_name}'s perspective and select the most appropriate classification label."""

        response_format = f"""Please respond in the following format:
【{role_name} Analysis】[Your professional analysis]
【Classification Label】[Must choose one from the above labels, exactly matching the label name]
【Confidence】[Your confidence in this classification, a decimal between 0.0-1.0, e.g., 0.85]"""

        return f"""Your current role is: {role_name}

{role_guidance}

{labels_section}

Requirement description to classify:
"{text}"

{response_format}"""

    def _build_rethinking_prompt(self, text: str, labels: List[str],
                                 human_label: str, previous_label: str,
                                 role_info: Dict) -> str:
        """Build rethinking prompt"""

        role_name = role_info["name"]

        labels_section = f"""Available classification labels (choose one strictly from the following labels):

{chr(10).join([f'• {label}' for label in labels])}"""

        rethinking_guidance = f"""
As a {role_name}, you need to rethink your previous classification.

Requirement description:
"{text}"

Your previous classification was: 【{previous_label}】
However, the human expert's annotation is: 【{human_label}】

Please re-analyze from the {role_name}'s professional perspective: why might the human expert have a different annotation?
Then provide the classification label you believe is most appropriate.

Please respond in the following format:
【{role_name} Rethinking Analysis】[Your analysis]
【Classification Label】[Must choose one from the above labels]
【Confidence】[Your confidence in this classification, a decimal between 0.0-1.0]"""

        return rethinking_guidance

    def _build_final_rethinking_prompt(self, text: str, labels: List[str],
                                       human_label: str, previous_label: str,
                                       role_info: Dict) -> str:
        """Build final rethinking prompt (requiring reason analysis)"""

        role_name = role_info["name"]

        labels_section = f"""Available classification labels (choose one strictly from the following labels):

{chr(10).join([f'• {label}' for label in labels])}"""

        final_guidance = f"""
As a {role_name}, this is your final opportunity to rethink.

Requirement description:
"{text}"

Your previous classification was: 【{previous_label}】
However, the human expert's annotation remains: 【{human_label}】

From the {role_name}'s professional perspective:
1. Provide the classification label you believe is most appropriate
2. Analyze why your classification differs from the human expert's

Please respond in the following format:
【{role_name} Final Analysis】[Your analysis, including possible perspective differences]
【Classification Label】[Must choose one from the above labels]
【Confidence】[Your confidence in this classification]
【Reason Analysis】[Explain why it differs from the human annotation, within 50 words]"""

        return final_guidance

    def _extract_label_from_response(self, response: str, labels: List[str]) -> str:
        """Extract label from response"""
        if not response or not labels:
            return labels[0] if labels else "Unknown"

        response_clean = response.strip()

        # 1. Try to extract from 【Classification Label】 format
        label_pattern = r'【Classification Label】\s*[:：]?\s*(.+)'
        label_match = re.search(label_pattern, response_clean, re.IGNORECASE | re.MULTILINE)
        if label_match:
            extracted = label_match.group(1).strip().strip('.,!?;:"\'')
            for label in labels:
                if extracted.lower() == label.lower():
                    return label
            for label in labels:
                if label.lower() in extracted.lower():
                    return label

        # 2. Search for labels in the entire response
        for label in labels:
            if label.lower() in response_clean.lower():
                return label

        return labels[0] if labels else "Unknown"

    def _extract_confidence_from_response(self, response: str) -> float:
        """Extract confidence from response"""
        if not response:
            return 0.5

        response_clean = response.strip()

        # Try to extract from 【Confidence】 format
        confidence_patterns = [
            r'【Confidence】\s*[:：]?\s*([0-9]*\.?[0-9]+)',
            r'Confidence[:：]\s*([0-9]*\.?[0-9]+)'
        ]

        for pattern in confidence_patterns:
            confidence_match = re.search(pattern, response_clean, re.IGNORECASE | re.MULTILINE)
            if confidence_match:
                try:
                    confidence = float(confidence_match.group(1))
                    # Ensure confidence is between 0-1
                    confidence = max(0.0, min(1.0, confidence))
                    return confidence
                except ValueError:
                    continue

        # If not explicitly found, try to find a number between 0-1
        number_pattern = r'(0?\.\d+|1\.0|0\.[0-9]+|1\.0+)'
        number_matches = re.findall(number_pattern, response_clean)

        for num_str in number_matches:
            try:
                num = float(num_str)
                if 0 <= num <= 1:
                    return num
            except ValueError:
                continue

        return 0.5  # Default value


class SingleRoleProcessor:
    """Single-role model processor - User Experience Designer only"""

    def __init__(self, single_role_client: SingleRoleGPTClient):
        self.single_role_client = single_role_client
        self.start_time = None
        self.end_time = None
        print(f"Single-role model processor initialized, using {single_role_client.role['name']} role")
        print(f"🔄 Retry mechanism: Vote once, compare with label, retry if inconsistent, up to 3 times")

    def process_dataset(self, dataset_with_labels: List[Dict], labels: List[str],
                        category_explanations: Dict[str, str] = None,
                        requirements_examples: Dict[str, List[str]] = None) -> List[DataPoint]:
        """Process dataset with labels"""
        data_points = []

        self.start_time = datetime.datetime.now()
        print(f"🚀 Single-role analysis start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"Starting to process {len(dataset_with_labels)} labeled software requirements...")
        print(f"Number of classification labels: {len(labels)}")
        print(f"🎭 Role configuration: User Experience Designer")
        print(f"🔄 Retry mechanism: 1 vote → Compare → Retry if inconsistent (up to 3 times)")

        for i, item in enumerate(dataset_with_labels):
            # Extract content and human annotation from data
            text = item.get('requirement', item.get('content', ''))
            human_label = item.get('label', None)  # Note: uses 'label' column

            if i % 2 == 0 and i > 0:
                elapsed_time = datetime.datetime.now() - self.start_time
                processed = i
                remaining = len(dataset_with_labels) - i
                avg_time_per_req = elapsed_time / processed if processed > 0 else elapsed_time
                est_remaining = avg_time_per_req * remaining

                print(f"  📈 Progress: {i}/{len(dataset_with_labels)} - Elapsed: {elapsed_time} - Estimated remaining: {est_remaining}")

            data_point = DataPoint(content=text, human_label=human_label)
            prediction = self.single_role_client.analyze_with_retry(
                text, labels, human_label, category_explanations, requirements_examples
            )

            data_point.prediction = prediction
            data_point.final_label = prediction.label
            data_points.append(data_point)

            if i < len(dataset_with_labels) - 1:
                time.sleep(1)  # Lower latency, GPT API may be faster

        self.end_time = datetime.datetime.now()
        return data_points

    def save_results(self, results: List[DataPoint], output_file: str):
        """Save final results to Excel file"""
        results_data = []

        for i, data_point in enumerate(results):
            # Build attempt records
            attempts_info = ""
            human_label = data_point.human_label or "N/A"
            match_status = "N/A"
            reason = ""
            confidence = 0.0
            attempts_summary = []

            if data_point.prediction:
                confidence = data_point.prediction.confidence

                if data_point.prediction.attempts:
                    attempts_summary = [
                        f"Attempt {a['attempt_number']}: {a['predicted_label']}({a['confidence']:.3f})"
                        for a in data_point.prediction.attempts
                    ]
                    attempts_info = "; ".join(attempts_summary)

                if data_point.human_label:
                    match_status = "✅ Consistent" if data_point.prediction.match_result else "❌ Inconsistent"
                    if data_point.prediction.final_reason and not data_point.prediction.match_result:
                        reason = data_point.prediction.final_reason

            row_data = {
                'Serial No.': i + 1,
                'Requirement Content': data_point.content,
                'Model Name': data_point.prediction.model_name if data_point.prediction else "N/A",
                'Human Annotation Label': human_label,
                'Model Final Prediction': data_point.final_label,
                'Consistent': match_status,
                'Total Attempts': data_point.prediction.total_attempts if data_point.prediction else 0,
                'Reason for Inconsistency': reason,
                'Attempt Records': attempts_info,
                'Final Confidence': round(confidence, 3),
                'Notes': f"Total {data_point.prediction.total_attempts} attempts" if data_point.prediction else ""
            }
            results_data.append(row_data)

        df = pd.DataFrame(results_data)
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ Single-role analysis results saved to: {output_file}")

        # Save detailed report
        self._save_detailed_report(results, output_file)

    def _save_detailed_report(self, results: List[DataPoint], output_file: str):
        """Save detailed analysis report"""
        report_file = output_file.replace('.xlsx', '_detailed_report.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("GPT-4.1-nano Model - User Experience Designer Single-Role Classification System Analysis Report\n")
            f.write("Special Configuration: Vote once, compare with label, retry if inconsistent, up to 3 times\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Analysis Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Requirements: {len(results)}\n")
            f.write("Professional Roles Used:\n")
            f.write("1. User Experience Designer\n\n")

            f.write("🔄 Retry Mechanism:\n")
            f.write("  1. First vote\n")
            f.write("  2. Compare with label column in dataset\n")
            f.write("  3. If inconsistent, proceed to second vote\n")
            f.write("  4. Compare again, if still inconsistent and not yet 3 attempts, proceed to third vote\n")
            f.write("  5. If still inconsistent after 3 attempts, record reason\n\n")

            f.write("🎯 Confidence Mechanism:\n")
            f.write("  - LLM self-assesses confidence for each vote\n")
            f.write("  - Final confidence uses confidence from the last vote\n\n")

            # Calculate consistency rate
            total_with_human_label = 0
            match_count = 0
            total_attempts = 0
            inconsistencies = []

            for i, dp in enumerate(results):
                if dp.human_label:
                    total_with_human_label += 1
                    if dp.prediction and dp.prediction.match_result:
                        match_count += 1
                    if dp.prediction:
                        total_attempts += dp.prediction.total_attempts
                    if dp.prediction and not dp.prediction.match_result and dp.prediction.final_reason:
                        inconsistencies.append({
                            'index': i + 1,
                            'model': dp.final_label,
                            'human': dp.human_label,
                            'confidence': dp.prediction.confidence if dp.prediction else 0.0,
                            'reason': dp.prediction.final_reason,
                            'attempts': dp.prediction.total_attempts if dp.prediction else 0
                        })

            f.write("📊 Consistency Rate Statistics:\n")
            f.write("-" * 80 + "\n")
            if total_with_human_label > 0:
                f.write(f"Total Samples (with human annotation): {total_with_human_label}\n")
                f.write(f"Consistent Samples: {match_count}\n")
                f.write(f"Consistency Rate: {(match_count / total_with_human_label * 100):.1f}%\n")
                f.write(f"Average Attempts: {(total_attempts / total_with_human_label):.2f}\n\n")
            else:
                f.write("No human annotation data available for consistency rate calculation\n\n")

            # Attempt distribution
            attempt_distribution = {1: 0, 2: 0, 3: 0}
            for dp in results:
                if dp.prediction:
                    attempts = dp.prediction.total_attempts
                    if attempts in attempt_distribution:
                        attempt_distribution[attempts] += 1

            f.write("🔄 Attempt Distribution:\n")
            for attempts, count in attempt_distribution.items():
                if count > 0:
                    percentage = count / len(results) * 100
                    f.write(f"  {attempts} attempt(s): {count} ({percentage:.1f}%)\n")
            f.write("\n")

            # Inconsistency reason analysis
            if inconsistencies:
                f.write("💭 Inconsistency Reason Analysis (still inconsistent after 3 attempts):\n")
                f.write("-" * 80 + "\n")
                for inc in inconsistencies:
                    f.write(f"Item {inc['index']}:\n")
                    f.write(f"  Model Prediction: {inc['model']} (Confidence: {inc['confidence']:.3f})\n")
                    f.write(f"  Human Annotation: {inc['human']}\n")
                    f.write(f"  Attempts: {inc['attempts']}\n")
                    f.write(f"  Reason: {inc['reason']}\n\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"✅ Detailed analysis report saved to: {report_file}")

    def print_statistics(self, results: List[DataPoint]):
        """Print statistical information"""
        print(f"\n{'=' * 60}")
        print("📊 GPT-4.1-nano Model - User Experience Designer Single-Role Classification Statistics")
        print(f"Special Configuration: Vote once, compare with label, retry if inconsistent, up to 3 times")
        print(f"{'=' * 60}")

        print(f"Total Data Processed: {len(results)} requirements")
        print(f"Role Used: User Experience Designer")

        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time
            print(f"\n⏰ Time Statistics:")
            print(f"  Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  End Time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Total Runtime: {total_duration}")

            if len(results) > 0:
                avg_time_per_req = total_duration / len(results)
                print(f"  Average Time per Requirement: {avg_time_per_req}")

        # Consistency rate statistics
        total_with_human_label = 0
        match_count = 0
        total_attempts = 0

        for dp in results:
            if dp.human_label:
                total_with_human_label += 1
                if dp.prediction and dp.prediction.match_result:
                    match_count += 1
                if dp.prediction:
                    total_attempts += dp.prediction.total_attempts

        print(f"\n🎯 Consistency Rate Statistics:")
        if total_with_human_label > 0:
            match_rate = match_count / total_with_human_label * 100
            print(f"  Samples with Human Annotation: {total_with_human_label}")
            print(f"  Consistent Samples: {match_count}")
            print(f"  Consistency Rate: {match_rate:.1f}%")
            print(f"  Total Attempts: {total_attempts}")
            print(f"  Average Attempts: {total_attempts / total_with_human_label:.2f}")
        else:
            print(f"  No Human Annotation Data")

        # Attempt distribution
        attempt_distribution = {1: 0, 2: 0, 3: 0}
        for dp in results:
            if dp.prediction:
                attempts = dp.prediction.total_attempts
                if attempts in attempt_distribution:
                    attempt_distribution[attempts] += 1

        print(f"\n🔄 Attempt Distribution:")
        for attempts, count in attempt_distribution.items():
            if count > 0:
                percentage = count / len(results) * 100
                print(f"  {attempts} attempt(s): {count} ({percentage:.1f}%)")


class DataLoader:
    """Data loader class"""

    @staticmethod
    def load_dataset_with_labels(file_path: str) -> List[Dict]:
        """Load requirements and human annotations from dataset file"""
        try:
            df = pd.read_excel(file_path)
            dataset_with_labels = []

            # Check required columns
            if 'requirement' not in df.columns:
                print(f"⚠️  File does not have a 'requirement' column, attempting to use first column as requirement content")
                requirement_col = df.columns[0]
            else:
                requirement_col = 'requirement'

            if 'label' not in df.columns:
                print(f"⚠️  File does not have a 'label' column, annotation comparison will not be possible")
                label_col = None
            else:
                label_col = 'label'

            for _, row in df.iterrows():
                if pd.notna(row[requirement_col]):
                    item = {
                        'requirement': str(row[requirement_col]).strip(),
                    }

                    if label_col and pd.notna(row.get(label_col)):
                        item['label'] = str(row[label_col]).strip()

                    dataset_with_labels.append(item)

            print(f"✅ Successfully loaded {len(dataset_with_labels)} labeled test requirements from {os.path.basename(file_path)}")
            if label_col:
                print(f"  Of which {sum(1 for item in dataset_with_labels if 'label' in item)} have annotations")
            return dataset_with_labels

        except Exception as e:
            print(f"❌ Error loading dataset file {os.path.basename(file_path)}: {e}")
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
            print(f"✅ Successfully loaded {len(categories)} classification labels from {os.path.basename(file_path)}")
            return categories
        except Exception as e:
            print(f"❌ Error loading category file {os.path.basename(file_path)}: {e}")
            return {}

    @staticmethod
    def load_requirements_examples(file_path: str) -> Dict[str, List[str]]:
        """Load requirement examples from examples file"""
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            requirements_examples = {}

            for _, row in df.iterrows():
                if 'boileplate type' in row and pd.notna(row['boileplate type']):
                    boilerplate_type = str(row['boileplate type']).strip()
                    examples = []

                    for i in range(1, 19):
                        col_name = f'example {i}'
                        if col_name in row and pd.notna(row[col_name]):
                            example = str(row[col_name]).strip()
                            if example and example != 'nan':
                                examples.append(example)

                    if boilerplate_type and examples:
                        requirements_examples[boilerplate_type] = examples

            print(f"✅ Successfully loaded {len(requirements_examples)} requirement examples by category from {os.path.basename(file_path)}")
            return requirements_examples

        except Exception as e:
            print(f"❌ Error loading requirement examples file {os.path.basename(file_path)}: {e}")
            return {}


def main():
    """Main function"""
    # File path configuration
    dataset_file = "1000dataset.xlsx"  # Must contain 'requirement' and 'label' columns
    concept_file = "1123Concept.xlsx"
    examples_file = "1122RequirementExamples.xlsx"
    output_file = "gpt_UXDesigner_retry_mechanism.xlsx"

    data_loader = DataLoader()

    print("=" * 80)
    print("🎨 GPT-4.1-nano Model - User Experience Designer Single-Role Classification System")
    print("Special Configuration: Vote once, compare with label column in dataset.xlsx")
    print("🔄 Retry Mechanism: If inconsistent, vote again, up to 3 times")
    print("📝 Reason Recording: If still inconsistent after 3 attempts, record detailed reason")
    print("=" * 80)

    # Check environment variables
    print(f"\n🔧 Environment Check:")
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OPENAI_API_KEY is set (length: {len(api_key)})")
    else:
        print(f"❌ OPENAI_API_KEY is not set")
        print(f"💡 Please set the OPENAI_API_KEY environment variable")
        return

    # Check if data files exist
    print(f"\n📁 Checking data files:")
    files_to_check = [dataset_file, concept_file, examples_file]
    all_files_exist = True

    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} does not exist")
            all_files_exist = False

    if not all_files_exist:
        print(f"💡 Please ensure the following files exist in the current directory:")
        for file_path in files_to_check:
            print(f"   - {file_path}")
        return

    try:
        # Create single-role GPT model client
        print(f"\n🚀 Creating single-role model client...")
        single_role_client = SingleRoleGPTClient(
            name="GPT-4.1-nano-UXDesigner",
            model_name="gpt-4.1-nano"
        )
    except Exception as e:
        print(f"❌ Failed to create single-role model client: {e}")
        return

    # Load categories, explanations, and examples
    print(f"\n📚 Loading data files...")
    category_explanations = data_loader.load_categories_and_explanations(concept_file)
    requirements_examples = data_loader.load_requirements_examples(examples_file)
    labels = list(category_explanations.keys())

    # Load labeled dataset
    dataset_with_labels = data_loader.load_dataset_with_labels(dataset_file)

    if not dataset_with_labels:
        print("❌ No test requirements found, program exiting")
        return

    if not labels:
        print("❌ No classification labels found, program exiting")
        return

    print(f"\n✅ Data loading complete:")
    print(f"  Classification labels: {len(labels)}")
    print(f"  Test requirements: {len(dataset_with_labels)}")

    # Count how many have annotations
    labeled_count = sum(1 for item in dataset_with_labels if 'label' in item)
    print(f"  Requirements with annotations: {labeled_count}")

    if labeled_count == 0:
        print(f"\n⚠️  Warning: The dataset does not have a 'label' column, annotation comparison will not be possible")
        print(f"  The retry mechanism will not be triggered")

    # Execute single-role model processing
    print(f"\n🚀 Starting GPT-4.1-nano model UX Designer single-role classification processing...")
    print(f"🎯 Note: Each vote will be compared with the label column in the dataset")
    print(f"🔄 Retry Mechanism: Inconsistent → Retry → Up to 3 times → Record reason")

    processor = SingleRoleProcessor(single_role_client)

    results = processor.process_dataset(dataset_with_labels, labels, category_explanations, requirements_examples)

    # Save and display results
    processor.save_results(results, output_file)
    processor.print_statistics(results)

    print(f"\n{'=' * 80}")
    print("🎉 GPT-4.1-nano Model UX Designer Single-Role Classification Processing Complete!")
    print(f"📊 Results file: {output_file}")
    print(f"📄 Detailed report: {output_file.replace('.xlsx', '_detailed_report.txt')}")
    print(f"🔄 Retry Mechanism: 1 vote → Compare with label → Retry if inconsistent (up to 3 times)")
    print(f"📝 Reason Recording: Still inconsistent after 3 attempts → Record detailed reason")
    print("=" * 80)


if __name__ == "__main__":
    main()
