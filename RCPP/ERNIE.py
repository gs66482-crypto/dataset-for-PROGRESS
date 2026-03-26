import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
import time
import re
import datetime
from collections import Counter
from openai import OpenAI

# Load environment variables
load_dotenv()


@dataclass
class ModelPrediction:
    """Store model prediction results"""
    label: str
    confidence: float
    model_name: str
    votes: List[str] = None
    vote_counts: Dict[str, int] = None
    role_votes: Dict[str, str] = None
    calculation_details: Dict[str, Any] = None


@dataclass
class DataPoint:
    """Data point class"""
    content: str
    prediction: ModelPrediction = None
    final_label: str = None


class ResearchPaperConfidenceCalculator:
    """Calculate confidence strictly according to the research paper method"""

    def __init__(self):
        """
        Strictly follow the paper formulas:
        C_base = V_top / N
        B_gap = min(0.2, (V_top - V_second) × 0.1)
        C_final = min(1.0, C_base + B_gap)

        Where:
        N = 5 (five role votes)
        K = 0.1 (scaling factor)
        U = 0.2 (Gap upper bound)
        """
        self.K = 0.1  # Scaling factor
        self.U = 0.2  # Gap upper bound

    def calculate_confidence(self, vote_counts: Dict[str, int]) -> Dict[str, Any]:
        """
        Calculate confidence according to the research paper method
        Returns a dictionary containing the calculation process
        """
        if not vote_counts:
            return {
                'final_confidence': 0.0,
                'c_base': 0.0,
                'b_gap': 0.0,
                'v_top': 0,
                'v_second': 0,
                'total_votes': 5,
                'gap': 0
            }

        N = 5  # Total votes fixed at 5
        sorted_counts = sorted(vote_counts.values(), reverse=True)

        V_top = sorted_counts[0]
        V_second = sorted_counts[1] if len(sorted_counts) > 1 else 0

        # 1. Calculate base confidence C_base (Formula 1)
        C_base = V_top / N

        # 2. Calculate Gap Bonus B_gap (Formula 2)
        gap = V_top - V_second
        B_gap = min(self.U, gap * self.K)

        # 3. Calculate final confidence C_final (Formula 3)
        C_final = min(1.0, C_base + B_gap)

        return {
            'final_confidence': C_final,
            'c_base': C_base,
            'b_gap': B_gap,
            'v_top': V_top,
            'v_second': V_second,
            'total_votes': N,
            'gap': gap
        }


class MultiRoleERNIEClient:
    """Multi-role ERNIE-4.5 model client class - Six professional roles based on research paper"""

    def __init__(self, name: str = "ERNIE-4.5", model_name: str = "baidu/ERNIE-4.5-300B-A47B"):
        self.name = name
        self.model_name = model_name
        self.confidence_calculator = ResearchPaperConfidenceCalculator()

        # Initialize ERNIE-4.5 client (via SiliconFlow)
        api_key = os.getenv("SILICON_FLOW_API_KEY")
        base_url = os.getenv("SILICON_FLOW_BASE_URL")

        if not api_key:
            raise ValueError("SILICON_FLOW_API_KEY environment variable not set")
        if not base_url:
            raise ValueError("SILICON_FLOW_BASE_URL environment variable not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        # Define six professional roles and their system prompts (including Linguist)
        self.roles = {
            "product_owner": {
                "name": "Product Owner",
                "system_prompt": """You are a Product Owner, focused on maximizing product value. You analyze requirements from the perspectives of business goals, market demands, return on investment (ROI), and product strategy. You focus on how requirements support business objectives, meet user needs, create business value, and determine requirement priorities. Your analysis focuses on "why" this feature should be developed and its contribution to the product vision.""",
                "focus_areas": ["Business Value", "Market Demand", "ROI", "Product Strategy", "Priority"]
            },
            "business_analyst": {
                "name": "Business Analyst",
                "system_prompt": """You are a Business Analyst, focused on requirements analysis, process optimization, and solution design. You analyze requirements from the perspectives of business processes, functional specifications, requirement completeness, and consistency. You focus on requirement clarity, testability, consistency with existing systems, and how to best meet business objectives. Your analysis focuses on "what" needs to be built and how to satisfy business needs.""",
                "focus_areas": ["Process Analysis", "Functional Specifications", "Requirement Consistency", "Solution Design"]
            },
            "system_architect": {
                "name": "System Architect",
                "system_prompt": """You are a System Architect, focused on technical feasibility, system design, and architectural constraints. You analyze requirements from the perspectives of technical implementation, scalability, performance, security, integration difficulty, and maintenance cost. You focus on the technical impact of requirements, architectural decisions, technical debt, and long-term sustainability. Your analysis focuses on "how" to technically implement this requirement.""",
                "focus_areas": ["Technical Architecture", "Scalability", "Performance", "Security", "Integration"]
            },
            "user_experience_designer": {
                "name": "User Experience Designer",
                "system_prompt": """You are a User Experience Designer, focusing on user interaction, interface design, and user experience optimization. You analyze requirements from the perspectives of user journey, interaction design, usability, accessibility, and emotional design. You focus on how requirements affect user satisfaction, ease of use, learning curve, and overall user experience. Your analysis focuses on how requirements are "perceived and used" by users.""",
                "focus_areas": ["User Experience", "Interaction Design", "Usability", "Accessibility", "User Satisfaction"]
            },
            "software_tester": {
                "name": "Software Tester",
                "system_prompt": """You are a Software Tester, focused on quality assurance, test design, and defect prevention. You analyze requirements from the perspectives of testability, test case design, boundary conditions, error handling, and quality standards. You focus on how requirements can be verified, test coverage, potential risk points, and quality assurance measures. Your analysis focuses on how to "validate" whether the requirement has been correctly implemented.""",
                "focus_areas": ["Testability", "Test Design", "Boundary Conditions", "Quality Assurance", "Validation Standards"]
            },
            "linguist": {
                "name": "Linguist",
                "system_prompt": """You are a Linguist, specializing in language structure, semantic analysis, and textual interpretation. You analyze requirements from the perspectives of linguistic precision, clarity of expression, ambiguity detection, terminology consistency, and communicative effectiveness. You focus on how the requirement is expressed in language, potential ambiguities in wording, the clarity of technical terms, and the overall linguistic quality of the requirement description. Your analysis focuses on how requirements are "communicated" through language and how linguistic features influence understanding and interpretation.""",
                "focus_areas": ["Linguistic Precision", "Ambiguity Detection", "Terminology Consistency", "Clarity of Expression", "Communicative Effectiveness"]
            }
        }

        print(f"✅ Successfully created multi-role ERNIE-4.5 model client: {self.name}")
        print(f"  Professional roles used: {', '.join([role['name'] for role in self.roles.values()])}")
        print(f"  Confidence calculation method: C_final = min(1.0, C_base + B_gap), where B_gap = min(0.2, (V_top - V_second) × 0.1)")

    def predict_with_multi_roles(self, text: str, labels: List[str],
                                 category_explanations: Dict[str, str] = None,
                                 requirements_examples: Dict[str, List[str]] = None) -> ModelPrediction:
        """Use six professional roles for voting prediction - strictly following the research paper method"""

        role_votes = {}  # Votes from each role

        print(f"\n🔍 Starting multi-role voting analysis: '{text[:50]}...'")
        print("-" * 60)

        # Let each role vote in order
        for role_key, role_info in self.roles.items():
            role_name = role_info["name"]
            print(f"  👤 {role_name} is analyzing...")

            for attempt in range(3):  # Retry mechanism
                try:
                    prompt = self._build_role_specific_prompt(
                        text, labels, role_info,
                        category_explanations, requirements_examples
                    )

                    # Use ERNIE-4.5 API (via SiliconFlow)
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": role_info["system_prompt"]},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=200
                    )

                    response_content = completion.choices[0].message.content.strip()
                    predicted_label = self._extract_label_from_response(response_content, labels)

                    if predicted_label:
                        role_votes[role_name] = predicted_label
                        print(f"    ✅ {role_name}: {predicted_label}")
                        break
                    else:
                        print(f"    ⚠️ {role_name} attempt {attempt + 1} failed - response format abnormal")
                        if attempt == 2:
                            role_votes[role_name] = labels[0] if labels else "Unknown"
                except Exception as e:
                    print(f"    ❌ {role_name} attempt {attempt + 1} exception: {str(e)[:100]}")
                    if attempt == 2:
                        role_votes[role_name] = labels[0] if labels else "Unknown"
                    time.sleep(2)

            time.sleep(1)  # Delay between roles

        # Collect all votes
        votes = list(role_votes.values())

        # Count voting results
        vote_counts = {}
        for vote in votes:
            vote_counts[vote] = vote_counts.get(vote, 0) + 1

        # Determine final label (majority vote)
        if vote_counts:
            final_label = max(vote_counts.items(), key=lambda x: x[1])[0]
        else:
            final_label = labels[0] if labels else "Unknown"

        # Calculate consensus confidence strictly according to the research paper method
        confidence_details = self.confidence_calculator.calculate_confidence(vote_counts)
        final_confidence = confidence_details['final_confidence']

        # Print calculation process
        print(f"\n📊 Vote statistics: {dict(vote_counts)}")
        print(f"📈 Confidence calculation process (according to paper formula):")
        print(
            f"  V_top = {confidence_details['v_top']}, V_second = {confidence_details['v_second']}, N = {confidence_details['total_votes']}")
        print(
            f"  C_base = V_top / N = {confidence_details['v_top']} / {confidence_details['total_votes']} = {confidence_details['c_base']:.2f}")
        print(
            f"  B_gap = min(0.2, (V_top - V_second) × 0.1) = min(0.2, ({confidence_details['v_top']} - {confidence_details['v_second']}) × 0.1) = {confidence_details['b_gap']:.2f}")
        print(
            f"  C_final = min(1.0, C_base + B_gap) = min(1.0, {confidence_details['c_base']:.2f} + {confidence_details['b_gap']:.2f}) = {final_confidence:.3f}")
        print(f"✅ Final classification: {final_label}, Confidence: {final_confidence:.3f}")

        return ModelPrediction(
            label=final_label,
            confidence=final_confidence,
            model_name=f"{self.name}-multi-role",
            votes=votes,
            vote_counts=vote_counts,
            role_votes=role_votes,
            calculation_details=confidence_details
        )

    def _build_role_specific_prompt(self, text: str, labels: List[str],
                                    role_info: Dict,
                                    category_explanations: Dict[str, str] = None,
                                    requirements_examples: Dict[str, List[str]] = None) -> str:
        """Build prompt for a specific role"""

        role_name = role_info["name"]

        labels_section = f"""Available classification labels (please strictly select one from the following labels):

{chr(10).join([f'• {label}' for label in labels])}

Important: Please ensure to return exactly the label name as shown, do not modify, abbreviate, or add any extra content."""

        explanations_section = ""
        if category_explanations:
            explanations_section = "\n\nDetailed category explanations:\n"
            for label in labels:
                if label in category_explanations:
                    explanations_section += f"\n[{label}]\n{category_explanations[label]}\n"

        examples_section = ""
        if requirements_examples:
            examples_section = "\n\nReference examples:\n"
            for label in labels:
                if label in requirements_examples and requirements_examples[label]:
                    examples = requirements_examples[label]
                    examples_section += f"\nTypical examples for [{label}]:"
                    for i, example in enumerate(examples[:2], 1):
                        examples_section += f"\n  {i}. {example}"
                    examples_section += "\n"

        role_guidance = f"""
As {role_name}, please analyze this requirement from your professional perspective:

Your professional focus areas: {', '.join(role_info['focus_areas'])}
Please think from {role_name}'s perspective and select the most appropriate classification label."""

        target_section = f"""
Requirement description to classify:
"{text}"

Please reply in the following format:
[{role_name} Analysis] [Your professional analysis]
[Classification Label] [Must select one from the above labels, exactly matching the label name]"""

        return f"""Your current role is: {role_name}

{role_guidance}

{labels_section}{explanations_section}{examples_section}{target_section}"""

    def _extract_label_from_response(self, response: str, labels: List[str]) -> str:
        """Extract label from response"""
        if not response or not labels:
            return labels[0] if labels else "Unknown"

        response_clean = response.strip()

        # 1. Try to extract from [Classification Label] format
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


class MultiRoleProcessor:
    """Multi-role model processor"""

    def __init__(self, multi_role_client: MultiRoleERNIEClient):
        self.multi_role_client = multi_role_client
        self.start_time = None
        self.end_time = None
        print(f"Multi-role model processor initialized, using {len(multi_role_client.roles)} professional roles")

    def process_dataset(self, dataset: List[str], labels: List[str],
                        category_explanations: Dict[str, str] = None,
                        requirements_examples: Dict[str, List[str]] = None) -> List[DataPoint]:
        """Process entire dataset using multi-role method"""
        data_points = []

        self.start_time = datetime.datetime.now()
        print(f"🚀 Multi-role analysis start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"Starting to process {len(dataset)} software requirements using 6 professional roles...")
        print(f"Number of classification labels: {len(labels)}")

        print(f"🎭 Professional role configuration:")
        for role_key, role_info in self.multi_role_client.roles.items():
            print(f"  • {role_info['name']}")

        for i, text in enumerate(dataset):
            if i % 2 == 0 and i > 0:
                elapsed_time = datetime.datetime.now() - self.start_time
                processed = i
                remaining = len(dataset) - i
                avg_time_per_req = elapsed_time / processed if processed > 0 else elapsed_time
                est_remaining = avg_time_per_req * remaining

                print(f"  📈 Progress: {i}/{len(dataset)} - Elapsed: {elapsed_time} - Estimated remaining: {est_remaining}")

            data_point = DataPoint(content=text)
            prediction = self.multi_role_client.predict_with_multi_roles(
                text, labels, category_explanations, requirements_examples
            )

            data_point.prediction = prediction
            data_point.final_label = prediction.label
            data_points.append(data_point)

            if i < len(dataset) - 1:
                time.sleep(2)

        self.end_time = datetime.datetime.now()
        return data_points

    def save_results(self, results: List[DataPoint], output_file: str):
        """Save final results to Excel file"""
        results_data = []

        for i, data_point in enumerate(results):
            # Build role vote details
            role_votes_detail = ""
            if data_point.prediction and data_point.prediction.role_votes:
                role_votes_detail = "; ".join(
                    [f"{role}:{vote}" for role, vote in data_point.prediction.role_votes.items()])

            # Vote distribution
            vote_distribution = ""
            if data_point.prediction and data_point.prediction.vote_counts:
                vote_distribution = ", ".join([f"{k}:{v}" for k, v in data_point.prediction.vote_counts.items()])

            # Confidence calculation details
            c_base = b_gap = 0.0
            v_top = v_second = 0
            if data_point.prediction and data_point.prediction.calculation_details:
                details = data_point.prediction.calculation_details
                c_base = details.get('c_base', 0.0)
                b_gap = details.get('b_gap', 0.0)
                v_top = details.get('v_top', 0)
                v_second = details.get('v_second', 0)

            row_data = {
                'Index': i + 1,
                'Requirement Content': data_point.content,
                'Model Name': data_point.prediction.model_name if data_point.prediction else "N/A",
                'Product Owner Vote': data_point.prediction.role_votes.get('Product Owner',
                                                                          'N/A') if data_point.prediction and data_point.prediction.role_votes else 'N/A',
                'Business Analyst Vote': data_point.prediction.role_votes.get('Business Analyst',
                                                                             'N/A') if data_point.prediction and data_point.prediction.role_votes else 'N/A',
                'System Architect Vote': data_point.prediction.role_votes.get('System Architect',
                                                                             'N/A') if data_point.prediction and data_point.prediction.role_votes else 'N/A',
                'User Experience Designer Vote': data_point.prediction.role_votes.get('User Experience Designer',
                                                                                     'N/A') if data_point.prediction and data_point.prediction.role_votes else 'N/A',
                'Software Tester Vote': data_point.prediction.role_votes.get('Software Tester',
                                                                            'N/A') if data_point.prediction and data_point.prediction.role_votes else 'N/A',
                'Linguist Vote': data_point.prediction.role_votes.get('Linguist',
                                                                     'N/A') if data_point.prediction and data_point.prediction.role_votes else 'N/A',
                'Role Vote Details': role_votes_detail,
                'Vote Distribution': vote_distribution,
                'Highest Votes (V_top)': v_top,
                'Second Highest Votes (V_second)': v_second,
                'Total Votes (N)': 6,
                'Base Confidence (C_base)': round(c_base, 3),
                'Gap Bonus (B_gap)': round(b_gap, 3),
                'Final Confidence (C_final)': data_point.prediction.confidence if data_point.prediction else 0.0,
                'Final Classification Label': data_point.final_label,
                'Confidence Formula': f"min(1.0, {c_base:.3f} + {b_gap:.3f}) = {data_point.prediction.confidence:.3f}" if data_point.prediction else "N/A"
            }
            results_data.append(row_data)

        df = pd.DataFrame(results_data)
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ Multi-role analysis results saved to: {output_file}")

        # Save detailed report
        self._save_detailed_report(results, output_file)

    def _save_detailed_report(self, results: List[DataPoint], output_file: str):
        """Save detailed analysis report"""
        report_file = output_file.replace('.xlsx', '_detailed_report.txt')

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ERNIE-4.5 Model - Six Professional Role Classification System Analysis Report\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Analysis Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Number of Requirements: {len(results)}\n")
            f.write("Six Professional Roles Used:\n")
            f.write("1. Product Owner\n")
            f.write("2. Business Analyst\n")
            f.write("3. System Architect\n")
            f.write("4. User Experience Designer\n")
            f.write("5. Software Tester\n")
            f.write("6. Linguist\n\n")

            f.write("Confidence Calculation Formula (based on research paper):\n")
            f.write("C_base = V_top / N  (where N=6)\n")
            f.write("B_gap = min(0.2, (V_top - V_second) × 0.1)\n")
            f.write("C_final = min(1.0, C_base + B_gap)\n\n")

            # Confidence statistics
            confidences = [dp.prediction.confidence for dp in results if dp.prediction]
            if confidences:
                f.write("📊 Confidence Statistics:\n")
                f.write("-" * 80 + "\n")
                f.write(f"Average Confidence: {np.mean(confidences):.3f}\n")
                f.write(f"Confidence Standard Deviation: {np.std(confidences):.3f}\n")
                f.write(f"Minimum Confidence: {np.min(confidences):.3f}\n")
                f.write(f"Maximum Confidence: {np.max(confidences):.3f}\n\n")

                # Confidence distribution
                confidence_bins = {
                    "0.9-1.0": 0, "0.8-0.9": 0, "0.7-0.8": 0,
                    "0.6-0.7": 0, "0.5-0.6": 0, "0.4-0.5": 0,
                    "0.3-0.4": 0, "0.2-0.3": 0, "0.0-0.2": 0
                }

                for conf in confidences:
                    if conf >= 0.9:
                        confidence_bins["0.9-1.0"] += 1
                    elif conf >= 0.8:
                        confidence_bins["0.8-0.9"] += 1
                    elif conf >= 0.7:
                        confidence_bins["0.7-0.8"] += 1
                    elif conf >= 0.6:
                        confidence_bins["0.6-0.7"] += 1
                    elif conf >= 0.5:
                        confidence_bins["0.5-0.6"] += 1
                    elif conf >= 0.4:
                        confidence_bins["0.4-0.5"] += 1
                    elif conf >= 0.3:
                        confidence_bins["0.3-0.4"] += 1
                    elif conf >= 0.2:
                        confidence_bins["0.2-0.3"] += 1
                    else:
                        confidence_bins["0.0-0.2"] += 1

                f.write("Confidence Distribution:\n")
                for bin_range, count in confidence_bins.items():
                    if count > 0:
                        percentage = count / len(confidences) * 100
                        f.write(f"  {bin_range}: {count} items ({percentage:.1f}%)\n")

                # Vote consistency statistics
                unanimous = sum(1 for dp in results if dp.prediction and dp.prediction.vote_counts and max(
                    dp.prediction.vote_counts.values()) == 6)
                majority_5 = sum(1 for dp in results if dp.prediction and dp.prediction.vote_counts and max(
                    dp.prediction.vote_counts.values()) == 5)
                majority_4 = sum(1 for dp in results if dp.prediction and dp.prediction.vote_counts and max(
                    dp.prediction.vote_counts.values()) == 4)

                f.write(f"\nVote Consistency Statistics:\n")
                f.write(f"  Unanimous (6:0): {unanimous} items ({(unanimous / len(results) * 100):.1f}%)\n")
                f.write(f"  Absolute Majority (5:1): {majority_5} items ({(majority_5 / len(results) * 100):.1f}%)\n")
                f.write(f"  Strong Majority (4:2): {majority_4} items ({(majority_4 / len(results) * 100):.1f}%)\n")

            f.write("\n" + "=" * 80 + "\n")

        print(f"✅ Detailed analysis report saved to: {report_file}")

    def print_statistics(self, results: List[DataPoint]):
        """Print statistics"""
        print(f"\n{'=' * 60}")
        print("📊 ERNIE-4.5 Model - Six Professional Role Classification Statistics")
        print(f"{'=' * 60}")

        print(f"Total Data Processed: {len(results)} requirements")
        print(f"Number of Roles Used: {len(self.multi_role_client.roles)} professional roles")

        if self.start_time and self.end_time:
            total_duration = self.end_time - self.start_time
            print(f"\n⏰ Time Statistics:")
            print(f"  Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  End Time: {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Total Runtime: {total_duration}")

            if len(results) > 0:
                avg_time_per_req = total_duration / len(results)
                print(f"  Average Processing Time per Requirement: {avg_time_per_req}")

        # Final label distribution
        final_labels = [dp.final_label for dp in results if dp.final_label]
        if final_labels:
            label_counts = {}
            for label in final_labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            print(f"\n🏷️ Final Classification Distribution:")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(results) * 100
                print(f"  {label}: {count} items ({percentage:.1f}%)")

        # Confidence statistics
        confidences = [dp.prediction.confidence for dp in results if dp.prediction]
        if confidences:
            print(f"\n🎯 Confidence Statistics (calculated according to paper formula):")
            print(f"  Average Confidence: {np.mean(confidences):.3f}")
            print(f"  Confidence Standard Deviation: {np.std(confidences):.3f}")

            # Calculate vote consistency statistics
            unanimous_count = sum(1 for dp in results if dp.prediction and dp.prediction.vote_counts and max(
                dp.prediction.vote_counts.values()) == 6)
            high_confidence_count = sum(1 for dp in results if dp.prediction and dp.prediction.confidence >= 0.8)
            medium_confidence_count = sum(
                1 for dp in results if dp.prediction and 0.6 <= dp.prediction.confidence < 0.8)
            low_confidence_count = sum(1 for dp in results if dp.prediction and dp.prediction.confidence < 0.6)

            print(f"\n📈 Confidence Distribution:")
            print(f"  High Confidence (≥0.8): {high_confidence_count} items ({(high_confidence_count / len(results) * 100):.1f}%)")
            print(
                f"  Medium Confidence (0.6-0.8): {medium_confidence_count} items ({(medium_confidence_count / len(results) * 100):.1f}%)")
            print(f"  Low Confidence (<0.6): {low_confidence_count} items ({(low_confidence_count / len(results) * 100):.1f}%)")
            print(f"  Unanimous (6:0): {unanimous_count} items ({(unanimous_count / len(results) * 100):.1f}%)")


class DataLoader:
    """Data loader class"""

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
            print(f"❌ Error loading categories file {os.path.basename(file_path)}: {e}")
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

            print(f"✅ Successfully loaded {len(requirements_examples)} categories with requirement examples from {os.path.basename(file_path)}")
            return requirements_examples

        except Exception as e:
            print(f"❌ Error loading requirement examples file {os.path.basename(file_path)}: {e}")
            return {}

    @staticmethod
    def load_test_requirements(file_path: str) -> List[str]:
        """Load requirements from dataset file"""
        try:
            df = pd.read_excel(file_path)
            requirements = []

            if 'requirement' in df.columns:
                requirements = [
                    str(row['requirement']).strip()
                    for _, row in df.iterrows()
                    if pd.notna(row['requirement']) and str(row['requirement']).strip()
                ]
            else:
                first_col = df.columns[0]
                requirements = [
                    str(row[first_col]).strip()
                    for _, row in df.iterrows()
                    if pd.notna(row[first_col]) and str(row[first_col]).strip()
                ]

            print(f"✅ Successfully loaded {len(requirements)} test requirements from {os.path.basename(file_path)}")
            return requirements
        except Exception as e:
            print(f"❌ Error loading test requirements file {os.path.basename(file_path)}: {e}")
            return []


def main():
    """Main function - using multi-role method"""
    # File path configuration
    dataset_file = "dataset.xlsx"
    concept_file = "1123Concept.xlsx"
    examples_file = "1122RequirementExamples.xlsx"
    output_file = "ernie_6_roles_confidence.xlsx"

    data_loader = DataLoader()

    print("=" * 60)
    print("🎭 ERNIE-4.5 Model - Six Professional Role Software Requirement Classification System")
    print("Based on Research Paper Method: Six Professional Roles Voting + Dual Confidence Calculation")
    print("Confidence Formula: C_final = min(1.0, C_base + B_gap)")
    print("Where: C_base = V_top/N, B_gap = min(0.2, (V_top - V_second)×0.1)")
    print("=" * 60)

    print("\n📁 Loading data...")
    print(f"  Dataset file: {dataset_file}")
    print(f"  Concept file: {concept_file}")
    print(f"  Examples file: {examples_file}")
    print(f"  Output file: {output_file}")

    # Create multi-role ERNIE-4.5 model client
    try:
        multi_role_client = MultiRoleERNIEClient(
            name="ERNIE-4.5-6Roles",
            model_name="baidu/ERNIE-4.5-300B-A47B"
        )
    except Exception as e:
        print(f"❌ Failed to create multi-role model client: {e}")
        return

    # Load categories, explanations, and examples
    category_explanations = data_loader.load_categories_and_explanations(concept_file)
    requirements_examples = data_loader.load_requirements_examples(examples_file)
    labels = list(category_explanations.keys())
    test_requirements = data_loader.load_test_requirements(dataset_file)

    if not test_requirements:
        print("❌ No test requirements found, program exiting")
        return

    if not labels:
        print("❌ No classification labels found, program exiting")
        return

    print(f"\n✅ Data loading completed:")
    print(f"  Classification labels: {len(labels)}")
    print(f"  Test requirements: {len(test_requirements)}")

    slash_labels = [label for label in labels if '/' in label]
    if slash_labels:
        print(f"  🔍 Labels containing '/': {', '.join(slash_labels)}")

    # Execute multi-role model processing
    print(f"\n🚀 Starting ERNIE-4.5 model six-role classification processing...")
    processor = MultiRoleProcessor(multi_role_client)
    results = processor.process_dataset(test_requirements, labels, category_explanations, requirements_examples)

    # Save and display results
    processor.save_results(results, output_file)
    processor.print_statistics(results)

    # Display detailed information for the first few results
    print(f"\n🔍 Detailed information for the first 3 results:")
    for i, data_point in enumerate(results[:3]):
        print(f"\n{'=' * 60}")
        print(f"{i + 1}. Requirement: {data_point.content[:100]}...")
        if data_point.prediction:
            print(f"   Final classification: {data_point.final_label}")
            print(f"   Final confidence: {data_point.prediction.confidence:.3f}")

            if data_point.prediction.role_votes:
                print(f"\n   Individual role votes:")
                for role, vote in data_point.prediction.role_votes.items():
                    print(f"     {role}: {vote}")

            if data_point.prediction.calculation_details:
                details = data_point.prediction.calculation_details
                print(f"   Confidence calculation process:")
                print(f"     V_top = {details['v_top']}, V_second = {details['v_second']}")
                print(f"     C_base = {details['v_top']}/6 = {details['c_base']:.2f}")
                print(f"     B_gap = min(0.2, ({details['v_top']}-{details['v_second']})×0.1) = {details['b_gap']:.2f}")
                print(
                    f"     C_final = min(1.0, {details['c_base']:.2f} + {details['b_gap']:.2f}) = {details['final_confidence']:.3f}")

    print(f"\n{'=' * 60}")
    print("🎉 ERNIE-4.5 Model six-role classification processing completed!")
    print(f"📊 Results file: {output_file}")
    print(f"📄 Detailed report: {output_file.replace('.xlsx', '_detailed_report.txt')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
