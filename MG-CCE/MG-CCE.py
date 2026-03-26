```python
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI
from volcenginesdkarkruntime import Ark
import time
import random
from collections import Counter

# Load environment variables
load_dotenv()


@dataclass
class ModelPrediction:
    """Store model prediction results"""
    label: str
    confidence: float
    model_name: str
    normalized_rank: float = None
    vote_details: Dict[str, int] = None  # New: vote details


@dataclass
class DataPoint:
    """Data point class"""
    content: str
    predictions: List[ModelPrediction] = None
    final_label: str = None

    def __post_init__(self):
        if self.predictions is None:
            self.predictions = []


class LLMClient:
    """LLM client class - with role-based voting"""

    def __init__(self, name: str, model_name: str, method_code: str, role: str = None):
        self.name = name
        self.model_name = model_name
        self.method_code = method_code
        self.role = role  # New: role identity
        self.client = self._create_client()

        # ERNIE model rate limiting configuration
        self.is_ernie = "ERNIE" in name.upper()
        self.last_request_time = 0
        self.min_request_interval = 2.0

    def _create_client(self):
        """Create client based on method column"""
        # Special handling for Doubao model
        if "Ark(api_key=" in self.method_code:
            api_key_env = self.method_code.split('os.getenv("')[1].split('")')[0]
            return Ark(api_key=os.getenv(api_key_env))

        # Other models use OpenAI compatible interface
        else:
            # Parse api_key environment variable name
            api_key_env = None
            if 'api_key=os.getenv(' in self.method_code:
                api_key_part = self.method_code.split('api_key=os.getenv(')[1].split(')')[0]
                api_key_env = api_key_part.strip('"\'')

            # Parse base_url environment variable name
            base_url_env = None
            if 'base_url=os.getenv(' in self.method_code:
                base_url_part = self.method_code.split('base_url=os.getenv(')[1].split(')')[0]
                base_url_env = base_url_part.strip('"\'')

            api_key = os.getenv(api_key_env) if api_key_env else None
            base_url = os.getenv(base_url_env) if base_url_env else None

            if not api_key:
                raise ValueError(f"API key environment variable not found: {api_key_env}")

            return OpenAI(api_key=api_key, base_url=base_url)

    def _rate_limit_delay(self):
        """ERNIE model rate limiting delay"""
        if self.is_ernie:
            current_time = time.time()
            time_since_last_request = current_time - self.last_request_time
            if time_since_last_request < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last_request
                print(f"  ERNIE rate limit: waiting {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
            self.last_request_time = time.time()

    def predict_with_role_voting(self, text: str, labels: List[str],
                                 category_explanations: Dict[str, str] = None,
                                 requirements_examples: Dict[str, List[str]] = None) -> ModelPrediction:
        """Role-based voting prediction with 5 votes"""

        # ERNIE model rate limiting
        self._rate_limit_delay()

        # Create role prompts
        role_prompts = self._create_role_prompts(category_explanations)

        # Conduct 5 votes with different roles
        votes = []
        vote_details = {}  # Record detailed information for each vote

        print(f"  {self.name} starting 5 role-based votes...")

        for i, (role_key, role_info) in enumerate(list(role_prompts.items())[:5]):  # Only take first 5 roles
            for attempt in range(3):  # Retry mechanism
                try:
                    role_name = role_info['name']
                    role_prompt = role_info['prompt']

                    prediction = self._single_prediction_with_role(
                        text, labels, role_name, role_prompt, f"vote_{i + 1}"
                    )

                    if prediction and prediction != "Analysis failed":
                        votes.append(prediction)
                        vote_details[f"{role_name}_vote_{i + 1}"] = prediction
                        print(f"    {role_name} vote: {prediction}")
                        break
                    else:
                        print(f"    {role_name} attempt {attempt + 1} failed")
                        time.sleep(1)
                except Exception as e:
                    print(f"    {role_name} vote exception: {str(e)}")
                    time.sleep(2)

            # Delay between votes
            time.sleep(0.5)

        # Calculate confidence based on voting results
        if votes:
            confidence = self._calculate_confidence_from_votes(votes)
            vote_counts = Counter(votes)
            final_label = vote_counts.most_common(1)[0][0]

            print(f"  Voting results: {dict(vote_counts)}")
            print(f"  Final classification: {final_label}")
            print(f"  Confidence: {confidence:.3f}")

            return ModelPrediction(
                label=final_label,
                confidence=confidence,
                model_name=f"{self.name}({self.role})" if self.role else self.name,
                vote_details=dict(vote_counts)  # Save vote details
            )
        else:
            print("  All votes failed")
            return ModelPrediction(
                label=labels[0] if labels else "Unknown",
                confidence=0.1,
                model_name=f"{self.name}({self.role})" if self.role else self.name,
                vote_details={}
            )

    def _create_role_prompts(self, category_explanations: Dict[str, str]) -> Dict:
        """Create role prompts"""
        explanations = ""
        if category_explanations:
            for label, explanation in category_explanations.items():
                explanations += f"- {label}: {explanation}\n"

        role_prompts = {
            'product_owner': {
                'name': 'Product Owner',
                'prompt': f"""As a Product Owner, you need to analyze requirements from the perspective of product value and business goals. Based on the following classification standards:

{explanations}

Please analyze the following requirement description from the perspective of product roadmap and market demands, and determine the most appropriate category.

Please reply directly with the category name without adding any other content."""
            },
            'business_analyst': {
                'name': 'Business Analyst',
                'prompt': f"""As a Business Analyst, you need to analyze requirements from the perspective of business processes and functional requirements. Based on the following classification standards:

{explanations}

Please analyze the following requirement description from the perspective of business process optimization and functional implementation, and determine the most appropriate category.

Please reply directly with the category name without adding any other content."""
            },
            'system_architect': {
                'name': 'System Architect',
                'prompt': f"""As a System Architect, you need to analyze requirements from the perspective of technical architecture and system design. Based on the following classification standards:

{explanations}

Please analyze the following requirement description from the perspective of technical feasibility, system scalability, and architectural impact, and determine the most appropriate category.

Please reply directly with the category name without adding any other content."""
            },
            'ux_designer': {
                'name': 'User Experience Designer',
                'prompt': f"""As a User Experience Designer, you need to analyze requirements from the perspective of user interaction and interface design. Based on the following classification standards:

{explanations}

Please analyze the following requirement description from the perspective of user experience, interface design, and user satisfaction, and determine the most appropriate category.

Please reply directly with the category name without adding any other content."""
            },
            'software_tester': {
                'name': 'Software Tester',
                'prompt': f"""As a Software Tester, you need to analyze requirements from the perspective of test feasibility and quality assurance. Based on the following classification standards:

{explanations}

Please analyze the following requirement description from the perspective of test coverage, quality requirements, and validation methods, and determine the most appropriate category.

Please reply directly with the category name without adding any other content."""
            }
        }

        return role_prompts

    def _single_prediction_with_role(self, text: str, labels: List[str],
                                     role_name: str, role_prompt: str, vote_name: str) -> str:
        """Single role-based vote prediction"""
        try:
            full_prompt = role_prompt + f'\nPlease analyze the following requirement description: "{text}"'

            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=10
            )

            result = completion.choices[0].message.content.strip()
            cleaned_result = self._extract_label_from_response(result, labels)
            return cleaned_result

        except Exception as e:
            print(f"    {vote_name}({role_name}) prediction failed: {str(e)}")
            return "Analysis failed"

    def _calculate_confidence_from_votes(self, votes: List[str]) -> float:
        """Calculate confidence based on voting results"""
        if not votes or len(votes) < 3:  # At least 3 valid votes needed
            return 0.1

        vote_counts = Counter(votes)
        total_votes = len(votes)

        # Get the two most frequent categories
        most_common = vote_counts.most_common(2)

        if len(most_common) == 0:
            return 0.1
        elif len(most_common) == 1:
            # Only one category was voted, confidence based on consistency
            top_count = most_common[0][1]
            return top_count / total_votes
        else:
            # Two or more categories, calculate relative advantage
            top1_count, top2_count = most_common[0][1], most_common[1][1]
            advantage_ratio = (top1_count - top2_count) / total_votes
            base_confidence = top1_count / total_votes

            # Combine base confidence and relative advantage
            confidence = base_confidence * (1 + advantage_ratio)
            return min(1.0, max(0.1, confidence))

    def _extract_label_from_response(self, response: str, labels: List[str]) -> str:
        """Extract label from response"""
        response_clean = response.strip().strip('.,!?;:"')

        # Exact match
        if response_clean in labels:
            return response_clean

        # Partial match
        for label in labels:
            if label in response_clean:
                return label

        # Default to first label
        return labels[0] if labels else "Unknown"


class DataLoader:
    """Data loader class"""

    @staticmethod
    def load_categories_and_explanations(file_path: str) -> Dict[str, str]:
        """Load categories and explanations from briefConcept file"""
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            return {
                str(row['category']).strip(): str(row['explanation']).strip()
                for _, row in df.iterrows()
                if pd.notna(row['category'])
            }
        except Exception as e:
            print(f"Error loading categories file: {e}")
            return {}

    @staticmethod
    def load_requirements_examples(file_path: str) -> Dict[str, List[str]]:
        """Load requirement examples from RequirementExamples file"""
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            requirements_examples = {}

            for _, row in df.iterrows():
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

            print(f"Successfully loaded {len(requirements_examples)} categories with requirement examples")
            return requirements_examples

        except Exception as e:
            print(f"Error loading requirement examples file: {e}")
            return {}

    @staticmethod
    def load_test_requirements(file_path: str) -> List[str]:
        """Load test requirements from sentencesForTest file"""
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            return [
                str(row['requirement']).strip()
                for _, row in df.iterrows()
                if 'requirement' in row and pd.notna(row['requirement'])
            ]
        except Exception as e:
            print(f"Error loading test requirements file: {e}")
            return []

    @staticmethod
    def load_llm_clients_from_excel(file_path: str) -> List[LLMClient]:
        """Load LLM clients from Excel file - supports role assignment"""
        try:
            df = pd.read_excel(file_path, sheet_name='Sheet1')
            print(f"Excel file columns: {list(df.columns)}")

            # Try different column name combinations
            possible_name_cols = ['模型名称', '模型名', 'name', 'Name', '模型']
            possible_model_cols = ['model', 'Model', '模型型号', '模型类型']
            possible_method_cols = ['method', 'Method', '调用方法', '方法']
            possible_role_cols = ['role', 'Role', '角色', '身份']  # New: role column

            # Find actual column names
            name_col, model_col, method_col, role_col = None, None, None, None

            for col in df.columns:
                if col in possible_name_cols:
                    name_col = col
                elif col in possible_model_cols:
                    model_col = col
                elif col in possible_method_cols:
                    method_col = col
                elif col in possible_role_cols:
                    role_col = col

            if not name_col or not model_col or not method_col:
                print(f"Required columns not found. Available columns: {list(df.columns)}")
                if len(df.columns) >= 3:
                    name_col = df.columns[0]
                    model_col = df.columns[1]
                    method_col = df.columns[2]
                    print(f"Using default columns: {name_col}, {model_col}, {method_col}")
                else:
                    return []

            clients = []

            for _, row in df.iterrows():
                name = str(row[name_col]).strip() if pd.notna(row[name_col]) else ""
                model_name = str(row[model_col]).strip() if pd.notna(row[model_col]) else ""
                method_code = str(row[method_col]).strip() if pd.notna(row[method_col]) else ""
                role = str(row[role_col]).strip() if role_col and pd.notna(row[role_col]) else None

                if name and model_name and method_code:
                    try:
                        client = LLMClient(name, model_name, method_code, role)
                        clients.append(client)
                        role_info = f"({role})" if role else ""
                        print(f"✓ Successfully loaded model: {name}{role_info}")
                    except Exception as e:
                        print(f"✗ Failed to load model {name}: {e}")

            return clients

        except Exception as e:
            print(f"Error loading model configuration file: {e}")
            return []


class ChainAlgorithm:
    """Chain algorithm main class - using role-based voting"""

    def __init__(self, llm_clients: List[LLMClient]):
        """Initialize chain algorithm"""
        self.llm_clients = llm_clients
        self.chain_length = len(llm_clients)
        print(f"Chain algorithm initialized, chain length: {self.chain_length}")

    def save_intermediate_results(self, data_points: List[DataPoint], chain_step: int,
                                  client_name: str, output_prefix: str = "ChainAlgorithm_Intermediate_Results"):
        """Save intermediate results to file"""
        output_file = f"{output_prefix}_Chain{chain_step}_{client_name}.xlsx"

        results_data = []
        for i, data_point in enumerate(data_points):
            row_data = {
                'Index': i + 1,
                'Requirement Content': data_point.content,
                'Current Chain Step': chain_step,
                'Current Model': client_name
            }

            # Add all completed prediction results
            for j, pred in enumerate(data_point.predictions):
                row_data[f'Model_{j + 1}_Name'] = pred.model_name
                row_data[f'Model_{j + 1}_Predicted_Label'] = pred.label
                row_data[f'Model_{j + 1}_Confidence'] = pred.confidence
                if pred.vote_details:
                    row_data[f'Model_{j + 1}_Vote_Details'] = str(pred.vote_details)
                if pred.normalized_rank is not None:
                    row_data[f'Model_{j + 1}_Normalized_Rank'] = pred.normalized_rank

            if data_point.final_label:
                row_data['Final_Label'] = data_point.final_label

            results_data.append(row_data)

        df = pd.DataFrame(results_data)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"✓ Chain {chain_step} intermediate results saved to: {output_file}")

    def process_dataset(self, dataset: List[str], labels: List[str],
                        category_explanations: Dict[str, str] = None,
                        requirements_examples: Dict[str, List[str]] = None) -> List[DataPoint]:
        """Process entire dataset - using role-based voting method"""
        data_points = [DataPoint(content=text) for text in dataset]

        print(f"Starting to process {len(data_points)} software requirements using chain architecture with {self.chain_length} models...")
        print(f"Using classification explanations: {len(category_explanations) if category_explanations else 0}")

        current_data = data_points.copy()

        for i, client in enumerate(self.llm_clients, 1):
            print(f"\nChain Step {i}/{self.chain_length} ({client.name}) processing {len(current_data)} data items...")

            # Current chain step processes data
            for j, data_point in enumerate(current_data):
                if j % 5 == 0 and j > 0:  # Display progress every 5 items
                    print(f"  Progress: {j}/{len(current_data)}")

                # Use role-based voting for prediction
                prediction = client.predict_with_role_voting(
                    data_point.content, labels, category_explanations, requirements_examples
                )
                data_point.predictions.append(prediction)

            # Save intermediate results
            self.save_intermediate_results(data_points, i, client.name)

            # Data routing
            if i < self.chain_length:
                confidences = [dp.predictions[-1].confidence for dp in current_data]

                pass_ratio = (self.chain_length - i) / self.chain_length
                min_pass_count = max(1, int(len(data_points) * 0.1))
                pass_count = max(min_pass_count, int(len(current_data) * pass_ratio))

                print(f"  Pass ratio: {pass_ratio:.1%} ({pass_count} items)")

                # Sort by confidence from low to high, pass low confidence data
                sorted_indices = np.argsort(confidences)
                next_chain_data = [current_data[idx] for idx in sorted_indices[:pass_count]]
                current_data = next_chain_data

                print(f"  Data passed to chain step {i + 1}: {len(current_data)}")

                if not current_data:
                    print("  No data to pass to the next chain step, terminating chain processing early")
                    break

        # Apply rank-based ensemble method
        self._rank_based_ensemble(data_points)
        return data_points

    def _rank_based_ensemble(self, data_points: List[DataPoint]):
        """Rank-based ensemble method"""
        for chain_idx in range(self.chain_length):
            chain_confidences = []
            valid_data_points = []

            for data_point in data_points:
                if len(data_point.predictions) > chain_idx:
                    confidence = data_point.predictions[chain_idx].confidence
                    chain_confidences.append(confidence)
                    valid_data_points.append(data_point)

            if not chain_confidences:
                continue

            ranks = np.argsort(np.argsort(chain_confidences)) + 1
            normalized_ranks = ranks / len(chain_confidences)

            for idx, data_point in enumerate(valid_data_points):
                data_point.predictions[chain_idx].normalized_rank = normalized_ranks[idx]

        # Final decision: select prediction with highest normalized rank
        for data_point in data_points:
            if not data_point.predictions:
                continue

            best_rank = -1
            best_label = None

            for prediction in data_point.predictions:
                if prediction.normalized_rank is not None and prediction.normalized_rank > best_rank:
                    best_rank = prediction.normalized_rank
                    best_label = prediction.label

            if best_label:
                data_point.final_label = best_label
            else:
                self._weighted_voting_fallback(data_point)

    def _weighted_voting_fallback(self, data_point: DataPoint):
        """Fallback method: weighted voting"""
        predictions = {}
        for pred in data_point.predictions:
            if pred.label not in predictions:
                predictions[pred.label] = 0
            predictions[pred.label] += pred.confidence

        if predictions:
            data_point.final_label = max(predictions.items(), key=lambda x: x[1])[0]

    def save_results(self, results: List[DataPoint], output_file: str):
        """Save final results to Excel file"""
        results_data = []

        for i, data_point in enumerate(results):
            row_data = {
                'Index': i + 1,
                'Requirement Content': data_point.content,
                'Final Classification Label': data_point.final_label
            }

            # Add predictions from each model
            for j, pred in enumerate(data_point.predictions):
                row_data[f'Model_{j + 1}_Name'] = pred.model_name
                row_data[f'Model_{j + 1}_Predicted_Label'] = pred.label
                row_data[f'Model_{j + 1}_Confidence'] = pred.confidence
                if pred.vote_details:
                    row_data[f'Model_{j + 1}_Vote_Details'] = str(pred.vote_details)
                if pred.normalized_rank is not None:
                    row_data[f'Model_{j + 1}_Normalized_Rank'] = pred.normalized_rank

            results_data.append(row_data)

        df = pd.DataFrame(results_data)
        df.to_excel(output_file, index=False, engine='openpyxl')
        print(f"\n✅ Final results saved to: {output_file}")

    def print_statistics(self, results: List[DataPoint]):
        """Print statistics"""
        print(f"\n📊 Chain Algorithm Statistics:")
        print(f"Total data processed: {len(results)}")

        final_labels = [dp.final_label for dp in results if dp.final_label]
        if final_labels:
            label_counts = {}
            for label in final_labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            print(f"\nFinal classification distribution:")
            for label, count in sorted(label_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {label}: {count} items ({count / len(results):.1%})")


def main():
    """Main function"""
    # File path configuration
    brief_concept_file = "1119Concept.xlsx"
    requirement_examples_file = "1119RequirementExamples.xlsx"
    test_requirements_file = "120sentencesForTest.xlsx"
    model_config_file = "模型调用方法对比.xlsx"
    output_file = "ChainAlgorithm_Classification_Results_RoleVoting1119.xlsx"

    data_loader = DataLoader()

    print("Loading data...")
    random.seed(42)

    # Load data
    test_df, concept_df, examples_df = load_data_files()
    print(f"Successfully loaded {len(test_df)} test data items")
    print(f"Successfully loaded {len(concept_df)} classification definitions")

    # Learn classification concepts
    category_explanations = data_loader.load_categories_and_explanations(brief_concept_file)
    requirements_examples = data_loader.load_requirements_examples(requirement_examples_file)
    labels = list(category_explanations.keys())
    test_requirements = data_loader.load_test_requirements(test_requirements_file)

    if not test_requirements:
        print("No test requirements found, program exiting")
        return

    if not labels:
        print("No classification labels found, program exiting")
        return

    # Load LLM clients
    llm_clients = data_loader.load_llm_clients_from_excel(model_config_file)
    if not llm_clients:
        print("No available LLM clients, program exiting")
        return

    print(f"\n✅ Data loading completed:")
    print(f"Available classification labels: {len(labels)}")
    print(f"Test requirements count: {len(test_requirements)}")
    print(f"Available models: {len(llm_clients)}")

    # Execute chain algorithm
    print(f"\n🚀 Starting chain algorithm processing (role-based voting method)...")
    chain_algo = ChainAlgorithm(llm_clients)
    results = chain_algo.process_dataset(test_requirements, labels, category_explanations, requirements_examples)

    # Save and display results
    chain_algo.save_results(results, output_file)
    chain_algo.print_statistics(results)

    # Display detailed information for the first few results
    print(f"\n🔍 Detailed information for the first 3 results:")
    for i, data_point in enumerate(results[:3]):
        print(f"\n{i + 1}. {data_point.content[:60]}...")
        print(f"   Final classification: {data_point.final_label}")
        for pred in data_point.predictions:
            vote_info = f" Votes: {pred.vote_details}" if pred.vote_details else ""
            rank_info = f" (Rank: {pred.normalized_rank:.3f})" if pred.normalized_rank is not None else ""
            print(f"   {pred.model_name}: {pred.label} (Confidence: {pred.confidence:.3f}{rank_info}{vote_info})")


def load_data_files():
    """Load data files"""
    test_df = pd.read_excel('120sentencesForTest.xlsx', sheet_name='Sheet1')
    concept_df = pd.read_excel('1119Concept.xlsx', sheet_name='Sheet1')
    examples_df = pd.read_excel('1119RequirementExamples.xlsx', sheet_name='Sheet1')
    return test_df, concept_df, examples_df


if __name__ == "__main__":
    main()
```
