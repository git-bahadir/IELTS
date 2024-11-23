import anthropic
import json
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import traceback

ANTHROPIC_API_KEY='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'
MODEL = "claude-3-opus-20240229"

class IELTSWritingAgent:
    # Class constants
    MINIMUM_WORDS = 150
    MAXIMUM_TOKENS = 1500
    TEMPERATURE = 0.7
    
    VISUAL_TYPES = [
        "bar graph",
        "line graph", 
        "pie chart",
        "mixed charts"
    ]
    
    def __init__(self):
        """Initialize the IELTS Writing Task 1 agent"""
        self.anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Get paths using os.path for better cross-platform compatibility
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        # Load templates and samples
        templates_path = os.path.join(parent_dir, 'writing', 'ielts_templates_writing.json')
        samples_path = os.path.join(parent_dir, 'writing', 'writing_samples.json')

        try:
            with open(templates_path, 'r') as f:
                self.templates = json.load(f)
            with open(samples_path, 'r') as f:
                self.samples = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required data files not found: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON data: {e}")
            
        self.current_question = None

    def get_new_question(self, visual_type=None):
        """
        Get a new IELTS Writing Task 1 question, utilizing existing samples for better structure
        """
        if visual_type and visual_type not in self.VISUAL_TYPES:
            raise ValueError(f"Invalid visual type. Must be one of: {', '.join(self.VISUAL_TYPES)}")
        
        if not visual_type:
            visual_type = random.choice(self.VISUAL_TYPES)
        
        # Get sample questions of the same type for reference
        relevant_samples = [
            q for q in self.samples['ielts_writing_task_1']['question_examples'] 
            if q['type'] == visual_type
        ]
        
        # Create a more detailed prompt using sample structure
        sample_question = random.choice(relevant_samples) if relevant_samples else None
        
        prompt = f"""You are an IELTS Writing Task 1 question generator. Create a new unique question for visual type: {visual_type}.

        Here's a sample question structure to follow:
        {json.dumps(sample_question, indent=2) if sample_question else 'No sample available'}

        Create a new question following these requirements:
        1. Use similar level of detail as the sample
        2. Include realistic numerical data
        3. Provide 6-8 key features
        4. Ensure data consistency
        5. Use appropriate time periods/categories
        6. Maintain academic style

        IMPORTANT: Return ONLY the JSON object with no additional text or explanation.
        Use this exact format:
        {{
            "description": "Write a detailed description of what the {visual_type} shows",
            "details": {{
                "time_span": {{
                    "start_year": "YYYY",
                    "end_year": "YYYY"
                }},
                "categories": ["category1", "category2"],
                "measurements": {{
                    "unit": "specify unit",
                    "range": "specify range"
                }}
            }},
            "data": {{
                "title": "Title for the {visual_type}",
                "x_axis": {{
                    "label": "X-axis label",
                    "categories": ["category1", "category2"]
                }},
                "y_axis": {{
                    "label": "Y-axis label",
                    "range": [min, max],
                    "unit": "unit"
                }},
                "series": [
                    {{
                        "name": "series name",
                        "values": [value1, value2],
                        "categories": ["category1", "category2"]
                    }}
                ]
            }},
            "key_features": [
                "Key feature 1",
                "Key feature 2"
            ],
            "expected_analysis": [
                "Analysis point 1",
                "Analysis point 2"
            ]
        }}"""

        if visual_type == "mixed charts":
            prompt += """
            You are an IELTS Writing Task 1 question generator. Create a new unique question showing two related charts (one bar chart and one line graph) about the same topic.

            The response must be a valid JSON object with this exact structure:
            {{
                "description": "Write a detailed description of what the two charts show",
                "details": {{
                    "time_span": {{
                        "start_year": "YYYY",
                        "end_year": "YYYY"
                    }},
                    "categories": ["category1", "category2", "category3"],
                    "measurements": {{
                        "unit": "specify unit",
                        "range": "specify range"
                    }}
                }},
                "data": {{
                    "title": "Overall title for both charts",
                    "x_axis": {{
                        "label": "Time period"
                    }},
                    "y_axis": {{
                        "label": "Values (specify unit)"
                    }},
                    "series": [
                        {{
                            "name": "Bar Chart Title",
                            "type": "bar",
                            "categories": ["2019", "2020", "2021"],
                            "values": [100, 150, 200]
                        }},
                        {{
                            "name": "Line Graph Title",
                            "type": "line",
                            "categories": ["2019", "2020", "2021"],
                            "values": [80, 120, 160]
                        }}
                    ]
                }},
                "key_features": [
                    "Key feature 1",
                    "Key feature 2",
                    "Key feature 3"
                ]
            }}

            Create realistic data that shows clear trends or patterns. Make sure both charts are related and tell a coherent story.
            IMPORTANT: Return ONLY the JSON object with no additional text."""


        try:
            response = self.anthropic.messages.create(
                model=MODEL,
                max_tokens=self.MAXIMUM_TOKENS,
                temperature=self.TEMPERATURE,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            
            # Extract text content from Claude's response
            response_text = response.content[0].text if isinstance(response.content, list) else response.content.text
            
            # Find and extract just the JSON portion
            try:
                # Try to find JSON object between curly braces
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    question_data = json.loads(json_str)
                else:
                    # If no curly braces found, try to parse the whole response
                    question_data = json.loads(response_text)
            except json.JSONDecodeError:
                # If both attempts fail, print the response for debugging
                print("Failed to parse JSON. Raw response:")
                print(response_text)
                raise
            
            # Generate visualization
            visual = self._generate_visualization(visual_type, question_data)
            
            # Store current question with additional metadata
            self.current_question = {
                "type": visual_type,
                "data": question_data,
                "visual": visual,
                "sample_reference": sample_question['id'] if sample_question else None,
                "expected_band_descriptors": self._get_band_descriptors_for_type(visual_type)
            }
            
            # Display question and visual
            self._display_question()
            
            return self._format_question_display()

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print("Raw response:", response_text)
            return "Error: Invalid JSON response from AI"
        except ValueError as e:
            print(f"Error validating question: {e}")
            print("Raw response:", response_text)
            return "Error: Invalid question structure"
        except Exception as e:
            print(f"Unexpected error: {e}")
            print("Raw response:", response_text)
            return "Error generating question. Please try again."

    def _parse_and_validate_question(self, response_content):
        """
        Parse and validate the question data from Claude's response
        """
        try:
            # Parse JSON response
            if isinstance(response_content, str):
                question_data = json.loads(response_content)
            else:
                question_data = response_content

            # Required fields and their types
            required_fields = {
                'description': str,
                'details': dict,
                'data': dict,
                'key_features': list
            }

            # Validate required fields and their types
            for field, field_type in required_fields.items():
                if field not in question_data:
                    raise ValueError(f"Missing required field: {field}")
                if not isinstance(question_data[field], field_type):
                    raise ValueError(f"Invalid type for field {field}: expected {field_type.__name__}")

            # Validate data structure for visualization
            data = question_data['data']
            if 'series' not in data:
                raise ValueError("Missing 'series' in data")
            if not isinstance(data['series'], list) or not data['series']:
                raise ValueError("'series' must be a non-empty list")

            # Validate each series
            for series in data['series']:
                if 'name' not in series:
                    raise ValueError("Missing 'name' in series")
                if 'values' not in series:
                    raise ValueError("Missing 'values' in series")
                if 'categories' not in series:
                    raise ValueError("Missing 'categories' in series")
                if len(series['values']) != len(series['categories']):
                    raise ValueError(f"Mismatch between values and categories length in series '{series['name']}'")

            return question_data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")

    def _get_band_descriptors_for_type(self, visual_type):
        """
        Get relevant band descriptors for the specific visual type
        """
        # Get band descriptors from templates
        descriptors = self.templates['writing']['types_of_questions']['academic_writing_task_1']['band_descriptors']
        
        # Filter/customize descriptors based on visual type if needed
        return descriptors

    def _format_question_display(self):
        """
        Format the question display with clear instructions and guidance
        """
        question = self.current_question
        
        formatted_output = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                           IELTS Writing Task 1                              ║
╠════════════════════════════════════════════════════════════════════════════╣
║ Type: {question['type']}
║
║ Task:
║ {question['data']['description']}
║
║ Key Features to Consider:
{self._format_key_features(question['data']['key_features'])}
║
║ Remember:
║ • Write at least {self.MINIMUM_WORDS} words
║ • Spend about 20 minutes on this task
║ • Include an overview of the main trends/features
║ • Select and compare key information
║ • Use appropriate language to describe the {question['type']}
╚═════════════════════════════════════════════════════════════════════════╝
"""
        return formatted_output

    def _format_key_features(self, features):
        """Helper method to format key features nicely"""
        return ''.join([f"║ • {feature}\n" for feature in features])

    def evaluate_answer(self, answer_text):
        """
        Evaluate a submitted answer for the current question
        """
        if not self.current_question:
            return "Error: No current question found. Please get a new question first."

        # Validate word count
        word_count = len(answer_text.split())
        if word_count < self.MINIMUM_WORDS:
            return f"Your answer is too short. Minimum {self.MINIMUM_WORDS} words required. Current word count: {word_count}"

        prompt = f"""Evaluate this IELTS Writing Task 1 answer. 
        
        Question type: {self.current_question['type']}
        Question description: {self.current_question['data']['description']}

        Student's answer:
        {answer_text}

        Provide a detailed evaluation including:
        1. Overall band score (0-9)
        2. Analysis of:
           - Task Achievement (how well they described the key features)
           - Coherence and Cohesion (organization and flow)
           - Lexical Resource (vocabulary usage)
           - Grammatical Range and Accuracy
        3. Specific examples from their writing to justify the score
        4. Suggestions for improvement
        
        Make the feedback constructive and specific."""

        response = self.anthropic.messages.create(
            model=MODEL,
            max_tokens=self.MAXIMUM_TOKENS,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._format_feedback(response.content)

    def _generate_visualization(self, visual_type, question_data):
        """Generate visualization based on type and data"""
        plt.style.use('classic')
        plt.close('all')
        
        try:
            if visual_type == "bar graph":
                fig = self._generate_bar_graph(question_data)
            elif visual_type == "line graph":
                fig = self._generate_line_graph(question_data)
            elif visual_type == "pie chart":
                fig = self._generate_pie_chart(question_data)
            elif visual_type == "mixed charts":
                fig = self._generate_mixed_charts(question_data)
            # elif visual_type == "map":
                # fig = self._generate_map_comparison(question_data)
            
            if fig:
                plt.show()
                return fig
            return None
        except Exception as e:
            print(f"Error in visualization generation: {e}")
            return None

    def _generate_line_graph(self, question_data):
        """Generate a line graph"""
        try:
            data = question_data['data']
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot each series
            for series in data['series']:
                ax.plot(series['categories'], series['values'], 
                       marker='o', label=series['name'])
            
            # Customize the graph
            ax.set_title(data['title'], pad=20)
            ax.set_xlabel(data['x_axis']['label'])
            ax.set_ylabel(data['y_axis']['label'])
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Rotate x-axis labels if needed
            plt.xticks(rotation=45)
            
            # Add legend
            ax.legend()
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error generating line graph: {e}")
            return None

    def _generate_bar_graph(self, question_data):
        """Generate a bar graph"""
        try:
            data = question_data['data']
            # Create figure with white background
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
            ax.set_facecolor('white')
            
            # Get series data
            series = data['series']
            
            # Print debug information
            print("Debug - Series data:", series)
            
            # Convert data to proper format
            categories = series[0]['categories']
            values = series[0]['values']
            
            # Create x positions for bars
            x = np.arange(len(categories))
            
            # Create bars
            bars = ax.bar(x, values, width=0.6)
                
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:,.1f}',
                       ha='center', va='bottom')
            
            # Customize the graph
            ax.set_title(data.get('title', ''), pad=20, fontsize=12, fontweight='bold')
            ax.set_xlabel(data.get('x_axis', {}).get('label', ''), fontsize=10)
            ax.set_ylabel(data.get('y_axis', {}).get('label', ''), fontsize=10)
            
            # Set x-axis labels
            ax.set_xticks(x)
            ax.set_xticklabels(categories, rotation=45, ha='right')
            
            # Add grid for better readability
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            print(f"Error generating bar graph: {e}")
            print("Debug - Question data:", json.dumps(question_data, indent=2))
            traceback.print_exc()
            return None

    def _generate_pie_chart(self, question_data):
        """Generate a pie chart"""
        try:
            data = question_data['data']
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get data from first series
            series = data['series'][0]
            
            # Create pie chart
            wedges, texts, autotexts = ax.pie(series['values'], 
                                             labels=series['categories'],
                                             autopct='%1.1f%%',
                                             startangle=90)
            
            # Customize the chart
            ax.set_title(data['title'], pad=20)
            
            # Make text more readable
            plt.setp(autotexts, size=8, weight="bold")
            plt.setp(texts, size=8)
            
            # Equal aspect ratio ensures that pie is drawn as a circle
            ax.axis('equal')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            print(f"Error generating pie chart: {e}")
            return None

    def _generate_mixed_charts(self, question_data):
        """Generate multiple charts (e.g., combination of bar and line)"""
        try:
            # Debug prints
            print("\n=== Debug Information for Mixed Charts ===")
            print("Full question_data structure:")
            print(json.dumps(question_data, indent=2))
            
            data = question_data['data']
            
            # Create figure and axes with white background
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='white')
            ax1.set_facecolor('white')
            ax2.set_facecolor('white')
            
            # Debug print series data
            print("\nSeries data:")
            print(json.dumps(data.get('series', []), indent=2))
            
            series = data.get('series', [])
            if not series:
                print("Warning: No series data found")
                return None

            # First chart (left side)
            if len(series) > 0:
                print("\nProcessing first chart...")
                first_series = series[0]
                print(f"First series data: {first_series}")
                
                categories = first_series['categories']
                values = first_series['values']
                x = np.arange(len(categories))
                
                # Create bars
                bars = ax1.bar(x, values, width=0.6, color='skyblue')
                ax1.set_title(f"{first_series['name']}", pad=20)
                ax1.set_xticks(x)
                ax1.set_xticklabels(categories, rotation=45, ha='right')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:,.1f}',
                            ha='center', va='bottom')

            # Second chart (right side)
            if len(series) > 1:
                print("\nProcessing second chart...")
                second_series = series[1]
                print(f"Second series data: {second_series}")
                
                categories = second_series['categories']
                values = second_series['values']
                
                # Create line plot
                x_points = np.arange(len(categories))
                ax2.plot(x_points, values, marker='o', linestyle='-', linewidth=2, color='forestgreen')
                ax2.set_title(f"{second_series['name']}", pad=20)
                ax2.set_xticks(x_points)
                ax2.set_xticklabels(categories, rotation=45, ha='right')
                
                # Add value labels on points
                for i, value in enumerate(values):
                    ax2.text(i, value, f'{value:,.1f}', 
                            ha='center', va='bottom')

            # Customize both charts
            for ax in [ax1, ax2]:
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.set_xlabel(data.get('x_axis', {}).get('label', ''))
                ax.set_ylabel(data.get('y_axis', {}).get('label', ''))
                
                # Remove top and right spines
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            # Add overall title
            fig.suptitle(data.get('title', ''), fontsize=12, fontweight='bold', y=1.05)
            
            # Adjust layout
            plt.tight_layout()
            
            print("\nVisualization generation completed")
            return fig
            
        except Exception as e:
            print(f"\nError generating mixed charts: {e}")
            traceback.print_exc()
            return None

    def _display_question(self):
        """Display the current question and visualization"""
        print("\n=== IELTS Writing Task 1 ===")
        print("\nQuestion:")
        print(self.current_question['data']['description'])
        print("\nKey Features to Consider:")
        for i, feature in enumerate(self.current_question['data']['key_features'], 1):
            print(f"{i}. {feature}")
        
        # Display visualization if available
        if self.current_question['visual']:
            plt.show()  # This should display the graph

    def _format_feedback(self, feedback):
        """Format the evaluation feedback in a clear, readable way"""
        formatted = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                           IELTS Writing Task 1 Evaluation                     ║
╠═══════════════════════════════════════════════════════════════════════════════╣
{}
╚══════════════════════════════════════════��═══════════════════════════════════╝
"""
        # Format the feedback content to fit within the box
        content = feedback.replace('\n', '\n║ ')
        return formatted.format(content)

# Initialize the agent
agent = IELTSWritingAgent()

# Or get a random type
question = agent.get_new_question(visual_type="mixed charts")

# # Later, when the student has written their answer:
# answer = """
# [Student's answer text here]
# """
# feedback = agent.evaluate_answer(answer)
# print(feedback)