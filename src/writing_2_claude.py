import anthropic
import json
import random
import os
from dotenv import load_dotenv
import logging
import traceback
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate API key
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

MODEL = "claude-3-5-sonnet-20241022"

class IELTSWritingTask2Agent:
    # Class constants
    MINIMUM_WORDS = 250
    MAXIMUM_TOKENS = 2000
    TEMPERATURE = 0.7
    
    QUESTION_TYPES = [
        "agree_disagree",
        "discuss_both_views",
        "advantages_disadvantages",
        "problem_solution",
        "positive_negative"
    ]
    
    def __init__(self):
        """Initialize the IELTS Writing Task 2 agent"""
        self.anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Get paths using os.path for better cross-platform compatibility
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        try:
            # Load templates and samples
            with open(os.path.join(parent_dir, 'standards', 'ielts_templates.json'), 'r') as f:
                self.templates = json.load(f)
            with open(os.path.join(parent_dir, 'writing', 'writing_2_samples.json'), 'r') as f:
                self.samples = json.load(f)
            with open(os.path.join(parent_dir, 'standards', 'cefr_standards.json'), 'r') as f:
                self.standards = json.load(f)
        except FileNotFoundError as e:
            logger.error(f"Required data file not found: {e}")
            self.templates = {}
            self.samples = {}
            self.standards = {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON data: {e}")
            self.templates = {}
            self.samples = {}
            self.standards = {}
        
        self.current_question = None

    def generate_question(self, question_type=None):
        """Generate a new IELTS Writing Task 2 question"""
        if not question_type:
            question_type = random.choice(self.QUESTION_TYPES)
            
        # Get sample questions of the same type
        sample_questions = [q for q in self.samples['ielts_writing_task_2']['question_examples'] 
                          if self._determine_question_type(q['description']) == question_type]
        
        prompt = self._create_question_prompt(question_type, sample_questions)
        
        try:
            response = self.anthropic.messages.create(
                model=MODEL,
                max_tokens=self.MAXIMUM_TOKENS,
                temperature=self.TEMPERATURE,
                messages=[{"role": "user", "content": prompt}]
            )
            
            question_data = self._parse_question_response(response)
            self.current_question = question_data
            
            return self._format_question_display()
            
        except Exception as e:
            logger.error(f"Error generating question: {e}")
            return None

    def evaluate_answer(self, answer_text):
        """Evaluate a submitted answer for the current question"""
        if not self.current_question:
            return "Error: No current question found. Please generate a question first."
        
        word_count = len(answer_text.split())
        if word_count < self.MINIMUM_WORDS:
            return f"Your answer is too short. Minimum {self.MINIMUM_WORDS} words required. Current word count: {word_count}"

        # First prompt: Get numerical scores only
        score_prompt = f"""You are an IELTS examiner. Evaluate this Writing Task 2 answer and provide ONLY numerical scores.

        Question: {self.current_question.get('description', '')}

        Student's answer ({word_count} words):
        {answer_text}

        Provide ONLY the scores in exactly this format (nothing else):
        Overall Band Score: [0.0-9.0]
        Task Response: [0.0-9.0]
        Coherence and Cohesion: [0.0-9.0]
        Lexical Resource: [0.0-9.0]
        Grammatical Range and Accuracy: [0.0-9.0]"""

        # Second prompt: Get detailed feedback
        feedback_prompt = f"""Now provide detailed feedback for this IELTS Writing Task 2 answer.

        Question: {self.current_question.get('description', '')}

        Student's answer ({word_count} words):
        {answer_text}

        Provide feedback in exactly this format:

        Key Strengths:
        • [specific strength 1]
        • [specific strength 2]
        • [specific strength 3]

        Areas for Improvement:
        • [specific area 1]
        • [specific area 2]
        • [specific area 3]

        Detailed Analysis:
        [Provide a paragraph-by-paragraph analysis of the essay]"""

        try:
            # Get scores
            score_response = self.anthropic.messages.create(
                model=MODEL,
                max_tokens=self.MAXIMUM_TOKENS,
                messages=[{"role": "user", "content": score_prompt}]
            )
            
            # Get feedback
            feedback_response = self.anthropic.messages.create(
                model=MODEL,
                max_tokens=self.MAXIMUM_TOKENS,
                messages=[{"role": "user", "content": feedback_prompt}]
            )

            # Extract content
            score_content = score_response.content[0].text if isinstance(score_response.content, list) else score_response.content
            feedback_content = feedback_response.content[0].text if isinstance(feedback_response.content, list) else feedback_response.content

            # Parse scores
            scores = self._parse_scores(score_content)
            
            # Parse feedback
            feedback = self._parse_feedback(feedback_content)

            # Combine results
            evaluation = {**scores, **feedback}

            # Format the evaluation
            return self._format_evaluation(evaluation)

        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            traceback.print_exc()
            return f"Error evaluating answer: {str(e)}"

    def _determine_question_type(self, description):
        """Determine the type of question based on its description"""
        text = ' '.join(description).lower()
        
        if "agree or disagree" in text:
            return "agree_disagree"
        elif "discuss both" in text:
            return "discuss_both_views"
        elif "advantages and disadvantages" in text:
            return "advantages_disadvantages"
        elif "positive or negative" in text:
            return "positive_negative"
        elif "problem" in text and "solution" in text:
            return "problem_solution"
        else:
            return "discuss_both_views"  # default type

    def _create_question_prompt(self, question_type, sample_questions):
        """Create a prompt for generating a new question"""
        sample = random.choice(sample_questions) if sample_questions else None
        
        prompt = f"""You are an IELTS Writing Task 2 question generator. Create a new unique question of type: {question_type}.

        Here's a sample question structure:
        {json.dumps(sample['description'], indent=2) if sample else 'No sample available'}

        Create a novel question following these requirements:
        1. Follow the {question_type} format but with a completely new topic
        2. Choose contemporary and engaging topics that test candidates' ability to:
           - Present and develop arguments
           - Show balanced viewpoints
           - Provide relevant examples
        3. Avoid controversial or culturally sensitive topics
        4. Include clear instructions about word count (minimum 250 words)
        5. Maintain academic style and clarity

        Return the question in this JSON format:
        {{
            "question_type": "{question_type}",
            "description": [
                "You should spend about 40 minutes on this task.",
                "Write about the following topic:",
                "YOUR_QUESTION_HERE",
                "YOUR_TASK_INSTRUCTION_HERE",
                "Give reasons for your answer and include any relevant examples from your own knowledge or experience.",
                "Write at least 250 words."
            ],
            "expected_elements": [
                "introduction with clear position",
                "main argument points",
                "supporting examples",
                "counter-arguments (if applicable)",
                "conclusion"
            ],
            "key_themes": [
                "theme1",
                "theme2"
            ],
            "suggested_structure": {{
                "introduction": "What should be included",
                "body_paragraphs": "How to organize main points",
                "conclusion": "How to summarize"
            }}
        }}"""

        return prompt

    def _create_evaluation_prompt(self, answer_text):
        """Create a prompt for evaluating an answer"""
        # Try to get sample answers, use empty list if not available
        try:
            sample_answers = self.samples.get('ielts_writing_task_2', {}).get('answer_examples', [])
            calibration_examples = [
                ans for ans in sample_answers 
                if ans.get('band_score') in [6.5, 7.0, 7.5]
            ][:2]  # Use 2 examples for calibration
        except (KeyError, AttributeError):
            calibration_examples = []
        
        # Get band descriptors with safe access
        try:
            band_descriptors = self.templates.get('writing', {}).get('types_of_questions', {}).get(
                'academic_writing_task_2', {}).get('band_descriptors', {})
        except (KeyError, AttributeError):
            band_descriptors = {}
        
        # Format band descriptors for the prompt if available
        band_descriptor_text = ""
        if band_descriptors:
            band_descriptor_text = "\nBand Descriptors:\n"
            for criterion, levels in band_descriptors.items():
                band_descriptor_text += f"\n{criterion.title()}:\n"
                for band, descriptor in levels.items():
                    band_descriptor_text += f"Band {band}: {descriptor}\n"
        
        prompt = (
            "You are an experienced IELTS examiner evaluating a Writing Task 2 answer.\n\n"
            f"Question:\n{self.current_question.get('description', '')}\n\n"
            f"Student's answer ({len(answer_text.split())} words):\n{answer_text}\n\n"
            + """Evaluate using official IELTS criteria. You MUST provide numerical scores for each criterion.
            
            Your response MUST include these exact sections with numerical scores (0.0-9.0):

            Overall Band Score: [numerical score]

            Detailed Scores:
            Task Response: [numerical score]
            Coherence and Cohesion: [numerical score]
            Lexical Resource: [numerical score]
            Grammatical Range and Accuracy: [numerical score]

            Then provide:
            Strengths:
            • [specific strength 1]
            • [specific strength 2]
            • [specific strength 3]

            Areas for Improvement:
            • [specific area 1]
            • [specific area 2]
            • [specific area 3]

            Detailed Analysis:
            [Provide a detailed paragraph-by-paragraph analysis]

            Remember:
            1. All scores MUST be numerical values between 0.0 and 9.0
            2. Scores must be in 0.5 increments (e.g., 6.0, 6.5, 7.0)
            3. Be specific and detailed in your feedback
            4. Provide concrete examples from the text"""
        )

        return prompt

    def _format_evaluation(self, evaluation):
        """Format the evaluation response with detailed feedback"""
        try:
            formatted_report = f"""
╔════════════════════════ IELTS Writing Task 2 Evaluation ════════════════════════╗
║ Overall Assessment
╠════════════════════════════════════════════════════════════════════════════════╣
║ Overall Band Score: {evaluation.get('band_score', 'N/A')}
║
║ Detailed Criteria Scores:
║ ▢ Task Response: {evaluation.get('tr_score', 'N/A')}
║ ▢ Coherence and Cohesion: {evaluation.get('cc_score', 'N/A')}
║ ▢ Lexical Resource: {evaluation.get('lr_score', 'N/A')}
║ ▢ Grammatical Range and Accuracy: {evaluation.get('gra_score', 'N/A')}
╠════════════════════════════════════════════════════════════════════════════════╣
║ Key Strengths
║ {self._format_bullet_points(evaluation.get('strengths', ['No specific strengths identified']))}
╠════════════════════════════════════════════════════════════════════════════════╣
║ Areas for Improvement
║ {self._format_bullet_points(evaluation.get('improvements', ['No specific improvements identified']))}
╠════════════════════════════════════════════════════════════════════════════════╣
║ Detailed Analysis
║ {self._format_paragraph(evaluation.get('detailed_feedback', 'No detailed feedback provided'))}
╚════════════════════════════════════════════════════════════════════════════════╝
"""
            return formatted_report
        
        except Exception as e:
            logger.error(f"Error formatting evaluation: {e}")
            return f"Error formatting evaluation: {str(e)}"

    def _parse_evaluation_content(self, content):
        """Parse the evaluation content into structured sections"""
        try:
            sections = {
                'band_score': None,
                'tr_score': None,
                'cc_score': None,
                'lr_score': None,
                'gra_score': None,
                'strengths': [],
                'improvements': [],
                'detailed_feedback': ''
            }

            # Split content into sections and process each section
            current_section = None
            
            for line in content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Extract scores using more flexible patterns
                score_patterns = {
                    'band_score': r'Overall Band Score:?\s*(\d+\.?\d*)',
                    'tr_score': r'Task Response:?\s*(\d+\.?\d*)',
                    'cc_score': r'Coherence and Cohesion:?\s*(\d+\.?\d*)',
                    'lr_score': r'Lexical Resource:?\s*(\d+\.?\d*)',
                    'gra_score': r'Grammatical Range and Accuracy:?\s*(\d+\.?\d*)'
                }
                
                for score_key, pattern in score_patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        try:
                            sections[score_key] = float(match.group(1))
                        except ValueError:
                            logger.warning(f"Could not convert score to float: {match.group(1)}")
                            
                # Extract strengths and improvements
                if 'Key Strengths' in line or 'Strengths:' in line:
                    current_section = 'strengths'
                    continue
                elif any(x in line for x in ['Areas for Improvement:', 'Improvements:', 'Weaknesses:']):
                    current_section = 'improvements'
                    continue
                elif 'Detailed Analysis:' in line:
                    current_section = 'detailed_feedback'
                    continue
                    
                # Collect bullet points and content
                if current_section:
                    if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                        if current_section in ['strengths', 'improvements']:
                            sections[current_section].append(line.lstrip('•-* ').strip())
                    elif current_section == 'detailed_feedback':
                        sections['detailed_feedback'] += line + '\n'

            # Clean up detailed feedback
            sections['detailed_feedback'] = sections['detailed_feedback'].strip()
            
            return sections

        except Exception as e:
            logger.error(f"Error parsing evaluation content: {e}")
            logger.debug(f"Content being parsed: {content}")
            return {}

    def _calibrate_scores(self, evaluation):
        """Calibrate scores based on IELTS standards and sample answers"""
        try:
            # Add null checks for scores
            scores = {
                'band_score': float(evaluation.get('band_score', 0) or 0),
                'tr_score': float(evaluation.get('tr_score', 0) or 0),
                'cc_score': float(evaluation.get('cc_score', 0) or 0),
                'lr_score': float(evaluation.get('lr_score', 0) or 0),
                'gra_score': float(evaluation.get('gra_score', 0) or 0)
            }
            
            # Get reference scores from sample answers
            sample_answers = self.samples.get('ielts_writing_task_2', {}).get('answer_examples', [])
            
            def adjust_score(score, criterion):
                if score == 0:  # If no score was provided
                    return None
                    
                # Get comparable sample answers
                comparable_samples = [
                    s for s in sample_answers 
                    if abs(float(s.get('band_score', 0)) - score) <= 1.0
                ]
                
                if not comparable_samples:
                    return score
                    
                # Calculate average difference for this criterion
                criterion_diffs = []
                for sample in comparable_samples:
                    criterion_score = sample.get('criteria_scores', {}).get(criterion)
                    if criterion_score is not None:
                        criterion_diffs.append(
                            float(criterion_score) - float(sample.get('band_score', 0))
                        )
                
                if not criterion_diffs:
                    return score
                    
                # Apply adjustment based on historical differences
                avg_diff = sum(criterion_diffs) / len(criterion_diffs)
                adjusted_score = round((score + avg_diff) * 2) / 2  # Round to nearest 0.5
                
                # Ensure score is within valid range
                return max(4.0, min(9.0, adjusted_score))
            
            # Adjust individual scores
            calibrated = {
                'band_score': scores['band_score'],
                'tr_score': adjust_score(scores['tr_score'], 'task_response'),
                'cc_score': adjust_score(scores['cc_score'], 'coherence_cohesion'),
                'lr_score': adjust_score(scores['lr_score'], 'lexical_resource'),
                'gra_score': adjust_score(scores['gra_score'], 'grammatical_range')
            }
            
            # Recalculate overall band score if all component scores are present
            if all(calibrated[k] is not None for k in ['tr_score', 'cc_score', 'lr_score', 'gra_score']):
                weighted_score = (
                    calibrated['tr_score'] * 0.3 +
                    calibrated['cc_score'] * 0.3 +
                    calibrated['lr_score'] * 0.2 +
                    calibrated['gra_score'] * 0.2
                )
                calibrated['band_score'] = round(weighted_score * 2) / 2  # Round to nearest 0.5
                
            return calibrated
            
        except Exception as e:
            logger.error(f"Error calibrating scores: {e}")
            return None

    def _format_bullet_points(self, items):
        """Format a list of items as bullet points"""
        return ''.join([f"║ • {item}\n" for item in items])

    def _format_paragraph(self, text):
        """Format a paragraph of text to fit within the evaluation box"""
        max_width = 76  # Maximum width for the content
        words = text.split()
        lines = []
        current_line = "║ "
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_width:
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = "║ " + word + " "
        
        if current_line:
            lines.append(current_line)
        
        return '\n'.join(lines)

    def _analyze_essay_structure(self, answer_text):
        """Analyze the structure of the essay"""
        paragraphs = [p.strip() for p in answer_text.split('\n\n') if p.strip()]
        
        analysis = {
            'paragraph_count': len(paragraphs),
            'has_introduction': False,
            'has_conclusion': False,
            'body_paragraphs': [],
            'cohesive_devices': [],
            'structure_score': 0
        }
        
        if len(paragraphs) >= 3:
            analysis['has_introduction'] = self._is_introduction(paragraphs[0])
            analysis['has_conclusion'] = self._is_conclusion(paragraphs[-1])
            analysis['body_paragraphs'] = paragraphs[1:-1]
            
            # Find cohesive devices
            analysis['cohesive_devices'] = self._find_cohesive_devices(answer_text)
            
            # Calculate structure score
            analysis['structure_score'] = self._calculate_structure_score(analysis)
        
        return analysis

    def _is_introduction(self, paragraph):
        """Check if paragraph is an introduction"""
        intro_markers = [
            'nowadays', 'recently', 'in recent years', 'it is often said',
            'many people believe', 'there is a growing concern',
            'there is considerable discussion', 'it is widely believed'
        ]
        
        para_lower = paragraph.lower()
        
        # Check for introduction markers
        has_marker = any(marker in para_lower for marker in intro_markers)
        
        # Check if it's presenting the topic
        presents_topic = len(paragraph.split()) >= 25 and '?' in paragraph
        
        return has_marker or presents_topic

    def _is_conclusion(self, paragraph):
        """Check if paragraph is a conclusion"""
        conclusion_markers = [
            'in conclusion', 'to conclude', 'to sum up', 'in summary',
            'overall', 'to summarize', 'in my opinion', 'finally'
        ]
        
        para_lower = paragraph.lower()
        return any(marker in para_lower for marker in conclusion_markers)

    def _find_cohesive_devices(self, text):
        """Find and categorize cohesive devices used in the text"""
        cohesive_devices = {
            'sequence': ['firstly', 'secondly', 'finally', 'next', 'then'],
            'addition': ['furthermore', 'moreover', 'in addition', 'also', 'besides'],
            'contrast': ['however', 'nevertheless', 'although', 'despite', 'while'],
            'example': ['for example', 'for instance', 'such as', 'particularly'],
            'result': ['therefore', 'thus', 'consequently', 'as a result'],
            'summary': ['in conclusion', 'to sum up', 'overall', 'in summary']
        }
        
        found_devices = {category: [] for category in cohesive_devices}
        text_lower = text.lower()
        
        for category, devices in cohesive_devices.items():
            for device in devices:
                if device in text_lower:
                    found_devices[category].append(device)
        
        return found_devices

    def _calculate_structure_score(self, analysis):
        """Calculate a score for essay structure (0-10)"""
        score = 0
        
        # Basic structure points
        if analysis['has_introduction']:
            score += 2
        if analysis['has_conclusion']:
            score += 2
        if analysis['paragraph_count'] >= 4:
            score += 2
        
        # Cohesive devices points
        device_count = sum(len(devices) for devices in analysis['cohesive_devices'].values())
        score += min(2, device_count / 5)  # Max 2 points for cohesive devices
        
        # Body paragraph analysis
        if len(analysis['body_paragraphs']) >= 2:
            score += 2
        
        return min(10, score)  # Cap at 10

    def generate_improvement_suggestions(self, answer_text, analysis_results):
        """Generate specific improvement suggestions using LLM"""
        
        prompt = f"""As an IELTS Writing Task 2 expert tutor, provide specific, actionable improvement suggestions for this essay.

        Original Question:
        {json.dumps(self.current_question['description'], indent=2)}

        Student's Answer:
        {answer_text}

        Essay Analysis:
        - Structure Score: {analysis_results['structure_score']}/10
        - Paragraph Count: {analysis_results['paragraph_count']}
        - Has Introduction: {analysis_results['has_introduction']}
        - Has Conclusion: {analysis_results['has_conclusion']}
        - Cohesive Devices Used: {json.dumps(analysis_results['cohesive_devices'], indent=2)}

        Provide detailed suggestions in these categories:
        1. Structure and Organization
        2. Argument Development
        3. Language Enhancement
        4. Academic Style
        5. Specific Examples and Evidence

        For each suggestion:
        - Identify the specific issue
        - Provide a concrete example of improvement
        - Show before/after comparisons where relevant
        - Include sample phrases or expressions

        Format your response as specific, actionable advice that the student can immediately apply.
        Focus on the most impactful improvements first."""

        try:
            response = self.anthropic.messages.create(
                model=MODEL,
                max_tokens=self.MAXIMUM_TOKENS,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._format_improvement_suggestions(response)
            
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}")
            return None

    def _format_improvement_suggestions(self, response):
        """Format the improvement suggestions in a clear, structured way"""
        content = response.content[0].text if isinstance(response.content, list) else response.content.text
        
        formatted_suggestions = f"""
╔═══════════════════ IELTS Writing Task 2 Improvement Guide ══════════════════╗
║                         Personalized Suggestions                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
{self._format_suggestion_section(content)}
╚════════════════════════════════════════════════════════════════════════════╝
"""
        return formatted_suggestions

    def generate_sample_improvements(self, answer_text, weak_areas):
        """Generate specific sample improvements for weak areas"""
        
        prompt = f"""As an IELTS Writing Task 2 expert, provide specific examples of how to improve these weak areas in the essay:
        {json.dumps(weak_areas, indent=2)}

        Original essay excerpt:
        {answer_text[:500]}... # First 500 characters for context

        For each weak area:
        1. Show the original text that needs improvement
        2. Provide 2-3 alternative versions with explanations
        3. Include advanced vocabulary and structures that could be used
        4. Explain why the improvements are effective

        Focus on practical, concrete examples that demonstrate:
        - Better academic vocabulary
        - More sophisticated sentence structures
        - Stronger argument development
        - More effective use of examples
        - Better cohesion and flow

        Format as clear before/after comparisons with explanations."""

        try:
            response = self.anthropic.messages.create(
                model=MODEL,
                max_tokens=self.MAXIMUM_TOKENS,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._format_sample_improvements(response)
            
        except Exception as e:
            logger.error(f"Error generating sample improvements: {e}")
            return None

    def generate_vocabulary_suggestions(self, answer_text):
        """Generate vocabulary improvement suggestions"""
        
        prompt = f"""As an IELTS vocabulary expert, analyze this essay and provide specific vocabulary improvements.

        Essay text:
        {answer_text}

        Please provide your suggestions in this exact format:

        Basic Vocabulary Improvements:
        • [original word/phrase] -> [better academic alternative] (Example usage)
        • [original word/phrase] -> [better academic alternative] (Example usage)

        Topic-Specific Vocabulary:
        • [suggested word/phrase] - [definition/usage note]
        • [suggested word/phrase] - [definition/usage note]

        Academic Collocations:
        • [suggested collocation] (Example sentence)
        • [suggested collocation] (Example sentence)

        Remember to:
        1. Focus on words/phrases actually used in the essay
        2. Provide natural, academic alternatives
        3. Include example sentences
        4. Explain why each suggestion would improve the writing

        Your suggestions must be specific and directly related to the essay content."""

        try:
            response = self.anthropic.messages.create(
                model=MODEL,
                max_tokens=self.MAXIMUM_TOKENS,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return self._format_vocabulary_suggestions(response)
            
        except Exception as e:
            logger.error(f"Error generating vocabulary suggestions: {e}")
            return None

    def _format_suggestion_section(self, content):
        """Format a section of suggestions with proper spacing and borders"""
        sections = {
            'Structure and Organization': [],
            'Argument Development': [],
            'Language Enhancement': [],
            'Academic Style': [],
            'Specific Examples': []
        }
        
        current_section = None
        current_content = []
        
        # Parse content into sections
        for line in content.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Check if this is a section header
            for section in sections.keys():
                if section.lower() in line.lower():
                    if current_section and current_content:
                        sections[current_section] = current_content
                    current_section = section
                    current_content = []
                    break
            else:
                if current_section:
                    current_content.append(line)
        
        # Add the last section
        if current_section and current_content:
            sections[current_section] = current_content
        
        # Format each section
        formatted_content = ""
        for section, suggestions in sections.items():
            if suggestions:
                formatted_content += f"║ {section}\n║ {'═' * len(section)}\n"
                for suggestion in suggestions:
                    formatted_content += self._wrap_suggestion_text(suggestion)
                formatted_content += "║\n"
        
        return formatted_content

    def _wrap_suggestion_text(self, text, width=76):
        """Wrap suggestion text to fit within the specified width"""
        import textwrap
        wrapped = textwrap.wrap(text, width=width-4)  # -4 for the border and spacing
        return ''.join([f"║  {line}\n" for line in wrapped])

    def _parse_question_response(self, response):
        """Parse the question response from Claude"""
        try:
            # Extract text content from response
            content = response.content[0].text if isinstance(response.content, list) else response.content.text
            
            # Try to find JSON object between curly braces
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                try:
                    question_data = json.loads(json_str)
                except json.JSONDecodeError:
                    logger.error("Failed to parse JSON from response")
                    logger.debug(f"Attempted to parse: {json_str}")
                    raise
            else:
                logger.error("No JSON object found in response")
                logger.debug(f"Full response: {content}")
                raise ValueError("No valid JSON found in response")

            # Validate required fields
            required_fields = ['question_type', 'description', 'expected_elements']
            for field in required_fields:
                if field not in question_data:
                    raise ValueError(f"Missing required field: {field}")

            return question_data

        except Exception as e:
            logger.error(f"Error parsing question response: {str(e)}")
            logger.debug(f"Full response content: {response.content}")
            raise

    def _format_question_display(self):
        """Format the question for display"""
        if not self.current_question:
            return "No question currently loaded"

        try:
            formatted_output = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           IELTS Writing Task 2                              ║
╠════════════════════════════════════════════════════════════════════════════╣
║ Type: {self.current_question['question_type']}
║
║ Task:
║ {self._format_description(self.current_question['description'])}
║
║ Expected Elements:
{self._format_elements(self.current_question['expected_elements'])}

║ Remember:
║ • Write at least {self.MINIMUM_WORDS} words
║ • Spend about 40 minutes on this task
║ • Plan your response before writing
║ • Include relevant examples and explanations
║ • Check your work when finished
╚════════════════════════════════════════════════════════════════════════════╝
"""
            return formatted_output

        except Exception as e:
            logger.error(f"Error formatting question display: {str(e)}")
            return "Error formatting question"

    def _format_description(self, description):
        """Format the question description"""
        if isinstance(description, list):
            return '\n║ '.join(description)
        return str(description)

    def _format_elements(self, elements):
        """Format the expected elements"""
        if isinstance(elements, list):
            return ''.join([f"║ • {element}\n" for element in elements])
        return f"║ • {str(elements)}\n"

    def _format_vocabulary_suggestions(self, response):
        """Format vocabulary suggestions in a clear structure"""
        try:
            content = response.content[0].text if isinstance(response.content, list) else response.content.text
            
            formatted_output = f"""
╔═══════════════════ Vocabulary Improvement Suggestions ══════════════════╗
║                                                                        ║
{self._format_suggestion_section(content)}
╚════════════════════════════════════════════════════════════════════════╝
"""
            return formatted_output
            
        except Exception as e:
            logger.error(f"Error formatting vocabulary suggestions: {e}")
            return "Error formatting vocabulary suggestions"

    def _parse_scores(self, content):
        """Parse only the numerical scores from content"""
        scores = {
            'band_score': None,
            'tr_score': None,
            'cc_score': None,
            'lr_score': None,
            'gra_score': None
        }
        
        patterns = {
            'band_score': r'Overall Band Score:\s*(\d+\.?\d*)',
            'tr_score': r'Task Response:\s*(\d+\.?\d*)',
            'cc_score': r'Coherence and Cohesion:\s*(\d+\.?\d*)',
            'lr_score': r'Lexical Resource:\s*(\d+\.?\d*)',
            'gra_score': r'Grammatical Range and Accuracy:\s*(\d+\.?\d*)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                try:
                    scores[key] = float(match.group(1))
                except ValueError:
                    logger.warning(f"Could not convert score to float for {key}")
        
        return scores

    def _parse_feedback(self, content):
        """Parse the detailed feedback sections"""
        feedback = {
            'strengths': [],
            'improvements': [],
            'detailed_feedback': ''
        }
        
        # Extract strengths
        strengths_match = re.search(r'Key Strengths:(.*?)(?=Areas for Improvement:|$)', content, re.DOTALL)
        if strengths_match:
            strengths = re.findall(r'[•\-\*]\s*([^\n]+)', strengths_match.group(1))
            feedback['strengths'] = [s.strip() for s in strengths if s.strip()]
        
        # Extract improvements
        improvements_match = re.search(r'Areas for Improvement:(.*?)(?=Detailed Analysis:|$)', content, re.DOTALL)
        if improvements_match:
            improvements = re.findall(r'[•\-\*]\s*([^\n]+)', improvements_match.group(1))
            feedback['improvements'] = [i.strip() for i in improvements if i.strip()]
        
        # Extract detailed feedback
        detailed_match = re.search(r'Detailed Analysis:(.*?)$', content, re.DOTALL)
        if detailed_match:
            feedback['detailed_feedback'] = detailed_match.group(1).strip()
        
        return feedback

def get_user_answer():
    """
    Get answer input from user with proper instructions and formatting.
    Allows for multiple lines of input until user indicates they're done.
    """
    print("\n╔════════════════════════════════════════════════════════════════════════════╗")
    print("║                           Write your answer below                            ║")
    print("║ Instructions:                                                               ║")
    print("║ - Write at least 250 words                                                 ║")
    print("║ - Type your answer, pressing Enter for new lines                           ║")
    print("║ - When finished, type 'DONE' on a new line and press Enter                 ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝\n")

    lines = []
    while True:
        line = input()
        if line.strip().upper() == 'DONE':
            break
        lines.append(line)
    
    return '\n'.join(lines)

def main():
    # Initialize the agent
    agent = IELTSWritingTask2Agent()
    
    # Get a new question
    question = agent.generate_question()
    print("\nGenerated Question:")
    print(question)
    
    # Get user's answer
    print("\nNow, write your answer. When finished, type 'DONE' on a new line.")
    answer = get_user_answer()
    
    # Show word count
    word_count = len(answer.split())
    print(f"\nWord count: {word_count}")
    
    if word_count < agent.MINIMUM_WORDS:
        print(f"\nWarning: Your answer is below the minimum requirement of {agent.MINIMUM_WORDS} words.")
        proceed = input("Would you like to proceed with evaluation anyway? (yes/no): ")
        if proceed.lower() != 'yes':
            print("Evaluation cancelled. Please try again with a longer answer.")
            return

    # Evaluate the answer
    print("\nEvaluating your answer...")
    evaluation = agent.evaluate_answer(answer)
    print(evaluation)
    
    # Generate improvement suggestions
    print("\nGenerating improvement suggestions...")
    analysis_results = agent._analyze_essay_structure(answer)
    suggestions = agent.generate_improvement_suggestions(answer, analysis_results)
    print(suggestions)
    
    # Generate vocabulary suggestions
    print("\nGenerating vocabulary suggestions...")
    vocab_suggestions = agent.generate_vocabulary_suggestions(answer)
    print(vocab_suggestions)

if __name__ == "__main__":
    main()