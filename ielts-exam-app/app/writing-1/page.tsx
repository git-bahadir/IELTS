'use client'

import { WritingTask } from "@/components/WritingTask"

const SAMPLE_RESPONSES = {
  task: `The graph below shows the number of international students studying in different faculties at Smith University in 2019.

Summarize the information by selecting and reporting the main features, and make comparisons where relevant.

Write at least 150 words.`,

  evaluation: `Here's my evaluation of your response:

Task Achievement: 7.0
✓ Covers key features effectively
✓ Clear overview provided
⚠ Some comparisons could be more explicit

Coherence and Cohesion: 6.5
✓ Logical organization
✓ Good use of paragraphing
⚠ Some transitions could be smoother

Lexical Resource: 7.0
✓ Good range of vocabulary
✓ Appropriate academic style
⚠ Few minor word choice issues

Would you like specific suggestions for improvement?`
}

export default function WritingTask1Page() {
  return (
    <WritingTask
      taskNumber={1}
      timeLimit={1200}
      minWords={150}
      sampleTask={SAMPLE_RESPONSES.task}
      sampleEvaluation={SAMPLE_RESPONSES.evaluation}
    />
  )
} 