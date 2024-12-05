'use client'

import { WritingTask } from "@/components/WritingTask"

const SAMPLE_RESPONSES = {
  task: `Some people believe that it is better to live in a big city, while others prefer to live in the countryside.

Discuss both views and give your own opinion.

Write at least 250 words.`,

  evaluation: `Here's my evaluation of your response:

Task Achievement: 7.0
✓ Clear position throughout
✓ Main ideas well-developed
⚠ Some supporting examples could be more specific

Coherence and Cohesion: 7.0
✓ Clear overall progression
✓ Effective paragraphing
✓ Good use of linking devices

Lexical Resource: 6.5
✓ Good range of vocabulary
✓ Topic-specific language used well
⚠ Few imprecise word choices

Grammatical Range and Accuracy: 7.0
✓ Mix of simple and complex structures
✓ Good control of grammar
⚠ Minor errors do not impede communication

Would you like specific suggestions for improvement?`
}

export default function WritingTask2Page() {
  return (
    <WritingTask
      taskNumber={2}
      timeLimit={2400}
      minWords={250}
      sampleTask={SAMPLE_RESPONSES.task}
      sampleEvaluation={SAMPLE_RESPONSES.evaluation}
    />
  )
} 