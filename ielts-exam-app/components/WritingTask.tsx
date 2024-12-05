'use client'

import { useState } from "react"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Timer } from "@/components/Timer"
import { BarChart2, Send, RefreshCw } from "lucide-react"

interface Message {
  id: string
  role: 'system' | 'user' | 'assistant'
  content: string
  isStreaming?: boolean
}

interface WritingTaskProps {
  taskNumber: 1 | 2
  timeLimit: number
  minWords: number
  sampleTask: string
  sampleEvaluation: string
}

export function WritingTask({ 
  taskNumber, 
  timeLimit, 
  minWords, 
  sampleTask, 
  sampleEvaluation 
}: WritingTaskProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [wordCount, setWordCount] = useState(0)
  const [taskStarted, setTaskStarted] = useState(false)

  const streamText = async (text: string, messageId: string) => {
    let currentText = ''
    const chars = text.split('')
    
    for (const char of chars) {
      currentText += char
      setMessages(prev => prev.map(msg => 
        msg.id === messageId 
          ? { ...msg, content: currentText, isStreaming: true }
          : msg
      ))
      await new Promise(resolve => setTimeout(resolve, 15 + Math.random() * 10))
    }

    setMessages(prev => prev.map(msg => 
      msg.id === messageId 
        ? { ...msg, isStreaming: false }
        : msg
    ))
  }

  const handleStartTask = async () => {
    setTaskStarted(true)
    setIsLoading(true)
    
    const messageId = Date.now().toString()
    const systemMessage: Message = {
      id: messageId,
      role: 'system',
      content: '',
      isStreaming: true
    }
    
    setMessages([systemMessage])

    try {
      await streamText(sampleTask, messageId)
    } catch (error) {
      console.error('Error starting task:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = async () => {
    if (!input.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input
    }

    const aiMessageId = (Date.now() + 1).toString()
    const aiMessage: Message = {
      id: aiMessageId,
      role: 'assistant',
      content: '',
      isStreaming: true
    }

    setMessages(prev => [...prev, userMessage, aiMessage])
    setInput('')
    setIsLoading(true)

    try {
      await streamText(sampleEvaluation, aiMessageId)
    } catch (error) {
      console.error('Error submitting response:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const updateWordCount = (text: string) => {
    const words = text.trim().split(/\s+/).filter(word => word.length > 0)
    setWordCount(words.length)
  }

  return (
    <div className="container mx-auto p-4 h-screen">
      <Card className="h-full">
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex justify-between items-center p-4 border-b">
            <div className="flex items-center gap-4">
              <h1 className="text-2xl font-bold">Writing Task {taskNumber}</h1>
              <Badge variant={taskStarted ? "default" : "secondary"}>
                {taskStarted ? "In Progress" : "Not Started"}
              </Badge>
            </div>
            
            <div className="flex items-center gap-4">
              <Timer timeLimit={timeLimit} />
            </div>
          </div>

          {/* Main Content */}
          <div className="flex flex-1 gap-4 p-4 min-h-0">
            {/* Chat/Instructions Area */}
            <div className="flex flex-col w-1/2 gap-4">
              <ScrollArea className="flex-1 p-4 border rounded-lg">
                {messages.map((msg) => (
                  <div
                    key={msg.id}
                    className={`mb-4 p-4 rounded-lg ${
                      msg.role === 'user' 
                        ? 'bg-primary/10 ml-auto max-w-[80%]' 
                        : 'bg-muted max-w-[80%]'
                    }`}
                  >
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                    {msg.isStreaming && (
                      <span className="inline-block w-1 h-4 ml-1 bg-primary animate-pulse" />
                    )}
                  </div>
                ))}
                {isLoading && !messages.some(msg => msg.isStreaming) && (
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <div className="animate-spin">
                      <RefreshCw className="h-4 w-4" />
                    </div>
                    Assistant is thinking...
                  </div>
                )}
              </ScrollArea>
            </div>

            {/* Writing Area */}
            <div className="flex flex-col w-1/2 gap-4">
              <Textarea
                className="flex-1 p-4 resize-none"
                placeholder="Write your response here..."
                value={input}
                onChange={(e) => {
                  setInput(e.target.value)
                  updateWordCount(e.target.value)
                }}
              />
              
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-4">
                  <Badge variant={wordCount >= minWords ? "success" : "secondary"}>
                    {wordCount} words
                  </Badge>
                  <Button variant="outline" size="sm">
                    <BarChart2 className="h-4 w-4 mr-2" />
                    Statistics
                  </Button>
                </div>
                
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    onClick={handleStartTask}
                    disabled={isLoading}
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    New Task
                  </Button>
                  <Button
                    onClick={handleSubmit}
                    disabled={isLoading || wordCount < minWords}
                  >
                    <Send className="h-4 w-4 mr-2" />
                    Submit
                  </Button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
} 