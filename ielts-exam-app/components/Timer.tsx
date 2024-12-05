"use client"

import { useState, useEffect } from 'react'
import { Button } from "@/components/ui/button"

interface TimerProps {
  timeLimit: number
}

export function Timer({ timeLimit }: TimerProps) {
  const [timeRemaining, setTimeRemaining] = useState(timeLimit)
  const [isRunning, setIsRunning] = useState(false)

  useEffect(() => {
    let interval: NodeJS.Timeout

    if (isRunning && timeRemaining > 0) {
      interval = setInterval(() => {
        setTimeRemaining((prev) => {
          if (prev <= 1) {
            setIsRunning(false)
            return 0
          }
          return prev - 1
        })
      }, 1000)
    }

    return () => clearInterval(interval)
  }, [isRunning, timeRemaining])

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
  }

  const handleStart = () => setIsRunning(true)
  const handlePause = () => setIsRunning(false)
  const handleReset = () => {
    setIsRunning(false)
    setTimeRemaining(timeLimit)
  }

  return (
    <div className="flex items-center gap-2">
      <span className="font-mono text-xl">{formatTime(timeRemaining)}</span>
      <div className="flex gap-1">
        {!isRunning ? (
          <Button 
            onClick={handleStart} 
            variant="outline" 
            size="sm"
            disabled={timeRemaining === 0}
          >
            Start
          </Button>
        ) : (
          <Button 
            onClick={handlePause} 
            variant="outline" 
            size="sm"
          >
            Pause
          </Button>
        )}
        <Button 
          onClick={handleReset} 
          variant="outline" 
          size="sm"
          disabled={timeRemaining === timeLimit && !isRunning}
        >
          Reset
        </Button>
      </div>
    </div>
  )
} 