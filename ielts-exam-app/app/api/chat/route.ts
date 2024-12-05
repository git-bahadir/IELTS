import { NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'

export async function POST(req: Request) {
  try {
    const { message, examMode, scriptPath } = await req.json()
    
    // Resolve the absolute path to the script
    const absoluteScriptPath = path.resolve(process.cwd(), scriptPath)
    console.log('Script path:', absoluteScriptPath) // For debugging

    return new Promise((resolve) => {
      const pythonProcess = spawn('python', [absoluteScriptPath, message])
      let result = ''

      pythonProcess.stdout.on('data', (data) => {
        result += data.toString()
      })

      pythonProcess.stderr.on('data', (data) => {
        console.error(`Error: ${data}`)
      })

      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          resolve(NextResponse.json({ error: 'Failed to process request' }, { status: 500 }))
        } else {
          resolve(NextResponse.json({ response: result.trim() }))
        }
      })
    })
  } catch (error) {
    console.error('Error processing request:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
} 