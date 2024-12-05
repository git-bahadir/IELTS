import Link from 'next/link'
import { Button } from "@/components/ui/button"
import { Card, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24 bg-background">
      <div className="w-full max-w-5xl space-y-8">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold tracking-tight">IELTS Practice Exam</h1>
          <p className="text-muted-foreground">Choose an exam mode to start practicing</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Reading Test</CardTitle>
              <CardDescription>60 minutes</CardDescription>
              <Button asChild className="w-full mt-4">
                <Link href="/reading">Start Practice</Link>
              </Button>
            </CardHeader>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Writing Task 1</CardTitle>
              <CardDescription>20 minutes</CardDescription>
              <Button asChild className="w-full mt-4">
                <Link href="/writing-1">Start Practice</Link>
              </Button>
            </CardHeader>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Writing Task 2</CardTitle>
              <CardDescription>40 minutes</CardDescription>
              <Button asChild className="w-full mt-4">
                <Link href="/writing-2">Start Practice</Link>
              </Button>
            </CardHeader>
          </Card>
        </div>
      </div>
    </main>
  )
}
