
import React, { useState } from 'react';
import { InvokeLLM } from '@/integrations/Core';
import { Detection } from '@/entities/Detection';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from '@/components/ui/input';
import { Bot, Loader2, AlertTriangle, CheckCircle, Search } from 'lucide-react';

export default function AnalysisForm({ onAnalysisComplete }) {
  const [inputText, setInputText] = useState('');
  const [inputUrl, setInputUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);

  const handleAnalyze = async (content, platform) => {
    setIsLoading(true);
    setResult(null);

    try {
        const prompt = `You are the 'WatchDog' verbal abuse detection system. Analyze the following text for verbal abuse (like hate speech, explicit language, aggression). Provide your analysis in a JSON format. The text to analyze is: "${content}". Respond ONLY with a JSON object with the following schema: { "is_abusive": boolean, "confidence_score": number (0.0 to 1.0), "detection_type": string ('Explicit', 'Sarcasm', 'Hate Speech', 'Aggressive Tone', or 'None'), "language": "string" ('Hindi', 'Punjabi', 'Hinglish', 'English', or 'Other') }`;
        const llmResponse = await InvokeLLM({
            prompt,
            response_json_schema: {
                type: 'object',
                properties: {
                    is_abusive: { type: 'boolean' },
                    confidence_score: { type: 'number' },
                    detection_type: { type: 'string', enum: ['Explicit', 'Sarcasm', 'Hate Speech', 'Aggressive Tone', 'None'] },
                    language: { type: 'string' }
                }
            }
        });

      if (llmResponse && llmResponse.is_abusive) {
        setResult({ ...llmResponse, status: 'abusive' });
        const newDetection = await Detection.create({
            content: content,
            source_url: platform === 'Manual Input' ? null : content,
            platform: platform,
            confidence_score: llmResponse.confidence_score,
            detection_type: llmResponse.detection_type,
            language: llmResponse.language
        });
        if(onAnalysisComplete) onAnalysisComplete(newDetection);
      } else {
         setResult({ status: 'safe', ...llmResponse });
      }
    } catch (error) {
      console.error("Analysis failed:", error);
      setResult({ status: 'error' });
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleSubmit = (e, type) => {
    e.preventDefault();
    if(type === 'text' && inputText) handleAnalyze(inputText, 'Manual Input');
    if(type === 'url' && inputUrl) handleAnalyze(inputUrl, 'YouTube'); // Assume URL is youtube for simplicity
  }

  return (
    <Card className="dark:bg-gray-950">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bot /> Manual Analysis
        </CardTitle>
        <CardDescription>
          Test the detection model by providing text or a URL.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="text">
            <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="text">Analyze Text</TabsTrigger>
                <TabsTrigger value="url">Analyze URL</TabsTrigger>
            </TabsList>
            <TabsContent value="text">
                <form onSubmit={(e) => handleSubmit(e, 'text')} className="space-y-4 pt-4">
                    <Textarea
                        placeholder="Enter text to analyze..."
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        className="min-h-[100px] dark:bg-gray-900"
                    />
                    <Button type="submit" disabled={isLoading || !inputText}>
                        {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
                        Analyze Text
                    </Button>
                </form>
            </TabsContent>
            <TabsContent value="url">
                <form onSubmit={(e) => handleSubmit(e, 'url')} className="space-y-4 pt-4">
                    <Input
                        placeholder="Enter YouTube or Instagram URL..."
                        value={inputUrl}
                        onChange={(e) => setInputUrl(e.target.value)}
                        className="dark:bg-gray-900"
                    />
                    <Button type="submit" disabled={isLoading || !inputUrl}>
                        {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
                        Analyze URL
                    </Button>
                </form>
            </TabsContent>
        </Tabs>
        
        {result && (
          <div className="mt-4 p-4 rounded-lg border dark:border-gray-700">
            {result.status === 'abusive' && (
              <div className="flex items-center gap-3 text-red-600 dark:text-red-400">
                <AlertTriangle className="h-6 w-6" />
                <div>
                  <p className="font-bold">Abusive Content Detected</p>
                  <p className="text-sm">Type: {result.detection_type}, Confidence: {(result.confidence_score * 100).toFixed(0)}%</p>
                </div>
              </div>
            )}
            {result.status === 'safe' && (
              <div className="flex items-center gap-3 text-green-600 dark:text-green-400">
                <CheckCircle className="h-6 w-6" />
                <div>
                  <p className="font-bold">Content Appears Safe</p>
                  <p className="text-sm">The model did not detect any abuse.</p>
                </div>
              </div>
            )}
             {result.status === 'error' && (
              <div className="flex items-center gap-3 text-yellow-600 dark:text-yellow-400">
                <AlertTriangle className="h-6 w-6" />
                <p className="font-bold">Analysis Failed. Please try again.</p>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
